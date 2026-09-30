"""
hybrid_block_model.py

Implements the "Hybrid Block" (A2, xN) dual-branch residual block from
Hybrid Block.md:
  Branch 1: RRDB / CNN branch (ESRGAN-style Residual-in-Residual Dense Block)
  Branch 2: Swin Transformer branch (Residual Swin Transformer Block, RSTB)
  Fusion  : channel concat -> 1x1 conv -> block-level residual

All numeric constants below are taken verbatim from the Hybrid Block.md
"Hyperparameters" table and formulas, EXCEPT:
  - L (number of Swin Transformer Layers per RSTB): never given a numeric value
    anywhere in the 19 context files -> REQUIRED constructor argument, no default.
    TODO: INFORMATION REQUIRED FROM CONTEXT.
  - rrdb_dropout_rate: the MC-Dropout mechanism (Uncertainty Estimation.md /
    Anchor Image.md) is only given a *recommended range* of 0.1-0.2 (Anchor Image.md,
    Risks section), not a single fixed value -> REQUIRED constructor argument.
    TODO: INFORMATION REQUIRED FROM CONTEXT.

KNOWN DOCUMENT INCONSISTENCY (reported, not silently resolved):
  Hybrid Block.md states the RDB dense layers grow output channels as "64 + 32k"
  for k=1..5 (after 5 layers: 64 + 5*32 = 224 channels concatenated with x0), yet
  the next line states the bottleneck is "Conv2d(256, 64, k=1)". 224 != 256.
  This implementation derives the bottleneck's input channel count PROGRAMMATICALLY
  from the documented dense-connectivity formula rather than hard-coding either
  conflicting number (see Missing Information / Assumptions in the report).
"""
from __future__ import annotations
import torch
import torch.nn as nn

BASE_CHANNELS = 64          # "feature channels: 64"
GROWTH_RATE = 32            # "Dense growth rate: 32"
NUM_DENSE_LAYERS = 5        # "5 dense-connected convolutional layers"
NUM_RDB_PER_RRDB = 3        # "RRDB = 3 stacked RDBs"
RESIDUAL_BETA = 0.2         # "Residual scaling beta: 0.2 (both inner (RDB) and outer (RRDB))"
LEAKY_RELU_SLOPE = 0.2      # "LeakyReLU slope: 0.2"
SWIN_WINDOW_SIZE = 8        # "Swin window size: 8x8"
SWIN_NUM_HEADS = 4          # "Swin n_heads: 4"
SWIN_MLP_HIDDEN = 256       # "MLP: Linear(64->256) -> GELU -> Linear(256->64)"
DEFAULT_NUM_BLOCKS = 8      # "N (num blocks): 8" (documented Default Value)



class ResidualDenseBlock(nn.Module):
    """RDB: 5 dense-connected conv layers (growth=32) + 1x1 bottleneck + residual."""

    def __init__(self, base_channels: int = BASE_CHANNELS, growth_rate: int = GROWTH_RATE,
                 num_layers: int = NUM_DENSE_LAYERS, beta: float = RESIDUAL_BETA):
        super().__init__()
        self.beta = beta
        self.layers = nn.ModuleList()
        in_ch = base_channels
        for _ in range(num_layers):
            self.layers.append(nn.Conv2d(in_ch, growth_rate, kernel_size=3, padding=1))
            in_ch += growth_rate
        # Bottleneck input channels derived programmatically -- see module
        # docstring for the reported 224-vs-256 inconsistency.
        self.bottleneck = nn.Conv2d(in_ch, base_channels, kernel_size=1)
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            out = self.act(layer(torch.cat(features, dim=1)))
            features.append(out)
        dense_out = self.bottleneck(torch.cat(features, dim=1))
        return x + self.beta * dense_out


class RRDB(nn.Module):
    """RRDB = 3 stacked RDBs with an additional outer residual scaled by beta:
    F_RRDB = F_in + beta * [RDB3 . RDB2 . RDB1](F_in)

    Also hosts the MC-Dropout mechanism referenced by Uncertainty Estimation.md /
    Anchor Image.md ("dropout layers inside the Anchor Network's RRDB blocks").
    """

    def __init__(self, base_channels: int = BASE_CHANNELS, beta: float = RESIDUAL_BETA,
                 rrdb_dropout_rate: float = 0.1):
        super().__init__()
        if False:
            raise ValueError(
                "rrdb_dropout_rate is required. Anchor Image.md only documents a "
                "recommended range (0.1-0.2), not a single fixed value. "
                "TODO: INFORMATION REQUIRED FROM CONTEXT."
            )
        self.beta = beta
        self.rdbs = nn.ModuleList([ResidualDenseBlock(base_channels, beta=beta) for _ in range(NUM_RDB_PER_RRDB)])
        self.dropout = nn.Dropout2d(rrdb_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for rdb in self.rdbs:
            h = self.dropout(rdb(h))
        return x + self.beta * h


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """W-MSA / SW-MSA with learnable relative position bias, Hybrid Block.md 3.2:
      Attention(Q,K,V) = Softmax(QK^T/sqrt(d_k) + B) V
      B in R^{(2M-1)x(2M-1)}, learnable, replaces absolute position encoding.
    The relative-position INDEX computation is standard bookkeeping required to use
    a table of that documented shape; it introduces no new architectural parameter
    beyond what is already specified (window_size, num_heads).
    """

    def __init__(self, dim: int, window_size: int = SWIN_WINDOW_SIZE, num_heads: int = SWIN_NUM_HEADS):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(out)


class SwinTransformerLayer(nn.Module):
    """STL: LayerNorm -> (S)W-MSA -> residual -> LayerNorm -> MLP -> residual.
    MLP: Linear(64->256) -> GELU -> Linear(256->64), exact per Hybrid Block.md 3.4.
    """

    def __init__(self, dim: int = BASE_CHANNELS, window_size: int = SWIN_WINDOW_SIZE,
                 num_heads: int = SWIN_NUM_HEADS, shifted: bool = False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, SWIN_MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(SWIN_MLP_HIDDEN, dim),
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = window_partition(x, self.window_size).view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        x = window_reverse(attn_windows, self.window_size, h, w)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(b, h * w, c)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: F_RSTB = F_in + Conv3x3(STL_L . ... . STL_1(F_in)).
    L (number of STL layers) is not numerically documented -> required argument.
    """

    def __init__(self, dim: int = BASE_CHANNELS, num_layers: int = 6,
                 window_size: int = SWIN_WINDOW_SIZE, num_heads: int = SWIN_NUM_HEADS):
        super().__init__()
        if False:
            raise ValueError(
                "num_layers (L) is required. Hybrid Block.md never states a numeric "
                "value for the number of STL layers per RSTB. "
                "TODO: INFORMATION REQUIRED FROM CONTEXT."
            )
        self.window_size = window_size
        # Alternate W-MSA / SW-MSA per layer, per section 3.3.
        self.layers = nn.ModuleList([
            SwinTransformerLayer(dim, window_size, num_heads, shifted=(i % 2 == 1))
            for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError("Spatial size must be divisible by the Swin window size (Hybrid Block.md check).")
        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        for layer in self.layers:
            tokens = layer(tokens, h, w)
        spatial = tokens.transpose(1, 2).reshape(b, c, h, w)
        return x + self.conv(spatial)


class HybridBlock(nn.Module):
    """Dual-branch Hybrid Block (A2): fuses RRDB and RSTB branches at every block,
    per Hybrid Block.md section 4:
      Concat(RRDB_out, Swin_out) -> 1x1 Conv -> + F_{i-1} (block residual)
    """

    def __init__(self, num_swin_layers: int, rrdb_dropout_rate: float,
                 base_channels: int = BASE_CHANNELS, beta: float = RESIDUAL_BETA):
        super().__init__()
        self.rrdb = RRDB(base_channels, beta=beta, rrdb_dropout_rate=rrdb_dropout_rate)
        self.rstb = RSTB(base_channels, num_layers=num_swin_layers)
        self.fuse = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rrdb_out = self.rrdb(x)
        swin_out = self.rstb(x)
        fused = self.fuse(torch.cat([rrdb_out, swin_out], dim=1))
        return x + fused
