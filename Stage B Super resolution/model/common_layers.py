"""
common_layers.py

Shared primitive building blocks reused across multiple TRIAD-SR stages.
Every primitive here is referenced by name in at least one context file:
- "ResBlock" appears in: Latent.md, VAE Decoder.md, Original-Image Cond. Features.md,
  Diffusion U-Net.md (Tech Stack: diffusers ResnetBlock2D "with GroupNorm and SiLU").
- "SelfAttention" appears in: Latent.md, VAE Decoder.md.
- "GroupNorm(32, C)" appears explicitly (num_groups=32) in: Latent.md,
  VAE Decoder.md, Original-Image Cond. Features.md. That constant (32) is reused
  wherever GroupNorm is required elsewhere (e.g. inside the Diffusion U-Net) since
  no other value is ever documented anywhere in the 19 context files.

Where a numeric hyperparameter (e.g. number of attention heads) is not given
anywhere in the context, the parameter is REQUIRED (no default) and flagged with a
TODO comment, per the anti-hallucination rules.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_NORM_GROUPS = 32  # reused constant from Latent.md / VAE Decoder.md / Original-Image Cond. Features.md


class ResBlock(nn.Module):
    """Conv(3x3) -> GroupNorm -> SiLU -> Conv(3x3) -> GroupNorm, + residual.
    Matches the "ResBlock" primitive referenced throughout Latent.md,
    VAE Decoder.md, Original-Image Cond. Features.md, and the Diffusion U-Net.md
    Tech Stack note: "ResnetBlock2D ... With GroupNorm and SiLU".
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(GROUP_NORM_GROUPS, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(GROUP_NORM_GROUPS, out_channels)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class Downsample(nn.Module):
    """Strided convolution downsampling, explicitly named in Latent.md:
    'Downsample(128, 128, stride=2)'."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2x(nn.Module):
    """Upsample by scale=2 followed by conv, per VAE Decoder.md:
    'Upsample(scale=2)' followed by 'Conv2d(...)'."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SelfAttention(nn.Module):
    """Self-attention at a fixed spatial bottleneck, named in Latent.md
    ('SelfAttention(512)') and VAE Decoder.md ('SelfAttention(512)').

    TODO: INFORMATION REQUIRED FROM CONTEXT - num_heads for this attention block is
    not specified anywhere in the 19 context files. Caller MUST supply it.
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, channels)
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)
        tokens = h_in.flatten(2).transpose(1, 2)  # [B, HW, C]
        attn_out, _ = self.mha(tokens, tokens, tokens)
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return x + attn_out


class SinusoidalPositionEmbedding(nn.Module):
    """Timestep.md section 2, exact formula:
    PE(t', 2i)   = sin(t' / 10000^(2i/d))
    PE(t', 2i+1) = cos(t' / 10000^(2i/d))
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Sinusoidal embedding dimension must be even (sin/cos pairs).")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        i = torch.arange(half, device=device, dtype=torch.float32)
        freqs = 1.0 / (10000 ** (2 * i / self.dim))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class AdaGN(nn.Module):
    """Adaptive Group Normalization, exact formula from Timestep.md section 3:
    AdaGN(h, t_emb) = y_s(t_emb) * GroupNorm(h) + y_b(t_emb)
    y_s, y_b are linear projections of the timestep embedding (scale/shift).
    """

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, channels)
        self.scale_proj = nn.Linear(time_emb_dim, channels)
        self.shift_proj = nn.Linear(time_emb_dim, channels)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        y_s = self.scale_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        y_b = self.shift_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        return y_s * self.norm(h) + y_b