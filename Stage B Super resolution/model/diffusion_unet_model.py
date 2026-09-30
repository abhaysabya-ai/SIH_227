"""
diffusion_unet_model.py

Implements the "Diffusion U-Net" (B4 / eps_theta), Diffusion U-Net.md, together with
the cross-attention conditioning mechanism documented in "Diffusion U-Net Fusion.md"
-- which that file itself describes as "a conceptual sub-node inside the Diffusion
U-Net", hence implemented here rather than as a standalone file.

Documented resolution levels for the 40x40 latent (Diffusion U-Net.md section 2):
    Level 1: 40x40, Level 2: 20x20, Level 3 / Bottleneck: 10x10
At each level: ResBlock(dim_in,dim_out) + timestep embedding (AdaGN)
               -> Self-Attention(dim_out) -> Cross-Attention(dim_out, c)

TODO: INFORMATION REQUIRED FROM CONTEXT (required constructor args, no defaults):
  - level_channels: channel width at each of the 3 levels (never given numerically)
  - num_heads: attention head count for self-/cross-attention inside the U-Net
"""
from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from model.common_layers import AdaGN, GROUP_NORM_GROUPS


class CrossAttention(nn.Module):
    """Diffusion U-Net Fusion.md:
    CrossAttn(F_unet, c) = Softmax(Q(F_unet) K(c)^T / sqrt(d)) V(c)
    Q from U-Net features; K, V from the spatial condition c.
    """

    def __init__(self, dim: int, cond_channels: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cond_channels, dim, bias=False)
        self.to_v = nn.Linear(cond_channels, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        x_in = x
        x_n = self.norm(x)
        q_tokens = x_n.flatten(2).transpose(1, 2)  # [B, HW, C]

        # "c is pooled/interpolated to match, or c remains 40x40 and the U-Net
        # attends to it" -- Diffusion U-Net Fusion.md section 2.
        if c.shape[-2:] != (h, w):
            c = Fn.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
        c_tokens = c.flatten(2).transpose(1, 2)  # [B, HW, Cc]

        q = self.to_q(q_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(c_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(c_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, -1, ch)
        out = self.to_out(out).transpose(1, 2).reshape(b, ch, h, w)
        return x_in + out


class SelfAttentionSpatial(nn.Module):
    """Spatial self-attention within the U-Net (Diffusion U-Net.md architecture list)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.GroupNorm(GROUP_NORM_GROUPS, dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)
        attn_out, _ = self.mha(tokens, tokens, tokens)
        return x + attn_out.transpose(1, 2).reshape(b, c, h, w)


class TimestepResBlock(nn.Module):
    """ResBlock with AdaGN timestep-embedding injection (Diffusion U-Net.md section 2
    + Timestep.md section 3)."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ada_gn1 = AdaGN(out_channels, time_emb_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.ada_gn2 = AdaGN(out_channels, time_emb_dim)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ada_gn1(self.conv1(x), t_emb))
        h = self.ada_gn2(self.conv2(h), t_emb)
        return self.act(h + self.skip(x))


class UNetLevel(nn.Module):
    """One resolution level: ResBlock(+AdaGN) -> Self-Attention -> Cross-Attention(c)."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 cond_channels: int, num_heads: int):
        super().__init__()
        self.res = TimestepResBlock(in_channels, out_channels, time_emb_dim)
        self.self_attn = SelfAttentionSpatial(out_channels, num_heads)
        self.cross_attn = CrossAttention(out_channels, cond_channels, num_heads)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        x = self.self_attn(x)
        x = self.cross_attn(x, c)
        return x


class DiffusionUNet(nn.Module):
    """eps_theta(z_t', t', c) -> predicted noise, same shape as z_t' ([B,4,40,40]).
    level_channels must have exactly 3 entries (40x40, 20x20, 10x10/bottleneck).
    """

    def __init__(self, latent_channels: int, level_channels: List[int], time_emb_dim: int,
                 cond_channels: int, num_heads: int):
        super().__init__()
        if len(level_channels) != 3:
            raise ValueError(
                "Diffusion U-Net.md documents exactly 3 resolution levels "
                "(40x40, 20x20, 10x10/bottleneck); level_channels must have length 3."
            )
        c1, c2, c3 = level_channels
        self.conv_in = nn.Conv2d(latent_channels, c1, kernel_size=3, padding=1)

        self.enc1 = UNetLevel(c1, c1, time_emb_dim, cond_channels, num_heads)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)  # 40 -> 20
        self.enc2 = UNetLevel(c2, c2, time_emb_dim, cond_channels, num_heads)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)  # 20 -> 10

        self.bottleneck = UNetLevel(c3, c3, time_emb_dim, cond_channels, num_heads)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)  # 10 -> 20
        self.dec2 = UNetLevel(c2 * 2, c2, time_emb_dim, cond_channels, num_heads)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)  # 20 -> 40
        self.dec1 = UNetLevel(c1 * 2, c1, time_emb_dim, cond_channels, num_heads)

        self.conv_out = nn.Conv2d(c1, latent_channels, kernel_size=3, padding=1)

        # Learned "null token" for Classifier-Free Guidance (Diffusion U-Net.md
        # section 7: "c is replaced with a learned null token"). Its exact broadcast
        # shape is not specified in context; see Assumptions in the report.
        self.null_cond = nn.Parameter(torch.zeros(1, cond_channels, 1, 1))

    def forward(self, z_t: torch.Tensor, t_emb: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        b, _, h, w = z_t.shape
        if c is None:
            c = self.null_cond.expand(b, -1, h, w)

        x1 = self.conv_in(z_t)
        x1 = self.enc1(x1, t_emb, c)
        x2 = self.down1(x1)
        x2 = self.enc2(x2, t_emb, c)
        x3 = self.down2(x2)
        x3 = self.bottleneck(x3, t_emb, c)

        u2 = self.up2(x3)
        u2 = self.dec2(torch.cat([u2, x2], dim=1), t_emb, c)
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, x1], dim=1), t_emb, c)

        return self.conv_out(u1)

    def forward_with_cfg(self, z_t: torch.Tensor, t_emb: torch.Tensor, c: torch.Tensor,
                          guidance_scale: float) -> torch.Tensor:
        """eps_hat = eps_uncond + w * (eps_cond - eps_uncond), section 7.
        guidance_scale (w): doc only gives an example ("e.g., w = 2.0"); not hard-coded here.
        """
        eps_uncond = self.forward(z_t, t_emb, c=None)
        eps_cond = self.forward(z_t, t_emb, c)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


def diffusion_mse_loss(eps: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
    """L_diffusion = E[||eps - eps_theta(z_t', t', c)||^2], section 3."""
    return torch.nn.functional.mse_loss(eps_pred, eps)