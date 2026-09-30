"""
latent_model.py

Implements the VAE Encoder architecture exactly as specified in Latent.md section 2,
plus the VAE training losses documented in Latent.md section 3 (KL divergence,
L1 reconstruction). The noise schedule constants documented in Latent.md sections
4-5 are centralized in noise_added_model.py (shared with Noise Added.md / DDIM
Steps.md, which document the same schedule) to avoid duplicating that math.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from model.common_layers import ResBlock, Downsample, SelfAttention, GROUP_NORM_GROUPS

LATENT_CHANNELS = 4  # "4-dimensional bottleneck ... 4 channels is the default"


class VAEEncoder(nn.Module):
    """E: R^{B x C x 320 x 320} -> (mu, logvar), each R^{B x 4 x 40 x 40}.

    Architecture copied verbatim from Latent.md section 2:
      Conv2d(C,128) -> ResBlock(128,128) -> Downsample(128,128,s=2)      [320->160]
      -> ResBlock(128,256) -> Downsample(256,256,s=2)                    [160->80]
      -> ResBlock(256,512) -> Downsample(512,512,s=2)                    [80->40]
      -> ResBlock(512,512) -> SelfAttention(512) -> ResBlock(512,512)
      -> GroupNorm(32,512) + SiLU -> Conv2d(512, 2*4, k=3, p=1)

    TODO: INFORMATION REQUIRED FROM CONTEXT - self-attention head count is not
    documented anywhere in the 19 context files.
    """

    def __init__(self, in_channels: int, self_attn_num_heads: int, latent_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.res1 = ResBlock(128, 128)
        self.down1 = Downsample(128, 128, stride=2)
        self.res2 = ResBlock(128, 256)
        self.down2 = Downsample(256, 256, stride=2)
        self.res3 = ResBlock(256, 512)
        self.down3 = Downsample(512, 512, stride=2)
        self.res4 = ResBlock(512, 512)
        self.self_attn = SelfAttention(512, num_heads=self_attn_num_heads)
        self.res5 = ResBlock(512, 512)
        self.norm_out = nn.GroupNorm(GROUP_NORM_GROUPS, 512)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 2 * latent_channels, kernel_size=3, padding=1)
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor):
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.down1(h)
        h = self.res2(h)
        h = self.down2(h)
        h = self.res3(h)
        h = self.down3(h)
        h = self.res4(h)
        h = self.self_attn(h)
        h = self.res5(h)
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z0 = mu + sigma * eps, sigma = exp(0.5 * logvar)."""
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + sigma * eps


def vae_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), Latent.md section 3:
    L_KL = -1/2 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3]).mean()


def vae_reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """L_recon = ||x - D(E(x))||_1, Latent.md section 3."""
    return torch.nn.functional.l1_loss(x_hat, x)

# TODO: INFORMATION REQUIRED FROM CONTEXT
# The perceptual loss term (VGG feature matching adapted for multispectral input via
# PCA projection to RGB) is named in Latent.md section 3 but no concrete VGG layer
# selection, PCA projection procedure, or weighting is given beyond the symbol
# lambda_perc. A perceptual-loss function is therefore intentionally NOT implemented.