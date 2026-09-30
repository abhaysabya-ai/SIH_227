"""
original_image_cond_features_model.py

Implements the Conditioning Adapter f_cond (B3), Original-Image Cond. Features.md
section 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common_layers import ResBlock, GROUP_NORM_GROUPS

COND_CHANNELS_DEFAULT = 256  # "Cc default: 256"
COND_SPATIAL = 40            # matches diffusion latent spatial size


class ConditioningAdapter(nn.Module):
    """I_LR [B,C,128,128] -> c [B,Cc,40,40].
    Architecture exact from doc section 2:
      Conv2d(C,64,k=3,s=1,p=1) -> ResBlock(64,64) -> GroupNorm(32,64)+SiLU
      -> Conv2d(64,128,k=3,s=2,p=1) -> ResBlock(128,128) -> GroupNorm(32,128)+SiLU
      -> Conv2d(128,Cc,k=3,s=2,p=1) -> ResBlock(Cc,Cc) -> interpolate to (40,40)
    """

    def __init__(self, in_channels: int, cond_channels: int = COND_CHANNELS_DEFAULT):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(64, 64)
        self.norm1 = nn.GroupNorm(GROUP_NORM_GROUPS, 64)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(128, 128)
        self.norm2 = nn.GroupNorm(GROUP_NORM_GROUPS, 128)
        self.down2 = nn.Conv2d(128, cond_channels, kernel_size=3, stride=2, padding=1)
        self.res3 = ResBlock(cond_channels, cond_channels)
        self.act = nn.SiLU()

    def forward(self, i_lr: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(i_lr)
        x = self.res1(x)
        x = self.act(self.norm1(x))
        x = self.down1(x)
        x = self.res2(x)
        x = self.act(self.norm2(x))
        x = self.down2(x)
        x = self.res3(x)
        x = F.interpolate(x, size=(int(i_lr.shape[2]*2.5/8), int(i_lr.shape[3]*2.5/8)), mode="bilinear", align_corners=False)
        return x
