"""
upsampling_model.py

Implements "Upsampling" (A4/A5): non-integer 2.5x scale via a two-step strategy
(integer PixelShuffle r=2, then bilinear + learned refine conv), followed by the
Anchor Reconstruction Head (A5).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PIXEL_SHUFFLE_R = 2
ANCHOR_SIZE = 320  # "320x320 anchor image"


class Upsampling(nn.Module):
    def __init__(self, channels: int = 64, out_channels: int = None):
        super().__init__()
        if out_channels is None:
            raise ValueError("out_channels (C, spectral band count) is required for the Anchor Head.")
        self.expand_conv = nn.Conv2d(channels, channels * (PIXEL_SHUFFLE_R ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(PIXEL_SHUFFLE_R)
        self.refine_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.anchor_head = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def upsample(self, f_agg: torch.Tensor) -> torch.Tensor:
        """F_agg [B,64,128,128] -> F_up [B,64,320,320]."""
        x = self.expand_conv(f_agg)
        x = self.pixel_shuffle(x)  # 128 -> 256
        x = F.interpolate(x, size=(int(f_agg.shape[2]*2.5), int(f_agg.shape[3]*2.5)), mode="bilinear", align_corners=False)
        x = self.refine_conv(x)
        return x

    def reconstruct_anchor(self, f_up: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        """F_up -> I_SR_anchor. No activation (linear projection). clamp=True applies
        the documented post-inference clamp to [0,1] ('not during training')."""
        out = self.anchor_head(f_up)
        if clamp:
            out = torch.clamp(out, 0.0, 1.0)
        return out

    def forward(self, f_agg: torch.Tensor, clamp: bool = False) -> torch.Tensor:
        return self.reconstruct_anchor(self.upsample(f_agg), clamp=clamp)
