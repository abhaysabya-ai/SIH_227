"""
globally_fused_features_model.py

Implements "Globally Fused Features" (A3 / F_agg):
    F_agg = F0 + Conv3x3(F_DF)
"""
import torch
import torch.nn as nn


class GloballyFusedFeatures(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.align_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, f0: torch.Tensor, f_df: torch.Tensor) -> torch.Tensor:
        """f0: shallow stem output [B,64,128,128] (long skip).
        f_df: final Hybrid Block output F_N [B,64,128,128]."""
        return f0 + self.align_conv(f_df)