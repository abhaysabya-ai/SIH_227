"""
shallow_convolution_model.py

Implements the "Shallow Convolution" node (A1 / F0), exact from Shallow Convolution.md.
"""
import torch
import torch.nn as nn


class ShallowConvolution(nn.Module):
    """Single 3x3 conv, stride=1, padding=1 (same-padding), bias=True, no activation,
    no normalization. Projects C-band input into a 64-channel embedding.
    """

    OUT_CHANNELS = 64  # "the fixed internal width of the entire Anchor Generation Network"

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, self.OUT_CHANNELS, kernel_size=3, stride=1, padding=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        # "Kaiming Uniform ... scaled for leaky_relu(a=0.2) ... Bias initialized to zero."
        nn.init.kaiming_uniform_(self.conv.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(self.conv.bias)

    def forward(self, i_lr: torch.Tensor) -> torch.Tensor:
        """i_lr: [B, C, 128, 128] -> F0: [B, 64, 128, 128] (no activation, unbounded range)."""
        return self.conv(i_lr)