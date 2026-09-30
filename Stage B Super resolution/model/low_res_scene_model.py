"""
low_res_scene_model.py

Data-artifact schema for the pipeline entry point I_LR, per Low-Res Scene.md.
This node performs no learned transformation itself; it is a pass-through artifact
consumed by two independent downstream paths:
  1) Preprocessing and Patching (SR) -> Shallow Convolution (B.1 primary path)
  2) Original-Image Cond. Features (B.2 conditioning path)
"""
from dataclasses import dataclass, field
from typing import List, Optional
import torch

# Default 6-band stack, explicit in Low-Res Scene.md section 5.
DEFAULT_BANDS: List[str] = ["B2", "B3", "B4", "B8", "B8A", "B11"]


@dataclass
class RawLowResScene:
    """Raw input schema (Low-Res Scene.md section 3, 'before preprocessing')."""
    data: torch.Tensor  # [H_raw, W_raw, C], dtype uint16, DN values 0-65535
    bands: List[str] = field(default_factory=lambda: list(DEFAULT_BANDS))
    crs: Optional[str] = None  # e.g. "EPSG:326xx" / "EPSG:327xx"

    def validate(self) -> None:
        if self.data.dim() != 3:
            raise ValueError("RawLowResScene.data must have shape [H_raw, W_raw, C].")
        if self.data.shape[-1] != len(self.bands):
            raise ValueError("Channel count does not match number of configured bands.")


@dataclass
class LowResScene:
    """Post-preprocessing schema (I_LR), consumed by all downstream Stage B nodes.

    Shape : [B, C, 128, 128]
    Dtype : float32
    Range : [0.0, 1.0] (per-band percentile-stretch normalization)
    """
    i_lr: torch.Tensor
    bands: List[str] = field(default_factory=lambda: list(DEFAULT_BANDS))

    def validate(self) -> None:
        if self.i_lr.dim() != 4:
            raise ValueError("LowResScene.i_lr must have shape [B, C, 128, 128].")
        b, c, h, w = self.i_lr.shape
        if (h, w) != (128, 128):
            raise ValueError(f"LowResScene.i_lr spatial size must be 128x128, got {(h, w)}.")
        if c != len(self.bands):
            raise ValueError("Channel count does not match number of configured bands.")

    @property
    def num_channels(self) -> int:
        return self.i_lr.shape[1]