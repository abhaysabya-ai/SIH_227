"""
anchor_image_model.py

Data-artifact schema for "Anchor Image" (I_SR_anchor), per Anchor Image.md.
The generation sequence itself is orchestrated by
controller/anchor_generation_controller.py.
"""
from dataclasses import dataclass
import torch


@dataclass
class AnchorImageArtifact:
    i_sr_anchor: torch.Tensor  # [B, C, 320, 320], float32, range ~[0,1]
    gsd_m: float = 4.0  # "~4 m/px (2.5x from 10m native)" -- documented approximate value

    def validate(self) -> None:
        if self.i_sr_anchor.dim() != 4 or self.i_sr_anchor.shape[-2:] != (320, 320):
            raise ValueError("AnchorImageArtifact.i_sr_anchor must be [B, C, 320, 320].")