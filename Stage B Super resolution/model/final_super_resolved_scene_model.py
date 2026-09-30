"""
final_super_resolved_scene_model.py

Data-artifact schema for "Final Super-Resolved Scene" (I_SR_refined), per
Final Super-Resolved Scene.md. The tensor itself is produced by
vae_decoder_model.VAEDecoder; this is a pure data container.
"""
from dataclasses import dataclass
import torch


@dataclass
class FinalSuperResolvedScene:
    i_sr_refined: torch.Tensor  # [B, C, 320, 320]
    scale_factor: float = 2.5   # "2.5x spatial enhancement over the native 10 m/px input"

    def validate(self) -> None:
        if self.i_sr_refined.dim() != 4 or self.i_sr_refined.shape[-2:] != (320, 320):
            raise ValueError("FinalSuperResolvedScene.i_sr_refined must be [B, C, 320, 320].")