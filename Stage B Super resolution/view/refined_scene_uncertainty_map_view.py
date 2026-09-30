"""
refined_scene_uncertainty_map_view.py

Packages the final deliverable exactly as described in
Refined Scene and Uncertainty Map.md: the Final Super-Resolved Scene plus the
Uncertainty Estimation maps, ready for hand-off to Stage C (out of scope -- Stage C
files are not among the 19 context files for this task).

The doc also mentions an OPTIONAL 'Physical Consistency Trust Map (from Module D)'.
Module D is never documented within the 19 context files, so that field is
intentionally omitted rather than invented.
TODO: INFORMATION REQUIRED FROM CONTEXT - Module D / Physical Consistency Guardrail.
"""
from typing import Optional
import torch

from model.final_super_resolved_scene_model import FinalSuperResolvedScene


def build_refined_scene_and_uncertainty_map(
    final_scene: FinalSuperResolvedScene,
    sigma2_structural: Optional[torch.Tensor],
    sigma2_texture: Optional[torch.Tensor],
) -> dict:
    """Packaged container documented as:
      1. The Image   : Final Super-Resolved Scene (I_SR_refined)
      2. The Meta-Data: Uncertainty Estimation maps (sigma^2_structural, sigma^2_texture)
    """
    final_scene.validate()
    return {
        "i_sr_refined": final_scene.i_sr_refined,
        "scale_factor": final_scene.scale_factor,
        "uncertainty_map_structural": sigma2_structural,
        "uncertainty_map_texture": sigma2_texture,
        # "physical_consistency_trust_map": omitted -- Module D not in context.
    }