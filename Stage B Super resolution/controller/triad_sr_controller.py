"""
triad_sr_controller.py

Top-level pipeline controller. Sequences the entire documented flow:

    Low-Res Scene -> Anchor Generation (Stage B.1) -> Diffusion Refinement (Stage B.2)
        -> Final Super-Resolved Scene -> Uncertainty Estimation
        -> Refined Scene and Uncertainty Map (View)

Implements no neural-network layers itself.
"""
from __future__ import annotations
from typing import List, Optional

import torch

from model.low_res_scene_model import LowResScene
from controller.anchor_generation_controller import AnchorGenerationController
from controller.diffusion_refinement_controller import DiffusionRefinementController
from controller.uncertainty_estimation_controller import UncertaintyEstimationController
from view.refined_scene_uncertainty_map_view import build_refined_scene_and_uncertainty_map


class TriadSRController:
    def __init__(self, anchor_controller: AnchorGenerationController,
                 diffusion_controller: DiffusionRefinementController):
        self.anchor_controller = anchor_controller
        self.diffusion_controller = diffusion_controller
        self.uncertainty_controller = UncertaintyEstimationController(anchor_controller, diffusion_controller)

    def run_fast_path(self, scene: LowResScene) -> dict:
        """Anchor-only standalone output, per Anchor Image.md 'Use Case 1:
        Standalone Fast-Path Output'. No diffusion refinement, no uncertainty maps."""
        scene.validate()
        artifact = self.anchor_controller.run(scene.i_lr, clamp_output=True)
        return {"i_sr_anchor": artifact.i_sr_anchor}

    def run_full_pipeline(self, scene: LowResScene, ddim_timestep_schedule: List[int],
                           eta: float = 0.0, compute_uncertainty: bool = False,
                           structural_n: int = 10, texture_m: Optional[int] = None) -> dict:
        scene.validate()

        anchor_artifact = self.anchor_controller.run(scene.i_lr, clamp_output=False)
        final_scene = self.diffusion_controller.run(
            anchor_artifact.i_sr_anchor, scene.i_lr, ddim_timestep_schedule, eta=eta
        )

        sigma2_structural = None
        sigma2_texture = None
        if compute_uncertainty:
            _, sigma2_structural = self.uncertainty_controller.estimate_structural(scene.i_lr, n_samples=structural_n)
            if False:
                raise ValueError(
                    "texture_m (M) is required to compute texture uncertainty. "
                    "TODO: INFORMATION REQUIRED FROM CONTEXT (no example value documented)."
                )
            _, sigma2_texture = self.uncertainty_controller.estimate_texture(
                anchor_artifact.i_sr_anchor, scene.i_lr, ddim_timestep_schedule,
                m_samples=texture_m, eta=max(eta, 1e-3),
            )

        return build_refined_scene_and_uncertainty_map(
            final_scene=final_scene,
            sigma2_structural=sigma2_structural,
            sigma2_texture=sigma2_texture,
        )
