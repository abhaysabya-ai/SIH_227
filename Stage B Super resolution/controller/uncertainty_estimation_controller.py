"""
uncertainty_estimation_controller.py

Orchestrates the two MC-sampling procedures documented in Uncertainty Estimation.md:
  E1 Structural Uncertainty : N stochastic passes through the Anchor Generation
                              Network with RRDB dropout active (MC-Dropout).
  E2 Texture Uncertainty    : M stochastic passes through the Diffusion Refinement
                              path with different random seeds (MC-Diffusion).
"""
from __future__ import annotations
from typing import List

import torch

from controller.anchor_generation_controller import AnchorGenerationController
from controller.diffusion_refinement_controller import DiffusionRefinementController
from model.uncertainty_estimation_model import (
    compute_structural_uncertainty,
    compute_texture_uncertainty,
    combine_uncertainty_maps,
)

# "The network runs N times (e.g., N=10)" -- documented example value, not a fixed
# requirement; kept as a named default so callers can override it explicitly.
DEFAULT_STRUCTURAL_SAMPLES_N = 10


class UncertaintyEstimationController:
    def __init__(self, anchor_controller: AnchorGenerationController,
                 diffusion_controller: DiffusionRefinementController):
        self.anchor_controller = anchor_controller
        self.diffusion_controller = diffusion_controller

    def estimate_structural(self, i_lr: torch.Tensor, n_samples: int = DEFAULT_STRUCTURAL_SAMPLES_N):
        # Caller is responsible for keeping the anchor controller's RRDB dropout
        # layers active (.train()) so MC-Dropout actually fires; Hybrid Block.md
        # documents a BatchNorm-free design, so no other train/eval-sensitive
        # layers exist in that path.
        samples = []
        for _ in range(n_samples):
            artifact = self.anchor_controller.run(i_lr, clamp_output=False)
            samples.append(artifact.i_sr_anchor)
        mean_anchor = torch.stack(samples, dim=0).mean(dim=0)
        sigma2_structural = compute_structural_uncertainty(samples)
        return mean_anchor, sigma2_structural

    def estimate_texture(self, i_sr_anchor: torch.Tensor, i_lr: torch.Tensor,
                          ddim_timestep_schedule: List[int], m_samples: int, eta: float):
        """m_samples (M) has no documented example value anywhere in the 19 context
        files -> REQUIRED argument, no default.
        TODO: INFORMATION REQUIRED FROM CONTEXT (M).
        eta must be > 0 for the DDIM sampler itself to be the stochasticity source,
        per doc ('if the DDIM sampler is slightly stochastic, eta>0'); the
        alternative documented source ('forward noise injection') is naturally
        present since truncated_forward_diffusion always samples eps.
        """
        samples = []
        for _ in range(m_samples):
            scene = self.diffusion_controller.run(i_sr_anchor, i_lr, ddim_timestep_schedule, eta=eta)
            samples.append(scene.i_sr_refined)
        mean_refined = torch.stack(samples, dim=0).mean(dim=0)
        sigma2_texture = compute_texture_uncertainty(samples)
        return mean_refined, sigma2_texture

    @staticmethod
    def combine(sigma2_structural: torch.Tensor, sigma2_texture: torch.Tensor) -> torch.Tensor:
        return combine_uncertainty_maps(sigma2_structural, sigma2_texture)