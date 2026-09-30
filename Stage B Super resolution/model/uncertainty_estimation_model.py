"""
uncertainty_estimation_model.py

Pure variance-computation formulas from Uncertainty Estimation.md. Orchestration
(running N/M stochastic forward passes) lives in
controller/uncertainty_estimation_controller.py, since that coordinates other
Models rather than performing a learned transformation itself.
"""
from typing import List
import torch


def compute_structural_uncertainty(samples: List[torch.Tensor]) -> torch.Tensor:
    """sigma^2_structural = (1/N) * sum_n (I_SR_anchor^(n) - mean)^2.
    samples: N tensors of I_SR_anchor, each [B, C, 320, 320].
    Returns [B, 1, 320, 320] by averaging over channels, matching the documented
    output shape ('[B, 1, 320, 320] heat map').
    """
    stacked = torch.stack(samples, dim=0)  # [N, B, C, H, W]
    mean = stacked.mean(dim=0)
    var = ((stacked - mean) ** 2).mean(dim=0)
    return var.mean(dim=1, keepdim=True)


def compute_texture_uncertainty(samples: List[torch.Tensor]) -> torch.Tensor:
    """sigma^2_texture = (1/M) * sum_m (I_SR_refined^(m) - mean)^2."""
    stacked = torch.stack(samples, dim=0)
    mean = stacked.mean(dim=0)
    var = ((stacked - mean) ** 2).mean(dim=0)
    return var.mean(dim=1, keepdim=True)


def combine_uncertainty_maps(structural: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
    """Doc: 'Combined: Added or concatenated' -- ambiguous ('or'). This
    implementation uses addition; concatenation is the other documented option.
    TODO: INFORMATION REQUIRED FROM CONTEXT - which of the two is the actual
    production choice is not specified.
    """
    return structural + texture