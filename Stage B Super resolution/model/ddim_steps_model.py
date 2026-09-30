"""
ddim_steps_model.py

Implements the DDIM sampler update rule, DDIM Steps.md sections 2-3.
"""
from __future__ import annotations
from typing import List
import torch

from model.noise_added_model import DDPMSchedule


def ddim_step(z_t: torch.Tensor, eps_pred: torch.Tensor, alpha_bar_t: torch.Tensor,
              alpha_bar_prev: torch.Tensor, eta: float = 0.0):
    """One DDIM update:
      z0_hat  = (z_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
      sigma_t = eta * sqrt((1-alpha_bar_prev)/(1-alpha_bar_t)) * sqrt(1-alpha_bar_t/alpha_bar_prev)
      z_{t-1} = sqrt(alpha_bar_prev)*z0_hat + sqrt(1-alpha_bar_prev-sigma_t^2)*eps + sigma_t*eps_t

    eta=0.0 is the documented TRIAD-SR default ("eta=0 is typically used for
    deterministic reconstruction").
    """
    ab_t = alpha_bar_t.view(-1, 1, 1, 1)
    ab_prev = alpha_bar_prev.view(-1, 1, 1, 1)

    z0_hat = (z_t - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)

    if eta == 0.0:
        sigma_t = torch.zeros_like(ab_t)
    else:
        sigma_t = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)

    direction = torch.sqrt((1 - ab_prev - sigma_t ** 2).clamp_min(0.0)) * eps_pred
    noise = torch.randn_like(z_t) if eta > 0.0 else 0.0
    z_prev = torch.sqrt(ab_prev) * z0_hat + direction + sigma_t * noise
    return z_prev, z0_hat


def ddim_sample_loop(unet_predict_fn, z_t_init: torch.Tensor, timestep_schedule: List[int],
                      schedule: DDPMSchedule, eta: float = 0.0) -> torch.Tensor:
    """Runs the DDIM loop over the caller-supplied truncated timestep schedule,
    e.g. [16, 14, 12, 10, 8, 6, 4, 2] (documented example).

    unet_predict_fn(z_t, t_batch) -> eps_pred wraps the Diffusion U-Net (including
    timestep-embedding lookup), kept generic so this Model file holds no U-Net logic
    itself (avoids duplicating Model logic across files).
    """
    z_t = z_t_init
    schedule_ext = list(timestep_schedule) + [0]
    z0_hat = z_t
    for t_cur, t_next in zip(schedule_ext[:-1], schedule_ext[1:]):
        b = z_t.shape[0]
        t_batch = torch.full((b,), t_cur, device=z_t.device, dtype=torch.long)
        eps_pred = unet_predict_fn(z_t, t_batch)
        alpha_bar_t = schedule.alpha_bar(t_batch)
        t_next_batch = torch.full((b,), max(t_next, 1), device=z_t.device, dtype=torch.long)
        alpha_bar_prev = schedule.alpha_bar(t_next_batch) if t_next > 0 else torch.ones_like(alpha_bar_t)
        z_t, z0_hat = ddim_step(z_t, eps_pred, alpha_bar_t, alpha_bar_prev, eta=eta)
    return z0_hat