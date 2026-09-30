"""
noise_added_model.py

Implements the DDPM linear noise schedule and truncated forward diffusion from
Noise Added.md (node B2). Shared by Latent.md, Diffusion U-Net.md, DDIM Steps.md,
all of which document this same schedule.
"""
from __future__ import annotations
import torch

DDPM_T = 1000          # "T (total schedule steps): 1000"
BETA_MIN = 1e-4        # "beta_min: 1e-4"
BETA_MAX = 2e-2        # "beta_max: 2e-2"
T_PRIME_MIN = 1
T_PRIME_MAX = 16       # "t' in {1, ..., 16}" truncated short chain


class DDPMSchedule:
    """Linear beta schedule: beta_t = beta_min + (t-1)/(T-1) * (beta_max - beta_min).
    alpha_t = 1 - beta_t; alpha_bar_t = prod_{s<=t} alpha_s.
    """

    def __init__(self, T: int = DDPM_T, beta_min: float = BETA_MIN, beta_max: float = BETA_MAX,
                 device: torch.device = torch.device("cpu")):
        self.T = T
        t = torch.arange(1, T + 1, dtype=torch.float64, device=device)
        betas = beta_min + (t - 1) / (T - 1) * (beta_max - beta_min)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)  # index 0 -> t=1

    def alpha_bar(self, t_prime: torch.Tensor) -> torch.Tensor:
        """t_prime is 1-indexed per the documented schedule (t' in [1, T])."""
        idx = (t_prime.long() - 1).clamp(min=0, max=self.T - 1)
        return self.alphas_cumprod.to(t_prime.device)[idx].float()


def truncated_forward_diffusion(z0_anchor: torch.Tensor, t_prime: torch.Tensor, schedule: DDPMSchedule):
    """z_t' = sqrt(alpha_bar_t') * z0_anchor + sqrt(1 - alpha_bar_t') * eps, eps ~ N(0,I).
    Exact closed-form reparameterized sampling, Noise Added.md section 3.
    """
    alpha_bar_t = schedule.alpha_bar(t_prime).view(-1, 1, 1, 1)
    eps = torch.randn_like(z0_anchor)
    z_t_prime = torch.sqrt(alpha_bar_t) * z0_anchor + torch.sqrt(1 - alpha_bar_t) * eps
    return z_t_prime, eps


def sample_training_timestep(batch_size: int, device: torch.device) -> torch.Tensor:
    """t' ~ U(1, 16), documented training-time sampling."""
    return torch.randint(T_PRIME_MIN, T_PRIME_MAX + 1, (batch_size,), device=device)