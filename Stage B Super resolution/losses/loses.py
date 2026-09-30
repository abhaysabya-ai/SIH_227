"""
losses.py

Training-only loss functions documented for the Anchor Generation Network
(Hybrid Block.md section 7). Losses tied to a specific other Model file are kept
alongside it instead (VAE KL/recon in latent_model.py, diffusion MSE in
diffusion_unet_model.py, RaGAN in patchgan_discriminator_model.py).
"""
import torch


def charbonnier_loss(i_sr: torch.Tensor, i_hr: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """L_Charb = (1/HW) * sum sqrt((I_SR - I_HR)^2 + eps^2), Hybrid Block.md section 7."""
    diff2 = (i_sr - i_hr) ** 2 + eps ** 2
    return torch.sqrt(diff2).mean()


def sam_loss(i_sr: torch.Tensor, i_hr: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Spectral Angle Mapper loss, Hybrid Block.md section 7:
    L_SAM = (1/HW) * sum arccos( <I_SR,I_HR> / (|I_SR| |I_HR|) ), over spectral dim.
    i_sr, i_hr: [B, C, H, W]
    """
    dot = (i_sr * i_hr).sum(dim=1)
    norm_sr = i_sr.norm(dim=1).clamp_min(epsilon)
    norm_hr = i_hr.norm(dim=1).clamp_min(epsilon)
    cos_angle = (dot / (norm_sr * norm_hr)).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.arccos(cos_angle).mean()