"""
diffusion_refinement_controller.py

Orchestrates Stage B.2 (Diffusion-Based Refinement):

    I_SR_anchor -> VAE Encoder (Latent) -> z0_anchor
    z0_anchor  -> Noise Added -> z_t'
    I_LR       -> Original-Image Cond. Features -> c
    (z_t', t', c) -> Diffusion U-Net (looped via DDIM Steps) -> z0_hat
    z0_hat     -> VAE Decoder -> I_SR_refined

Contains no neural-network layer implementations of its own.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from model.latent_model import VAEEncoder
from model.noise_added_model import DDPMSchedule, truncated_forward_diffusion
from model.original_image_cond_features_model import ConditioningAdapter
from model.timestep_model import TimestepEmbedder
from model.diffusion_unet_model import DiffusionUNet
from model.ddim_steps_model import ddim_sample_loop
from model.vae_decoder_model import VAEDecoder
from model.final_super_resolved_scene_model import FinalSuperResolvedScene


class DiffusionRefinementController:
    def __init__(self, in_channels: int, vae_self_attn_heads: int, unet_level_channels: List[int],
                 unet_time_emb_dim: int, unet_mlp_hidden_dim: int, unet_num_heads: int,
                 cond_channels: int, device: torch.device = torch.device("cpu")):
        self.device = device
        self.schedule = DDPMSchedule(device=device)

        self.vae_encoder = VAEEncoder(in_channels, self_attn_num_heads=vae_self_attn_heads)
        self.cond_adapter = ConditioningAdapter(in_channels, cond_channels=cond_channels)
        self.timestep_embedder = TimestepEmbedder(unet_time_emb_dim, unet_mlp_hidden_dim)
        self.unet = DiffusionUNet(
            latent_channels=4,
            level_channels=unet_level_channels,
            time_emb_dim=unet_time_emb_dim,
            cond_channels=cond_channels,
            num_heads=unet_num_heads,
        )
        self.vae_decoder = VAEDecoder(in_channels, self_attn_num_heads=vae_self_attn_heads)

    def modules(self) -> List[nn.Module]:
        return [self.vae_encoder, self.cond_adapter, self.timestep_embedder, self.unet, self.vae_decoder]

    def _unet_predict_fn(self, c: torch.Tensor):
        def predict(z_t: torch.Tensor, t_batch: torch.Tensor) -> torch.Tensor:
            t_emb = self.timestep_embedder(t_batch)
            return self.unet(z_t, t_emb, c)
        return predict

    def encode_anchor(self, i_sr_anchor: torch.Tensor):
        mu, logvar = self.vae_encoder(i_sr_anchor)
        z0_anchor = self.vae_encoder.reparameterize(mu, logvar)
        return z0_anchor, mu, logvar

    def run(self, i_sr_anchor: torch.Tensor, i_lr: torch.Tensor,
            ddim_timestep_schedule: List[int], eta: float = 0.0) -> FinalSuperResolvedScene:
        z0_anchor, _, _ = self.encode_anchor(i_sr_anchor)

        t_start = torch.full((i_sr_anchor.shape[0],), ddim_timestep_schedule[0], device=i_sr_anchor.device)
        z_t_init, _ = truncated_forward_diffusion(z0_anchor, t_start, self.schedule)

        c = self.cond_adapter(i_lr)
        predict_fn = self._unet_predict_fn(c)
        z0_hat = ddim_sample_loop(predict_fn, z_t_init, ddim_timestep_schedule, self.schedule, eta=eta)

        i_sr_refined = self.vae_decoder(z0_hat)
        return FinalSuperResolvedScene(i_sr_refined=i_sr_refined)