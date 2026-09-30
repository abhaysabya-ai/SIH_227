"""
vae_decoder_model.py

Implements the VAE Decoder (B5 / D(z)) exactly as specified in VAE Decoder.md
section 2.
"""
import torch
import torch.nn as nn

from model.common_layers import ResBlock, SelfAttention, Upsample2x, GROUP_NORM_GROUPS


class VAEDecoder(nn.Module):
    """z_hat_0 [B,4,40,40] -> I_SR_refined [B,C,320,320].

    TODO: INFORMATION REQUIRED FROM CONTEXT - self-attention head count is not
    documented anywhere in the 19 context files.
    """

    def __init__(self, out_channels: int, self_attn_num_heads: int, latent_channels: int = 4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, 512, kernel_size=3, padding=1)
        self.res1 = ResBlock(512, 512)
        self.self_attn = SelfAttention(512, num_heads=self_attn_num_heads)
        self.res2 = ResBlock(512, 512)

        self.up1 = Upsample2x(512)          # 40 -> 80
        self.conv_mid1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.res3 = ResBlock(512, 256)

        self.up2 = Upsample2x(256)          # 80 -> 160
        self.conv_mid2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res4 = ResBlock(256, 128)

        self.up3 = Upsample2x(128)          # 160 -> 320
        self.conv_mid3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res5 = ResBlock(128, 128)

        self.norm_out = nn.GroupNorm(GROUP_NORM_GROUPS, 128)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def forward(self, z_hat_0: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z_hat_0)
        h = self.res1(h)
        h = self.self_attn(h)
        h = self.res2(h)

        h = self.up1(h)
        h = self.conv_mid1(h)
        h = self.res3(h)

        h = self.up2(h)
        h = self.conv_mid2(h)
        h = self.res4(h)

        h = self.up3(h)
        h = self.conv_mid3(h)
        h = self.res5(h)

        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)