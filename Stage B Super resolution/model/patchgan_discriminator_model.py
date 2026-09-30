"""
patchgan_discriminator_model.py

Implements the PatchGAN Discriminator (C1), PatchGAN Discriminator.md.
TRAINING-ONLY component (doc: "not used at inference").
"""
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 receptive-field PatchGAN, BatchNorm-free (per ESRGAN convention),
    architecture exact from doc section 2."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ragan_generator_loss(c_real: torch.Tensor, c_fake: torch.Tensor) -> torch.Tensor:
    """Relativistic average GAN generator loss, section 3:
      D_Ra(xr,xf) = sigma(C(xr) - E[C(xf)])
      D_Ra(xf,xr) = sigma(C(xf) - E[C(xr)])
      L_Adv = -E[log D_Ra(xf,xr)] - E[log(1 - D_Ra(xr,xf))]
    """
    d_ra_real = torch.sigmoid(c_real - c_fake.mean())
    d_ra_fake = torch.sigmoid(c_fake - c_real.mean())
    eps = 1e-8
    return -torch.log(d_ra_fake + eps).mean() - torch.log(1 - d_ra_real + eps).mean()