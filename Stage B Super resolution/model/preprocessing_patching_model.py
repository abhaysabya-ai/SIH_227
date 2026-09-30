"""
preprocessing_patching_model.py

Implements the ordered steps documented in 'Preprocessing and Patching (SR).md'
(Stage B.1 entry node), in the documented order:
  1. Band co-registration & resampling (bilinear upsample of coarser bands to 10 m)
  2. Atmospheric correction verification (external tool call; only the CLI flag is
     documented, full processor configuration is TODO)
  3. Cloud & shadow masking (SCL-based binary mask)
  4. Per-band radiometric normalization (2nd/98th percentile stretch)
  5. Tiling into 128x128 patches, stride 96, 32 px overlap (+ Hann-window reconstruction)
  6. Batch assembly
"""
from __future__ import annotations
import subprocess
from typing import Tuple, List

import torch
import torch.nn.functional as F

PATCH_SIZE = 128
STRIDE = 96
OVERLAP = 32

# SCL classes, exact from doc Step 3.
SCL_VALID_CLASSES = {4, 5, 6, 7, 11}
SCL_MASKED_CLASSES = {3, 8, 9, 10}


def resample_band_to_10m(band: torch.Tensor, native_gsd_m: float, target_gsd_m: float = 10.0) -> torch.Tensor:
    """Step 1: bilinear upsample coarser-GSD bands (20 m / 60 m) to the common 10 m grid.
    band: [H, W] single-band tensor.
    """
    if native_gsd_m == target_gsd_m:
        return band
    scale = native_gsd_m / target_gsd_m
    h, w = band.shape[-2:]
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return F.interpolate(band[None, None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0, 0]


def verify_or_apply_atmospheric_correction(scene_path: str, is_level2a: bool) -> None:
    """Step 2: confirm Level-2A (BOA reflectance). If Level-1C, invoke Sen2Cor.
    The only concretely documented detail is the CLI invocation:
        'L2A_Process --resolution 10'
    TODO: INFORMATION REQUIRED FROM CONTEXT - full Sen2Cor/MAJA/FORCE configuration.
    """
    if is_level2a:
        return
    subprocess.run(["L2A_Process", "--resolution", "10", scene_path], check=True)


def cloud_mask_from_scl(scl: torch.Tensor) -> torch.Tensor:
    """Step 3: binary validity mask M in {0,1}^{H,W} from the Sentinel-2 SCL band.
    VALID  = SCL in {4,5,6,7,11}
    MASKED = SCL in {3,8,9,10}
    """
    valid = torch.zeros_like(scl, dtype=torch.bool)
    for cls in SCL_VALID_CLASSES:
        valid |= scl == cls
    return valid.float()


def percentile_stretch_normalize(x: torch.Tensor, p_low: float = 2.0, p_high: float = 98.0) -> Tuple[torch.Tensor, dict]:
    """Step 4: x_norm = (x - p2) / (p98 - p2), clipped to [0, 1].
    Percentiles computed per band, within the scene, as documented.
    x: [C, H, W]
    """
    c = x.shape[0]
    params = {}
    out = torch.empty_like(x)
    for ci in range(c):
        band = x[ci]
        p2 = torch.quantile(band, p_low / 100.0)
        p98 = torch.quantile(band, p_high / 100.0)
        norm = (band - p2) / (p98 - p2 + 1e-12)
        out[ci] = torch.clamp(norm, 0.0, 1.0)
        params[ci] = {"p2": p2.item(), "p98": p98.item()}
    return out, params


def hann_window_2d(patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Exact Hann-window formula, Step 5:
    w(i,j) = sin^2(pi*i/P) * sin^2(pi*j/P), i, j in [0, P-1]
    """
    idx = torch.arange(patch_size, dtype=torch.float32)
    w1d = torch.sin(torch.pi * idx / patch_size) ** 2
    return torch.outer(w1d, w1d)


def extract_patches(x: torch.Tensor, patch_size: int = PATCH_SIZE, stride: int = STRIDE) -> torch.Tensor:
    """Step 5: tile a scene into overlapping patch_size x patch_size patches.
    x: [C, H, W] -> patches: [N, C, patch_size, patch_size]
    """
    c, h, w = x.shape
    patches: List[torch.Tensor] = []
    for top in range(0, max(h - patch_size, 0) + 1, stride):
        for left in range(0, max(w - patch_size, 0) + 1, stride):
            patch = x[:, top:top + patch_size, left:left + patch_size]
            if patch.shape[-2:] == (patch_size, patch_size):
                patches.append(patch)
    return torch.stack(patches, dim=0)


def reconstruct_from_patches(
    patches: torch.Tensor,
    positions: List[Tuple[int, int]],
    output_size: Tuple[int, int],
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Inverse of extract_patches using Hann-window weighted blending, per doc:
    'Weighted average blending using a Hann window in the overlap region'."""
    c = patches.shape[1]
    h, w = output_size
    canvas = torch.zeros(c, h, w)
    weight = torch.zeros(1, h, w)
    win = hann_window_2d(patch_size)
    for patch, (top, left) in zip(patches, positions):
        canvas[:, top:top + patch_size, left:left + patch_size] += patch * win
        weight[:, top:top + patch_size, left:left + patch_size] += win
    return canvas / weight.clamp_min(1e-8)


def assemble_batch(patches: torch.Tensor) -> torch.Tensor:
    """Step 6: batch assembly (pass-through; patches already [N, C, 128, 128])."""
    return patches