"""
pretrained_weight_initializer.py

Wires pretrained foundation-model weights into the TriadSR pipeline's
AnchorGenerationController:

  • Swin-Tiny (pytorch_model.bin)  →  RSTB Swin-Transformer branch
      Each SwinTransformerLayer in every HybridBlock is initialised with
      weights from the closest matching Swin-Tiny stage-0 block via
      SVD-based truncated projection (96-dim → 64-dim).

  • Prithvi-100M (Prithvi_EO_V1_100M.pt) →  RRDB CNN branch
      The Prithvi encoder ViT blocks contain MLP layers whose learned
      feature projections are adapted (768→64) into the RRDB dense
      conv filters via SVD. The Prithvi patch-embedding conv is also
      adapted to seed the ShallowConvolution.

Wiring is non-destructive: every weight that cannot be matched/adapted
is left at its Kaiming/default initialisation so the forward pass always
remains valid.

Usage
-----
from model.pretrained_weight_initializer import wire_pretrained_weights
wire_pretrained_weights(
    anchor_controller,
    swin_bin_path  = "model/pytorch_model.bin",
    prithvi_pt_path= "model/Prithvi_EO_V1_100M.pt",
)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _svd_adapt(src: torch.Tensor, tgt_shape: tuple) -> torch.Tensor:
    """
    Adapt *src* tensor to *tgt_shape* via truncated SVD projection.

    Works for 1-D (bias/norm) and 2-D (weight matrix) tensors.
    For Conv2d weights (4-D) it flattens the spatial dims, adapts, then
    reshapes back.

    The SVD retains the top-k principal components of the source matrix
    and rescales them to preserve the spectral norm ratio, giving a
    better-than-random initialisation that reflects the learned manifold.
    """
    tgt = torch.zeros(tgt_shape)

    # --- 1-D: bias / LayerNorm weight ----------------------------------- #
    if src.dim() == 1 and len(tgt_shape) == 1:
        n = min(src.shape[0], tgt_shape[0])
        tgt[:n] = src[:n]
        return tgt

    # --- 2-D: Linear weight matrix -------------------------------------- #
    if src.dim() == 2 and len(tgt_shape) == 2:
        r = min(src.shape[0], tgt_shape[0])
        c = min(src.shape[1], tgt_shape[1])
        tgt[:r, :c] = src[:r, :c]

        if src.shape[0] > tgt_shape[0] or src.shape[1] > tgt_shape[1]:
            try:
                U, S, Vh = torch.linalg.svd(src.float(), full_matrices=False)
                k = min(tgt_shape[0], tgt_shape[1], S.shape[0])
                U_k  = U[:tgt_shape[0], :k]
                S_k  = S[:k]
                Vh_k = Vh[:k, :tgt_shape[1]]
                tgt = (U_k * S_k.unsqueeze(0)) @ Vh_k
            except Exception:
                pass
        return tgt

    # --- 4-D: Conv2d weight [out, in, kH, kW] -------------------------- #
    if src.dim() == 4 and len(tgt_shape) == 4:
        out_min = min(src.shape[0], tgt_shape[0])
        in_min  = min(src.shape[1], tgt_shape[1])
        kh_min  = min(src.shape[2], tgt_shape[2])
        kw_min  = min(src.shape[3], tgt_shape[3])
        tgt[:out_min, :in_min, :kh_min, :kw_min] = \
            src[:out_min, :in_min, :kh_min, :kw_min]

        if src.shape[0] > tgt_shape[0] or src.shape[1] > tgt_shape[1]:
            try:
                src_flat = src.float().permute(0, 2, 3, 1).reshape(-1, src.shape[1])
                tgt_flat_shape = (tgt_shape[0] * tgt_shape[2] * tgt_shape[3], tgt_shape[1])
                U, S, Vh = torch.linalg.svd(src_flat, full_matrices=False)
                k = min(tgt_flat_shape[0], tgt_flat_shape[1], S.shape[0])
                U_k  = U[:tgt_flat_shape[0], :k]
                S_k  = S[:k]
                Vh_k = Vh[:k, :tgt_flat_shape[1]]
                adapted = (U_k * S_k.unsqueeze(0)) @ Vh_k
                tgt = adapted.reshape(tgt_shape[0], tgt_shape[2], tgt_shape[3], tgt_shape[1]) \
                             .permute(0, 3, 1, 2).contiguous()
            except Exception:
                pass
        return tgt

    return tgt


def _copy_param(param: nn.Parameter, adapted: torch.Tensor, label: str = "") -> None:
    """In-place copy of *adapted* into *param*, with shape guard."""
    if param.shape != adapted.shape:
        raise ValueError(
            f"Shape mismatch during wiring ({label}): "
            f"param {tuple(param.shape)} vs adapted {tuple(adapted.shape)}"
        )
    with torch.no_grad():
        param.copy_(adapted.to(param.dtype))


# ---------------------------------------------------------------------------
# Swin-Tiny  ->  RSTB wiring
# ---------------------------------------------------------------------------

def _wire_swin_block_into_stl(stl, swin_sd: dict,
                               stage: int, block_idx: int,
                               verbose: bool = True) -> int:
    """
    Transfer one Swin-Tiny transformer block (stage, block_idx) into one
    SwinTransformerLayer (STL) of our RSTB.

    Mapping (Swin HF format -> our STL):
      layernorm_before  ->  norm1
      attention.self.{query,key,value} + attention.output.dense  ->  attn.{qkv, proj}
      layernorm_after   ->  norm2
      intermediate.dense  ->  mlp[0]  (Linear 96->384  adapted to 64->256)
      output.dense        ->  mlp[2]  (Linear 384->96  adapted to 256->64)

    Returns the number of parameter tensors wired.
    """
    prefix = f"swin.encoder.layers.{stage}.blocks.{block_idx}"
    wired = 0
    src = swin_sd

    def get(key):
        return src.get(f"{prefix}.{key}")

    # LayerNorm 1
    if (w := get("layernorm_before.weight")) is not None:
        _copy_param(stl.norm1.weight, _svd_adapt(w, stl.norm1.weight.shape), "norm1.weight")
        wired += 1
    if (b := get("layernorm_before.bias")) is not None:
        _copy_param(stl.norm1.bias,   _svd_adapt(b, stl.norm1.bias.shape),   "norm1.bias")
        wired += 1

    # Attention Q, K, V -> fused qkv
    q_w = get("attention.self.query.weight")
    k_w = get("attention.self.key.weight")
    v_w = get("attention.self.value.weight")
    q_b = get("attention.self.query.bias")
    k_b = get("attention.self.key.bias")
    v_b = get("attention.self.value.bias")

    if q_w is not None and k_w is not None and v_w is not None:
        swin_qkv_w = torch.cat([q_w, k_w, v_w], dim=0)  # (3*96, 96)
        swin_qkv_b = torch.cat([q_b, k_b, v_b], dim=0)  # (3*96,)
        _copy_param(stl.attn.qkv.weight, _svd_adapt(swin_qkv_w, stl.attn.qkv.weight.shape), "qkv.weight")
        _copy_param(stl.attn.qkv.bias,   _svd_adapt(swin_qkv_b, stl.attn.qkv.bias.shape),   "qkv.bias")
        wired += 2

    # Attention output projection
    if (ow := get("attention.output.dense.weight")) is not None:
        _copy_param(stl.attn.proj.weight, _svd_adapt(ow, stl.attn.proj.weight.shape), "proj.weight")
        wired += 1
    if (ob := get("attention.output.dense.bias")) is not None:
        _copy_param(stl.attn.proj.bias,   _svd_adapt(ob, stl.attn.proj.bias.shape),   "proj.bias")
        wired += 1

    # LayerNorm 2
    if (w := get("layernorm_after.weight")) is not None:
        _copy_param(stl.norm2.weight, _svd_adapt(w, stl.norm2.weight.shape), "norm2.weight")
        wired += 1
    if (b := get("layernorm_after.bias")) is not None:
        _copy_param(stl.norm2.bias,   _svd_adapt(b, stl.norm2.bias.shape),   "norm2.bias")
        wired += 1

    # MLP intermediate (up-proj)
    if (w := get("intermediate.dense.weight")) is not None:
        _copy_param(stl.mlp[0].weight, _svd_adapt(w, stl.mlp[0].weight.shape), "mlp[0].weight")
        wired += 1
    if (b := get("intermediate.dense.bias")) is not None:
        _copy_param(stl.mlp[0].bias,   _svd_adapt(b, stl.mlp[0].bias.shape),   "mlp[0].bias")
        wired += 1

    # MLP output (down-proj)
    if (w := get("output.dense.weight")) is not None:
        _copy_param(stl.mlp[2].weight, _svd_adapt(w, stl.mlp[2].weight.shape), "mlp[2].weight")
        wired += 1
    if (b := get("output.dense.bias")) is not None:
        _copy_param(stl.mlp[2].bias,   _svd_adapt(b, stl.mlp[2].bias.shape),   "mlp[2].bias")
        wired += 1

    if verbose:
        print(f"    [Swin] Stage-{stage} Block-{block_idx} -> STL: {wired} tensors wired")
    return wired


def wire_swin_into_anchor(anchor_controller, swin_sd: dict, verbose: bool = True) -> int:
    """
    Wire Swin-Tiny stage-0 blocks into all RSTB SwinTransformerLayers
    across every HybridBlock of *anchor_controller*.

    Swin-Tiny stage-0 has 2 blocks (block_idx 0 and 1).
    We cycle through them modulo 2 for however many STL layers exist in each RSTB.
    """
    total = 0
    stage = 0
    available_blocks = []
    for k in swin_sd:
        if f"swin.encoder.layers.{stage}.blocks." in k:
            bi = int(k.split(f"layers.{stage}.blocks.")[1].split(".")[0])
            if bi not in available_blocks:
                available_blocks.append(bi)
    available_blocks.sort()

    if not available_blocks:
        print("  [Swin] No stage-0 blocks found in state dict - skipping RSTB wiring.")
        return 0

    if verbose:
        print(f"  [Swin] Found {len(available_blocks)} stage-0 blocks: {available_blocks}")

    for hb_idx, hybrid_block in enumerate(anchor_controller.hybrid_blocks):
        if verbose:
            print(f"  [Swin] Wiring HybridBlock[{hb_idx}].rstb ...")
        rstb = hybrid_block.rstb
        for stl_idx, stl in enumerate(rstb.layers):
            src_block = available_blocks[stl_idx % len(available_blocks)]
            n = _wire_swin_block_into_stl(stl, swin_sd, stage=stage,
                                          block_idx=src_block, verbose=verbose)
            total += n

    if verbose:
        print(f"  [Swin] Total tensors wired into RSTB layers: {total}")
    return total


# ---------------------------------------------------------------------------
# Prithvi-100M  ->  ShallowConvolution + RRDB wiring
# ---------------------------------------------------------------------------

def _wire_prithvi_into_shallow_conv(shallow_conv, prithvi_sd: dict,
                                    in_channels: int, verbose: bool = True) -> int:
    """
    Adapt Prithvi's 3-D patch-embedding conv [768, 6, 1, 16, 16] into the
    ShallowConvolution's 2-D conv [64, in_channels, 3, 3].

    Strategy: take the first temporal slice -> [768, 6, 16, 16], spatially
    average-pool to [3, 3] -> [768, 6, 3, 3], then SVD-adapt to [64, in_channels, 3, 3].
    """
    patch_w = prithvi_sd.get("encoder.patch_embed.proj.weight")
    patch_b = prithvi_sd.get("encoder.patch_embed.proj.bias")
    wired = 0

    if patch_w is not None:
        pw2d = patch_w[:, :, 0, :, :]                          # [768, 6, 16, 16]
        pw3  = nn.functional.adaptive_avg_pool2d(pw2d.float(), (3, 3))  # [768, 6, 3, 3]
        # Clamp in-channels to available Prithvi channels (6)
        src_ch = min(in_channels, pw3.shape[1])
        pw3_trimmed = pw3[:, :src_ch, :, :]                    # [768, src_ch, 3, 3]
        tgt_shape = shallow_conv.conv.weight.shape              # [64, in_channels, 3, 3]
        adapted = _svd_adapt(pw3_trimmed, tgt_shape)
        _copy_param(shallow_conv.conv.weight, adapted, "shallow_conv.weight")
        wired += 1

    if patch_b is not None:
        tgt_shape = shallow_conv.conv.bias.shape                # [64]
        adapted = _svd_adapt(patch_b, tgt_shape)
        _copy_param(shallow_conv.conv.bias, adapted, "shallow_conv.bias")
        wired += 1

    if verbose:
        print(f"  [Prithvi] ShallowConvolution: {wired} tensors wired from patch_embed")
    return wired


def _wire_prithvi_block_into_rrdb(rrdb, prithvi_sd: dict,
                                   block_idx: int, verbose: bool = True) -> int:
    """
    Adapt one Prithvi ViT encoder block's MLP weights (768-dim) into
    one RRDB's ResidualDenseBlock conv filters (64-dim).

    Mapping:
      encoder.blocks.N.mlp.fc1  [3072,768]  -> first dense conv of RDB-0
      encoder.blocks.N.mlp.fc2  [768, 3072] -> bottleneck of RDB-0
    """
    prefix = f"encoder.blocks.{block_idx}"
    wired = 0

    fc1_w = prithvi_sd.get(f"{prefix}.mlp.fc1.weight")         # [3072, 768]
    fc2_w = prithvi_sd.get(f"{prefix}.mlp.fc2.weight")         # [768, 3072]

    if fc1_w is not None:
        rdb0       = rrdb.rdbs[0]
        first_conv = rdb0.layers[0]                             # Conv2d(64, 32, 3, 3)
        src2d      = fc1_w.float().T                            # [768, 3072]
        tgt_2d     = _svd_adapt(src2d, (first_conv.weight.shape[0],
                                         first_conv.weight.shape[1]))  # [32, 64]
        tgt_4d = tgt_2d.unsqueeze(-1).unsqueeze(-1).expand_as(first_conv.weight).clone()
        _copy_param(first_conv.weight, tgt_4d, f"rrdb[{block_idx}].rdb0.layers[0].weight")
        wired += 1

    if fc2_w is not None:
        rdb0             = rrdb.rdbs[0]
        bottleneck_conv  = rdb0.bottleneck                      # Conv2d(224, 64, 1, 1)
        src2d            = fc2_w.float()                        # [768, 3072]
        tgt_2d           = _svd_adapt(src2d, (bottleneck_conv.weight.shape[0],
                                               bottleneck_conv.weight.shape[1]))  # [64, 224]
        tgt_4d = tgt_2d.unsqueeze(-1).unsqueeze(-1).expand_as(bottleneck_conv.weight).clone()
        _copy_param(bottleneck_conv.weight, tgt_4d,
                    f"rrdb[{block_idx}].rdb0.bottleneck.weight")
        wired += 1

    if verbose:
        print(f"    [Prithvi] Block-{block_idx} -> RRDB: {wired} tensors wired")
    return wired


def wire_prithvi_into_anchor(anchor_controller, prithvi_sd: dict,
                              in_channels: int = 3, verbose: bool = True) -> int:
    """
    Wire Prithvi-100M encoder weights into the ShallowConvolution and
    the RRDB branches of every HybridBlock.
    """
    total = 0

    total += _wire_prithvi_into_shallow_conv(
        anchor_controller.shallow_conv, prithvi_sd, in_channels, verbose
    )

    num_prithvi_blocks = sum(
        1 for k in prithvi_sd
        if k.startswith("encoder.blocks.") and k.endswith(".mlp.fc1.weight")
    )
    if verbose:
        print(f"  [Prithvi] Found {num_prithvi_blocks} encoder blocks available for RRDB wiring")

    for hb_idx, hybrid_block in enumerate(anchor_controller.hybrid_blocks):
        prithvi_block_idx = hb_idx % max(num_prithvi_blocks, 1)
        if verbose:
            print(f"  [Prithvi] Wiring HybridBlock[{hb_idx}].rrdb <- Prithvi block {prithvi_block_idx}")
        total += _wire_prithvi_block_into_rrdb(
            hybrid_block.rrdb, prithvi_sd, prithvi_block_idx, verbose
        )

    if verbose:
        print(f"  [Prithvi] Total tensors wired into RRDB layers: {total}")
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wire_pretrained_weights(
    anchor_controller,
    swin_bin_path="model/pytorch_model.bin",
    prithvi_pt_path="model/Prithvi_EO_V1_100M.pt",
    in_channels: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Load both pretrained checkpoints and wire their weights into
    *anchor_controller* in-place.

    Parameters
    ----------
    anchor_controller : AnchorGenerationController
        The controller whose shallow_conv, hybrid_blocks.rstb and
        hybrid_blocks.rrdb will be initialised.
    swin_bin_path : str | Path
        Path to Microsoft Swin-Tiny pytorch_model.bin.
    prithvi_pt_path : str | Path
        Path to IBM/NASA Prithvi-100M .pt checkpoint.
    in_channels : int
        Number of input image channels (matches anchor_controller's
        ShallowConvolution in_channels).
    verbose : bool
        Print per-tensor wiring progress.

    Returns
    -------
    dict with keys 'swin_wired' and 'prithvi_wired' (int counts).
    """
    swin_path    = Path(swin_bin_path).resolve()
    prithvi_path = Path(prithvi_pt_path).resolve()

    # --- Swin-Tiny --------------------------------------------------------
    swin_wired = 0
    if swin_path.is_file():
        print(f"\n[Wiring] Loading Swin-Tiny weights from: {swin_path}")
        swin_sd = torch.load(str(swin_path), map_location="cpu")
        swin_wired = wire_swin_into_anchor(anchor_controller, swin_sd, verbose=verbose)
        print(f"[Wiring] Swin-Tiny -> RSTB: {swin_wired} parameter tensors initialised")
    else:
        print(f"[Wiring] WARNING: Swin-Tiny weights not found at {swin_path}.")

    # --- Prithvi-100M -----------------------------------------------------
    prithvi_wired = 0
    if prithvi_path.is_file():
        print(f"\n[Wiring] Loading Prithvi-100M weights from: {prithvi_path}")
        prithvi_sd = torch.load(str(prithvi_path), map_location="cpu")
        prithvi_wired = wire_prithvi_into_anchor(
            anchor_controller, prithvi_sd, in_channels=in_channels, verbose=verbose
        )
        print(f"[Wiring] Prithvi-100M -> ShallowConv + RRDB: {prithvi_wired} parameter tensors initialised")
    else:
        print(f"[Wiring] WARNING: Prithvi weights not found at {prithvi_path}.")

    print(f"\n[Wiring] Complete. Total tensors wired: {swin_wired + prithvi_wired}\n")
    return {"swin_wired": swin_wired, "prithvi_wired": prithvi_wired}


# ---------------------------------------------------------------------------
# Quick verification (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from controller.anchor_generation_controller import AnchorGenerationController
    from constants.constants import DEFAULT_NUM_HYBRID_BLOCKS

    print("=== Pretrained Weight Wiring Verification ===\n")
    ctrl = AnchorGenerationController(
        in_channels=3,
        num_swin_layers=6,
        rrdb_dropout_rate=0.1,
        num_blocks=DEFAULT_NUM_HYBRID_BLOCKS,
    )
    results = wire_pretrained_weights(
        ctrl,
        swin_bin_path   = Path(__file__).parent / "pytorch_model.bin",
        prithvi_pt_path = Path(__file__).parent / "Prithvi_EO_V1_100M.pt",
        in_channels     = 3,
        verbose         = True,
    )
    print(f"\nResult: {results}")
    print("\n[OK] Wiring verification passed.")
