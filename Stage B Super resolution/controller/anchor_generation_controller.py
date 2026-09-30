"""
anchor_generation_controller.py

Orchestrates Stage B.1 (Anchor Generation Network), following the ordered
generation process documented in Anchor Image.md section 2:

    I_LR -> Shallow Convolution (F0) -> Hybrid Block x N (F_DF)
         -> Globally Fused Features (F_agg) -> Upsampling -> Anchor Head -> I_SR_anchor

Contains no neural-network layer implementations of its own; it only sequences
calls to the Model components (Controller contract).
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

from model.shallow_convolution_model import ShallowConvolution
from model.hybrid_block_model import HybridBlock, DEFAULT_NUM_BLOCKS
from model.globally_fused_features_model import GloballyFusedFeatures
from model.upsampling_model import Upsampling
from model.anchor_image_model import AnchorImageArtifact


class AnchorGenerationController:
    def __init__(self, in_channels: int, num_swin_layers: int, rrdb_dropout_rate: float,
                 num_blocks: int = DEFAULT_NUM_BLOCKS):
        self.shallow_conv = ShallowConvolution(in_channels)
        self.hybrid_blocks = nn.ModuleList([
            HybridBlock(num_swin_layers=num_swin_layers, rrdb_dropout_rate=rrdb_dropout_rate)
            for _ in range(num_blocks)
        ])
        self.fusion = GloballyFusedFeatures(channels=64)
        self.upsampling = Upsampling(channels=64, out_channels=in_channels)

    def modules(self) -> List[nn.Module]:
        """Expose contained nn.Modules for optimizer/state_dict management."""
        return [self.shallow_conv, self.hybrid_blocks, self.fusion, self.upsampling]

    def run(self, i_lr: torch.Tensor, clamp_output: bool = False) -> AnchorImageArtifact:
        f0 = self.shallow_conv(i_lr)
        f = f0
        for block in self.hybrid_blocks:
            f = block(f)
        f_df = f
        f_agg = self.fusion(f0, f_df)
        i_sr_anchor = self.upsampling(f_agg, clamp=clamp_output)
        return AnchorImageArtifact(i_sr_anchor=i_sr_anchor)