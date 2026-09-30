"""
timestep_model.py

Implements the "Timestep" node (t'): sinusoidal embedding + MLP, per Timestep.md.

TODO: INFORMATION REQUIRED FROM CONTEXT - neither the embedding dimension 'd' nor
the MLP's hidden-layer width is given a numeric value anywhere in the 19 context
files. Both are required constructor arguments.
"""
import torch
import torch.nn as nn

from model.common_layers import SinusoidalPositionEmbedding, AdaGN  # re-exported for convenience

__all__ = ["TimestepEmbedder", "AdaGN"]


class TimestepEmbedder(nn.Module):
    """e_t' = SinusoidalPositionEmbedding(t'); t_emb = MLP(e_t')."""

    def __init__(self, embedding_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, embedding_dim),
        )

    def forward(self, t_prime: torch.Tensor) -> torch.Tensor:
        e = self.sinusoidal(t_prime)
        return self.mlp(e)