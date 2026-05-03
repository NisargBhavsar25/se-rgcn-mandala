"""SDGNN baseline (Huang et al., AAAI 2021) -- "SDGNN: Learning Node
Representation for Signed Directed Networks".

Citation: Huang, J., Shen, H., Hou, L., & Cheng, X. (2021). SDGNN: Learning
Node Representation for Signed Directed Networks. AAAI 2021.

DEVIATION FROM CANONICAL: SDGNN was designed for SIGNED DIRECTED graphs with
4 edge types (positive-incoming, positive-outgoing, negative-incoming,
negative-outgoing). Our signed graphs are UNDIRECTED (alliance and conflict
edges between dyads are symmetric). We retain SDGNN's 4-weight-matrix
structure -- the model has the capacity for 4 distinct directed
aggregations -- but feed it undirected edges (each (i,j) appears as both
(i,j) and (j,i)). Effectively the model has twice the weight capacity of
SGCN per layer but sees the pos-in / pos-out (and neg-in / neg-out) edge
sets as identical. This is the cleanest adaptation; documenting that
forcing direction on undirected dyad data would be arbitrary.

Three configurations (project plan section 2.3): AS_PUBLISHED, IDENTITY_FREE,
IDENTITY_ONLY. Identity-permutation probe supported via embed_perm argument.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.common import BaselineConfig


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean over neighbors -- aggregates from src into dst."""
    n = x.shape[0]
    if edge_index.shape[1] == 0:
        return torch.zeros_like(x)
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros(n, device=x.device)
    deg.index_add_(0, dst, torch.ones(src.shape[0], device=x.device))
    return out / deg.clamp(min=1).unsqueeze(-1)


class SDGNNLayer(nn.Module):
    """Four-stream aggregation: pos-out, pos-in, neg-out, neg-in."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_pos_out = nn.Linear(in_dim, out_dim)
        self.W_pos_in = nn.Linear(in_dim, out_dim)
        self.W_neg_out = nn.Linear(in_dim, out_dim)
        self.W_neg_in = nn.Linear(in_dim, out_dim)
        self.combine = nn.Linear(4 * out_dim + in_dim, out_dim)

    def forward(self, x, edge_index_pos, edge_index_neg):
        rev_pos = (
            torch.flip(edge_index_pos, dims=[0]) if edge_index_pos.shape[1] > 0
            else edge_index_pos
        )
        rev_neg = (
            torch.flip(edge_index_neg, dims=[0]) if edge_index_neg.shape[1] > 0
            else edge_index_neg
        )
        agg_pos_out = _mean_aggregate(self.W_pos_out(x), edge_index_pos)
        agg_pos_in = _mean_aggregate(self.W_pos_in(x), rev_pos)
        agg_neg_out = _mean_aggregate(self.W_neg_out(x), edge_index_neg)
        agg_neg_in = _mean_aggregate(self.W_neg_in(x), rev_neg)
        h = torch.cat([agg_pos_out, agg_pos_in, agg_neg_out, agg_neg_in, x], dim=-1)
        return F.relu(self.combine(h))


class SDGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        n_dyad_features: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        config: BaselineConfig = BaselineConfig.AS_PUBLISHED,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_dyad_features = n_dyad_features

        in_dim = 0
        if config in (BaselineConfig.AS_PUBLISHED, BaselineConfig.IDENTITY_ONLY):
            self.id_embedding = nn.Embedding(n_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.id_embedding.weight)
            in_dim += hidden_dim
        else:
            self.id_embedding = None
        if config in (BaselineConfig.AS_PUBLISHED, BaselineConfig.IDENTITY_FREE):
            self.feature_proj = nn.Linear(n_features, hidden_dim)
            in_dim += hidden_dim
        else:
            self.feature_proj = None

        # Project input dim to hidden_dim before layers, so all layers are hidden->hidden.
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([SDGNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])

        # Single-stream head: SDGNN produces ONE representation per node (no balanced/unbalanced split).
        head_in = 2 * hidden_dim + n_dyad_features
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def _build_input(self, x_features, embed_perm):
        parts = []
        if self.id_embedding is not None:
            parts.append(
                self.id_embedding.weight if embed_perm is None
                else self.id_embedding.weight[embed_perm]
            )
        if self.feature_proj is not None:
            proj = self.feature_proj(x_features)
            parts.append(torch.zeros_like(proj) if self.config == BaselineConfig.IDENTITY_ONLY else proj)
        return torch.cat(parts, dim=-1)

    def encode(self, x_features, edge_index_pos, edge_index_neg, *, embed_perm=None):
        x = self._build_input(x_features, embed_perm)
        h = self.input_proj(x)
        for layer in self.layers:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(h, edge_index_pos, edge_index_neg)
        return h

    def score_dyads(self, h, src, dst, dyad_features):
        if self.config == BaselineConfig.IDENTITY_ONLY:
            dyad_features = torch.zeros_like(dyad_features)
        return self.head(torch.cat([h[src], h[dst], dyad_features], dim=-1)).squeeze(-1)

    def forward(
        self, x_features, edge_index_pos, edge_index_neg,
        src, dst, dyad_features, *, embed_perm=None,
    ):
        h = self.encode(x_features, edge_index_pos, edge_index_neg, embed_perm=embed_perm)
        return self.score_dyads(h, src, dst, dyad_features)
