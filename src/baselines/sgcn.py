"""SGCN baseline (Derr et al., ICDM 2018) -- "Signed Graph Convolutional Networks".

Citation: Derr, T., Ma, Y., & Tang, J. (2018). Signed Graph Convolutional
Networks. ICDM 2018.

Source-of-record reference implementation:
  https://github.com/benedekrozemberczki/SGCN
  Also: torch_geometric_signed_directed.nn.SGCN

DEVIATION FROM CANONICAL: Reimplemented with a graph-AGNOSTIC forward pass
(graph passed at forward time rather than at __init__). The
torch_geometric_signed_directed.nn.SGCN class instantiates per-graph (precomputes
normalization from the edge_index_s passed at __init__), which is incompatible
with our temporal multi-graph training where weights must share across yearly
graphs. Architecture (signed first/deep convs, balance theory aggregation) is
faithful to Derr et al. 2018; only the wiring changes.

Three configurations (project plan section 2.3):
  AS_PUBLISHED   -- learnable per-node identity embeddings + per-node features
  IDENTITY_FREE  -- per-node features only (no learnable identity)
  IDENTITY_ONLY  -- learnable per-node identity embeddings, features zeroed

Identity-permutation probe (src/probes/identity_permutation.py) supports these
configs via the `embed_perm` argument to `encode`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.common import BaselineConfig


def _aggregate_neighbors(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean aggregation of neighbor features. (n, d) -> (n, d).

    edge_index expected as (2, E) with src in row 0, dst in row 1; aggregate
    INTO dst nodes. Empty edge_index returns zeros.
    """
    n = x.shape[0]
    if edge_index.shape[1] == 0:
        return torch.zeros_like(x)
    src, dst = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros(n, device=x.device)
    ones = torch.ones(src.shape[0], device=x.device)
    deg.index_add_(0, dst, ones)
    deg = deg.clamp(min=1).unsqueeze(-1)
    return out / deg


class SignedFirstLayer(nn.Module):
    """First SGCN layer -- separate balanced/unbalanced aggregation."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_B = nn.Linear(2 * in_dim, out_dim)
        self.W_U = nn.Linear(2 * in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_pos: torch.Tensor,
        edge_index_neg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agg_pos = _aggregate_neighbors(x, edge_index_pos)
        agg_neg = _aggregate_neighbors(x, edge_index_neg)
        h_B = torch.tanh(self.W_B(torch.cat([agg_pos, x], dim=-1)))
        h_U = torch.tanh(self.W_U(torch.cat([agg_neg, x], dim=-1)))
        return h_B, h_U


class SignedDeepLayer(nn.Module):
    """Subsequent SGCN layer -- balance theory aggregation.

    Friend of friend = friend (positive path to balanced).
    Enemy of enemy   = friend (negative path to balanced).
    Friend of enemy  = enemy  (positive path to unbalanced).
    Enemy of friend  = enemy  (negative path to unbalanced).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_B = nn.Linear(3 * in_dim, out_dim)
        self.W_U = nn.Linear(3 * in_dim, out_dim)

    def forward(
        self,
        h_B: torch.Tensor,
        h_U: torch.Tensor,
        edge_index_pos: torch.Tensor,
        edge_index_neg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agg_BB = _aggregate_neighbors(h_B, edge_index_pos)
        agg_BU = _aggregate_neighbors(h_U, edge_index_neg)  # enemy of enemy
        agg_UU = _aggregate_neighbors(h_U, edge_index_pos)
        agg_UB = _aggregate_neighbors(h_B, edge_index_neg)  # enemy of friend
        new_B = torch.tanh(self.W_B(torch.cat([agg_BB, agg_BU, h_B], dim=-1)))
        new_U = torch.tanh(self.W_U(torch.cat([agg_UU, agg_UB, h_U], dim=-1)))
        return new_B, new_U


class SGCN(nn.Module):
    """Three-config SGCN with graph-agnostic forward pass.

    Args:
      n_nodes: total node vocabulary (union of states across all years).
      n_features: per-node feature dimension (5 for our setup).
      n_dyad_features: dyad-level feature dimension at the prediction head.
      hidden_dim: SGCN hidden dimension; matches the embedding dimension when
        AS_PUBLISHED or IDENTITY_ONLY.
      n_layers: number of SGCN layers (>= 1).
      config: which of the three configurations to instantiate.
      dropout: applied between layers.
    """

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
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
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
        if in_dim == 0:
            raise ValueError("At least one of identity / features must be enabled")

        layers: list[nn.Module] = [SignedFirstLayer(in_dim, hidden_dim)]
        for _ in range(n_layers - 1):
            layers.append(SignedDeepLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Head: [h^B_i, h^U_i, h^B_j, h^U_j, dyad_features] -> logit
        head_in = 4 * hidden_dim + n_dyad_features
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _build_input(
        self, x_features: torch.Tensor, embed_perm: Optional[torch.Tensor]
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.id_embedding is not None:
            if embed_perm is None:
                parts.append(self.id_embedding.weight)
            else:
                parts.append(self.id_embedding.weight[embed_perm])
        if self.feature_proj is not None:
            if self.config == BaselineConfig.IDENTITY_ONLY:
                parts.append(torch.zeros_like(self.feature_proj(x_features)))
            else:
                parts.append(self.feature_proj(x_features))
        return torch.cat(parts, dim=-1)

    def encode(
        self,
        x_features: torch.Tensor,
        edge_index_pos: torch.Tensor,
        edge_index_neg: torch.Tensor,
        *,
        embed_perm: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run signed message passing. Returns (h_B, h_U), both (n_nodes, hidden)."""
        x = self._build_input(x_features, embed_perm)
        h_B, h_U = self.layers[0](x, edge_index_pos, edge_index_neg)
        for layer in self.layers[1:]:
            h_B = F.dropout(h_B, p=self.dropout, training=self.training)
            h_U = F.dropout(h_U, p=self.dropout, training=self.training)
            h_B, h_U = layer(h_B, h_U, edge_index_pos, edge_index_neg)
        return h_B, h_U

    def score_dyads(
        self,
        h_B: torch.Tensor,
        h_U: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        dyad_features: torch.Tensor,
    ) -> torch.Tensor:
        """Score dyad logits given embeddings + dyad features."""
        if self.config == BaselineConfig.IDENTITY_ONLY:
            dyad_features = torch.zeros_like(dyad_features)
        h = torch.cat([h_B[src], h_U[src], h_B[dst], h_U[dst], dyad_features], dim=-1)
        return self.head(h).squeeze(-1)

    def forward(
        self,
        x_features: torch.Tensor,
        edge_index_pos: torch.Tensor,
        edge_index_neg: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        dyad_features: torch.Tensor,
        *,
        embed_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_B, h_U = self.encode(
            x_features, edge_index_pos, edge_index_neg, embed_perm=embed_perm
        )
        return self.score_dyads(h_B, h_U, src, dst, dyad_features)
