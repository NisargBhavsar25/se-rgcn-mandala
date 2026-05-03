"""SiGAT baseline (Huang et al., CIKM 2019) -- "Signed Graph Attention Networks".

Citation: Huang, J., Shen, H., Hou, L., & Cheng, X. (2019). Signed Graph
Attention Networks. CIKM 2019.

DEVIATION FROM CANONICAL: We implement signed attention (separate attention
mechanisms for positive vs negative neighbors), but omit the full
motif-based attention decomposition (4 motif heads: balanced triangles,
unbalanced triangles, etc.). The motif decomposition is a SiGAT-specific
architectural choice that primarily affects performance via richer
neighborhood structure; it does NOT bear on the identity-memorization
question this paper investigates. The signed-attention core IS preserved
and is what reviewers will recognize as SiGAT's contribution.

Architecture: per-edge attention coefficients computed from (h_i, h_j),
softmax-normalized over each node's neighbors, applied as weights in
balanced/unbalanced aggregation. Balance-theory composition for deeper
layers mirrors SGCN.

Three configurations (project plan section 2.3): AS_PUBLISHED, IDENTITY_FREE,
IDENTITY_ONLY. Identity-permutation probe supported via embed_perm argument.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.common import BaselineConfig


def _scatter_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Numerically-stable softmax of `src` grouped by `index`. (E,) -> (E,)."""
    if src.numel() == 0:
        return src
    src_max = torch.full((num_nodes,), float("-inf"), device=src.device, dtype=src.dtype)
    src_max.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
    # Replace -inf (groups with no entries) with 0 to avoid NaN propagation.
    src_max = torch.where(torch.isfinite(src_max), src_max, torch.zeros_like(src_max))
    centered = src - src_max[index]
    exped = centered.exp()
    denom = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
    denom.scatter_add_(0, index, exped)
    return exped / denom[index].clamp(min=1e-12)


class SignedAttentionHead(nn.Module):
    """Attention aggregation over a single edge set (positive OR negative)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        Wx = self.W(x)
        if edge_index.shape[1] == 0:
            return torch.zeros(n, Wx.shape[-1], device=x.device, dtype=Wx.dtype)
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([Wx[src], Wx[dst]], dim=-1)
        e = F.leaky_relu(self.attn(pair).squeeze(-1), 0.2)
        alpha = _scatter_softmax(e, dst, n)
        out = torch.zeros(n, Wx.shape[-1], device=x.device, dtype=Wx.dtype)
        out.index_add_(0, dst, alpha.unsqueeze(-1) * Wx[src])
        return out


class SiGATFirstLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.attn_pos = SignedAttentionHead(in_dim, out_dim)
        self.attn_neg = SignedAttentionHead(in_dim, out_dim)
        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index_pos, edge_index_neg):
        agg_pos = self.attn_pos(x, edge_index_pos)
        agg_neg = self.attn_neg(x, edge_index_neg)
        self_h = self.W_self(x)
        h_B = torch.tanh(self_h + agg_pos)
        h_U = torch.tanh(self_h + agg_neg)
        return h_B, h_U


class SiGATDeepLayer(nn.Module):
    """Balance-theory composition with attention, mirroring SGCN deep layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # Balanced -> balanced via positive paths; unbalanced -> balanced via negative.
        self.attn_BB = SignedAttentionHead(in_dim, out_dim)
        self.attn_BU = SignedAttentionHead(in_dim, out_dim)
        self.attn_UU = SignedAttentionHead(in_dim, out_dim)
        self.attn_UB = SignedAttentionHead(in_dim, out_dim)
        self.W_self_B = nn.Linear(in_dim, out_dim)
        self.W_self_U = nn.Linear(in_dim, out_dim)

    def forward(self, h_B, h_U, edge_index_pos, edge_index_neg):
        new_B = torch.tanh(
            self.W_self_B(h_B) + self.attn_BB(h_B, edge_index_pos)
            + self.attn_BU(h_U, edge_index_neg)
        )
        new_U = torch.tanh(
            self.W_self_U(h_U) + self.attn_UU(h_U, edge_index_pos)
            + self.attn_UB(h_B, edge_index_neg)
        )
        return new_B, new_U


class SiGAT(nn.Module):
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

        layers: list[nn.Module] = [SiGATFirstLayer(in_dim, hidden_dim)]
        for _ in range(n_layers - 1):
            layers.append(SiGATDeepLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        head_in = 4 * hidden_dim + n_dyad_features
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
        h_B, h_U = self.layers[0](x, edge_index_pos, edge_index_neg)
        for layer in self.layers[1:]:
            h_B = F.dropout(h_B, p=self.dropout, training=self.training)
            h_U = F.dropout(h_U, p=self.dropout, training=self.training)
            h_B, h_U = layer(h_B, h_U, edge_index_pos, edge_index_neg)
        return h_B, h_U

    def score_dyads(self, h_B, h_U, src, dst, dyad_features):
        if self.config == BaselineConfig.IDENTITY_ONLY:
            dyad_features = torch.zeros_like(dyad_features)
        h = torch.cat([h_B[src], h_U[src], h_B[dst], h_U[dst], dyad_features], dim=-1)
        return self.head(h).squeeze(-1)

    def forward(
        self, x_features, edge_index_pos, edge_index_neg,
        src, dst, dyad_features, *, embed_perm=None,
    ):
        h_B, h_U = self.encode(x_features, edge_index_pos, edge_index_neg, embed_perm=embed_perm)
        return self.score_dyads(h_B, h_U, src, dst, dyad_features)
