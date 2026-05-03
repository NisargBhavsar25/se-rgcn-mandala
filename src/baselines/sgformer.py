"""SignedTransformer baseline -- generic signed-graph transformer.

The project plan listed "SGformer 2024" as a baseline. This name does not
refer to one canonical paper -- there is SGFormer (Wu et al., NeurIPS 2023)
for unsigned graphs, plus several signed-graph transformer papers from
2023-2024 (Bagasrawala et al., signed graph transformer; etc.). Rather than
pick one and risk misrepresentation, this file implements a clean
"signed-transformer-of-record" combining standard architectural choices
from the genre:

  - Multi-head self-attention over all nodes (full attention, not localized)
  - Sign-aware additive bias on attention scores: positive edges get a
    learnable positive bias, negative edges get a learnable negative bias,
    no edge gets a learnable neutral bias.
  - Residual + LayerNorm + FFN per layer (transformer block standard)
  - Per-node output goes to the same dyad-prediction head as the other
    baselines.

The test of identity memorization is invariant to which specific signed
transformer paper this is -- the architectural family is what matters for
the question. Documented in the file docstring as such.

Three configurations (project plan section 2.3): AS_PUBLISHED, IDENTITY_FREE,
IDENTITY_ONLY. Identity-permutation probe supported via embed_perm argument.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.common import BaselineConfig


class SignedTransformerLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.dropout = nn.Dropout(dropout)
        # Sign-aware additive biases on attention scores.
        # bias[0] = no edge, bias[1] = positive edge, bias[2] = negative edge.
        self.sign_bias = nn.Parameter(torch.zeros(3))

    def _build_sign_mask(
        self, n: int, edge_index_pos: torch.Tensor, edge_index_neg: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.full((n, n), self.sign_bias[0].item(), device=device)
        mask.fill_(self.sign_bias[0].item())
        # Use functional indexing with parameter so gradients flow.
        if edge_index_pos.shape[1] > 0:
            mask[edge_index_pos[0], edge_index_pos[1]] = self.sign_bias[1]
        if edge_index_neg.shape[1] > 0:
            mask[edge_index_neg[0], edge_index_neg[1]] = self.sign_bias[2]
        return mask

    def forward(self, x, edge_index_pos, edge_index_neg):
        n = x.shape[0]
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape for multi-head: (n, dim) -> (heads, n, head_dim)
        q = q.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.n_heads, self.head_dim).transpose(0, 1)
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Sign-aware additive bias (broadcast across heads)
        sign_mask = self._build_sign_mask(n, edge_index_pos, edge_index_neg, x.device)
        scores = scores + sign_mask.unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        out_attn = torch.matmul(attn, v)  # (heads, n, head_dim)
        out_attn = out_attn.transpose(0, 1).reshape(n, self.dim)
        out_attn = self.dropout(self.out_proj(out_attn))
        x = x + out_attn  # residual
        x = x + self.dropout(self.ffn(self.norm2(x)))  # FFN with norm + residual
        return x


class SignedTransformer(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        n_dyad_features: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
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

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            SignedTransformerLayer(hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

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
