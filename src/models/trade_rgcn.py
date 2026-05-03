"""Trade-only RGCN baseline (reviewer Week 2 spec).

Architecture: per-state learnable embeddings -> RGCN with one relation
(trade) -> per-dyad MLP head taking [h_i, h_j, raw_log_trade_ij] -> logit.

Why include raw `log_trade_ij` at the head: PyG RGCNConv ignores edge
weights (only relation type matters), so without re-injecting trade
magnitude at the head, the model would lose the gravity-model signal
entirely and the "trade-only baseline" would understate trade's
predictive power.

Sized for 4 GB VRAM: ~200 nodes x 64-dim hidden -> trivially fits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import RGCNConv


class TradeOnlyRGCN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(n_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=1)
            for _ in range(n_layers)
        ])
        self.dropout = dropout
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Run RGCN message passing over the trade graph; return node embeddings."""
        h = self.node_emb.weight
        for conv in self.convs:
            h = conv(h, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def score_dyads(
        self,
        node_h: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        log_trade_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Score dyads given precomputed node embeddings; returns logits (B,)."""
        edge_h = torch.cat(
            [node_h[src_idx], node_h[dst_idx], log_trade_ij.unsqueeze(-1)],
            dim=-1,
        )
        return self.edge_mlp(edge_h).squeeze(-1)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        log_trade_ij: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end forward: encode + score."""
        h = self.encode(edge_index, edge_type)
        return self.score_dyads(h, src_idx, dst_idx, log_trade_ij)
