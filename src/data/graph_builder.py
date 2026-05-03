"""Per-year multiplex graph constructor for SE-RGCN.

Builds a per-year PyG-friendly graph object. For the trade-only baseline,
the graph has a single relation (trade); the same scaffold accepts
additional relations (military, spatial) once the trade-only gate is
passed and SE-RGCN proper is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class YearlyGraph:
    """Per-year graph plus the local-node-id <-> GW-code mapping."""

    year: int
    node_to_gwcode: dict[int, int]
    gwcode_to_node: dict[int, int]
    edge_index: torch.Tensor      # (2, E)
    edge_type: torch.Tensor       # (E,) -- relation index
    edge_attr: torch.Tensor       # (E, F)

    @property
    def num_nodes(self) -> int:
        return len(self.node_to_gwcode)


def build_yearly_trade_graph(
    trade_year: pd.DataFrame,
    states: list[int],
    *,
    log_transform: bool = True,
) -> YearlyGraph:
    """Build a single-relation (trade) graph for one year.

    Args:
      trade_year: rows for ONE year only with columns
        ['year', 'gwcode_i', 'gwcode_j', 'total_trade'] (canonical i < j).
      states: GW codes that should be nodes (typically the PRD-active set
        for that year). States with no trade still appear as isolated nodes.
      log_transform: if True, edge_attr = log(1 + total_trade).
    """
    sorted_states = sorted(set(states))
    gw_to_node = {gw: i for i, gw in enumerate(sorted_states)}
    node_to_gw = {i: gw for gw, i in gw_to_node.items()}

    year = int(trade_year["year"].iloc[0]) if len(trade_year) else 0

    if not states:
        return YearlyGraph(
            year=year,
            node_to_gwcode=node_to_gw,
            gwcode_to_node=gw_to_node,
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_type=torch.empty(0, dtype=torch.long),
            edge_attr=torch.empty(0, 1, dtype=torch.float32),
        )

    valid = trade_year[
        trade_year["gwcode_i"].isin(gw_to_node) &
        trade_year["gwcode_j"].isin(gw_to_node)
    ]

    if len(valid) == 0:
        return YearlyGraph(
            year=year,
            node_to_gwcode=node_to_gw,
            gwcode_to_node=gw_to_node,
            edge_index=torch.empty(2, 0, dtype=torch.long),
            edge_type=torch.empty(0, dtype=torch.long),
            edge_attr=torch.empty(0, 1, dtype=torch.float32),
        )

    src = valid["gwcode_i"].map(gw_to_node).to_numpy()
    dst = valid["gwcode_j"].map(gw_to_node).to_numpy()
    weights = valid["total_trade"].to_numpy(dtype=np.float32)
    if log_transform:
        weights = np.log1p(weights)

    # Undirected: add reverse edges.
    fwd = np.stack([src, dst])
    rev = np.stack([dst, src])
    ei = np.concatenate([fwd, rev], axis=1)
    ea = np.concatenate([weights, weights])

    return YearlyGraph(
        year=year,
        node_to_gwcode=node_to_gw,
        gwcode_to_node=gw_to_node,
        edge_index=torch.from_numpy(ei).long(),
        edge_type=torch.zeros(ei.shape[1], dtype=torch.long),
        edge_attr=torch.from_numpy(ea).float().unsqueeze(-1),
    )


def lookup_dyad_trade(
    trade_year: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    log_transform: bool = True,
) -> np.ndarray:
    """Look up trade volume for an arbitrary set of (i, j) dyads.

    Args:
      trade_year: trade rows for one year (canonical i < j).
      pairs: DataFrame with columns ['gwcode_i', 'gwcode_j'] in canonical
        ordering.

    Returns:
      Float array of length len(pairs) with log(1 + total_trade) for each
      dyad (or raw total_trade if log_transform=False). Dyads not present
      in trade_year get 0.
    """
    lookup = trade_year.set_index(["gwcode_i", "gwcode_j"])["total_trade"]
    keys = list(zip(pairs["gwcode_i"].to_numpy(), pairs["gwcode_j"].to_numpy()))
    raw = np.fromiter(
        (lookup.get(k, 0.0) for k in keys), dtype=np.float32, count=len(keys)
    )
    return np.log1p(raw) if log_transform else raw
