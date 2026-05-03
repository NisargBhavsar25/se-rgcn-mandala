"""Tests for the per-year multiplex graph builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data.graph_builder import build_yearly_trade_graph, lookup_dyad_trade


def _trade_df(year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "year":        [year, year, year],
        "gwcode_i":    [   2,    2,  200],
        "gwcode_j":    [ 200,  365,  365],
        "total_trade":[100.0, 50.0,  10.0],
    })


def test_node_to_gwcode_mapping_is_sorted():
    g = build_yearly_trade_graph(_trade_df(2000), states=[365, 2, 200])
    assert g.node_to_gwcode == {0: 2, 1: 200, 2: 365}
    assert g.gwcode_to_node == {2: 0, 200: 1, 365: 2}


def test_edges_are_undirected_with_both_directions():
    g = build_yearly_trade_graph(_trade_df(2000), states=[2, 200, 365])
    # 3 trade rows -> 6 edges (undirected x2)
    assert g.edge_index.shape == (2, 6)
    assert g.edge_type.shape == (6,)
    assert g.edge_attr.shape == (6, 1)
    # Type is all-zero (single relation = trade)
    assert (g.edge_type == 0).all()


def test_log_transform_applied_when_requested():
    g = build_yearly_trade_graph(_trade_df(2000), states=[2, 200, 365])
    # First row (2,200) -> trade 100 -> log(101)
    expected = np.log1p(100.0)
    found = float(g.edge_attr[g.edge_attr.argmax()])
    assert abs(found - expected) < 1e-5


def test_states_with_no_trade_appear_as_isolated_nodes():
    g = build_yearly_trade_graph(_trade_df(2000), states=[2, 200, 365, 999])
    assert g.num_nodes == 4
    # Node 3 (gw 999) has no edges
    nodes_with_edges = set(g.edge_index.flatten().tolist())
    assert 3 not in nodes_with_edges


def test_trade_dyads_outside_state_set_dropped():
    df = _trade_df(2000)
    # Only include states 2 and 200; the 200-365 and 2-365 rows should drop
    g = build_yearly_trade_graph(df, states=[2, 200])
    # Only one trade dyad (2,200) -> 2 edges undirected
    assert g.edge_index.shape == (2, 2)


def test_empty_graph_when_no_trade():
    g = build_yearly_trade_graph(_trade_df(2000), states=[])
    assert g.num_nodes == 0
    assert g.edge_index.shape == (2, 0)


def test_lookup_dyad_trade_returns_zero_for_missing_pairs():
    df = _trade_df(2000)
    pairs = pd.DataFrame({
        "gwcode_i": [2, 2, 999],
        "gwcode_j": [200, 999, 1000],
    })
    out = lookup_dyad_trade(df, pairs, log_transform=False)
    assert out[0] == 100.0
    assert out[1] == 0.0      # 2-999 not in trade
    assert out[2] == 0.0      # 999-1000 not in trade


def test_lookup_dyad_trade_log_transform():
    df = _trade_df(2000)
    pairs = pd.DataFrame({"gwcode_i": [2], "gwcode_j": [200]})
    out = lookup_dyad_trade(df, pairs, log_transform=True)
    assert abs(out[0] - np.log1p(100.0)) < 1e-5
