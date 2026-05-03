"""Adapter / common.py tests with hand-coded fixture graphs.

Per project plan section 2.2: "Test this adapter with a fixture graph (10
nodes, hand-coded signs) before running on real data. The adapter is the
most likely source of silent bugs."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.baselines.common import (
    BaselineConfig, PER_NODE_FEATURE_COLS,
    active_mid_pairs, build_per_node_features, build_signed_graph,
)


# ---------- BaselineConfig ---------------------------------------------------

def test_baseline_config_values():
    assert BaselineConfig.AS_PUBLISHED.value == "as_published"
    assert BaselineConfig.IDENTITY_FREE.value == "identity_free"
    assert BaselineConfig.IDENTITY_ONLY.value == "identity_only"


# ---------- active_mid_pairs -------------------------------------------------

def test_active_mid_pairs_includes_ongoing():
    onsets = pd.DataFrame({
        "dispnum":     [1, 2, 3],
        "onset_year":  [2005, 2008, 2012],
        "end_year":    [2010, 2008, 2012],
        "gwcode_i":    [   2,  200,  140],
        "gwcode_j":    [ 200,  365,  160],
        "hostlev_max": [   4,    3,    4],
    })
    # 2007: dispute 1 (2005-2010) is active. Disputes 2, 3 not yet.
    out = active_mid_pairs(onsets, 2007)
    pairs = set(zip(out.gwcode_i, out.gwcode_j))
    assert (2, 200) in pairs
    assert len(pairs) == 1


def test_active_mid_pairs_excludes_finished_and_future():
    onsets = pd.DataFrame({
        "dispnum":     [1, 2, 3],
        "onset_year":  [2005, 2008, 2012],
        "end_year":    [2010, 2008, 2012],
        "gwcode_i":    [   2,  200,  140],
        "gwcode_j":    [ 200,  365,  160],
        "hostlev_max": [   4,    3,    4],
    })
    out_2011 = active_mid_pairs(onsets, 2011)
    pairs = set(zip(out_2011.gwcode_i, out_2011.gwcode_j))
    assert (2, 200) not in pairs   # ended 2010
    assert (200, 365) not in pairs # ended 2008
    assert (140, 160) not in pairs # starts 2012


# ---------- build_signed_graph -----------------------------------------------

def test_signed_graph_pos_neg_edges_undirected():
    """ATOP-allies = +1, MID-active = -1, both encoded undirected."""
    atop = pd.DataFrame({
        "year":         [2010, 2010, 2010],
        "gwcode_i":     [   2,    2,  100],
        "gwcode_j":     [ 200,  365,  101],
        "edge_present": [   1,    1,    1],
    })
    onsets = pd.DataFrame({
        "dispnum":     [1, 2],
        "onset_year":  [2008, 2009],
        "end_year":    [2012, 2010],
        "gwcode_i":    [   2,  100],
        "gwcode_j":    [ 365,  200],
        "hostlev_max": [   4,    3],
    })
    states = [2, 100, 101, 200, 365]
    sg = build_signed_graph(2010, states, atop, onsets)
    # 3 ATOP edges undirected -> 6
    assert sg.edge_index_pos.shape == (2, 6)
    # 2 active MIDs -> 4 edges
    assert sg.edge_index_neg.shape == (2, 4)
    # Node mapping is sorted by GW code
    assert sg.node_to_gwcode == {0: 2, 1: 100, 2: 101, 3: 200, 4: 365}


def test_signed_graph_drops_dyads_outside_state_set():
    atop = pd.DataFrame({
        "year":         [2010, 2010],
        "gwcode_i":     [   2,    2],
        "gwcode_j":     [ 200,  999],   # 999 not in states
        "edge_present": [   1,    1],
    })
    onsets = pd.DataFrame(columns=[
        "dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max",
    ])
    sg = build_signed_graph(2010, [2, 200], atop, onsets)
    assert sg.edge_index_pos.shape == (2, 2)  # only (2, 200) survives, undirected


def test_signed_graph_uses_provided_gw_to_node():
    """gw_to_node should win over the implicit sorted-states mapping."""
    atop = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200], "edge_present": [1],
    })
    onsets = pd.DataFrame(columns=[
        "dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max",
    ])
    custom = {2: 7, 200: 3}
    sg = build_signed_graph(2010, [2, 200], atop, onsets, gw_to_node=custom)
    # Local IDs should be 7 and 3 per the custom map.
    pairs_in_graph = set()
    for e in range(sg.edge_index_pos.shape[1]):
        a = int(sg.edge_index_pos[0, e])
        b = int(sg.edge_index_pos[1, e])
        pairs_in_graph.add((a, b))
    assert (7, 3) in pairs_in_graph
    assert (3, 7) in pairs_in_graph


# ---------- build_per_node_features -----------------------------------------

def test_per_node_features_have_correct_shape_and_cols():
    distance = pd.DataFrame({
        "year": [2010] * 3,
        "gwcode_i": [2, 2, 100],
        "gwcode_j": [200, 365, 101],
        "d_km": [5500, 8000, 0],
    })
    trade = pd.DataFrame({
        "year": [2009] * 3,
        "gwcode_i": [2, 2, 100],
        "gwcode_j": [200, 365, 101],
        "total_trade": [500.0, 50.0, 10.0],
    })
    atop = pd.DataFrame({
        "year": [2010, 2010],
        "gwcode_i": [2, 100],
        "gwcode_j": [200, 101],
        "edge_present": [1, 1],
    })
    onsets = pd.DataFrame({
        "dispnum": [1],
        "onset_year": [2007],
        "end_year": [2007],
        "gwcode_i": [2],
        "gwcode_j": [200],
        "hostlev_max": [4],
    })
    states = [2, 100, 101, 200, 365]
    feats = build_per_node_features(2010, states, distance, trade, atop, onsets)
    assert feats.shape == (5, 5)
    # Row order: sorted states [2, 100, 101, 200, 365]
    # USA (state 2) is_major_power = 1 in 2010
    assert feats[0, 0] == 1.0
    # State 100 not major
    assert feats[1, 0] == 0.0
    # USA log_total_trade > 0 (sum over (2, 200) and (2, 365) at 2009)
    assert feats[0, 1] > 0
    # 100-101 is contiguous (d_km = 0); both should have contiguous_degree >= 1
    assert feats[1, 2] >= 1.0
    assert feats[2, 2] >= 1.0
    # USA had a 2007 conflict -> rivalry_degree at 2010 = 1 (in [2005, 2009])
    assert feats[0, 3] == 1.0
    # USA log_alliance_degree > 0 (allied with 200 at 2010)
    assert feats[0, 4] > 0


def test_per_node_features_inactive_states_get_zeros():
    feats = build_per_node_features(
        2010, [2, 200],
        distance=pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "d_km"]),
        trade=pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "total_trade"]),
        atop=pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "edge_present"]),
        mid_onsets=pd.DataFrame(columns=[
            "dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max",
        ]),
    )
    # USA (state 2) still major in 2010
    assert feats[0, 0] == 1.0
    # All other features = 0 because no data
    assert feats[0, 1] == 0.0
    assert feats[0, 2] == 0.0
    assert feats[0, 3] == 0.0
    assert feats[0, 4] == 0.0
