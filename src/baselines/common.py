"""Shared infrastructure for the Track B published-baseline replication.

Contents:

  BaselineConfig          -- enum {AS_PUBLISHED, IDENTITY_FREE, IDENTITY_ONLY};
                             parameterizes the three configurations every
                             baseline runs in (see project plan section 2.3).

  build_per_node_features -- compute the 5 per-node features that mirror the
                             5 dyad-level F-LR features. Used as input to
                             Configs 1 and 2 (zeroed in Config 3).

  build_signed_graph      -- per-year signed edge index: ATOP allies = +1,
                             MID-active dyads = -1.

  active_mid_pairs        -- helper: dyads with an ongoing MID dispute at year y
                             (started <= y, ended >= y). Used to mark - edges.

  build_run_manifest      -- experiment provenance dict (git hash, deps, seed,
                             timestamp, hostname). Embedded in every result JSON.

The per-node features mirror the dyad-level F-LR features but aggregated:

  dyad-level F-LR feature       per-node analog
  ─────────────────────────────────────────────────────────
  log_d_km(i, j)                (no per-node analog; dyad-only)
  log_trade(i, j)               log(1 + sum_k trade(i, k))
  contiguous(i, j)              contiguous_degree(i)
  both_major(i, j)              is_major_power(i)
  rivalry_count(i, j)           rivalry_degree(i)

Plus log_alliance_degree (count of current ATOP allies) as a fifth feature
to keep parity with the dyad feature count. The dyad-level features are
NOT replaced -- they're appended at the prediction head.
"""

from __future__ import annotations

import logging
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.prd import major_powers_in_year

logger = logging.getLogger(__name__)


class BaselineConfig(Enum):
    AS_PUBLISHED = "as_published"   # learnable identity emb + per-node features
    IDENTITY_FREE = "identity_free"  # per-node features only (no learnable identity)
    IDENTITY_ONLY = "identity_only"  # learnable identity emb only (zero features)


PER_NODE_FEATURE_COLS = (
    "is_major_power",
    "log_total_trade",
    "contiguous_degree",
    "rivalry_degree",
    "log_alliance_degree",
)


def active_mid_pairs(onsets: pd.DataFrame, year: int) -> pd.DataFrame:
    """Dyads with a MID dispute active at `year` (started <= year, ended >= year).

    Returns columns ['gwcode_i', 'gwcode_j'] (canonical i < j; deduplicated).
    """
    mask = (onsets["onset_year"] <= year) & (onsets["end_year"] >= year)
    return onsets.loc[mask, ["gwcode_i", "gwcode_j"]].drop_duplicates()


@dataclass
class YearlySignedGraph:
    """Per-year signed graph. node_to_gwcode is local-id -> GW code mapping."""
    year: int
    node_to_gwcode: dict[int, int]
    gwcode_to_node: dict[int, int]
    edge_index_pos: torch.Tensor  # (2, E_pos)
    edge_index_neg: torch.Tensor  # (2, E_neg)

    @property
    def num_nodes(self) -> int:
        return len(self.node_to_gwcode)


def build_signed_graph(
    year: int,
    states: list[int],
    atop: pd.DataFrame,
    mid_onsets: pd.DataFrame,
    *,
    gw_to_node: Optional[dict[int, int]] = None,
) -> YearlySignedGraph:
    """Construct the signed graph at `year`.

    Positive edges: ATOP-allied dyads at year (atop.year == year, edge_present == 1).
    Negative edges: dyads with an MID dispute active at year.
    Both stored as undirected (both directions), so source = i->j and j->i.
    """
    if gw_to_node is None:
        sorted_states = sorted(set(states))
        gw_to_node = {gw: i for i, gw in enumerate(sorted_states)}
    node_to_gw = {i: gw for gw, i in gw_to_node.items()}

    # Positive: ATOP at this year
    atop_y = atop[atop["year"] == year]
    pos_pairs = atop_y[atop_y["edge_present"] == 1][["gwcode_i", "gwcode_j"]]
    pos_pairs = pos_pairs[
        pos_pairs["gwcode_i"].isin(gw_to_node) & pos_pairs["gwcode_j"].isin(gw_to_node)
    ]

    # Negative: active MIDs at this year
    neg_pairs = active_mid_pairs(mid_onsets, year)
    neg_pairs = neg_pairs[
        neg_pairs["gwcode_i"].isin(gw_to_node) & neg_pairs["gwcode_j"].isin(gw_to_node)
    ]

    def to_undirected_edge_index(pairs: pd.DataFrame) -> torch.Tensor:
        if len(pairs) == 0:
            return torch.empty(2, 0, dtype=torch.long)
        src = pairs["gwcode_i"].map(gw_to_node).to_numpy()
        dst = pairs["gwcode_j"].map(gw_to_node).to_numpy()
        ei = np.concatenate(
            [np.stack([src, dst]), np.stack([dst, src])], axis=1
        )
        return torch.from_numpy(ei).long()

    return YearlySignedGraph(
        year=year,
        node_to_gwcode=node_to_gw,
        gwcode_to_node=gw_to_node,
        edge_index_pos=to_undirected_edge_index(pos_pairs),
        edge_index_neg=to_undirected_edge_index(neg_pairs),
    )


def build_per_node_features(
    year: int,
    states: list[int],
    distance: pd.DataFrame,
    trade: pd.DataFrame,
    atop: pd.DataFrame,
    mid_onsets: pd.DataFrame,
    *,
    contiguity_threshold_km: float = 241.0,
    rivalry_window: int = 5,
) -> np.ndarray:
    """Per-node feature matrix at `year`. Rows ordered by sorted `states`.

    Columns (in order):
      is_major_power, log_total_trade, contiguous_degree, rivalry_degree,
      log_alliance_degree
    """
    sorted_states = sorted(set(states))
    n = len(sorted_states)
    state_to_idx = {gw: i for i, gw in enumerate(sorted_states)}

    feats = np.zeros((n, len(PER_NODE_FEATURE_COLS)), dtype=np.float32)

    # is_major_power
    mp = major_powers_in_year(year)
    for s in sorted_states:
        if s in mp:
            feats[state_to_idx[s], 0] = 1.0

    # log_total_trade (year - 1)
    trade_prior = trade[trade["year"] == year - 1]
    by_state: dict[int, float] = {}
    for r in trade_prior.itertuples():
        by_state[int(r.gwcode_i)] = by_state.get(int(r.gwcode_i), 0.0) + float(r.total_trade)
        by_state[int(r.gwcode_j)] = by_state.get(int(r.gwcode_j), 0.0) + float(r.total_trade)
    for s, v in by_state.items():
        if s in state_to_idx:
            feats[state_to_idx[s], 1] = np.log1p(v)

    # contiguous_degree
    dist_y = distance[distance["year"] == year]
    cont = dist_y[dist_y["d_km"] <= contiguity_threshold_km]
    deg: dict[int, int] = {}
    for r in cont.itertuples():
        deg[int(r.gwcode_i)] = deg.get(int(r.gwcode_i), 0) + 1
        deg[int(r.gwcode_j)] = deg.get(int(r.gwcode_j), 0) + 1
    for s, v in deg.items():
        if s in state_to_idx:
            feats[state_to_idx[s], 2] = float(v)

    # rivalry_degree (count of past conflict involvements in [year - W, year - 1])
    past = mid_onsets[
        (mid_onsets["onset_year"] >= year - rivalry_window)
        & (mid_onsets["onset_year"] < year)
    ]
    riv: dict[int, int] = {}
    for r in past.itertuples():
        riv[int(r.gwcode_i)] = riv.get(int(r.gwcode_i), 0) + 1
        riv[int(r.gwcode_j)] = riv.get(int(r.gwcode_j), 0) + 1
    for s, v in riv.items():
        if s in state_to_idx:
            feats[state_to_idx[s], 3] = float(v)

    # log_alliance_degree
    atop_y = atop[(atop["year"] == year) & (atop["edge_present"] == 1)]
    ally: dict[int, int] = {}
    for r in atop_y.itertuples():
        ally[int(r.gwcode_i)] = ally.get(int(r.gwcode_i), 0) + 1
        ally[int(r.gwcode_j)] = ally.get(int(r.gwcode_j), 0) + 1
    for s, v in ally.items():
        if s in state_to_idx:
            feats[state_to_idx[s], 4] = np.log1p(float(v))

    return feats


def build_run_manifest(seed: int, config_dict: dict) -> dict:
    """Reproducibility manifest -- embed in every result JSON."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=str(__file__).rsplit("src", 1)[0],
        ).decode().strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        commit += "-dirty" if dirty else ""
    except Exception:
        pass
    return {
        "git_commit": commit,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": seed,
        "config": config_dict,
    }
