"""Loaders for the COW + ATOP datasets in canonical dyad-year shape.

Both alliance loaders return a DataFrame keyed by (year, gwcode_i, gwcode_j)
with gwcode_i < gwcode_j (canonical undirected ordering) and edge_present = 1.
Only positive (allied) rows are returned. Construction of the full PRD
universe with binary labels is the job of `src.data.edge_table.build_edge_table`.

ATOP is the project's primary alliance source (covers 1815-2018);
COW Alliance v4.1 is the robustness-check source (caps at 2012). Both are
included so the kernel-ablation grid can swap sources without code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.gw_cow_mapping import cow_to_gw_series

logger = logging.getLogger(__name__)


def _drop_self_loops(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Remove (i, j) rows with i == j (artifacts of COW->GW collapsing).

    Example: COW Alliance v4.1 occasionally codes the same German entity as
    both ccode 255 (Germany / unified) and ccode 260 (FRG) within the same
    dyad-year coverage; both translate to GW 260, producing a self-loop.
    These rows are dropped (a state cannot meaningfully be allied with
    itself) and the count is logged for audit.
    """
    mask_self = df["gwcode_i"] == df["gwcode_j"]
    n_self = int(mask_self.sum())
    if n_self > 0:
        logger.info(
            "[%s] dropping %d self-loop rows after COW->GW translation",
            source, n_self,
        )
    return df.loc[~mask_self].reset_index(drop=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ATOP_DEFAULT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "atop_5.1" / "ATOP 5.1 (.csv)" / "atop5_1dy.csv"
)
COW_ALLIANCE_DEFAULT_PATH = (
    PROJECT_ROOT
    / "data" / "raw" / "cow_alliance_v4.1" / "version4.1_csv"
    / "alliance_v4.1_by_dyad_yearly.csv"
)


def _canonicalize_dyad(df: pd.DataFrame, code1: str, code2: str) -> pd.DataFrame:
    """Add gwcode_i / gwcode_j with i < j ordering."""
    pair = df[[code1, code2]].to_numpy()
    df = df.copy()
    df["gwcode_i"] = pair.min(axis=1).astype(int)
    df["gwcode_j"] = pair.max(axis=1).astype(int)
    return df


def load_atop_alliance_edges(
    years: Optional[range] = None,
    *,
    atop_path: Path = ATOP_DEFAULT_PATH,
    require_treaty_type: Optional[str] = None,
) -> pd.DataFrame:
    """ATOP 5.1 alliance edges, GW-coded, canonical (i < j).

    Args:
      years: optional year filter; row kept iff years.start <= year < years.stop.
      atop_path: ATOP dyad-year CSV path.
      require_treaty_type: optional column name (e.g., 'defense', 'nonagg')
        that must equal 1; default None means any ATOP-recorded alliance.

    Returns:
      DataFrame with ['year', 'gwcode_i', 'gwcode_j', 'edge_present',
      'defense', 'offense', 'neutral', 'nonagg', 'consul']. Only positive
      (allied) rows are returned.
    """
    df = pd.read_csv(atop_path, encoding="latin-1", low_memory=False)
    df = df[df["atopally"] == 1].copy()
    if require_treaty_type is not None:
        df = df[df[require_treaty_type] == 1].copy()

    df["mem1_gw"] = cow_to_gw_series(df["mem1"], df["year"])
    df["mem2_gw"] = cow_to_gw_series(df["mem2"], df["year"])
    df = _canonicalize_dyad(df, "mem1_gw", "mem2_gw")

    df["edge_present"] = 1

    out = df[
        [
            "year", "gwcode_i", "gwcode_j", "edge_present",
            "defense", "offense", "neutral", "nonagg", "consul",
        ]
    ].copy()

    if years is not None:
        out = out[(out["year"] >= years.start) & (out["year"] < years.stop)]

    # Some ATOP rows can collapse to the same (year, i, j) under translation
    # if multiple treaties cover the same dyad-year; deduplicate by ORing the
    # type flags and keeping a single row.
    agg = {
        "edge_present": "max",
        "defense": "max", "offense": "max", "neutral": "max",
        "nonagg": "max", "consul": "max",
    }
    out = (
        out.groupby(["year", "gwcode_i", "gwcode_j"], as_index=False).agg(agg)
    )
    return _drop_self_loops(out, "ATOP 5.1")


def load_cow_alliance_edges(
    years: Optional[range] = None,
    *,
    cow_path: Path = COW_ALLIANCE_DEFAULT_PATH,
    treaty_types: tuple[str, ...] = ("defense", "nonaggression", "entente"),
) -> pd.DataFrame:
    """COW Alliance v4.1 edges, GW-coded, canonical (i < j).

    Args:
      years: optional year filter.
      cow_path: COW Alliance dyad-year CSV path.
      treaty_types: an alliance is present iff ANY of these flags is 1.
        Default excludes 'neutrality' (matches the standard "active alliance"
        definition in the literature).

    Returns:
      DataFrame with ['year', 'gwcode_i', 'gwcode_j', 'edge_present',
      'defense', 'neutrality', 'nonaggression', 'entente']. Only positive rows.
    """
    df = pd.read_csv(cow_path, encoding="latin-1", low_memory=False)
    type_cols = list(treaty_types)
    df["edge_present"] = (df[type_cols].sum(axis=1) > 0).astype(int)
    df = df[df["edge_present"] == 1].copy()

    df["c1_gw"] = cow_to_gw_series(df["ccode1"], df["year"])
    df["c2_gw"] = cow_to_gw_series(df["ccode2"], df["year"])
    df = _canonicalize_dyad(df, "c1_gw", "c2_gw")

    keep_flags = ["defense", "neutrality", "nonaggression", "entente"]
    out = df[
        ["year", "gwcode_i", "gwcode_j", "edge_present"] + keep_flags
    ].copy()

    if years is not None:
        out = out[(out["year"] >= years.start) & (out["year"] < years.stop)]

    agg = {
        "edge_present": "max",
        "defense": "max", "neutrality": "max",
        "nonaggression": "max", "entente": "max",
    }
    out = (
        out.groupby(["year", "gwcode_i", "gwcode_j"], as_index=False).agg(agg)
    )
    return _drop_self_loops(out, "COW Alliance v4.1")
