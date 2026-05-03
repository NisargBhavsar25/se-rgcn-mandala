"""Tests for ATOP / COW Alliance loaders and the edge-table builder.

Loader tests skip cleanly if the raw files are absent (mirrors the distance
test pattern). Edge-table tests use synthetic data so they always run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.edge_table import build_edge_table
from src.data.loaders import (
    ATOP_DEFAULT_PATH,
    COW_ALLIANCE_DEFAULT_PATH,
    load_atop_alliance_edges,
    load_cow_alliance_edges,
)


# ---------- edge_table (synthetic) ------------------------------------------

def test_build_edge_table_fills_missing_with_zero():
    universe = pd.DataFrame({
        "year":     [2000, 2000, 2001, 2001],
        "gwcode_i": [   2,    2,    2,    2],
        "gwcode_j": [ 200,  365,  200,  365],
    })
    positive = pd.DataFrame({
        "year":         [2000, 2001],
        "gwcode_i":     [   2,    2],
        "gwcode_j":     [ 200,  365],
        "edge_present": [   1,    1],
    })
    out = build_edge_table(universe, positive)
    rows = {(r.year, r.gwcode_i, r.gwcode_j): r.edge_present for r in out.itertuples()}
    assert rows[(2000, 2, 200)] == 1
    assert rows[(2000, 2, 365)] == 0
    assert rows[(2001, 2, 200)] == 0
    assert rows[(2001, 2, 365)] == 1


def test_build_edge_table_drops_positives_outside_universe():
    """Edges outside the PRD universe are silently dropped (intentional)."""
    universe = pd.DataFrame({
        "year": [2000], "gwcode_i": [2], "gwcode_j": [200],
    })
    positive = pd.DataFrame({
        "year":         [2000, 2000],
        "gwcode_i":     [   2,    2],
        "gwcode_j":     [ 200,  999],   # 999 is not in the universe
        "edge_present": [   1,    1],
    })
    out = build_edge_table(universe, positive)
    assert len(out) == 1
    assert int(out.iloc[0].edge_present) == 1


def test_build_edge_table_preserves_universe_columns():
    universe = pd.DataFrame({
        "year": [2000], "gwcode_i": [2], "gwcode_j": [200],
        "d_km": [5500.0], "border_type": ["oceanic"],
    })
    positive = pd.DataFrame({
        "year": [2000], "gwcode_i": [2], "gwcode_j": [200],
        "edge_present": [1],
    })
    out = build_edge_table(universe, positive)
    assert "d_km" in out.columns
    assert "border_type" in out.columns


# ---------- loaders (real data, skip if absent) ------------------------------

@pytest.fixture(scope="module")
def atop_edges() -> pd.DataFrame:
    if not ATOP_DEFAULT_PATH.exists():
        pytest.skip(f"{ATOP_DEFAULT_PATH} not found.")
    return load_atop_alliance_edges(years=range(1950, 2019))


@pytest.fixture(scope="module")
def cow_edges() -> pd.DataFrame:
    if not COW_ALLIANCE_DEFAULT_PATH.exists():
        pytest.skip(f"{COW_ALLIANCE_DEFAULT_PATH} not found.")
    return load_cow_alliance_edges(years=range(1950, 2013))


def test_atop_canonical_dyad_ordering(atop_edges: pd.DataFrame):
    assert (atop_edges.gwcode_i < atop_edges.gwcode_j).all()


def test_atop_year_window_respected(atop_edges: pd.DataFrame):
    assert atop_edges.year.min() >= 1950
    assert atop_edges.year.max() <= 2018


def test_atop_germany_translated_to_gw_260(atop_edges: pd.DataFrame):
    """No row should reference COW Germany code 255 in our window;
    all should have been translated to GW 260."""
    germany_rows = atop_edges[
        (atop_edges.gwcode_i == 255) | (atop_edges.gwcode_j == 255)
    ]
    assert len(germany_rows) == 0
    # And we should see Germany 260 actually appearing post-1991.
    assert (
        ((atop_edges.gwcode_i == 260) | (atop_edges.gwcode_j == 260))
        & (atop_edges.year >= 1991)
    ).any()


def test_atop_edge_present_is_one(atop_edges: pd.DataFrame):
    assert (atop_edges.edge_present == 1).all()


def test_atop_known_alliance_present():
    """Sanity check: NATO members US-UK should be ATOP-allied throughout 1950-2018."""
    df = load_atop_alliance_edges(years=range(1950, 2019))
    us_uk = df[(df.gwcode_i == 2) & (df.gwcode_j == 200)]
    # NATO 1949 onwards -> all years 1950-2018 should be present.
    assert len(us_uk) >= 60


def test_cow_canonical_dyad_ordering(cow_edges: pd.DataFrame):
    assert (cow_edges.gwcode_i < cow_edges.gwcode_j).all()


def test_cow_germany_translated_to_gw_260(cow_edges: pd.DataFrame):
    germany_255 = cow_edges[
        (cow_edges.gwcode_i == 255) | (cow_edges.gwcode_j == 255)
    ]
    assert len(germany_255) == 0
