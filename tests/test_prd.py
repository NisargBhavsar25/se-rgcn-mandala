"""Pinned tests for the PRD filter."""

from __future__ import annotations

import pandas as pd

from src.data.prd import major_powers_in_year, prd_dyads, prd_dyads_all_years


# ---------- major-power membership ------------------------------------------

def test_major_powers_1980_excludes_germany_and_japan():
    """Per COW canon, FRG and Japan are NOT major in 1980 (only post-1991)."""
    mp = major_powers_in_year(1980)
    assert mp == {2, 200, 220, 365, 710}


def test_major_powers_2000_includes_unified_germany_and_japan():
    mp = major_powers_in_year(2000)
    assert mp == {2, 200, 220, 260, 365, 710, 740}


def test_major_powers_1900_no_china_no_germany():
    mp = major_powers_in_year(1900)
    assert 710 not in mp  # PRC didn't exist
    assert 260 not in mp  # Pre-WWII Germany used GW=255 in CShapes coding


# ---------- single-year filter ----------------------------------------------

def test_prd_keeps_major_power_dyads_even_when_distant():
    df = pd.DataFrame({
        "gwcode_i": [2, 100, 100, 100],
        "gwcode_j": [220, 200, 110, 105],
        "d_km":     [6000, 8000, 12000, 15000],
    })
    out = prd_dyads(df, year=1980)
    # USA-France: both major, distant -> kept (major-power dyad).
    assert ((out.gwcode_i == 2) & (out.gwcode_j == 220)).any()
    # 100-200: UK is major, kept.
    assert ((out.gwcode_i == 100) & (out.gwcode_j == 200)).any()
    # 100-110, 100-105: neither major, far -> dropped.
    assert not ((out.gwcode_j == 110) | (out.gwcode_j == 105)).any()


def test_prd_keeps_contiguous_non_major_dyads():
    df = pd.DataFrame({
        "gwcode_i": [100, 110, 120],
        "gwcode_j": [101, 111, 130],
        "d_km":     [0.0, 50.0, 5000.0],
    })
    out = prd_dyads(df, year=1980)
    assert len(out) == 2
    assert (out.d_km <= 241).all()


def test_prd_threshold_is_inclusive():
    df = pd.DataFrame({
        "gwcode_i": [100, 100],
        "gwcode_j": [101, 102],
        "d_km":     [241.0, 241.5],
    })
    out = prd_dyads(df, year=1980, contiguity_threshold_km=241.0)
    pairs = set(zip(out.gwcode_i, out.gwcode_j))
    assert (100, 101) in pairs       # at threshold -> included
    assert (100, 102) not in pairs   # past threshold -> excluded


def test_prd_custom_major_powers_override():
    """Caller can inject a custom major-power set (e.g., for ablation runs)."""
    df = pd.DataFrame({
        "gwcode_i": [100, 200],
        "gwcode_j": [101, 300],
        "d_km":     [9000, 9000],
    })
    out = prd_dyads(df, year=1980, major_powers={300})
    pairs = set(zip(out.gwcode_i, out.gwcode_j))
    # 100-101: not contiguous, neither in {300} -> excluded.
    # 200-300: not contiguous, but 300 is major -> kept.
    assert (200, 300) in pairs
    assert (100, 101) not in pairs


def test_prd_preserves_extra_columns():
    df = pd.DataFrame({
        "gwcode_i": [100],
        "gwcode_j": [101],
        "d_km":     [0.0],
        "border_type": ["land"],
        "d_capital_km": [500.0],
    })
    out = prd_dyads(df, year=1980)
    assert "border_type" in out.columns
    assert "d_capital_km" in out.columns


# ---------- multi-year sweep -------------------------------------------------

def test_prd_all_years_respects_year_dependent_major_status():
    """Germany (260) becomes major in 1991. A distant 260-X dyad should be
    filtered OUT in 1980 but kept IN in 2000."""
    df = pd.DataFrame({
        "year":     [1980, 2000],
        "gwcode_i": [260, 260],
        "gwcode_j": [100, 100],
        "d_km":     [5000, 5000],
    })
    out = prd_dyads_all_years(df)
    assert not ((out.year == 1980)).any()
    assert ((out.year == 2000)).any()


def test_prd_all_years_requires_year_column():
    df = pd.DataFrame({"gwcode_i": [1], "gwcode_j": [2], "d_km": [0.0]})
    import pytest
    with pytest.raises(ValueError, match="year"):
        prd_dyads_all_years(df)
