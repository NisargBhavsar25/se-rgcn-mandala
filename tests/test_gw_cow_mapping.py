"""Pinned tests for GW <-> COW state-code translation."""

from __future__ import annotations

import pandas as pd

from src.data.gw_cow_mapping import cow_to_gw, cow_to_gw_series


def test_identity_default_for_undocumented_codes():
    assert cow_to_gw(2, 2000) == 2          # USA
    assert cow_to_gw(200, 1980) == 200      # UK
    assert cow_to_gw(710, 2010) == 710      # PRC
    assert cow_to_gw(365, 1985) == 365      # USSR / Russia


def test_germany_post_1949_translates_to_260():
    """COW codes FRG and Unified Germany as 255; GW uses 260."""
    assert cow_to_gw(255, 1980) == 260      # FRG
    assert cow_to_gw(255, 1995) == 260      # Unified Germany
    assert cow_to_gw(255, 2018) == 260


def test_germany_pre_1949_unchanged():
    """The Empire / Weimar / Nazi Germany period is not in our window
    and shares COW=GW=255 by convention; mapping is identity."""
    assert cow_to_gw(255, 1940) == 255
    assert cow_to_gw(255, 1816) == 255


def test_germany_threshold_year_inclusive_lower():
    """1949 is the first year of FRG and the start of the divergence."""
    assert cow_to_gw(255, 1948) == 255
    assert cow_to_gw(255, 1949) == 260


def test_cow_to_gw_series_vectorized():
    cow = pd.Series([255, 200, 255, 365, 255])
    year = pd.Series([1980, 1980, 1940, 2000, 2018])
    result = cow_to_gw_series(cow, year).tolist()
    assert result == [260, 200, 255, 365, 260]


def test_unified_yemen_post_1990_translates_to_678():
    """COW 679 (Unified Yemen) -> GW 678 (continues North Yemen code)."""
    assert cow_to_gw(679, 1995) == 678
    assert cow_to_gw(679, 2010) == 678


def test_yemen_codes_pre_1990_unchanged():
    """Pre-unification Yemen codes (678 North, 680 South) are identity."""
    assert cow_to_gw(678, 1985) == 678  # YAR
    assert cow_to_gw(680, 1985) == 680  # YPR
