"""Pinned tests for COW MID 5.0 dyadic-onset derivation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.mid import (
    MIDB_DEFAULT_PATH,
    build_mid_onset_edge_table,
    derive_mid_onsets,
    load_mid_onsets,
)


def midb_row(
    dispnum: int, ccode: int, sidea: int, orig: int,
    *, hostlev: int = 3, styear: int = 2010, endyear: int = 2010,
) -> dict:
    """Synthetic MIDB row with all required schema columns populated."""
    return {
        "dispnum": dispnum, "stabb": f"X{ccode}", "ccode": ccode,
        "stday": 1, "stmon": 1, "styear": styear,
        "endday": 31, "endmon": 12, "endyear": endyear,
        "sidea": sidea, "revstate": 0, "revtype1": 1, "revtype2": 0,
        "fatality": 0, "fatalpre": 0, "hiact": hostlev * 2,
        "hostlev": hostlev, "orig": orig, "version": 5,
    }


# ---------- derive_mid_onsets (synthetic) -----------------------------------

def test_simple_originator_dyad_captured():
    """Two originators on opposing sides at hostility 3 -> one onset."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1),
        midb_row(1000, 200, sidea=0, orig=1),
    ])
    out = derive_mid_onsets(midb)
    assert len(out) == 1
    r = out.iloc[0]
    assert (r.gwcode_i, r.gwcode_j) == (2, 200)
    assert r.onset_year == 2010
    assert r.hostlev_max == 3


def test_hostility_threshold_filters_low_intensity_disputes():
    """Hostility 2 (display of force) is below default threshold of 3."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1, hostlev=2),
        midb_row(1000, 200, sidea=0, orig=1, hostlev=2),
    ])
    out = derive_mid_onsets(midb, hostility_threshold=3)
    assert len(out) == 0


def test_hostility_threshold_uses_dispute_max():
    """Dispute qualifies if ANY participant's hostlev >= threshold."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1, hostlev=2),    # below
        midb_row(1000, 200, sidea=0, orig=1, hostlev=4),  # above; pulls dispute in
    ])
    out = derive_mid_onsets(midb, hostility_threshold=3)
    assert len(out) == 1
    assert out.iloc[0].hostlev_max == 4


def test_joiners_excluded_when_originators_only():
    """A joiner (orig=0) on side B should NOT pair with originator on side A."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1),    # USA originator
        midb_row(1000, 200, sidea=0, orig=1),  # UK originator
        midb_row(1000, 220, sidea=0, orig=0),  # France joiner
    ])
    out = derive_mid_onsets(midb)
    pairs = set(zip(out.gwcode_i, out.gwcode_j))
    assert (2, 200) in pairs       # originator-originator captured
    assert (2, 220) not in pairs   # USA-France (joiner) excluded
    assert (200, 220) not in pairs # UK-France (joiner) excluded


def test_same_side_dyad_not_captured():
    """Two states on the same side cannot be a conflict-onset dyad."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1),
        midb_row(1000, 200, sidea=1, orig=1),  # same side as USA
        midb_row(1000, 365, sidea=0, orig=1),  # opposing side
    ])
    out = derive_mid_onsets(midb)
    pairs = set(zip(out.gwcode_i, out.gwcode_j))
    assert (2, 200) not in pairs   # same side -> excluded
    assert (2, 365) in pairs
    assert (200, 365) in pairs


def test_germany_translated_to_gw_260_at_onset_year():
    """COW 255 in 1995 should translate to GW 260 (post-reunification)."""
    midb = pd.DataFrame([
        midb_row(1000, 255, sidea=1, orig=1, styear=1995, endyear=1995),
        midb_row(1000, 365, sidea=0, orig=1, styear=1995, endyear=1995),
    ])
    out = derive_mid_onsets(midb)
    assert (out.iloc[0].gwcode_i, out.iloc[0].gwcode_j) == (260, 365)


def test_multi_originator_pairs_emitted_correctly():
    """3 originators on side A vs 2 on side B -> 6 dyad-disputes."""
    midb = pd.DataFrame([
        midb_row(1000, 2, sidea=1, orig=1),
        midb_row(1000, 200, sidea=1, orig=1),
        midb_row(1000, 220, sidea=1, orig=1),
        midb_row(1000, 365, sidea=0, orig=1),
        midb_row(1000, 710, sidea=0, orig=1),
    ])
    out = derive_mid_onsets(midb)
    assert len(out) == 6


# ---------- build_mid_onset_edge_table (synthetic) --------------------------

def test_edge_table_marks_onset_year_only_as_positive_and_censors_rest():
    """A 3-year dispute -> only onset is positive, years +1 and +2 censored."""
    onsets = pd.DataFrame([{
        "dispnum": 1000, "onset_year": 2010, "end_year": 2012,
        "gwcode_i": 2, "gwcode_j": 200, "hostlev_max": 4,
    }])
    universe = pd.DataFrame({
        "year":     [2009, 2010, 2011, 2012, 2013],
        "gwcode_i": [   2,    2,    2,    2,    2],
        "gwcode_j": [ 200,  200,  200,  200,  200],
    })
    out = build_mid_onset_edge_table(universe, onsets)
    by_year = {int(r.year): (int(r.edge_present), int(r.censored)) for r in out.itertuples()}
    assert by_year[2009] == (0, 0)  # before dispute -> normal negative
    assert by_year[2010] == (1, 0)  # onset
    assert by_year[2011] == (0, 1)  # censored
    assert by_year[2012] == (0, 1)  # censored
    assert by_year[2013] == (0, 0)  # after dispute -> back to candidate


def test_edge_table_handles_pre_window_onset_with_in_window_censoring():
    """Dispute starts before the PRD window but extends into it -> censor in window."""
    onsets = pd.DataFrame([{
        "dispnum": 999, "onset_year": 1955, "end_year": 2008,
        "gwcode_i": 2, "gwcode_j": 365, "hostlev_max": 5,
    }])
    universe = pd.DataFrame({
        "year":     [2006, 2007, 2008, 2009],
        "gwcode_i": [   2,    2,    2,    2],
        "gwcode_j": [ 365,  365,  365,  365],
    })
    out = build_mid_onset_edge_table(universe, onsets)
    by_year = {int(r.year): (int(r.edge_present), int(r.censored)) for r in out.itertuples()}
    assert by_year[2006] == (0, 1)  # ongoing -> censored
    assert by_year[2007] == (0, 1)
    assert by_year[2008] == (0, 1)
    assert by_year[2009] == (0, 0)  # after end -> normal candidate


# ---------- real-data smoke tests -------------------------------------------

@pytest.fixture(scope="module")
def real_onsets() -> pd.DataFrame:
    if not MIDB_DEFAULT_PATH.exists():
        pytest.skip(f"{MIDB_DEFAULT_PATH} not found.")
    return load_mid_onsets()


def test_real_canonical_dyad_ordering(real_onsets):
    assert (real_onsets.gwcode_i < real_onsets.gwcode_j).all()


def test_real_onsets_have_reasonable_count(real_onsets):
    """Sanity: MIDB 5.0 has ~2400 disputes; with hostility >=3 and originators
    only, we should see at least a few hundred dyad-disputes."""
    assert 200 < len(real_onsets) < 10_000


def test_real_no_germany_255_post_1949(real_onsets):
    """Pre-1949 Germany correctly stays as GW 255 (Empire/Weimar/Nazi era,
    where both COW and GW agree on 255). Only the post-1949 FRG/Unified
    period diverges -- those rows must translate to GW 260."""
    germany_255_post = real_onsets[
        ((real_onsets.gwcode_i == 255) | (real_onsets.gwcode_j == 255))
        & (real_onsets.onset_year >= 1949)
    ]
    assert len(germany_255_post) == 0
