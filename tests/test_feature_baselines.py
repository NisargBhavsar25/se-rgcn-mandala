"""Tests for hand-crafted feature baselines (F + H)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.feature_baselines import (
    FeatureLR,
    add_distance_features,
    add_major_power_flag,
    add_rivalry_count,
    add_trade_features,
    build_feature_table,
    rivalry_only_score,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "year":         [2010, 2010, 2010],
        "gwcode_i":     [   2,    2,  100],
        "gwcode_j":     [ 200,  365,  101],
        "edge_present": [   1,    0,    0],
    })


# ---------- distance features -----------------------------------------------

def test_distance_features_log_and_contiguity():
    cand = _candidates()
    dist = pd.DataFrame({
        "year":     [2010, 2010, 2010],
        "gwcode_i": [   2,    2,  100],
        "gwcode_j": [ 200,  365,  101],
        "d_km":     [5500, 8000,    0],
    })
    out = add_distance_features(cand, dist)
    assert (out.log_d_km == np.log1p(out.d_km).astype("float32")).all()
    # 100-101 at d_km=0 -> contiguous
    row = out[(out.gwcode_i == 100) & (out.gwcode_j == 101)].iloc[0]
    assert int(row.contiguous) == 1
    # 2-200 at 5500 km -> not contiguous
    row = out[(out.gwcode_i == 2) & (out.gwcode_j == 200)].iloc[0]
    assert int(row.contiguous) == 0


def test_distance_missing_dyad_filled_with_far_value():
    cand = _candidates()
    dist = pd.DataFrame({"year": [2010], "gwcode_i": [2], "gwcode_j": [200], "d_km": [100.0]})
    out = add_distance_features(cand, dist)
    # 2-365 has no distance row -> should default to 20000 km, not_contiguous
    row = out[(out.gwcode_i == 2) & (out.gwcode_j == 365)].iloc[0]
    assert row.d_km == 20_000.0
    assert int(row.contiguous) == 0


# ---------- trade features --------------------------------------------------

def test_trade_features_use_year_minus_one():
    """Trade volume from year y-1 should land on candidate year y."""
    cand = pd.DataFrame({
        "year":     [2010],
        "gwcode_i": [   2],
        "gwcode_j": [ 200],
        "edge_present": [1],
    })
    trade = pd.DataFrame({
        "year":         [2009, 2010],
        "gwcode_i":     [   2,    2],
        "gwcode_j":     [ 200,  200],
        "total_trade": [500.0, 700.0],
    })
    out = add_trade_features(cand, trade)
    # candidate year 2010 should pull trade from 2009
    assert out.iloc[0].total_trade == 500.0
    assert abs(float(out.iloc[0].log_trade) - np.log1p(500.0)) < 1e-5


# ---------- major power flag ------------------------------------------------

def test_major_power_flag_year_dependent():
    cand = pd.DataFrame({
        "year":         [1980, 1980, 1995, 1995],
        "gwcode_i":     [   2,    2,    2,  260],
        "gwcode_j":     [ 200,  260,  260,  365],
        "edge_present": [   1,    0,    0,    0],
    })
    out = add_major_power_flag(cand)
    by = {(int(r.year), int(r.gwcode_i), int(r.gwcode_j)): int(r.both_major)
          for r in out.itertuples()}
    # USA-UK 1980: both major
    assert by[(1980, 2, 200)] == 1
    # USA-Germany 1980: Germany NOT major in 1980 -> 0
    assert by[(1980, 2, 260)] == 0
    # USA-Germany 1995: Germany IS major post-1991 -> 1
    assert by[(1995, 2, 260)] == 1


# ---------- rivalry count ---------------------------------------------------

def test_rivalry_count_window_and_dyad_specific():
    onsets = pd.DataFrame({
        "dispnum":     [1, 2, 3, 4],
        "onset_year":  [2005, 2007, 2009, 2008],
        "end_year":    [2005, 2007, 2009, 2008],
        "gwcode_i":    [   2,    2,    2,  200],
        "gwcode_j":    [ 200,  200,  200,  365],
        "hostlev_max": [   4,    4,    4,    4],
    })
    cand = pd.DataFrame({
        "year":         [2010, 2010, 2010, 2011],
        "gwcode_i":     [   2,  200,  100,    2],
        "gwcode_j":     [ 200,  365,  101,  200],
        "edge_present": [   0,    0,    0,    0],
    })
    out = add_rivalry_count(cand, onsets, window=5)
    by = {(int(r.year), int(r.gwcode_i), int(r.gwcode_j)): int(r.rivalry_count)
          for r in out.itertuples()}
    # USA-UK 2010 -> onsets at 2005, 2007, 2009 are in [2005, 2009] window -> 3
    assert by[(2010, 2, 200)] == 3
    # UK-Russia 2010 -> onset at 2008 in window -> 1
    assert by[(2010, 200, 365)] == 1
    # 100-101: no onsets ever -> 0
    assert by[(2010, 100, 101)] == 0
    # USA-UK 2011 -> onsets at 2007, 2009 in [2006, 2010] window -> 2 (2005 falls out)
    assert by[(2011, 2, 200)] == 2


def test_rivalry_count_handles_empty_onsets():
    cand = _candidates()
    out = add_rivalry_count(cand, pd.DataFrame(columns=[
        "dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max",
    ]), window=5)
    assert (out.rivalry_count == 0).all()


# ---------- end-to-end build + LR fit/predict --------------------------------

def test_build_feature_table_and_fit_predict():
    distance = pd.DataFrame({
        "year":     [2010, 2010, 2010],
        "gwcode_i": [   2,    2,  100],
        "gwcode_j": [ 200,  365,  101],
        "d_km":     [5500, 8000,    0],
    })
    trade = pd.DataFrame({
        "year":         [2009, 2009, 2009],
        "gwcode_i":     [   2,    2,  100],
        "gwcode_j":     [ 200,  365,  101],
        "total_trade": [500.0, 50.0, 10.0],
    })
    onsets = pd.DataFrame({
        "dispnum":     [1],
        "onset_year":  [2008],
        "end_year":    [2008],
        "gwcode_i":    [   2],
        "gwcode_j":    [ 200],
        "hostlev_max": [   4],
    })
    cand = _candidates()
    feats = build_feature_table(cand, distance, trade, onsets)
    assert {"log_d_km", "log_trade", "contiguous", "both_major", "rivalry_count"
            }.issubset(feats.columns)

    # Fit LR (just ensure it runs end-to-end on a tiny example).
    # With 1 positive and 2 negatives, LR will be degenerate, but should not crash.
    train = feats.copy()
    model = FeatureLR.fit(train)
    probs = model.predict_proba(train)
    assert probs.shape == (3,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_rivalry_only_score_returns_count():
    feats = pd.DataFrame({"rivalry_count": [0, 2, 5, 1]})
    out = rivalry_only_score(feats)
    np.testing.assert_array_equal(out, [0.0, 2.0, 5.0, 1.0])


# ---------- rich-features extension -----------------------------------------

def test_rich_feature_table_has_all_12_columns():
    from src.models.feature_baselines import (
        FEATURE_COLS_RICH, build_feature_table_rich,
    )
    distance = pd.DataFrame({
        "year":          [2010, 2010, 2010],
        "gwcode_i":      [   2,    2,  100],
        "gwcode_j":      [ 200,  365,  101],
        "d_km":          [5500, 8000,    0],
        "d_capital_km":  [5800, 8200,  300],
    })
    trade = pd.DataFrame({
        "year":         [2008, 2008, 2008, 2009, 2009, 2009],
        "gwcode_i":     [   2,    2,  100,    2,    2,  100],
        "gwcode_j":     [ 200,  365,  101,  200,  365,  101],
        "total_trade": [400.0, 40.0,  8.0, 500.0, 50.0, 10.0],
    })
    onsets = pd.DataFrame({
        "dispnum":     [1, 2],
        "onset_year":  [2007, 2008],
        "end_year":    [2007, 2008],
        "gwcode_i":    [   2,  200],
        "gwcode_j":    [ 200,  365],
        "hostlev_max": [   4,    4],
    })
    atop = pd.DataFrame({
        "year":         [2009, 2009],
        "gwcode_i":     [   2,    2],
        "gwcode_j":     [ 200,  100],
        "edge_present": [   1,    1],
    })
    cand = _candidates()
    feats = build_feature_table_rich(cand, distance, trade, onsets, atop)
    for col in FEATURE_COLS_RICH:
        assert col in feats.columns, f"missing column {col}"


def test_rich_features_rivalry_at_multiple_windows():
    from src.models.feature_baselines import build_feature_table_rich
    distance = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200],
        "d_km": [5500.0], "d_capital_km": [5800.0],
    })
    trade = pd.DataFrame({
        "year": [2008, 2009], "gwcode_i": [2, 2], "gwcode_j": [200, 200],
        "total_trade": [400.0, 500.0],
    })
    onsets = pd.DataFrame({
        "dispnum":     [1, 2, 3, 4],
        "onset_year":  [2009, 2007, 2005, 2001],  # 1y, 3y, 5y, 10y windows
        "end_year":    [2009, 2007, 2005, 2001],
        "gwcode_i":    [2, 2, 2, 2],
        "gwcode_j":    [200, 200, 200, 200],
        "hostlev_max": [4, 4, 4, 4],
    })
    atop = pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "edge_present"])
    cand = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200], "edge_present": [0],
    })
    feats = build_feature_table_rich(cand, distance, trade, onsets, atop)
    # Window [2010-1, 2009] (1y) -> only the 2009 onset -> 1
    assert int(feats.iloc[0].rivalry_count_1y) == 1
    # Window [2007, 2009] (3y) -> 2007 + 2009 -> 2
    assert int(feats.iloc[0].rivalry_count_3y) == 2
    # Window [2005, 2009] (5y) -> 2005 + 2007 + 2009 -> 3
    assert int(feats.iloc[0].rivalry_count) == 3
    # Window [2000, 2009] (10y) -> 2001 + 2005 + 2007 + 2009 -> 4
    assert int(feats.iloc[0].rivalry_count_10y) == 4


def test_rich_features_common_rivals():
    from src.models.feature_baselines import build_feature_table_rich
    distance = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200],
        "d_km": [5500.0], "d_capital_km": [5800.0],
    })
    trade = pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "total_trade"])
    # Both 2 and 200 fought 365 in last 5 years -> common_rivals = 1
    onsets = pd.DataFrame({
        "dispnum":     [1, 2],
        "onset_year":  [2007, 2008],
        "end_year":    [2007, 2008],
        "gwcode_i":    [2, 200],
        "gwcode_j":    [365, 365],
        "hostlev_max": [4, 4],
    })
    atop = pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "edge_present"])
    cand = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200], "edge_present": [0],
    })
    feats = build_feature_table_rich(cand, distance, trade, onsets, atop)
    assert int(feats.iloc[0].common_rivals_5y) == 1


def test_rich_features_common_allies():
    from src.models.feature_baselines import build_feature_table_rich
    distance = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200],
        "d_km": [5500.0], "d_capital_km": [5800.0],
    })
    trade = pd.DataFrame(columns=["year", "gwcode_i", "gwcode_j", "total_trade"])
    onsets = pd.DataFrame(columns=[
        "dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max",
    ])
    # Both 2 and 200 ally with 100 in 2009 (which is candidate-year - 1) -> common_allies = 1
    atop = pd.DataFrame({
        "year":         [2009, 2009],
        "gwcode_i":     [   2,  100],
        "gwcode_j":     [ 100,  200],
        "edge_present": [   1,    1],
    })
    cand = pd.DataFrame({
        "year": [2010], "gwcode_i": [2], "gwcode_j": [200], "edge_present": [0],
    })
    feats = build_feature_table_rich(cand, distance, trade, onsets, atop)
    assert int(feats.iloc[0].common_allies) == 1
