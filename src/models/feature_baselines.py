"""Hand-crafted feature baselines for the conflict-onset task.

Two baselines, in order of complexity:

  (H) Rivalry-history-only  -- score = count of onsets between (i,j) in
      [y - rivalry_window, y - 1]. No training. Tests whether the rivalry
      literature's empirical regularity ("5% of dyads account for 70% of
      disputes") is the entire story.

  (F) Hand-crafted-feature logistic regression -- five features per
      dyad-year: log(1+d_km), log(1+total_trade_{y-1}), contiguity flag,
      both-major-power flag, rivalry_count in [y-W, y-1]. Trained on the
      full PRD-uncensored train set with class_weight='balanced'. The
      standard quantitative-IR baseline in the gravity / rivalry literature.

Both are evaluated against the SAME test set as the trade-only RGCN and
the identity-only ablation, so the gate question is well-posed: does
SE-RGCN beat the most informative feature-based baseline available?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.prd import major_powers_in_year

logger = logging.getLogger(__name__)

KEY = ["year", "gwcode_i", "gwcode_j"]
FEATURE_COLS = ["log_d_km", "log_trade", "contiguous", "both_major", "rivalry_count"]

# Rich feature set used when --rich-features is enabled. Adds:
#   - rivalry counts at multiple windows (1y, 3y, 10y) in addition to default 5y
#   - log_trade_growth: yoy change in dyad trade
#   - common_rivals_5y: count of states both i and j have fought in last 5y
#   - common_allies: count of states both i and j currently ally with
#   - log_d_capital_km: capital-to-capital distance (diagnostic column from
#     distance matrix)
FEATURE_COLS_RICH = [
    "log_d_km", "log_trade", "log_trade_growth",
    "contiguous", "both_major",
    "rivalry_count_1y", "rivalry_count_3y", "rivalry_count",  # 5y is the original
    "rivalry_count_10y",
    "common_rivals_5y", "common_allies",
    "log_d_capital_km",
]


def add_distance_features(
    candidates: pd.DataFrame, distance: pd.DataFrame, *, contiguity_threshold_km: float = 241.0
) -> pd.DataFrame:
    """Merge in d_km at the target year; derive log_d_km and contiguous flag."""
    dist = distance[KEY + ["d_km"]]
    out = candidates.merge(dist, on=KEY, how="left")
    # Dyads missing from distance matrix get a max value (effectively far).
    out["d_km"] = out["d_km"].fillna(20_000.0)
    out["log_d_km"] = np.log1p(out["d_km"]).astype("float32")
    out["contiguous"] = (out["d_km"] <= contiguity_threshold_km).astype("int8")
    return out


def add_trade_features(candidates: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    """Merge in total_trade at year-1; derive log_trade."""
    trade_prior = trade.rename(columns={"year": "year_t"})
    trade_prior["year"] = trade_prior["year_t"] + 1
    trade_prior = trade_prior[["year", "gwcode_i", "gwcode_j", "total_trade"]]
    out = candidates.merge(trade_prior, on=KEY, how="left")
    out["total_trade"] = out["total_trade"].fillna(0.0)
    out["log_trade"] = np.log1p(out["total_trade"]).astype("float32")
    return out


def add_major_power_flag(candidates: pd.DataFrame) -> pd.DataFrame:
    """Both-major-power binary flag per dyad-year (year-dependent membership)."""
    out = candidates.copy()
    out["both_major"] = 0
    for year in candidates["year"].unique():
        mp = major_powers_in_year(int(year))
        mask = (
            (candidates["year"] == year)
            & candidates["gwcode_i"].isin(mp)
            & candidates["gwcode_j"].isin(mp)
        )
        out.loc[mask, "both_major"] = 1
    out["both_major"] = out["both_major"].astype("int8")
    return out


def add_rivalry_count(
    candidates: pd.DataFrame, onsets: pd.DataFrame, *, window: int = 5
) -> pd.DataFrame:
    """For each (year y, i, j), count onsets between i and j in [y-window, y-1]."""
    # Expand each onset to the W years it contributes to as "past" rivalry signal.
    # Vectorized: for onset at year=k, contributes to candidate years k+1..k+W.
    if len(onsets) == 0:
        out = candidates.copy()
        out["rivalry_count"] = 0
        return out

    rows = []
    for r in onsets.itertuples():
        for y_target in range(int(r.onset_year) + 1, int(r.onset_year) + window + 1):
            rows.append((y_target, int(r.gwcode_i), int(r.gwcode_j)))
    exp = pd.DataFrame(rows, columns=KEY)
    rivalry = (
        exp.groupby(KEY, as_index=False).size().rename(columns={"size": "rivalry_count"})
    )
    out = candidates.merge(rivalry, on=KEY, how="left")
    out["rivalry_count"] = out["rivalry_count"].fillna(0).astype("int16")
    return out


def build_feature_table(
    candidates: pd.DataFrame,
    distance: pd.DataFrame,
    trade: pd.DataFrame,
    onsets: pd.DataFrame,
    *,
    rivalry_window: int = 5,
    contiguity_threshold_km: float = 241.0,
) -> pd.DataFrame:
    """Materialize all five features for every candidate dyad-year.

    Args:
      candidates: PRD-uncensored dyad-year table. Must have KEY + 'edge_present'.
      distance, trade, onsets: loaded by their respective loaders.
      rivalry_window: look-back window in years for rivalry_count.
      contiguity_threshold_km: PRD contiguity threshold (used for the binary flag).

    Returns:
      candidates plus FEATURE_COLS plus carried 'edge_present'.
    """
    out = add_distance_features(
        candidates, distance, contiguity_threshold_km=contiguity_threshold_km
    )
    out = add_trade_features(out, trade)
    out = add_major_power_flag(out)
    out = add_rivalry_count(out, onsets, window=rivalry_window)
    return out


# ---------------------------------------------------------------------------
# Rich-features extension (12 features instead of 5).
# Used when --rich-features is passed to runners. Tests whether the
# identity-memorization pattern is robust to a richer feature substrate or
# whether GNNs pull ahead with more information available.
# ---------------------------------------------------------------------------


def add_rivalry_count_at_window(
    candidates: pd.DataFrame, onsets: pd.DataFrame, *, window: int, col_name: str
) -> pd.DataFrame:
    """Same logic as add_rivalry_count but writes to a configurable column."""
    if len(onsets) == 0:
        out = candidates.copy()
        out[col_name] = 0
        return out
    rows = []
    for r in onsets.itertuples():
        for y_target in range(int(r.onset_year) + 1, int(r.onset_year) + window + 1):
            rows.append((y_target, int(r.gwcode_i), int(r.gwcode_j)))
    exp = pd.DataFrame(rows, columns=KEY)
    riv = exp.groupby(KEY, as_index=False).size().rename(columns={"size": col_name})
    out = candidates.merge(riv, on=KEY, how="left")
    out[col_name] = out[col_name].fillna(0).astype("int16")
    return out


def add_trade_growth(candidates: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    """log( (1 + trade_{y-1}) / (1 + trade_{y-2}) ) -- yoy growth rate.

    Uses information available at prediction time (both year-1 and year-2 trade).
    """
    cur = trade.rename(columns={"year": "year_t", "total_trade": "trade_t"})
    cur["year"] = cur["year_t"] + 1  # trade_t lands on candidate year (y-1)
    cur = cur[["year", "gwcode_i", "gwcode_j", "trade_t"]]

    prev = trade.rename(columns={"year": "year_t", "total_trade": "trade_tm1"})
    prev["year"] = prev["year_t"] + 2  # trade_tm1 lands on candidate year (y-2)
    prev = prev[["year", "gwcode_i", "gwcode_j", "trade_tm1"]]

    out = candidates.merge(cur, on=KEY, how="left").merge(prev, on=KEY, how="left")
    out["trade_t"] = out["trade_t"].fillna(0.0)
    out["trade_tm1"] = out["trade_tm1"].fillna(0.0)
    out["log_trade_growth"] = (
        np.log1p(out["trade_t"]) - np.log1p(out["trade_tm1"])
    ).astype("float32")
    return out.drop(columns=["trade_t", "trade_tm1"])


def add_capital_distance(
    candidates: pd.DataFrame, distance: pd.DataFrame
) -> pd.DataFrame:
    """Capital-to-capital distance from the existing distance matrix."""
    if "d_capital_km" not in distance.columns:
        out = candidates.copy()
        out["log_d_capital_km"] = 0.0
        return out
    cap = distance[KEY + ["d_capital_km"]]
    out = candidates.merge(cap, on=KEY, how="left")
    out["d_capital_km"] = out["d_capital_km"].fillna(20_000.0)
    out["log_d_capital_km"] = np.log1p(out["d_capital_km"]).astype("float32")
    return out


def add_common_rivals_and_allies(
    candidates: pd.DataFrame,
    onsets: pd.DataFrame,
    atop: pd.DataFrame,
    *,
    rivalry_window: int = 5,
) -> pd.DataFrame:
    """For each (year y, i, j), count states k such that:
       - common_rivals_5y: k has fought BOTH i and j in [y - W, y - 1]
       - common_allies: k currently allies with BOTH i and j at year y - 1.
    """
    out = candidates.copy()
    out["common_rivals_5y"] = 0
    out["common_allies"] = 0

    # Pre-index past onsets by year and state for fast lookup.
    # For each year y, find the set of past-fight partners for every state.
    states_with_past_rivals: dict[tuple[int, int], set[int]] = {}
    for r in onsets.itertuples():
        oy = int(r.onset_year)
        i, j = int(r.gwcode_i), int(r.gwcode_j)
        for y_target in range(oy + 1, oy + rivalry_window + 1):
            states_with_past_rivals.setdefault((y_target, i), set()).add(j)
            states_with_past_rivals.setdefault((y_target, j), set()).add(i)

    states_with_allies: dict[tuple[int, int], set[int]] = {}
    if "edge_present" in atop.columns:
        for r in atop.itertuples():
            if int(getattr(r, "edge_present", 1)) != 1:
                continue
            y_atop = int(r.year) + 1  # alliance at y-1 contributes to candidate year y
            i, j = int(r.gwcode_i), int(r.gwcode_j)
            states_with_allies.setdefault((y_atop, i), set()).add(j)
            states_with_allies.setdefault((y_atop, j), set()).add(i)

    rivals_arr = np.zeros(len(out), dtype=np.int32)
    allies_arr = np.zeros(len(out), dtype=np.int32)
    yrs = out["year"].to_numpy()
    is_ = out["gwcode_i"].to_numpy()
    js_ = out["gwcode_j"].to_numpy()
    for idx in range(len(out)):
        y, i, j = int(yrs[idx]), int(is_[idx]), int(js_[idx])
        ri = states_with_past_rivals.get((y, i), set())
        rj = states_with_past_rivals.get((y, j), set())
        rivals_arr[idx] = len(ri & rj)
        ai = states_with_allies.get((y, i), set())
        aj = states_with_allies.get((y, j), set())
        allies_arr[idx] = len(ai & aj)

    out["common_rivals_5y"] = rivals_arr
    out["common_allies"] = allies_arr
    return out


def build_feature_table_rich(
    candidates: pd.DataFrame,
    distance: pd.DataFrame,
    trade: pd.DataFrame,
    onsets: pd.DataFrame,
    atop: pd.DataFrame,
    *,
    contiguity_threshold_km: float = 241.0,
) -> pd.DataFrame:
    """Materialize the rich 12-feature set per dyad-year.

    Args:
      candidates: PRD-uncensored dyad-year table.
      distance, trade, onsets, atop: loaded by their respective loaders.
      contiguity_threshold_km: PRD contiguity threshold.

    Returns:
      candidates plus FEATURE_COLS_RICH plus carried 'edge_present'.
    """
    out = add_distance_features(
        candidates, distance, contiguity_threshold_km=contiguity_threshold_km
    )
    out = add_capital_distance(out, distance)
    out = add_trade_features(out, trade)
    out = add_trade_growth(out, trade)
    out = add_major_power_flag(out)
    # Rivalry counts at multiple windows (1y / 3y / 5y / 10y)
    out = add_rivalry_count_at_window(out, onsets, window=1, col_name="rivalry_count_1y")
    out = add_rivalry_count_at_window(out, onsets, window=3, col_name="rivalry_count_3y")
    out = add_rivalry_count(out, onsets, window=5)  # writes "rivalry_count" (=5y)
    out = add_rivalry_count_at_window(out, onsets, window=10, col_name="rivalry_count_10y")
    out = add_common_rivals_and_allies(out, onsets, atop, rivalry_window=5)
    return out


# ---------------------------------------------------------------------------


@dataclass
class FeatureLR:
    """Wrapper around sklearn LogisticRegression with a configurable feature set."""

    model: LogisticRegression
    feature_cols: tuple[str, ...] = tuple(FEATURE_COLS)

    @classmethod
    def fit(
        cls, train_features: pd.DataFrame,
        *,
        feature_cols: tuple[str, ...] = tuple(FEATURE_COLS),
    ) -> "FeatureLR":
        X = train_features[list(feature_cols)].to_numpy()
        y = train_features["edge_present"].to_numpy().astype(int)
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", n_jobs=1
        )
        model.fit(X, y)
        return cls(model=model, feature_cols=feature_cols)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[list(self.feature_cols)].to_numpy()
        return self.model.predict_proba(X)[:, 1]


def rivalry_only_score(features: pd.DataFrame) -> np.ndarray:
    """Baseline (H): score = rivalry_count. No training; pure rule."""
    return features["rivalry_count"].to_numpy().astype(float)
