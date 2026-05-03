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


@dataclass
class FeatureLR:
    """Wrapper around sklearn LogisticRegression with the locked feature set."""

    model: LogisticRegression
    feature_cols: tuple[str, ...] = tuple(FEATURE_COLS)

    @classmethod
    def fit(cls, train_features: pd.DataFrame) -> "FeatureLR":
        X = train_features[list(FEATURE_COLS)].to_numpy()
        y = train_features["edge_present"].to_numpy().astype(int)
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs", n_jobs=1
        )
        model.fit(X, y)
        return cls(model=model)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[list(self.feature_cols)].to_numpy()
        return self.model.predict_proba(X)[:, 1]


def rivalry_only_score(features: pd.DataFrame) -> np.ndarray:
    """Baseline (H): score = rivalry_count. No training; pure rule."""
    return features["rivalry_count"].to_numpy().astype(float)
