"""Evaluation metrics for SE-RGCN.

The reviewer-mandated metric set for severe class imbalance:

  - pr_auc            -- HEADLINE metric. Average precision; not ROC-AUC.
                         ROC-AUC of 0.95 is achievable on 0.1%-positive tasks
                         by a model that is functionally useless.
  - recall_at_k       -- Fraction of positives in the top-k highest scores.
                         Pair k with the actual annual base rate when reporting.
  - lift_at_k         -- precision_at_k / overall_base_rate. Random predictor
                         has lift = 1.0 by construction; any contribution must
                         exceed it.
  - brier_positives   -- Brier restricted to positives. Calibration on the
                         rare class, where it actually matters.
  - bootstrap_ci      -- CI helper. Default row-bootstrap is misleading when
                         positives cluster across years (NATO 1949, USSR
                         collapse 1991, post-9/11 2001-2002); pass
                         block_by_year=True for paper-grade CIs.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import average_precision_score


def _validate(y_true: np.ndarray, y_score: np.ndarray) -> None:
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(f"expected 1-D arrays, got {y_true.ndim}-D")


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC (average precision). NaN if no positives."""
    _validate(y_true, y_score)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _expected_top_k_positives(
    y_true: np.ndarray, y_score: np.ndarray, k: int
) -> float:
    """Expected count of positives in top-k under uniform random tie-breaking.

    Closed-form: positives strictly above the kth-largest score, plus a
    proportional share of positives tied at that score. Required for sane
    behavior on heavily-tied scores (constant predictors, persistence
    baseline, models early in training).
    """
    n = len(y_score)
    if k >= n:
        return float(y_true.sum())
    order = np.argsort(-y_score, kind="stable")
    sorted_score = y_score[order]
    sorted_true = y_true[order]
    cutoff = sorted_score[k - 1]
    above = sorted_score > cutoff
    at = sorted_score == cutoff
    n_above = int(above.sum())
    pos_above = float(sorted_true[above].sum())
    pos_at = float(sorted_true[at].sum())
    n_at = int(at.sum())
    return pos_above + (k - n_above) / n_at * pos_at


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of positives captured in the top-k highest-scored predictions.

    Ties at the kth score are handled by expected value over uniform random
    tie-breaking (deterministic, closed-form).
    """
    _validate(y_true, y_score)
    if k <= 0:
        raise ValueError("k must be positive")
    n_positives = int(y_true.sum())
    if n_positives == 0:
        return float("nan")
    k_eff = min(k, len(y_score))
    return _expected_top_k_positives(y_true, y_score, k_eff) / n_positives


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Precision-at-k divided by the overall positive base rate.

    Ties at the kth score are handled by expected value over uniform random
    tie-breaking. A constant-score predictor therefore returns lift = 1.0
    exactly, as the mathematical definition requires.
    """
    _validate(y_true, y_score)
    if k <= 0:
        raise ValueError("k must be positive")
    base_rate = float(y_true.mean())
    if base_rate == 0.0:
        return float("nan")
    k_eff = min(k, len(y_score))
    precision = _expected_top_k_positives(y_true, y_score, k_eff) / k_eff
    return precision / base_rate


def brier_positives(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score restricted to positives -- calibration on the rare class."""
    _validate(y_true, y_score)
    pos_mask = y_true == 1
    if not pos_mask.any():
        return float("nan")
    return float(np.mean((1.0 - y_score[pos_mask]) ** 2))


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[..., float],
    *,
    years: Optional[np.ndarray] = None,
    block_by_year: bool = False,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    metric_kwargs: Optional[dict] = None,
) -> tuple[float, float, float]:
    """Bootstrap CI. Returns (point_estimate, lower, upper).

    Set block_by_year=True (and pass `years`) to resample years rather than
    individual rows. Row-bootstrap on transition data understates uncertainty
    because positives cluster within historical events.
    """
    _validate(y_true, y_score)
    metric_kwargs = metric_kwargs or {}
    point = metric_fn(y_true, y_score, **metric_kwargs)
    rng = np.random.default_rng(seed)

    samples: list[float] = []
    if block_by_year:
        if years is None:
            raise ValueError("block_by_year=True requires `years`")
        unique_years = np.unique(years)
        for _ in range(n_bootstrap):
            chosen = rng.choice(unique_years, size=len(unique_years), replace=True)
            mask = np.isin(years, chosen)
            samples.append(metric_fn(y_true[mask], y_score[mask], **metric_kwargs))
    else:
        warnings.warn(
            "Row-bootstrap on transition data understates uncertainty when "
            "events cluster across years. Use block_by_year=True for paper CIs.",
            stacklevel=2,
        )
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            samples.append(metric_fn(y_true[idx], y_score[idx], **metric_kwargs))

    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))
