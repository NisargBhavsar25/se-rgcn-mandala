"""Pinned tests for the evaluation harness.

Covers:
  - Each metric on a known degenerate input (perfect / constant / worst).
  - The persistence -> transitions -> PR-AUC pipeline returns base rate.
    This is the load-bearing assertion for the entire week-2 baseline:
    the floor PR-AUC must equal the transition base rate, otherwise our
    metric pipeline is silently broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    bootstrap_ci,
    brier_positives,
    lift_at_k,
    pr_auc,
    recall_at_k,
)
from src.evaluation.transitions import derive_transitions
from src.models.persistence import persistence_predict


# ---------- pr_auc -----------------------------------------------------------

def test_pr_auc_perfect_predictor():
    y_true = np.array([1, 0, 1, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.3])
    assert pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_pr_auc_constant_predictor_equals_base_rate():
    rng = np.random.default_rng(0)
    y_true = (rng.random(2000) < 0.05).astype(int)
    y_score = np.full_like(y_true, 0.5, dtype=float)
    assert pr_auc(y_true, y_score) == pytest.approx(y_true.mean(), abs=1e-9)


def test_pr_auc_no_positives_is_nan():
    assert np.isnan(pr_auc(np.zeros(10, dtype=int), np.random.random(10)))


# ---------- recall_at_k ------------------------------------------------------

def test_recall_at_k_perfect_ordering():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.4])
    assert recall_at_k(y_true, y_score, 3) == pytest.approx(1.0)


def test_recall_at_k_partial_capture():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.05, 0.2, 0.3])
    assert recall_at_k(y_true, y_score, 2) == pytest.approx(0.5)


# ---------- lift_at_k --------------------------------------------------------

def test_lift_at_k_perfect_predictor():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # base rate 0.2
    y_score = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05, 0.0])
    assert lift_at_k(y_true, y_score, 2) == pytest.approx(5.0)


def test_lift_at_k_random_is_approximately_one():
    rng = np.random.default_rng(0)
    y_true = (rng.random(20_000) < 0.05).astype(int)
    y_score = rng.random(20_000)
    assert 0.7 < lift_at_k(y_true, y_score, 1000) < 1.3


# ---------- brier_positives --------------------------------------------------

def test_brier_positives_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_positives(y_true, y_score) == pytest.approx(0.0)


def test_brier_positives_worst():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_positives(y_true, y_score) == pytest.approx(1.0)


# ---------- bootstrap_ci -----------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.1).astype(int)
    y_score = rng.random(500) + y_true * 0.5  # slight positive signal
    point, lo, hi = bootstrap_ci(
        y_true, y_score, pr_auc, n_bootstrap=200, seed=0
    )
    assert lo <= point <= hi


def test_bootstrap_ci_block_by_year_requires_years():
    y_true = np.array([1, 0, 1])
    y_score = np.array([0.9, 0.1, 0.8])
    with pytest.raises(ValueError, match="block_by_year"):
        bootstrap_ci(y_true, y_score, pr_auc, block_by_year=True)


# ---------- transitions ------------------------------------------------------

def test_derive_transitions_formation_and_dissolution():
    edges = pd.DataFrame({
        "year":        [2000, 2001, 2002, 2000, 2001, 2002],
        "gwcode_i":    [   2,    2,    2,  200,  200,  200],
        "gwcode_j":    [ 200,  200,  200,  365,  365,  365],
        "edge_present":[   0,    1,    1,    1,    1,    0],
    })
    trans = derive_transitions(edges)

    row = trans[(trans.year == 2001) & (trans.gwcode_i == 2)].iloc[0]
    assert row.formation == 1 and row.dissolution == 0

    row = trans[(trans.year == 2002) & (trans.gwcode_i == 200)].iloc[0]
    assert row.formation == 0 and row.dissolution == 1

    # 2000 has no prior year in the table; must be dropped.
    assert (trans.year == 2000).sum() == 0


# ---------- persistence end-to-end ------------------------------------------

def test_persistence_carries_state_forward():
    edges = pd.DataFrame({
        "year":         [2000, 2000, 2000],
        "gwcode_i":     [   2,    2,  200],
        "gwcode_j":     [ 200,  365,  365],
        "edge_present": [   1,    0,    1],
    })
    preds = persistence_predict(edges)
    assert (preds.year == 2001).all()
    expected = edges["edge_present"].astype(float).values
    actual = preds.sort_values(["gwcode_i", "gwcode_j"])["pred_score"].values
    np.testing.assert_array_equal(actual, expected)


def test_persistence_pr_auc_on_transitions_equals_base_rate():
    """Persistence predicts no transitions; PR-AUC must equal the base rate.

    This is the floor any SE-RGCN configuration must beat. If the assertion
    drifts, the metric pipeline is silently miscounting.
    """
    rng = np.random.default_rng(0)
    n_dyads, n_years = 50, 20
    rows = []
    for d in range(n_dyads):
        e = 0
        for y in range(n_years):
            if rng.random() < 0.05:
                e = 1 - e
            rows.append({
                "year": y,
                "gwcode_i": d,
                "gwcode_j": d + 200,
                "edge_present": e,
            })
    edges = pd.DataFrame(rows)
    trans = derive_transitions(edges)
    trans["any_transition"] = (
        (trans.formation == 1) | (trans.dissolution == 1)
    ).astype(int)

    # Persistence prediction for "transition occurred" is constant 0.
    y_true = trans.any_transition.values
    y_score = np.zeros_like(y_true, dtype=float)

    base_rate = float(y_true.mean())
    assert pr_auc(y_true, y_score) == pytest.approx(base_rate, abs=1e-9)
    # Constant-score predictor must give lift = 1.0 EXACTLY under expected-value
    # tie-breaking. This is the load-bearing invariant for the floor baseline.
    assert lift_at_k(y_true, y_score, k=int(y_true.sum())) == pytest.approx(1.0, abs=1e-9)
