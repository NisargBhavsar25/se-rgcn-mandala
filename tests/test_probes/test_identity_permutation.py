"""Tests for the identity permutation probe."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.probes.identity_permutation import identity_permutation_probe


def test_perfect_predictor_with_no_identity_collapse_zero():
    """A score function invariant to embed_perm should give collapse_ratio ~ 0."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    def score_fn(perm):
        # Always returns scores ranking positives first; ignores perm entirely.
        return np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5])

    out = identity_permutation_probe(
        score_fn, y_true, n_nodes=8, n_permutations=10, seed=0,
    )
    assert out["baseline_pr_auc"] == pytest.approx(1.0)
    assert abs(out["collapse_ratio"]) < 1e-9


def test_identity_dependent_predictor_collapses():
    """A score function that uses perm should show collapse."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    # Embed-permutation-dependent scorer: returns high scores at positions
    # specified by `perm[:3]` if perm is provided; baseline ranks positives.
    rng = np.random.default_rng(0)

    def score_fn(perm):
        if perm is None:
            return np.array([0.9, 0.85, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5])
        # When permuted, scores get scrambled relative to labels:
        scores = np.array([0.9, 0.85, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5])
        return scores[perm.cpu().numpy()]

    out = identity_permutation_probe(
        score_fn, y_true, n_nodes=8, n_permutations=50, seed=0,
    )
    assert out["baseline_pr_auc"] == pytest.approx(1.0)
    # Scrambled scores should give expected PR-AUC near base rate (3/8 = 0.375)
    # so collapse should be significant.
    assert out["permuted_pr_auc_mean"] < out["baseline_pr_auc"]
    assert out["collapse_ratio"] > 0.3


def test_probe_handles_nan_permuted_pr_auc():
    """If a permutation gives all-zero positives in the data view, that
    iteration's PR-AUC may be NaN; the probe must skip it gracefully."""
    y_true = np.zeros(8, dtype=int)  # No positives anywhere -- pr_auc returns NaN.

    def score_fn(perm):
        return np.random.rand(8)

    out = identity_permutation_probe(
        score_fn, y_true, n_nodes=8, n_permutations=5, seed=0,
    )
    # Baseline pr_auc is NaN because no positives.
    assert np.isnan(out["baseline_pr_auc"])
    assert out["n_permutations_valid"] == 0


def test_probe_returns_required_keys():
    y_true = np.array([1, 0, 1, 0])

    def score_fn(perm):
        return np.array([0.9, 0.1, 0.8, 0.2])

    out = identity_permutation_probe(
        score_fn, y_true, n_nodes=4, n_permutations=3, seed=0,
    )
    for key in ("baseline_pr_auc", "permuted_pr_auc_mean", "permuted_pr_auc_std",
                "collapse_ratio", "n_permutations_valid"):
        assert key in out
