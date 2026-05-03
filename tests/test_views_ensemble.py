"""Tests for ViEWS-style ensemble baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.feature_baselines import FEATURE_COLS
from src.models.views_ensemble import ViEWSEnsemble


def _train_set() -> pd.DataFrame:
    """Tiny synthetic feature table with a few positives for fit() to converge."""
    rng = np.random.default_rng(0)
    n = 50
    rivalry = rng.integers(0, 6, n)
    return pd.DataFrame({
        "year":          [2010] * n,
        "gwcode_i":      list(range(n)),
        "gwcode_j":      [n + i for i in range(n)],
        "edge_present":  (rivalry > 3).astype(int),  # ~10-20 positives
        "log_d_km":      rng.uniform(0, 10, n),
        "log_trade":     rng.uniform(0, 12, n),
        "contiguous":    rng.integers(0, 2, n),
        "both_major":    rng.integers(0, 2, n),
        "rivalry_count": rivalry,
    })


def test_fit_and_predict_shape():
    train = _train_set()
    model = ViEWSEnsemble.fit(train, seed=0)
    p = model.predict_proba(train)
    assert p.shape == (len(train),)
    assert ((p >= 0) & (p <= 1)).all()


def test_per_learner_predictions():
    train = _train_set()
    model = ViEWSEnsemble.fit(train, seed=0)
    per = model.predict_proba_per_learner(train)
    assert set(per.keys()) == {"gbm", "rf", "lr"}
    for name, p in per.items():
        assert p.shape == (len(train),), f"{name} shape mismatch"
        assert ((p >= 0) & (p <= 1)).all(), f"{name} probs out of range"


def test_average_is_actual_average():
    """Ensemble probability should equal mean of base learner probabilities."""
    train = _train_set()
    model = ViEWSEnsemble.fit(train, seed=0)
    avg = model.predict_proba(train)
    per = model.predict_proba_per_learner(train)
    expected = (per["gbm"] + per["rf"] + per["lr"]) / 3.0
    np.testing.assert_allclose(avg, expected, rtol=1e-6)


def test_uses_feature_cols_from_FEATURE_COLS():
    train = _train_set()
    model = ViEWSEnsemble.fit(train, seed=0)
    assert tuple(model.feature_cols) == tuple(FEATURE_COLS)
