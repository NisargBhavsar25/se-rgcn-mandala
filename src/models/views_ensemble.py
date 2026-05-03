"""ViEWS-style ensemble baseline -- gradient boosting + random forest + LR averaged.

The IR-forecasting community (ViEWS, Hegre et al., JPR 2019; Brandt et al.,
Int'l Interactions 2022) uses ensemble classical ML (gradient boosting + RFs)
as the standard non-deep baseline. Reviewers from this community will ask
why we compared GNNs only to logistic regression and not to a
production-grade classical-ML ensemble. This module fills that gap.

Design choice: same 5 F-LR features (log_d_km, log_trade, contiguous,
both_major, rivalry_count) as input. This isolates "model class" (LR vs
GBM vs RF) from "feature engineering" (which would be a separate ablation).
If the ViEWS ensemble matches F-LR on identical features, the bottleneck
is features, not model class.

All three base learners use class_weight='balanced' to match F-LR's
treatment of the 1.4% positive base rate. Predictions are averaged
uniformly across the three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.feature_baselines import FEATURE_COLS


@dataclass
class ViEWSEnsemble:
    gbm: HistGradientBoostingClassifier
    rf: RandomForestClassifier
    lr: LogisticRegression
    feature_cols: tuple[str, ...] = tuple(FEATURE_COLS)

    @classmethod
    def fit(cls, train_features: pd.DataFrame, *, seed: int = 0) -> "ViEWSEnsemble":
        X = train_features[list(FEATURE_COLS)].to_numpy()
        y = train_features["edge_present"].to_numpy().astype(int)

        # HistGBM doesn't accept class_weight='balanced' in older sklearn; use
        # sample_weight instead. Compute the balanced weight per sample.
        n_pos = max(int(y.sum()), 1)
        n_neg = max(len(y) - n_pos, 1)
        sample_weight = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

        gbm = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=seed,
        )
        gbm.fit(X, y, sample_weight=sample_weight)

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X, y)

        lr = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver="lbfgs",
        )
        lr.fit(X, y)

        return cls(gbm=gbm, rf=rf, lr=lr)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = features[list(self.feature_cols)].to_numpy()
        p_gbm = self.gbm.predict_proba(X)[:, 1]
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_lr = self.lr.predict_proba(X)[:, 1]
        return (p_gbm + p_rf + p_lr) / 3.0

    def predict_proba_per_learner(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        """Per-learner predictions for diagnostic comparison."""
        X = features[list(self.feature_cols)].to_numpy()
        return {
            "gbm": self.gbm.predict_proba(X)[:, 1],
            "rf": self.rf.predict_proba(X)[:, 1],
            "lr": self.lr.predict_proba(X)[:, 1],
        }
