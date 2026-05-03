"""Time-aware hard-negative sampling for rare-event link prediction.

Reviewer-mandated approach (REVIEWER_NOTES.md, section 1):

  - For each positive event at year t, sample negatives from year t +/- w.
    This controls for temporal confounders (era, system size); a positive in
    1962 paired with negatives from 1962 forces the model to learn what was
    special about *this dyad*, not what was special about the 1960s.

  - Hard-negative mining via a hardness-scoring callable. Without this,
    random negatives are dominated by uninformative dyads ("Burkina Faso and
    Bolivia stayed unallied") and AUC inflates artificially. Pass a callable
    that scores candidate dyads by how plausibly they could have formed an
    alliance (high trade volume, geographic proximity, shared rivals).

This module is hardness-agnostic; the project supplies the scoring function
elsewhere (see configs/ when the trade-only baseline is wired up).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


def time_aware_negatives(
    positives: pd.DataFrame,
    candidates: pd.DataFrame,
    n_per_positive: int,
    *,
    year_window: int = 2,
    hardness_score: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    hardness_top_p: float = 0.1,
    seed: int = 0,
) -> pd.DataFrame:
    """For each positive, sample negatives from a +/- year_window time window.

    Args:
      positives: DataFrame with ['year', 'gwcode_i', 'gwcode_j'].
      candidates: pool of all dyad-years with at minimum
        ['year', 'gwcode_i', 'gwcode_j', 'edge_present'] plus any features
        the hardness_score callable needs.
      n_per_positive: number of negatives sampled per positive.
      year_window: candidate years are [t - w, t + w] inclusive.
      hardness_score: maps a candidate DataFrame to a hardness vector
        (higher = harder, better as a negative). None for uniform sampling.
      hardness_top_p: when hardness_score is provided, restrict the per-positive
        candidate pool to the top-p fraction by hardness before uniform sampling.
      seed: RNG seed.

    Returns:
      DataFrame with ['year', 'gwcode_i', 'gwcode_j', 'positive_year',
      'positive_i', 'positive_j']. The positive_* columns let downstream
      eval stratify negatives by which positive event anchored them.
    """
    rng = np.random.default_rng(seed)
    pool_all = candidates[candidates["edge_present"] == 0]

    samples: list[pd.DataFrame] = []
    for _, pos in positives.iterrows():
        t = int(pos["year"])
        pi, pj = int(pos["gwcode_i"]), int(pos["gwcode_j"])
        pool = pool_all[
            (pool_all["year"] >= t - year_window)
            & (pool_all["year"] <= t + year_window)
            & ~((pool_all["gwcode_i"] == pi) & (pool_all["gwcode_j"] == pj))
        ]
        if len(pool) == 0:
            continue

        if hardness_score is not None:
            scores = np.asarray(hardness_score(pool), dtype=float)
            cutoff = float(np.quantile(scores, 1.0 - hardness_top_p))
            pool = pool.loc[scores >= cutoff]
            if len(pool) == 0:
                continue

        n = min(n_per_positive, len(pool))
        idx = rng.choice(len(pool), size=n, replace=False)
        chosen = pool.iloc[idx][["year", "gwcode_i", "gwcode_j"]].copy()
        chosen["positive_year"] = t
        chosen["positive_i"] = pi
        chosen["positive_j"] = pj
        samples.append(chosen)

    if not samples:
        return pd.DataFrame(
            columns=[
                "year", "gwcode_i", "gwcode_j",
                "positive_year", "positive_i", "positive_j",
            ]
        )
    return pd.concat(samples, ignore_index=True)
