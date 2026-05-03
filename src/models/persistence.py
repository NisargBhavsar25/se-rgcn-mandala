"""Persistence baseline: predict E_{t+1} = E_t.

Reviewer's bottom-line baseline (week 2). On the EDGE-EXISTENCE task,
persistence is hard to beat -- alliances persist (this is the trap the
proposal exists to escape). On the TRANSITION task (formations +
dissolutions), persistence predicts ZERO events by construction, so its
PR-AUC equals the transition base rate and its lift = 1.0.

Any SE-RGCN configuration claiming a contribution must outperform this
floor on the transition task. If it cannot, the architecture is not
contributing the structural reasoning the paper claims.
"""

from __future__ import annotations

import pandas as pd

DYAD_KEY = ["year", "gwcode_i", "gwcode_j"]


def persistence_predict(edges_t: pd.DataFrame) -> pd.DataFrame:
    """Predict edge state at year t+1 by carrying forward year t.

    Args:
      edges_t: dyad-year edge presence table with columns
        ['year', 'gwcode_i', 'gwcode_j', 'edge_present'].

    Returns:
      DataFrame with the same dyad rows but year incremented and a
      'pred_score' column equal to the previous year's edge_present
      cast to float (0.0 or 1.0).
    """
    out = edges_t[DYAD_KEY + ["edge_present"]].copy()
    out["year"] = out["year"].astype(int) + 1
    out["pred_score"] = out["edge_present"].astype(float)
    return out[DYAD_KEY + ["pred_score"]]
