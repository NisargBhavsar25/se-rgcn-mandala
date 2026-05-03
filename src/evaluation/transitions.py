"""Edge-state -> transition labels.

Converts a long-form dyad-year edge presence table into transition labels
(formation / dissolution / no_change). All evaluation in this project
operates on transitions, not steady-state edges -- this is the primary
target of the SE-RGCN forecasting task.
"""

from __future__ import annotations

import pandas as pd

DYAD_KEY = ["year", "gwcode_i", "gwcode_j"]


def derive_transitions(edges: pd.DataFrame, *, lag: int = 1) -> pd.DataFrame:
    """Convert dyad-year edge presence into transition labels.

    For each (gwcode_i, gwcode_j, year) where year-lag is also observed,
    emit:
      formation: 1 if edge present at year and absent at year-lag, else 0.
      dissolution: 1 if absent at year and present at year-lag, else 0.

    Rows where year-lag is not observed are dropped (no label can be derived).
    """
    a = edges[DYAD_KEY + ["edge_present"]].copy()
    b = edges[DYAD_KEY + ["edge_present"]].copy()
    b["year"] = b["year"] + lag
    b = b.rename(columns={"edge_present": "edge_prev"})
    merged = a.merge(b, on=DYAD_KEY, how="inner")
    merged["formation"] = (
        (merged["edge_present"] == 1) & (merged["edge_prev"] == 0)
    ).astype("int8")
    merged["dissolution"] = (
        (merged["edge_present"] == 0) & (merged["edge_prev"] == 1)
    ).astype("int8")
    return merged[
        DYAD_KEY + ["edge_prev", "edge_present", "formation", "dissolution"]
    ]
