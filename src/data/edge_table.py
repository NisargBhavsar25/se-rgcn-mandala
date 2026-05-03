"""Build full dyad-year edge tables (PRD universe x edge presence).

Loaders return only positive (edge_present=1) rows. The PRD universe table
provides the full (year, gwcode_i, gwcode_j) candidate space. This module
joins them so every PRD dyad-year has a binary edge_present label.
"""

from __future__ import annotations

import pandas as pd

KEY = ["year", "gwcode_i", "gwcode_j"]


def build_edge_table(
    prd_universe: pd.DataFrame,
    positive_edges: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the full PRD-year edge presence table.

    Args:
      prd_universe: PRD-filtered dyad universe with at minimum
        ['year', 'gwcode_i', 'gwcode_j']. Other columns are preserved.
      positive_edges: DataFrame of recorded positive edges with at minimum
        ['year', 'gwcode_i', 'gwcode_j', 'edge_present'].

    Returns:
      prd_universe with a new 'edge_present' column (0 or 1). Positives that
      sit OUTSIDE the PRD universe are silently dropped (they shouldn't drive
      the model and including them would inflate the negative pool spuriously).
    """
    pos = (
        positive_edges[KEY + ["edge_present"]]
        .drop_duplicates(KEY, keep="first")
    )
    out = prd_universe.merge(pos, on=KEY, how="left")
    out["edge_present"] = out["edge_present"].fillna(0).astype("int8")
    return out
