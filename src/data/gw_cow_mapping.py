"""Gleditsch-Ward (GW) <-> Correlates of War (COW) state-code translation.

CShapes 2.0 (and therefore our distance matrix) is GW-coded. COW Alliance v4,
COW MID 5, COW Trade 4, and ATOP 5.1 are all COW-coded. To join, we translate
COW -> GW per the small set of known divergences. Codes not in the table
default to identity (the common case for ~95% of states).

VERIFIED divergences (1950-2018 window):

  Germany (post-1949):  COW 255  ->  GW 260
    COW codes both FRG (1949-1990) and Unified Germany (1990-) as 255.
    GW/CShapes uses 260 for both. The pre-1945 Germany Empire / Weimar / Nazi
    uses GW 255 and COW 255 (identity, not in this table).

TODO: verify against the CShapes 2.0 codebook (Schvitz et al. 2022, supplementary
materials) and add if they affect dyad-year counts in our window:

  Yemen (1990+ unification):
    Suspected COW 679 (Unified Yemen) -> GW 678 (continues North Yemen code).
  Vietnam (1976+ unification):
    Suspected COW 818 (Unified Vietnam) -> GW 816 (continues North Vietnam code).
  Serbia / Yugoslavia successors (2003+):
    Multiple successor-state code reassignments.

Until verified, those COWs map to themselves (identity). The diagnostic
script in tests/test_gw_cow_mapping.py prints any unmapped COW codes that
appear in ATOP / COW datasets but not in CShapes; surprise diffs land there.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# (cow_code, year_start, year_end_inclusive_or_None, gw_code).
# Identity is the default; only divergences are listed.
_COW_TO_GW: list[tuple[int, int, Optional[int], int]] = [
    (255, 1949, None, 260),  # Germany: FRG (1949-1990) and Unified (1990-)
]


def cow_to_gw(cow_code: int, year: int) -> int:
    """Translate a COW state code to its GW equivalent for the given year.

    Returns the COW code unchanged if no divergence is registered.
    """
    for cow, y_start, y_end, gw in _COW_TO_GW:
        if cow == cow_code and y_start <= year and (y_end is None or year <= y_end):
            return gw
    return cow_code


def cow_to_gw_series(cow_series: pd.Series, year_series: pd.Series) -> pd.Series:
    """Vectorized COW -> GW translation. Inputs must be equal-length pandas Series."""
    out = cow_series.copy()
    for cow, y_start, y_end, gw in _COW_TO_GW:
        if y_end is None:
            mask = (cow_series == cow) & (year_series >= y_start)
        else:
            mask = (
                (cow_series == cow)
                & (year_series >= y_start)
                & (year_series <= y_end)
            )
        out.loc[mask] = gw
    return out
