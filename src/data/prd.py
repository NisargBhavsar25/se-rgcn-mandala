"""Politically Relevant Dyad (PRD) filter -- Maoz 2011 convention.

A dyad (i, j) is PRD if:
  - At least one of (i, j) is a COW major power for that year, OR
  - The two states are directly or near-directly contiguous
    (default threshold: 241 km, ~= 150 miles, matching COW direct-contiguity
    level 4 "separated by at most 150 miles of water").

Random negatives over the full Cartesian universe are uninformative -- the
reviewer specifically called out "Burkina Faso and Bolivia stayed unallied"
as a useless negative that inflates AUC. Filtering to PRD before training
keeps the candidate pool meaningful.

The matrix is GW-coded (CShapes-native); the major-power table below is
GW-coded throughout. For 1950-2018 the only GW vs COW divergence in the
major-power list is unified Germany (GW 260, COW 255).
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


# COW major-power list expressed in GW codes.
# Each tuple: (gwcode, year_in, year_out_inclusive_or_None).
# Source: COW Major Power list (codebook v6).
_MAJOR_POWERS_GW: list[tuple[int, int, Optional[int]]] = [
    (2,   1898, None),   # USA
    (200, 1816, None),   # UK
    (220, 1816, 1940),   # France pre-WWII
    (220, 1945, None),   # France post-WWII
    (260, 1991, None),   # Germany -- COW codes FRG/GDR (1949-1990) as non-major;
                         # unified Germany regains major status in 1991.
    (365, 1816, None),   # USSR / Russia (continuous)
    (710, 1950, None),   # PRC China
    (740, 1991, None),   # Japan -- COW codes Japan as non-major 1946-1990;
                         # restored to major status in 1991.
]


def major_powers_in_year(year: int) -> set[int]:
    """COW major-power GW codes active in the given year."""
    return {
        gw
        for gw, y_in, y_out in _MAJOR_POWERS_GW
        if y_in <= year and (y_out is None or year <= y_out)
    }


def prd_dyads(
    distance_year: pd.DataFrame,
    year: int,
    *,
    contiguity_threshold_km: float = 241.0,
    major_powers: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Filter a single year's distance table to politically relevant dyads.

    Args:
      distance_year: DataFrame with at minimum ['gwcode_i', 'gwcode_j', 'd_km'].
        Other columns are preserved.
      year: the year being filtered (drives the default major-power list).
      contiguity_threshold_km: dyads with d_km <= this are PRD by contiguity.
        Default 241 km matches COW direct-contiguity level 4 (~150 miles).
      major_powers: optional override of the major-power GW code set;
        default is COW-canonical via `major_powers_in_year`.

    Returns:
      Subset of `distance_year` retaining only PRD rows.
    """
    mp = set(major_powers) if major_powers is not None else major_powers_in_year(year)
    is_contiguous = distance_year["d_km"] <= contiguity_threshold_km
    is_major_dyad = (
        distance_year["gwcode_i"].isin(mp) | distance_year["gwcode_j"].isin(mp)
    )
    return distance_year.loc[is_contiguous | is_major_dyad].copy()


def prd_dyads_all_years(
    distance_matrix: pd.DataFrame,
    *,
    contiguity_threshold_km: float = 241.0,
) -> pd.DataFrame:
    """Apply the PRD filter year-by-year over the full multi-year distance matrix."""
    if "year" not in distance_matrix.columns:
        raise ValueError("distance_matrix must have a 'year' column")
    parts: list[pd.DataFrame] = []
    for year, sub in distance_matrix.groupby("year"):
        parts.append(
            prd_dyads(sub, int(year), contiguity_threshold_km=contiguity_threshold_km)
        )
    return pd.concat(parts, ignore_index=True) if parts else distance_matrix.iloc[0:0].copy()
