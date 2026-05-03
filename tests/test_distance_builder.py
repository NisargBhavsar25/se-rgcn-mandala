"""Pinned regression tests for the CShapes 2.0 distance matrix.

Loads the built distance matrix at data/processed/distance_matrix.parquet and
asserts that five well-known dyad-year distances fall in expected ranges.
Skips when the parquet does not yet exist; fails loudly when a refactor
silently changes the geometry, the dissolve, or the as-of-date filter.

Bounds are wide enough to absorb minor CShapes-revision differences but tight
enough that any real bug (km becomes degree, sign flip, missing dissolve,
self-intersection artifact) trips the test.
"""

from pathlib import Path

import pandas as pd
import pytest

DISTANCE_PARQUET = Path("data/processed/distance_matrix.parquet")

# Gleditsch-Ward state codes (CShapes-native; do not confuse with COW codes --
# notably, unified Germany is GW 260 vs COW 255).
USA = 2
UK = 200
FRANCE = 220
GERMANY = 260
USSR_RUSSIA = 365
INDIA = 750
PAKISTAN = 770
BRAZIL = 140
ARGENTINA = 160

# (year, ccode_a, ccode_b, lo_km, hi_km, expected_border_type)
PINNED = [
    # Reunified Germany shares a long land border with France. Use 1995 to
    # dodge the 1990 reunification transition year.
    (1995, FRANCE, GERMANY, 0.0, 5.0, "land"),
    # Bering Strait: mainland Alaska <-> mainland Chukotka is ~85 km;
    # Little Diomede <-> Big Diomede is ~4 km. Whether CShapes includes the
    # Diomedes drives the spread, so we allow [3, 100].
    (1980, USA, USSR_RUSSIA, 3.0, 100.0, "maritime"),
    # India and Pakistan share a long land border across Punjab and Kashmir.
    (1970, INDIA, PAKISTAN, 0.0, 5.0, "land"),
    # English Channel at the Strait of Dover is ~33 km.
    (1965, UK, FRANCE, 20.0, 50.0, "maritime"),
    # Brazil-Argentina share a land border at Iguazu.
    (2000, BRAZIL, ARGENTINA, 0.0, 5.0, "land"),
]


@pytest.fixture(scope="module")
def matrix() -> pd.DataFrame:
    if not DISTANCE_PARQUET.exists():
        pytest.skip(
            f"{DISTANCE_PARQUET} not found; "
            "run scripts/build_distance_matrix.py before this test."
        )
    return pd.read_parquet(DISTANCE_PARQUET)


def _lookup(df: pd.DataFrame, year: int, a: int, b: int) -> pd.Series:
    lo, hi = (a, b) if a < b else (b, a)
    rows = df[(df["year"] == year) & (df["gwcode_i"] == lo) & (df["gwcode_j"] == hi)]
    assert len(rows) == 1, (
        f"expected 1 row for (year={year}, {lo}<->{hi}), got {len(rows)}"
    )
    return rows.iloc[0]


@pytest.mark.parametrize("year,a,b,lo_km,hi_km,border", PINNED)
def test_pinned_distance(matrix, year, a, b, lo_km, hi_km, border):
    row = _lookup(matrix, year, a, b)
    assert lo_km <= row["d_km"] <= hi_km, (
        f"{year} {a}<->{b}: d_km={row['d_km']:.2f} outside [{lo_km}, {hi_km}]"
    )
    assert row["border_type"] == border, (
        f"{year} {a}<->{b}: border_type={row['border_type']!r} expected {border!r}"
    )


def test_dyad_ordering_is_canonical(matrix):
    """Every row must satisfy gwcode_i < gwcode_j (canonical undirected dyad)."""
    bad = matrix[matrix["gwcode_i"] >= matrix["gwcode_j"]]
    assert bad.empty, f"{len(bad)} rows violate gwcode_i < gwcode_j ordering"


def test_distances_are_nonnegative(matrix):
    assert (matrix["d_km"] >= 0).all(), "negative distances present"


def test_distances_are_bounded(matrix):
    # Earth circumference / 2 ~= 20,037 km. Add 1% slack for floating point.
    assert (matrix["d_km"] <= 20_200).all(), "distance exceeds antipodal bound"
