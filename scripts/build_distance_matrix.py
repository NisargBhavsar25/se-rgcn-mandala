"""Build the CShapes 2.0 dyad-year distance matrix and persist as parquet.

Run from the project root:

    python scripts/build_distance_matrix.py
    python scripts/build_distance_matrix.py --start 1990 --end 2001

The default arguments reproduce the locked study window (1950-2018 inclusive).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.distance_builder import DistanceConfig, build_distance_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CShapes 2.0 distance matrix.")
    p.add_argument(
        "--cshapes",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "CShapes-2.0" / "CShapes-2.0.shp",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "distance_matrix.parquet",
    )
    p.add_argument("--start", type=int, default=1950, help="First year (inclusive).")
    p.add_argument("--end", type=int, default=2019, help="Last year (exclusive).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    cfg = DistanceConfig(
        cshapes_path=args.cshapes,
        output_path=args.out,
        years=range(args.start, args.end),
    )
    df = build_distance_matrix(cfg)
    print(
        f"Wrote {len(df):,} rows "
        f"(years {df['year'].min()}-{df['year'].max()}) to {args.out}"
    )


if __name__ == "__main__":
    main()
