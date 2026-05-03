"""CShapes 2.0 -> dyad-year geodesic distance matrix builder.

Implements the locked design decisions documented in
docs/distance_decisions.md:

  1. Time-varying borders   -- CShapes' native state-period rows, filtered to a
                               mid-year as-of date (June 30) within each year.
  2. Maritime adjacency     -- closest-point-on-coast between actual polygons.
  3. Trans-oceanic distance -- geodesic on WGS84 via pyproj.Geod.
  4. Split / non-contiguous -- minimum distance across the union of all polygons
                               (handled natively by shapely.nearest_points on
                                multi-polygon state geometries).
  5. State-ID system        -- Gleditsch-Ward codes (CShapes-native, lossless).
                               GW->COW translation is applied at the join layer
                               when merging with COW Alliance / MID / Trade.

Schema verified against CShapes 2.0 release dated 2021-05-20.

Output is a single parquet keyed by (year, gwcode_i, gwcode_j). The build is
intended to be run ONCE; downstream pipelines consume the saved parquet.
Changing any decision here invalidates every prior experimental result, so any
rebuild must be paired with a docs/distance_decisions.md update and a re-run
of tests/test_distance_builder.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import Geod
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points
from shapely.validation import make_valid
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Geodesic computations are always on WGS84. Never use shapely.distance on
# lat/lon coordinates -- it returns degree units, the most common silent bug
# in this kind of pipeline.
GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class DistanceConfig:
    cshapes_path: Path
    output_path: Path
    years: range = range(1950, 2019)
    # CShapes 2.0 schema (verified against the 2021-05-20 release).
    state_id_col: str = "gwcode"
    start_date_col: str = "gwsdate"
    end_date_col: str = "gwedate"
    capital_lon_col: str = "caplong"
    capital_lat_col: str = "caplat"
    # As-of date within each year for selecting valid state-period rows.
    # Mid-year (June 30) is the locked compromise vs year-start / year-end.
    as_of_month: int = 6
    as_of_day: int = 30
    # Diagnostic border classification only; not a model feature.
    maritime_threshold_km: float = 400.0


def load_cshapes(path: Path) -> gpd.GeoDataFrame:
    """Load CShapes 2.0 in EPSG:4326 with validated geometries."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"CShapes file at {path} has no CRS metadata.")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    # CShapes polygons can have self-intersections; make_valid is mandatory
    # before any nearest_points / intersects op or shapely will crash or lie.
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    return gdf


def states_at_date(
    gdf: gpd.GeoDataFrame, asof: pd.Timestamp, cfg: DistanceConfig
) -> gpd.GeoDataFrame:
    """Subset to state-period rows valid at `asof`; dissolve by state ID.

    CShapes encodes border-change events as separate state-period rows
    (e.g., USA has distinct rows for pre-Alaska, Alaska-only, Alaska+Hawaii).
    Filtering by a specific date selects exactly the row valid then. The
    dissolve is a no-op in the common case but defends against rare encodings
    where one state has multiple polygon rows valid at the same date.
    """
    mask = (gdf[cfg.start_date_col] <= asof) & (gdf[cfg.end_date_col] >= asof)
    cols = [
        cfg.state_id_col,
        cfg.capital_lon_col,
        cfg.capital_lat_col,
        "geometry",
    ]
    sub = gdf.loc[mask, cols].copy()
    return sub.dissolve(by=cfg.state_id_col, as_index=False, aggfunc="first")


def min_geodesic_km(geom_i: BaseGeometry, geom_j: BaseGeometry) -> float:
    """Minimum geodesic distance (km) between two (Multi)Polygons on WGS84.

    Adjacent or overlapping polygons return 0.0. The Mandala kernel
    S(d) = cos(beta * d) * exp(-alpha * d) gives S(0) = 1, the correct
    max-coupling value, so no distance floor is applied.
    """
    if geom_i.intersects(geom_j):
        return 0.0
    p_i, p_j = nearest_points(geom_i, geom_j)
    _, _, dist_m = GEOD.inv(p_i.x, p_i.y, p_j.x, p_j.y)
    return dist_m / 1000.0


def capital_geodesic_km(
    row_i: pd.Series, row_j: pd.Series, cfg: DistanceConfig
) -> float:
    """Diagnostic capital-to-capital geodesic distance (km).

    Stored alongside the polygon distance for sanity-checking; never consumed
    as a model feature.
    """
    _, _, dist_m = GEOD.inv(
        row_i[cfg.capital_lon_col], row_i[cfg.capital_lat_col],
        row_j[cfg.capital_lon_col], row_j[cfg.capital_lat_col],
    )
    return dist_m / 1000.0


def classify_border(
    geom_i: BaseGeometry,
    geom_j: BaseGeometry,
    d_km: float,
    maritime_threshold_km: float,
) -> str:
    """Diagnostic land / maritime / oceanic classification."""
    if d_km == 0.0 and geom_i.touches(geom_j):
        return "land"
    if d_km <= maritime_threshold_km:
        return "maritime"
    return "oceanic"


def build_year(
    gdf: gpd.GeoDataFrame, year: int, cfg: DistanceConfig
) -> pd.DataFrame:
    """Symmetric dyad distance table for a single year (gwcode_i < gwcode_j)."""
    asof = pd.Timestamp(year=year, month=cfg.as_of_month, day=cfg.as_of_day)
    states = states_at_date(gdf, asof, cfg)
    n = len(states)

    rows: list[dict] = []
    for i in range(n):
        ri = states.iloc[i]
        gi = ri.geometry
        ci = int(ri[cfg.state_id_col])
        for j in range(i + 1, n):
            rj = states.iloc[j]
            gj = rj.geometry
            cj = int(rj[cfg.state_id_col])
            lo_id, hi_id = (ci, cj) if ci < cj else (cj, ci)
            d_km = min_geodesic_km(gi, gj)
            rows.append(
                {
                    "year": year,
                    "gwcode_i": lo_id,
                    "gwcode_j": hi_id,
                    "d_km": d_km,
                    "d_capital_km": capital_geodesic_km(ri, rj, cfg),
                    "border_type": classify_border(
                        gi, gj, d_km, cfg.maritime_threshold_km
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_distance_matrix(cfg: DistanceConfig) -> pd.DataFrame:
    """Build the full (year, gwcode_i, gwcode_j) distance matrix; persist as parquet."""
    gdf = load_cshapes(cfg.cshapes_path)

    frames: list[pd.DataFrame] = []
    for year in tqdm(cfg.years, desc="years"):
        frames.append(build_year(gdf, year, cfg))

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype("int16")
    out["gwcode_i"] = out["gwcode_i"].astype("int32")
    out["gwcode_j"] = out["gwcode_j"].astype("int32")
    out["d_km"] = out["d_km"].astype("float32")
    out["d_capital_km"] = out["d_capital_km"].astype("float32")
    out["border_type"] = out["border_type"].astype("category")

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.output_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), cfg.output_path)
    return out
