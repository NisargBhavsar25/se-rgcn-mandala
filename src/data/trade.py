"""COW Trade 4.0 dyad-year loader.

The COW Trade codebook recommends smoothflow over the raw flow columns
because the latter has more -9 sentinel ("missing") values. Even smoothflow
has some -9s, which we replace with 0. The output is a single canonical
dyad-year table with one symmetric trade-volume column (sum of both
directional flows).

Logic mirrors the alliance loaders: COW -> GW translation, canonical
(i < j) ordering, self-loop drop, year-window filter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.gw_cow_mapping import cow_to_gw_series

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COW_TRADE_DEFAULT_PATH = (
    PROJECT_ROOT / "data" / "raw" / "cow_trade_4.0"
    / "COW_Trade_4.0" / "Dyadic_COW_4.0.csv"
)


def load_cow_trade_edges(
    *,
    cow_trade_path: Path = COW_TRADE_DEFAULT_PATH,
    years: Optional[range] = None,
) -> pd.DataFrame:
    """Load COW Trade 4.0, GW-coded, canonical (i < j).

    Returns columns:
      year, gwcode_i, gwcode_j, total_trade  (millions USD; smoothed; symmetric).
    """
    df = pd.read_csv(cow_trade_path, encoding="latin-1", low_memory=False)

    for col in ("smoothflow1", "smoothflow2"):
        df.loc[df[col] < 0, col] = 0.0
    df["total_trade"] = (df["smoothflow1"] + df["smoothflow2"]).astype("float32")

    df["gw1"] = cow_to_gw_series(df["ccode1"], df["year"])
    df["gw2"] = cow_to_gw_series(df["ccode2"], df["year"])
    pair = df[["gw1", "gw2"]].to_numpy()
    df["gwcode_i"] = pair.min(axis=1).astype(int)
    df["gwcode_j"] = pair.max(axis=1).astype(int)

    n_self = int((df["gwcode_i"] == df["gwcode_j"]).sum())
    if n_self > 0:
        logger.info("[COW Trade] dropping %d self-loop rows after COW->GW", n_self)
    df = df[df["gwcode_i"] < df["gwcode_j"]]

    if years is not None:
        df = df[(df["year"] >= years.start) & (df["year"] < years.stop)]

    out = (
        df.groupby(["year", "gwcode_i", "gwcode_j"], as_index=False)
          .agg(total_trade=("total_trade", "sum"))
    )
    out["year"] = out["year"].astype("int16")
    out["gwcode_i"] = out["gwcode_i"].astype("int32")
    out["gwcode_j"] = out["gwcode_j"].astype("int32")
    return out.reset_index(drop=True)
