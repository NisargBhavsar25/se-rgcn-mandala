"""COW MID 5.0 -> dyadic conflict-onset derivation.

The MID 5.0 release does not include a pre-built dyadic file -- dyads must
be derived from MIDB (state-dispute participants). The locked rules,
documented in docs/mid_derivation.md:

  - Granularity: dispute-level (MIDB), NOT incident-level (MIDI). Incidents
    over-count because a single dispute can include many.
  - Onset coding: a dyad-year (i, j, y) is a positive ONSET only if the
    dispute's styear == y for both participants. Subsequent dispute-years
    are NOT positives -- they are CENSORED and excluded from prediction.
  - Hostility threshold: dispute-level max hostlev >= 3 ("use of force")
    by default. Levels 1-2 are threats / demonstrations; too noisy.
  - Originators only: orig == 1 on both participants. Joiners enter via
    different dynamics (alliance-pull, bandwagoning) and would conflate
    the prediction.
  - Translation: COW -> GW (Germany 255 -> 260, etc.) at the onset year.
  - Self-loops dropped post-translation, same as alliance loaders.

The censoring rule -- excluding ongoing-dispute years from the prediction
candidate set -- is what avoids the "duration trap" the staff feedback
called out: predicting onsets while sweeping ongoing disputes into the
candidate pool double-counts and biases the metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.gw_cow_mapping import cow_to_gw_series

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIDB_DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "cow_mid_5" / "MIDB 5.0.csv"

KEY = ["year", "gwcode_i", "gwcode_j"]


def derive_mid_onsets(
    midb: pd.DataFrame,
    *,
    hostility_threshold: int = 3,
    originators_only: bool = True,
) -> pd.DataFrame:
    """Self-join MIDB into one row per (dispute, opposing-originator-dyad).

    Returns columns:
      dispnum, onset_year, end_year, gwcode_i, gwcode_j, hostlev_max
    """
    df = midb

    if hostility_threshold is not None:
        max_host = df.groupby("dispnum")["hostlev"].max()
        qual_disputes = max_host[max_host >= hostility_threshold].index
        df = df[df["dispnum"].isin(qual_disputes)]

    if originators_only:
        df = df[df["orig"] == 1]

    cols = ["dispnum", "ccode", "sidea", "styear", "endyear", "hostlev"]
    a = df[cols].rename(columns={
        "ccode": "ccode_a", "sidea": "sidea_a",
        "styear": "styear_a", "endyear": "endyear_a",
        "hostlev": "hostlev_a",
    })
    b = df[cols].rename(columns={
        "ccode": "ccode_b", "sidea": "sidea_b",
        "styear": "styear_b", "endyear": "endyear_b",
        "hostlev": "hostlev_b",
    })
    pairs = a.merge(b, on="dispnum")
    pairs = pairs[pairs["sidea_a"] != pairs["sidea_b"]]
    # Canonical i < j by COW code first; deduplicates (a,b) vs (b,a) self-join rows.
    pairs = pairs[pairs["ccode_a"] < pairs["ccode_b"]]

    # For originators on opposing sides, both styears should equal the dispute
    # styear; min/max defends against any data weirdness.
    pairs["onset_year"] = pairs[["styear_a", "styear_b"]].min(axis=1)
    pairs["end_year"] = pairs[["endyear_a", "endyear_b"]].max(axis=1)
    pairs["hostlev_max"] = pairs[["hostlev_a", "hostlev_b"]].max(axis=1)

    pairs["gw_a"] = cow_to_gw_series(pairs["ccode_a"], pairs["onset_year"])
    pairs["gw_b"] = cow_to_gw_series(pairs["ccode_b"], pairs["onset_year"])
    pair_arr = pairs[["gw_a", "gw_b"]].to_numpy()
    pairs["gwcode_i"] = pair_arr.min(axis=1).astype(int)
    pairs["gwcode_j"] = pair_arr.max(axis=1).astype(int)

    n_self = int((pairs["gwcode_i"] == pairs["gwcode_j"]).sum())
    if n_self > 0:
        logger.info(
            "[MID] dropping %d self-loop dyad-disputes after COW->GW", n_self
        )
    pairs = pairs[pairs["gwcode_i"] < pairs["gwcode_j"]]

    pairs = pairs.drop_duplicates(["dispnum", "gwcode_i", "gwcode_j"])

    return pairs[
        ["dispnum", "onset_year", "end_year", "gwcode_i", "gwcode_j", "hostlev_max"]
    ].reset_index(drop=True)


def load_mid_onsets(
    *,
    midb_path: Path = MIDB_DEFAULT_PATH,
    hostility_threshold: int = 3,
    originators_only: bool = True,
) -> pd.DataFrame:
    """Read MIDB CSV and derive onsets. No year filter -- pre-window onsets
    are needed for the censoring step in build_mid_onset_edge_table."""
    df = pd.read_csv(midb_path, encoding="latin-1", low_memory=False)
    return derive_mid_onsets(
        df,
        hostility_threshold=hostility_threshold,
        originators_only=originators_only,
    )


def build_mid_onset_edge_table(
    prd_universe: pd.DataFrame,
    onsets: pd.DataFrame,
) -> pd.DataFrame:
    """Build the conflict-onset prediction table over the PRD universe.

    Returns the prd_universe rows with two new columns:
      edge_present : 1 iff that dyad-year is a conflict ONSET (year == styear).
      censored     : 1 iff that dyad-year is INSIDE an ongoing dispute
                     (year strictly after onset_year, through end_year).
    Censored rows must be excluded from prediction; they are not candidates
    for new onsets and including them as negatives biases the base rate.
    """
    pos = (
        onsets.rename(columns={"onset_year": "year"})[KEY]
        .drop_duplicates(KEY)
        .copy()
    )
    pos["edge_present"] = 1

    cens_rows: list[tuple[int, int, int]] = []
    for r in onsets.itertuples():
        for y in range(int(r.onset_year) + 1, int(r.end_year) + 1):
            cens_rows.append((y, int(r.gwcode_i), int(r.gwcode_j)))
    if cens_rows:
        cens = pd.DataFrame(cens_rows, columns=KEY).drop_duplicates(KEY)
        cens["censored"] = 1
    else:
        cens = pd.DataFrame(columns=KEY + ["censored"])

    out = prd_universe.merge(pos, on=KEY, how="left")
    out = out.merge(cens, on=KEY, how="left")
    out["edge_present"] = out["edge_present"].fillna(0).astype("int8")
    out["censored"] = out["censored"].fillna(0).astype("int8")

    # Onset overrides censoring: if a NEW dispute starts in year y on the
    # same dyad as an ongoing dispute, that year is still a valid onset
    # observation. Censoring applies only to "pure ongoing-dispute" years
    # where no new onset is recorded. Without this rule, ~15% of onsets
    # are silently dropped from prediction whenever dyads have overlapping
    # disputes (common for India-Pakistan, USA-Iran, etc.).
    out.loc[out["edge_present"] == 1, "censored"] = 0
    return out
