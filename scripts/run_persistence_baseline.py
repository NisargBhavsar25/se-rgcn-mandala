"""End-to-end persistence baseline (reviewer Week 2 gate).

Pipeline:
  distance_matrix -> PRD filter -> ATOP positives -> binary edge table
    -> derive transitions -> persistence prediction -> stratified metrics
    with block-by-year bootstrap CIs.

Persistence predicts E_{t+1} = E_t, so on the TRANSITION task its score is
constant 0 -> PR-AUC equals the transition base rate and lift = 1.0 by
construction. This is the floor every SE-RGCN config must beat.

Output: stdout report + JSON artifact at docs/results/persistence_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edge_table import build_edge_table  # noqa: E402
from src.data.loaders import load_atop_alliance_edges  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.evaluation.transitions import derive_transitions  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run persistence baseline end-to-end.")
    p.add_argument(
        "--distance",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "distance_matrix.parquet",
    )
    p.add_argument("--train-end", type=int, default=2000)
    p.add_argument("--val-end", type=int, default=2005)
    p.add_argument("--test-end", type=int, default=2014,
                   help="Truncated to 2014 because COW MID/Trade end here.")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "docs" / "results" / "persistence_baseline.json",
    )
    return p.parse_args()


def evaluate_target(
    df: pd.DataFrame,
    target_col: str,
    y_score: np.ndarray,
    n_bootstrap: int,
) -> dict:
    """PR-AUC + Recall@k + lift + Brier on a single target column.

    Bootstrap CIs are block-by-year because positives cluster within
    historical events (NATO 1949, USSR 1991, post-9/11 2001-2002).
    """
    y_true = df[target_col].to_numpy().astype(int)
    years = df["year"].to_numpy()
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean())
    # Recall@k where k matches the actual annual positive count, summed over years.
    k = max(n_pos, 1)

    pr, pr_lo, pr_hi = bootstrap_ci(
        y_true, y_score, pr_auc,
        years=years, block_by_year=True, n_bootstrap=n_bootstrap,
    )
    rec, rec_lo, rec_hi = bootstrap_ci(
        y_true, y_score, recall_at_k,
        years=years, block_by_year=True, n_bootstrap=n_bootstrap,
        metric_kwargs={"k": k},
    )
    lift, lift_lo, lift_hi = bootstrap_ci(
        y_true, y_score, lift_at_k,
        years=years, block_by_year=True, n_bootstrap=n_bootstrap,
        metric_kwargs={"k": k},
    )
    brier_p = brier_positives(y_true, y_score) if n_pos > 0 else float("nan")

    return {
        "n_dyad_years": int(len(y_true)),
        "n_positives": n_pos,
        "base_rate": base_rate,
        "k_for_topk": k,
        "pr_auc": {"point": pr, "lo95": pr_lo, "hi95": pr_hi},
        "recall_at_k": {"point": rec, "lo95": rec_lo, "hi95": rec_hi},
        "lift_at_k": {"point": lift, "lo95": lift_lo, "hi95": lift_hi},
        "brier_positives": brier_p,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    logger.info("Loading distance matrix from %s", args.distance)
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= 1949) & (distance.year <= args.test_end)
    ]

    logger.info("Applying PRD filter")
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
    logger.info("PRD universe: %d dyad-years", len(prd))

    logger.info("Loading ATOP alliance positives")
    atop = load_atop_alliance_edges(years=range(1949, args.test_end + 1))
    atop = atop[["year", "gwcode_i", "gwcode_j", "edge_present"]]
    logger.info("ATOP positives in window: %d", len(atop))

    logger.info("Building binary edge table")
    edges = build_edge_table(prd, atop)

    logger.info("Deriving transitions (lag=1)")
    trans = derive_transitions(edges, lag=1)
    trans["any_transition"] = (
        (trans.formation == 1) | (trans.dissolution == 1)
    ).astype(int)

    # Test split per the proposal's chronological OOD design.
    test = trans[
        (trans.year >= 2006) & (trans.year <= args.test_end)
    ].reset_index(drop=True)
    logger.info(
        "Test window %d-%d: %d dyad-years",
        2006, args.test_end, len(test),
    )

    # Persistence on transition prediction is constant zero.
    y_score = np.zeros(len(test), dtype=float)

    results = {
        "config": {
            "distance_matrix": str(args.distance),
            "alliance_source": "ATOP_5.1",
            "test_window": [2006, args.test_end],
            "train_window": [1950, args.train_end],
            "val_window": [args.train_end + 1, args.val_end],
            "n_bootstrap": args.bootstrap,
            "block_by_year": True,
        },
        "formation": evaluate_target(test, "formation", y_score, args.bootstrap),
        "dissolution": evaluate_target(test, "dissolution", y_score, args.bootstrap),
        "any_transition": evaluate_target(
            test, "any_transition", y_score, args.bootstrap
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", args.out)

    # Console report
    print()
    print("=" * 70)
    print("Persistence Baseline -- Week 2 Gate")
    print(f"  Test window: 2006-{args.test_end}, n = {len(test):,} dyad-years")
    print("=" * 70)
    for target in ("formation", "dissolution", "any_transition"):
        r = results[target]
        print(f"\n[{target}]  positives = {r['n_positives']}, "
              f"base rate = {r['base_rate']:.5f}")
        for m in ("pr_auc", "recall_at_k", "lift_at_k"):
            v = r[m]
            print(f"  {m:14s} = {v['point']:.4f}  "
                  f"[95% CI {v['lo95']:.4f} - {v['hi95']:.4f}]")
        print(f"  brier_positives = {r['brier_positives']:.4f}")
    print()
    print("By construction (constant-score predictor):")
    print("  pr_auc -> base_rate ; lift_at_k -> 1.0 ; recall_at_k -> k * base_rate / n_pos")
    print("If any of these fail to match, the pipeline is silently broken.")


if __name__ == "__main__":
    main()
