"""End-to-end persistence baseline -- multi-task gate (Week 2).

Two parallel tasks evaluated:

  ALLIANCE-FORMATION:  ATOP 5.1 alliance edges -> derive transitions ->
                       persistence on formation prediction.

  CONFLICT-ONSET:      COW MIDB 5.0 -> derive opposing-originator dyad-disputes
                       (hostlev >= 3) -> mark onset-year positives, censor
                       ongoing-dispute years -> persistence on onset prediction.

Persistence predicts no transition / no onset -> y_score is constant 0 ->
PR-AUC equals base rate, lift = 1.0 by construction. This is the floor any
SE-RGCN configuration must beat on each task.

The two tasks are reported in parallel (never aggregated). The Mandala
hypothesis is most directly a claim about *conflict* topology (Ari/enemy =
bordering state); the conflict task is the strongest test of the spatial
prior. Reporting both lets us test the dissociation.

Output: stdout report + JSON artifact at docs/results/persistence_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.edge_table import build_edge_table  # noqa: E402
from src.data.loaders import load_atop_alliance_edges  # noqa: E402
from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.evaluation.transitions import derive_transitions  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task persistence baseline (alliance + conflict)."
    )
    p.add_argument(
        "--distance", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "distance_matrix.parquet",
    )
    p.add_argument("--test-start", type=int, default=2006)
    p.add_argument("--test-end", type=int, default=2014,
                   help="Truncated to 2014 because COW MID/Trade end here.")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--hostility-threshold", type=int, default=3)
    p.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "docs" / "results" / "persistence_baseline.json",
    )
    return p.parse_args()


def evaluate_target(
    df: pd.DataFrame, target_col: str, y_score: np.ndarray, n_bootstrap: int,
) -> dict:
    """PR-AUC + Recall@k + lift + Brier on a single target column."""
    y_true = df[target_col].to_numpy().astype(int)
    years = df["year"].to_numpy()
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if len(y_true) > 0 else float("nan")
    k = max(n_pos, 1)

    pr, pr_lo, pr_hi = bootstrap_ci(
        y_true, y_score, pr_auc, years=years, block_by_year=True,
        n_bootstrap=n_bootstrap,
    )
    rec, rec_lo, rec_hi = bootstrap_ci(
        y_true, y_score, recall_at_k, years=years, block_by_year=True,
        n_bootstrap=n_bootstrap, metric_kwargs={"k": k},
    )
    lift, lift_lo, lift_hi = bootstrap_ci(
        y_true, y_score, lift_at_k, years=years, block_by_year=True,
        n_bootstrap=n_bootstrap, metric_kwargs={"k": k},
    )
    brier = brier_positives(y_true, y_score) if n_pos > 0 else float("nan")

    return {
        "n_dyad_years": int(len(y_true)),
        "n_positives": n_pos,
        "base_rate": base_rate,
        "k_for_topk": k,
        "pr_auc": {"point": pr, "lo95": pr_lo, "hi95": pr_hi},
        "recall_at_k": {"point": rec, "lo95": rec_lo, "hi95": rec_hi},
        "lift_at_k": {"point": lift, "lo95": lift_lo, "hi95": lift_hi},
        "brier_positives": brier,
    }


def evaluate_alliance_task(
    prd: pd.DataFrame, args: argparse.Namespace,
) -> dict:
    logger.info("[alliance] loading ATOP positives")
    atop = load_atop_alliance_edges(years=range(1949, args.test_end + 1))
    atop = atop[["year", "gwcode_i", "gwcode_j", "edge_present"]]

    edges = build_edge_table(prd, atop)
    trans = derive_transitions(edges, lag=1)
    trans["any_transition"] = (
        (trans.formation == 1) | (trans.dissolution == 1)
    ).astype(int)

    test = trans[
        (trans.year >= args.test_start) & (trans.year <= args.test_end)
    ].reset_index(drop=True)
    logger.info("[alliance] test n=%d dyad-years", len(test))

    y_score = np.zeros(len(test), dtype=float)
    return {
        "config": {
            "source": "ATOP_5.1",
            "test_window": [args.test_start, args.test_end],
            "n_dyad_years": int(len(test)),
        },
        "formation": evaluate_target(test, "formation", y_score, args.bootstrap),
        "dissolution": evaluate_target(test, "dissolution", y_score, args.bootstrap),
        "any_transition": evaluate_target(
            test, "any_transition", y_score, args.bootstrap
        ),
    }


def evaluate_conflict_task(
    prd: pd.DataFrame, args: argparse.Namespace,
) -> dict:
    logger.info("[conflict] deriving MID onsets (hostility >= %d, originators)",
                args.hostility_threshold)
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)
    n_onsets_total = len(onsets)
    onsets_in_test = onsets[
        (onsets.onset_year >= args.test_start)
        & (onsets.onset_year <= args.test_end)
    ]

    table = build_mid_onset_edge_table(prd, onsets)

    test_full = table[
        (table.year >= args.test_start) & (table.year <= args.test_end)
    ].reset_index(drop=True)

    # Apply censoring: ongoing-dispute years are NOT prediction candidates.
    test = test_full[test_full.censored == 0].reset_index(drop=True)
    n_censored = int((test_full.censored == 1).sum())

    # PRD retention sanity check: how many of the in-window onsets actually
    # land inside the PRD universe?
    in_window_pos_in_prd = int(test["edge_present"].sum())
    in_window_pos_total = int(len(onsets_in_test))
    retention = (
        in_window_pos_in_prd / in_window_pos_total
        if in_window_pos_total else float("nan")
    )
    logger.info(
        "[conflict] PRD retention: %d/%d = %.1f%% of in-window onsets",
        in_window_pos_in_prd, in_window_pos_total, retention * 100,
    )

    y_score = np.zeros(len(test), dtype=float)
    return {
        "config": {
            "source": "COW_MID_5.0",
            "hostility_threshold": args.hostility_threshold,
            "originators_only": True,
            "test_window": [args.test_start, args.test_end],
            "n_dyad_years_uncensored": int(len(test)),
            "n_dyad_years_censored": n_censored,
            "n_onsets_total_all_years": int(n_onsets_total),
            "n_onsets_in_test_window": in_window_pos_total,
            "prd_retention": retention,
        },
        "onset": evaluate_target(test, "edge_present", y_score, args.bootstrap),
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

    results = {
        "config": {
            "distance_matrix": str(args.distance),
            "test_window": [args.test_start, args.test_end],
            "n_bootstrap": args.bootstrap,
            "block_by_year": True,
            "prd_universe_size": int(len(prd)),
        },
        "alliance_task": evaluate_alliance_task(prd, args),
        "conflict_task": evaluate_conflict_task(prd, args),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results to %s", args.out)

    # Console report
    print()
    print("=" * 76)
    print("Persistence Baseline -- Multi-task Week 2 Gate")
    print(f"  Test window: {args.test_start}-{args.test_end}")
    print(f"  PRD universe: {len(prd):,} dyad-years")
    print("=" * 76)

    print("\n[ALLIANCE TASK -- ATOP 5.1]")
    a = results["alliance_task"]
    print(f"  test n = {a['config']['n_dyad_years']:,} dyad-years")
    for tgt in ("formation", "dissolution", "any_transition"):
        r = a[tgt]
        print(f"\n  ({tgt})  n_pos = {r['n_positives']}, "
              f"base_rate = {r['base_rate']:.5f}")
        for m in ("pr_auc", "recall_at_k", "lift_at_k"):
            v = r[m]
            print(f"    {m:14s} = {v['point']:.4f}  "
                  f"[95% CI {v['lo95']:.4f} - {v['hi95']:.4f}]")
        print(f"    brier_positives = {r['brier_positives']:.4f}")

    print("\n[CONFLICT TASK -- COW MID 5.0, hostlev>=", args.hostility_threshold,
          ", originators]")
    c = results["conflict_task"]
    cc = c["config"]
    print(f"  uncensored test n = {cc['n_dyad_years_uncensored']:,} dyad-years")
    print(f"  censored (excluded) = {cc['n_dyad_years_censored']:,} dyad-years")
    print(f"  PRD retention of in-window onsets = "
          f"{cc['prd_retention']*100:.1f}% "
          f"({int(cc['prd_retention']*cc['n_onsets_in_test_window'])}"
          f"/{cc['n_onsets_in_test_window']})")
    r = c["onset"]
    print(f"\n  (onset)  n_pos = {r['n_positives']}, "
          f"base_rate = {r['base_rate']:.5f}")
    for m in ("pr_auc", "recall_at_k", "lift_at_k"):
        v = r[m]
        print(f"    {m:14s} = {v['point']:.4f}  "
              f"[95% CI {v['lo95']:.4f} - {v['hi95']:.4f}]")
    print(f"    brier_positives = {r['brier_positives']:.4f}")

    print()
    print("By construction (constant-score predictor):")
    print("  pr_auc -> base_rate ; lift_at_k -> 1.0 exactly.")
    print("If those don't match, the pipeline is silently broken.")


if __name__ == "__main__":
    main()
