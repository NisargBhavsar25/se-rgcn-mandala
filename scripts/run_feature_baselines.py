"""Hand-crafted feature baselines (F + H) for the conflict-onset gate.

Runs in parallel:
  (F) Logistic regression on five hand-crafted features:
        log(1+d_km), log(1+total_trade_{y-1}), contiguity, both-major, rivalry_count.
      Trained on full PRD-uncensored 1950-2000 with class_weight='balanced'.
  (H) Pure rivalry-history scorer: score = onset count in [y-W, y-1].
      No training; tests whether the rivalry-literature regularity is the
      entire signal.

Both evaluated on the same test set (2006-2014 uncensored, conflict onset
hostlev>=3 originators) as the trade-only RGCN and identity-only ablation.

Output: stdout report + docs/results/feature_baselines.json.
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

from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.data.trade import load_cow_trade_edges  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.data.loaders import load_atop_alliance_edges  # noqa: E402
from src.models.feature_baselines import (  # noqa: E402
    FEATURE_COLS, FEATURE_COLS_RICH,
    FeatureLR, build_feature_table, build_feature_table_rich, rivalry_only_score,
)
from src.models.views_ensemble import ViEWSEnsemble  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--distance", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "distance_matrix.parquet")
    p.add_argument("--train-start", type=int, default=1950)
    p.add_argument("--train-end", type=int, default=2000)
    p.add_argument("--val-start", type=int, default=2001)
    p.add_argument("--val-end", type=int, default=2005)
    p.add_argument("--test-start", type=int, default=2006)
    p.add_argument("--test-end", type=int, default=2014)
    p.add_argument("--hostility-threshold", type=int, default=3)
    p.add_argument("--rivalry-window", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "docs" / "results" / "feature_baselines.json")
    p.add_argument(
        "--rich-features", action="store_true",
        help="Use the 12-feature substrate (rivalry counts at 1y/3y/5y/10y, "
             "trade growth, common rivals, common allies, capital distance) "
             "instead of the default 5 features. Output goes to "
             "feature_baselines_rich.json by default.",
    )
    return p.parse_args()


def report_metrics(y_true: np.ndarray, y_score: np.ndarray, years: np.ndarray,
                   n_bootstrap: int) -> dict:
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
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
        "pr_auc": {"point": pr, "lo95": pr_lo, "hi95": pr_hi},
        "recall_at_k": {"point": rec, "lo95": rec_lo, "hi95": rec_hi},
        "lift_at_k": {"point": lift, "lo95": lift_lo, "hi95": lift_hi},
        "brier_positives": brier,
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"\n[{label}] n_pos = {m['n_positives']}, base_rate = {m['base_rate']:.5f}")
    for name in ("pr_auc", "recall_at_k", "lift_at_k"):
        v = m[name]
        print(f"  {name:14s} = {v['point']:.4f}  [95% CI {v['lo95']:.4f} - {v['hi95']:.4f}]")
    print(f"  brier_positives = {m['brier_positives']:.4f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    logger.info("Loading distance / PRD / MID / trade / atop")
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= args.train_start - 2) & (distance.year <= args.test_end)
    ]
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)
    trade = load_cow_trade_edges(years=range(args.train_start - 2, args.test_end + 1))
    atop = load_atop_alliance_edges(years=range(args.train_start - 1, args.test_end + 1))

    logger.info("Building MID-onset labels over PRD universe")
    mid_table = build_mid_onset_edge_table(prd, onsets)
    candidates = mid_table[mid_table.censored == 0][
        ["year", "gwcode_i", "gwcode_j", "edge_present"]
    ].copy()

    if args.rich_features:
        logger.info("Materializing %d candidates with RICH features (12 cols)",
                    len(candidates))
        feats = build_feature_table_rich(
            candidates, distance, trade, onsets, atop,
        )
        feature_cols = tuple(FEATURE_COLS_RICH)
        # Auto-rename output unless user overrode --out
        if args.out.name == "feature_baselines.json":
            args.out = args.out.with_name("feature_baselines_rich.json")
    else:
        logger.info("Materializing %d candidates with default features (5 cols)",
                    len(candidates))
        feats = build_feature_table(
            candidates, distance, trade, onsets,
            rivalry_window=args.rivalry_window,
        )
        feature_cols = tuple(FEATURE_COLS)

    train = feats[(feats.year >= args.train_start) & (feats.year <= args.train_end)]
    val = feats[(feats.year >= args.val_start) & (feats.year <= args.val_end)]
    test = feats[(feats.year >= args.test_start) & (feats.year <= args.test_end)]

    logger.info("train n=%d (pos %d) | val n=%d (pos %d) | test n=%d (pos %d)",
                len(train), int(train.edge_present.sum()),
                len(val), int(val.edge_present.sum()),
                len(test), int(test.edge_present.sum()))

    # --- (F) Hand-crafted feature LR ---
    logger.info("Training feature LR (n_features=%d)", len(feature_cols))
    lr = FeatureLR.fit(train, feature_cols=feature_cols)
    coef = dict(zip(lr.feature_cols, lr.model.coef_[0].tolist()))
    intercept = float(lr.model.intercept_[0])
    logger.info("LR coefficients: %s ; intercept = %.4f", coef, intercept)

    val_y = val.edge_present.to_numpy().astype(int)
    test_y = test.edge_present.to_numpy().astype(int)
    val_yrs = val.year.to_numpy()
    test_yrs = test.year.to_numpy()

    val_lr_score = lr.predict_proba(val)
    test_lr_score = lr.predict_proba(test)

    val_lr = report_metrics(val_y, val_lr_score, val_yrs, args.bootstrap)
    test_lr = report_metrics(test_y, test_lr_score, test_yrs, args.bootstrap)

    # --- (H) Rivalry-only ---
    val_riv_score = rivalry_only_score(val)
    test_riv_score = rivalry_only_score(test)
    val_riv = report_metrics(val_y, val_riv_score, val_yrs, args.bootstrap)
    test_riv = report_metrics(test_y, test_riv_score, test_yrs, args.bootstrap)

    # --- (V) ViEWS-style ensemble (gradient boosting + RF + LR averaged) ---
    logger.info("Training ViEWS-style ensemble (GBM + RF + LR)")
    views = ViEWSEnsemble.fit(train, seed=0, feature_cols=feature_cols)
    val_views_score = views.predict_proba(val)
    test_views_score = views.predict_proba(test)
    val_views = report_metrics(val_y, val_views_score, val_yrs, args.bootstrap)
    test_views = report_metrics(test_y, test_views_score, test_yrs, args.bootstrap)
    # Per-learner test PR-AUC for diagnostic
    per_learner_test = {}
    per_learner_scores = views.predict_proba_per_learner(test)
    for name, scores in per_learner_scores.items():
        per_learner_test[name] = report_metrics(
            test_y, scores, test_yrs, args.bootstrap,
        )

    persistence_floor = 0.01415
    identity_only_floor = 0.1567

    results = {
        "config": vars(args) | {
            "distance": str(args.distance), "out": str(args.out),
        },
        "lr_coefficients": coef,
        "lr_intercept": intercept,
        "F_feature_lr": {"val": val_lr, "test": test_lr},
        "H_rivalry_only": {"val": val_riv, "test": test_riv},
        "V_views_ensemble": {
            "val": val_views, "test": test_views,
            "per_learner_test": per_learner_test,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote results to %s", args.out)

    print()
    print("=" * 76)
    print("Hand-crafted feature baselines -- Conflict onset (hostlev>=", args.hostility_threshold, ")")
    print(f"  rivalry window = {args.rivalry_window} years")
    print(f"  test {args.test_start}-{args.test_end}; n_test = {len(test):,}")
    print("=" * 76)

    print("\n>>> (F) Logistic regression on 5 hand-crafted features")
    print("    LR coefficients:")
    for k, v in coef.items():
        print(f"      {k:18s} = {v:+.4f}")
    print(f"      intercept          = {intercept:+.4f}")
    print_metrics("F-LR / VAL", val_lr)
    print_metrics("F-LR / TEST", test_lr)

    print("\n>>> (H) Rivalry-history-only (no training)")
    print_metrics("H-RIV / VAL", val_riv)
    print_metrics("H-RIV / TEST", test_riv)

    print("\n>>> (V) ViEWS-style ensemble (GBM + RF + LR averaged)")
    print_metrics("V-VIEWS / VAL", val_views)
    print_metrics("V-VIEWS / TEST", test_views)
    print("    Per-learner test PR-AUC (diagnostic):")
    for name in ("gbm", "rf", "lr"):
        v = per_learner_test[name]["pr_auc"]
        print(f"      {name:6s} = {v['point']:.4f} [{v['lo95']:.4f}, {v['hi95']:.4f}]")

    print()
    print("=" * 76)
    print("Comparison vs. previously-locked floors (test PR-AUC):")
    print(f"  Persistence (no model)         : {persistence_floor:.4f}  (lift 1.00)")
    print(f"  Trade-only RGCN                : 0.1414  (lift 12.67)")
    print(f"  Identity-only RGCN ablation    : {identity_only_floor:.4f}  (lift 14.00)")
    print(f"  (H) rivalry-only               : {test_riv['pr_auc']['point']:.4f}  "
          f"(lift {test_riv['lift_at_k']['point']:.2f})")
    print(f"  (F) feature LR                 : {test_lr['pr_auc']['point']:.4f}  "
          f"(lift {test_lr['lift_at_k']['point']:.2f})")
    print(f"  (V) ViEWS ensemble             : {test_views['pr_auc']['point']:.4f}  "
          f"(lift {test_views['lift_at_k']['point']:.2f})")
    print()
    print("  Best identity-free GNN (SiGAT) : 0.3754  (lift 30.34)")
    print("  Best as-published GNN (SDGNN)  : 0.3741  (lift 31.00)")
    print("=" * 76)


if __name__ == "__main__":
    main()
