"""Tier-2 kernel ablation grid via kernel-augmented logistic regression.

Trains four variants on the same five-feature substrate (with distance
swapped for each):

  raw_distance   -- log(1 + d_km)                  (= F-LR sanity check)
  decay_only     -- exp(-alpha * d_mm)             (learnable alpha)
  periodic_only  -- cos(beta * d_mm)               (learnable beta)
  full_kernel    -- cos(beta*d) * exp(-alpha*d)    (learnable alpha, beta)

The test of the Mandala hypothesis is whether full_kernel beats raw_distance
and decay_only by a meaningful margin. The decay-only vs full-kernel
contrast isolates the value of periodic structure specifically.

Output: stdout report + docs/results/kernel_lr.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.data.trade import load_cow_trade_edges  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.models.feature_baselines import build_feature_table  # noqa: E402
from src.models.kernel_lr import (  # noqa: E402
    KERNEL_TYPES, KernelLR, TrainConfig, predict_proba, train_kernel_lr,
)

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
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "docs" / "results" / "kernel_lr.json")
    return p.parse_args()


def report_metrics(y_true, y_score, years, n_bootstrap):
    n_pos = int(y_true.sum())
    base_rate = float(y_true.mean()) if len(y_true) else float("nan")
    k = max(n_pos, 1)
    pr, pr_lo, pr_hi = bootstrap_ci(
        y_true, y_score, pr_auc, years=years, block_by_year=True, n_bootstrap=n_bootstrap,
    )
    rec, rec_lo, rec_hi = bootstrap_ci(
        y_true, y_score, recall_at_k, years=years, block_by_year=True, n_bootstrap=n_bootstrap,
        metric_kwargs={"k": k},
    )
    lift, lift_lo, lift_hi = bootstrap_ci(
        y_true, y_score, lift_at_k, years=years, block_by_year=True, n_bootstrap=n_bootstrap,
        metric_kwargs={"k": k},
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


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    logger.info("Loading data and building features")
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= args.train_start - 1) & (distance.year <= args.test_end)
    ]
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)
    trade = load_cow_trade_edges(years=range(args.train_start - 1, args.test_end + 1))

    mid_table = build_mid_onset_edge_table(prd, onsets)
    candidates = mid_table[mid_table.censored == 0][
        ["year", "gwcode_i", "gwcode_j", "edge_present"]
    ].copy()
    feats = build_feature_table(
        candidates, distance, trade, onsets, rivalry_window=args.rivalry_window,
    )

    train = feats[(feats.year >= args.train_start) & (feats.year <= args.train_end)]
    val = feats[(feats.year >= args.val_start) & (feats.year <= args.val_end)]
    test = feats[(feats.year >= args.test_start) & (feats.year <= args.test_end)]
    logger.info("train n=%d (pos %d) | val n=%d (pos %d) | test n=%d (pos %d)",
                len(train), int(train.edge_present.sum()),
                len(val), int(val.edge_present.sum()),
                len(test), int(test.edge_present.sum()))

    val_yrs = val.year.to_numpy()
    test_yrs = test.year.to_numpy()
    val_y = val.edge_present.to_numpy().astype(int)
    test_y = test.edge_present.to_numpy().astype(int)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr,
                      weight_decay=args.weight_decay, seed=args.seed)

    results: dict = {"variants": {}}
    for kernel_type in KERNEL_TYPES:
        logger.info("=== Training %s ===", kernel_type)
        model = KernelLR(kernel_type=kernel_type)
        t0 = time.time()
        model = train_kernel_lr(model, train, cfg=cfg, device=device)
        elapsed = time.time() - t0

        val_score = predict_proba(model, val, device)
        test_score = predict_proba(model, test, device)
        val_metrics = report_metrics(val_y, val_score, val_yrs, args.bootstrap)
        test_metrics = report_metrics(test_y, test_score, test_yrs, args.bootstrap)

        results["variants"][kernel_type] = {
            "training_seconds": elapsed,
            "kernel_params_final": model.kernel_params_summary(),
            "linear_weights": model.linear.weight.detach().cpu().numpy().squeeze().tolist(),
            "linear_bias": float(model.linear.bias.detach().cpu().numpy().squeeze()),
            "val": val_metrics,
            "test": test_metrics,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results | {"config": vars(args) | {
            "distance": str(args.distance), "out": str(args.out),
            "device": str(args.device),
        }}, f, indent=2, default=str)
    logger.info("Wrote results to %s", args.out)

    # Console comparison
    print()
    print("=" * 78)
    print("Tier-2 Kernel Ablation Grid -- KernelLR on conflict onset")
    print(f"  test {args.test_start}-{args.test_end}, n_test={len(test):,}, n_pos={int(test.edge_present.sum())}")
    print("=" * 78)
    print()
    print(f"  {'variant':<16}{'PR-AUC (test)':<18}{'lift':<10}{'alpha':<10}{'beta':<10}")
    print("  " + "-" * 64)
    for kt in KERNEL_TYPES:
        r = results["variants"][kt]
        v = r["test"]
        params = r["kernel_params_final"]
        a = f"{params.get('alpha', float('nan')):.3f}" if "alpha" in params else "  -  "
        b = f"{params.get('beta', float('nan')):.3f}" if "beta" in params else "  -  "
        print(f"  {kt:<16}"
              f"{v['pr_auc']['point']:.4f} [{v['pr_auc']['lo95']:.4f}, {v['pr_auc']['hi95']:.4f}]   "
              f"{v['lift_at_k']['point']:>5.2f}     {a:<10}{b:<10}")

    print()
    print("Comparison vs. previously-locked floors (test PR-AUC):")
    print(f"  Persistence (no model)       : 0.0142  (lift  1.00)")
    print(f"  Trade-only RGCN              : 0.1414  (lift 12.67)")
    print(f"  Identity-only RGCN ablation  : 0.1567  (lift 14.00)")
    print(f"  (H) rivalry-only             : 0.2929  (lift 29.04)")
    print(f"  (F) feature LR (sklearn)     : 0.3501  (lift 30.04)")
    raw = results["variants"]["raw_distance"]["test"]["pr_auc"]["point"]
    full = results["variants"]["full_kernel"]["test"]["pr_auc"]["point"]
    decay = results["variants"]["decay_only"]["test"]["pr_auc"]["point"]
    periodic = results["variants"]["periodic_only"]["test"]["pr_auc"]["point"]
    print(f"  KernelLR raw_distance        : {raw:.4f}   (sanity-check vs F-LR sklearn)")
    print(f"  KernelLR decay_only          : {decay:.4f}")
    print(f"  KernelLR periodic_only       : {periodic:.4f}")
    print(f"  KernelLR full_kernel         : {full:.4f}")
    print()
    if full > raw * 1.10:
        print(f"  -> full_kernel beats raw_distance by {(full - raw) / raw * 100:.1f}%."
              f" Mandala hypothesis SUPPORTED on this substrate.")
    elif full > raw:
        print(f"  -> full_kernel beats raw_distance by {(full - raw) / raw * 100:.1f}%"
              f" (sub-10% -- ambiguous, would not survive a hard reviewer).")
    else:
        print(f"  -> full_kernel does NOT beat raw_distance ({full:.4f} vs {raw:.4f})."
              f" Mandala hypothesis NOT supported on this substrate."
              f" That is a clean negative result and a publishable finding.")


if __name__ == "__main__":
    main()
