"""Track B Week 2 -- unified runner for all four signed-GNN baselines.

Runs SGCN, SiGAT, SDGNN, SignedTransformer on the conflict-onset task,
each in three configurations (AS_PUBLISHED, IDENTITY_FREE, IDENTITY_ONLY)
with the identity permutation probe applied to each cell.

Headline output: docs/results/track_b/all_baselines_summary.json with the
4 x 3 = 12 cells, plus F-LR (0.3501) as the reference row -- the 13-row
table that anchors the paper.

Drops alliance formation (n=96 events makes three-config CIs too wide to
discriminate). Conflict onset is the only task with sufficient power for
the three-config diagnostic.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import (  # noqa: E402
    BaselineConfig, build_per_node_features, build_run_manifest, build_signed_graph,
)
from src.baselines.sdgnn import SDGNN  # noqa: E402
from src.baselines.sgcn import SGCN  # noqa: E402
from src.baselines.sgformer import SignedTransformer  # noqa: E402
from src.baselines.sigat import SiGAT  # noqa: E402
from src.data.loaders import load_atop_alliance_edges  # noqa: E402
from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.data.trade import load_cow_trade_edges  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.evaluation.negative_sampling import time_aware_negatives  # noqa: E402
from src.models.feature_baselines import (  # noqa: E402
    FEATURE_COLS, FEATURE_COLS_RICH,
    build_feature_table, build_feature_table_rich,
)
from src.probes.identity_permutation import identity_permutation_probe  # noqa: E402

logger = logging.getLogger(__name__)

BASELINE_REGISTRY: dict[str, Type] = {
    "sgcn": SGCN,
    "sigat": SiGAT,
    "sdgnn": SDGNN,
    "sgformer": SignedTransformer,
}


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
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--neg-per-pos", type=int, default=10)
    p.add_argument("--year-window", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--probe-permutations", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baselines", nargs="+",
                   choices=list(BASELINE_REGISTRY.keys()) + ["all"],
                   default=["all"])
    p.add_argument("--configs", nargs="+",
                   choices=[c.value for c in BaselineConfig],
                   default=[c.value for c in BaselineConfig])
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "docs" / "results" / "track_b")
    p.add_argument("--early-stop", action="store_true",
                   help="Track val PR-AUC every epoch and restore best-val "
                        "checkpoint at end of training. Off by default so "
                        "default results are comparable to fixed-epoch Week 2 "
                        "runs.")
    p.add_argument("--track-val", action="store_true",
                   help="Track val PR-AUC trajectory across epochs (saved to "
                        "JSON). Implied by --early-stop.")
    p.add_argument("--rich-features", action="store_true",
                   help="Use the 12-feature rich substrate (rivalry counts at "
                        "1y/3y/5y/10y, trade growth, common rivals, common "
                        "allies, capital distance) instead of the default 5 "
                        "features. Same per-node features in either case.")
    return p.parse_args()


def build_data_bundle(args) -> dict:
    """Same data prep as run_sgcn.py -- centralized for all baselines."""
    logger.info("Loading data layers")
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= args.train_start - 1) & (distance.year <= args.test_end)
    ]
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)
    trade = load_cow_trade_edges(years=range(args.train_start - 2, args.test_end + 1))
    atop_full = load_atop_alliance_edges(years=range(args.train_start - 1, args.test_end + 1))
    atop = atop_full[["year", "gwcode_i", "gwcode_j", "edge_present"]]

    mid_table = build_mid_onset_edge_table(prd, onsets)
    candidates = mid_table[mid_table.censored == 0][
        ["year", "gwcode_i", "gwcode_j", "edge_present"]
    ].copy()
    if args.rich_features:
        logger.info("Building RICH 12-feature dyad table")
        feats = build_feature_table_rich(
            candidates, distance, trade, onsets, atop_full,
        )
        dyad_feature_cols = tuple(FEATURE_COLS_RICH)
    else:
        feats = build_feature_table(
            candidates, distance, trade, onsets, rivalry_window=args.rivalry_window,
        )
        dyad_feature_cols = tuple(FEATURE_COLS)

    all_states = sorted(set(prd["gwcode_i"]).union(prd["gwcode_j"]))
    gw_to_node = {gw: i for i, gw in enumerate(all_states)}
    n_nodes = len(all_states)
    logger.info("Vocab: %d unique states", n_nodes)

    by_year: dict[int, dict] = {}
    for y in range(args.train_start - 1, args.test_end + 1):
        prd_y = prd[prd.year == y]
        states_y = sorted(set(prd_y.gwcode_i).union(prd_y.gwcode_j))
        if not states_y:
            continue
        sg = build_signed_graph(y, states_y, atop, onsets, gw_to_node=gw_to_node)
        pn = np.zeros((n_nodes, 5), dtype=np.float32)
        active_pn = build_per_node_features(
            y, states_y, distance, trade, atop, onsets,
            rivalry_window=args.rivalry_window,
        )
        for local_i, gw in enumerate(sorted(set(states_y))):
            pn[gw_to_node[gw]] = active_pn[local_i]
        by_year[y] = {
            "edge_index_pos": sg.edge_index_pos,
            "edge_index_neg": sg.edge_index_neg,
            "per_node_features": torch.from_numpy(pn),
        }

    return {
        "by_year": by_year,
        "feature_table": feats,
        "all_states": all_states,
        "gw_to_node": gw_to_node,
        "n_nodes": n_nodes,
        "dyad_feature_cols": dyad_feature_cols,
    }


def make_dyad_tensors(rows, gw_to_node, dyad_feature_cols, device):
    src = torch.tensor([gw_to_node[int(c)] for c in rows["gwcode_i"]],
                       dtype=torch.long, device=device)
    dst = torch.tensor([gw_to_node[int(c)] for c in rows["gwcode_j"]],
                       dtype=torch.long, device=device)
    dyad_x = torch.tensor(
        rows[list(dyad_feature_cols)].to_numpy(dtype=np.float32), device=device,
    )
    return {"src": src, "dst": dst, "dyad_x": dyad_x}


def make_year_batch(target_year, bundle, args, *, seed):
    feats = bundle["feature_table"]
    candidates = feats[
        (feats.year >= target_year - args.year_window)
        & (feats.year <= target_year + args.year_window)
    ]
    pos = feats[(feats.year == target_year) & (feats.edge_present == 1)]
    if len(pos) == 0:
        return None
    sampled = time_aware_negatives(
        positives=pos[["year", "gwcode_i", "gwcode_j"]],
        candidates=candidates[["year", "gwcode_i", "gwcode_j", "edge_present"]],
        n_per_positive=args.neg_per_pos, year_window=args.year_window, seed=seed,
    )
    if len(sampled) == 0:
        return None
    pos_full = pos.copy()
    pos_full["label"] = 1
    neg_keys = sampled[["year", "gwcode_i", "gwcode_j"]]
    neg_full = neg_keys.merge(feats, on=["year", "gwcode_i", "gwcode_j"], how="left")
    neg_full["label"] = 0
    return pd.concat([pos_full, neg_full], ignore_index=True)


def train(model, optim, bundle, args, device):
    """Returns the val-PR-AUC trajectory (list of dicts) if tracking is on."""
    train_targets = list(range(args.train_start + 1, args.train_end + 1))
    rng = np.random.default_rng(args.seed)

    track_val = args.track_val or args.early_stop
    val_traj: list[dict] = []
    best_val_pr = -float("inf")
    best_state = None

    for epoch in range(args.epochs):
        order = train_targets.copy()
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        model.train()
        for tgt in order:
            graph_year = tgt - 1
            if graph_year not in bundle["by_year"]:
                continue
            batch = make_year_batch(tgt, bundle, args, seed=args.seed + epoch * 1000 + tgt)
            if batch is None or len(batch) == 0:
                continue
            g = bundle["by_year"][graph_year]
            tensors = make_dyad_tensors(batch, bundle["gw_to_node"], bundle["dyad_feature_cols"], device)
            labels = torch.tensor(
                batch["label"].to_numpy(dtype=np.float32), device=device,
            )
            optim.zero_grad()
            logits = model(
                g["per_node_features"].to(device),
                g["edge_index_pos"].to(device),
                g["edge_index_neg"].to(device),
                tensors["src"], tensors["dst"], tensors["dyad_x"],
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)

        # Optional val-PR-AUC tracking. Cheap but not free; only run when asked.
        val_pr_this_epoch = None
        if track_val:
            val_yt, val_ys, _ = score_split(
                model, bundle, args, device,
                year_start=args.val_start, year_end=args.val_end,
            )
            val_pr_this_epoch = float(pr_auc(val_yt, val_ys))
            val_traj.append({
                "epoch": epoch, "avg_train_loss": avg_loss,
                "val_pr_auc": val_pr_this_epoch,
            })
            if args.early_stop and val_pr_this_epoch > best_val_pr:
                best_val_pr = val_pr_this_epoch
                best_state = copy.deepcopy(model.state_dict())

        logger.info(
            "[%s/%s] epoch %02d  avg_loss=%.4f  val_pr=%s",
            type(model).__name__, model.config.value, epoch, avg_loss,
            f"{val_pr_this_epoch:.4f}" if val_pr_this_epoch is not None else "n/a",
        )

    if args.early_stop and best_state is not None:
        logger.info(
            "[%s/%s] early-stop: restoring best-val checkpoint (val_pr=%.4f)",
            type(model).__name__, model.config.value, best_val_pr,
        )
        model.load_state_dict(best_state)

    return val_traj


@torch.no_grad()
def score_split(model, bundle, args, device, *, year_start, year_end,
                embed_perm_for_year=None):
    model.eval()
    feats = bundle["feature_table"]
    test = feats[(feats.year >= year_start) & (feats.year <= year_end)]
    all_y_true: list[int] = []
    all_y_score: list[float] = []
    all_years: list[int] = []
    for tgt in range(year_start, year_end + 1):
        graph_year = tgt - 1
        if graph_year not in bundle["by_year"]:
            continue
        rows = test[test.year == tgt]
        if len(rows) == 0:
            continue
        g = bundle["by_year"][graph_year]
        tensors = make_dyad_tensors(rows, bundle["gw_to_node"], bundle["dyad_feature_cols"], device)
        perm = embed_perm_for_year.get(tgt) if embed_perm_for_year else None
        logits = model(
            g["per_node_features"].to(device),
            g["edge_index_pos"].to(device),
            g["edge_index_neg"].to(device),
            tensors["src"], tensors["dst"], tensors["dyad_x"],
            embed_perm=perm,
        )
        scores = torch.sigmoid(logits).cpu().numpy()
        all_y_true.extend(rows["edge_present"].tolist())
        all_y_score.extend(scores.tolist())
        all_years.extend([tgt] * len(rows))
    return (
        np.asarray(all_y_true, dtype=int),
        np.asarray(all_y_score, dtype=float),
        np.asarray(all_years, dtype=int),
    )


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
        "n_dyad_years": int(len(y_true)), "n_positives": n_pos, "base_rate": base_rate,
        "pr_auc": {"point": pr, "lo95": pr_lo, "hi95": pr_hi},
        "recall_at_k": {"point": rec, "lo95": rec_lo, "hi95": rec_hi},
        "lift_at_k": {"point": lift, "lo95": lift_lo, "hi95": lift_hi},
        "brier_positives": brier,
    }


def run_one_cell(baseline_name, baseline_cls, config, bundle, args, device, manifest):
    logger.info("=== %s / %s ===", baseline_name, config.value)
    model = baseline_cls(
        n_nodes=bundle["n_nodes"], n_features=5,
        n_dyad_features=len(bundle["dyad_feature_cols"]),
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, config=config, dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    t0 = time.time()
    val_traj = train(model, optim, bundle, args, device)
    elapsed = time.time() - t0

    val_yt, val_ys, val_yrs = score_split(
        model, bundle, args, device, year_start=args.val_start, year_end=args.val_end,
    )
    test_yt, test_ys, test_yrs = score_split(
        model, bundle, args, device, year_start=args.test_start, year_end=args.test_end,
    )
    val_m = report_metrics(val_yt, val_ys, val_yrs, args.bootstrap)
    test_m = report_metrics(test_yt, test_ys, test_yrs, args.bootstrap)

    def score_fn_test(perm):
        if perm is None:
            perm_dict = None
        else:
            perm_dict = {y: perm for y in range(args.test_start, args.test_end + 1)}
        _, ys, _ = score_split(
            model, bundle, args, device, year_start=args.test_start, year_end=args.test_end,
            embed_perm_for_year=perm_dict,
        )
        return ys

    probe = identity_permutation_probe(
        score_fn_test, test_yt, n_nodes=bundle["n_nodes"],
        n_permutations=args.probe_permutations, seed=args.seed, device=device,
    )

    return {
        "baseline": baseline_name, "config": config.value,
        "training_seconds": elapsed,
        "val": val_m, "test": test_m, "identity_probe": probe,
        "val_trajectory": val_traj,  # empty list unless --track-val or --early-stop
        "early_stop_used": bool(args.early_stop),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    bundle = build_data_bundle(args)
    manifest = build_run_manifest(args.seed, vars(args) | {
        "distance": str(args.distance), "out_dir": str(args.out_dir),
    })

    baselines = list(BASELINE_REGISTRY.keys()) if "all" in args.baselines else args.baselines
    configs = [BaselineConfig(c) for c in args.configs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"manifest": manifest, "cells": {}}
    for bname in baselines:
        for cfg in configs:
            cell = run_one_cell(bname, BASELINE_REGISTRY[bname], cfg, bundle, args, device, manifest)
            key = f"{bname}_{cfg.value}"
            summary["cells"][key] = cell
            with (args.out_dir / f"{key}.json").open("w") as f:
                json.dump({"manifest": manifest, **cell}, f, indent=2, default=str)

    with (args.out_dir / "all_baselines_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Headline 13-row table
    print()
    print("=" * 88)
    print("Track B Week 2 -- Headline 4-baseline x 3-config table (conflict onset)")
    print("=" * 88)
    print()
    print(f"  {'baseline':<12}{'config':<16}{'PR-AUC (test)':<28}{'lift':<8}{'collapse':<10}")
    print("  " + "-" * 78)
    for bname in baselines:
        for cfg in configs:
            r = summary["cells"][f"{bname}_{cfg.value}"]
            v = r["test"]; p = r["identity_probe"]
            print(f"  {bname:<12}{cfg.value:<16}"
                  f"{v['pr_auc']['point']:.4f} [{v['pr_auc']['lo95']:.4f}, {v['pr_auc']['hi95']:.4f}]   "
                  f"{v['lift_at_k']['point']:>5.2f}    "
                  f"{p['collapse_ratio']:+.4f}")
    print()
    print("References (existing):")
    print("  Persistence floor    : PR-AUC = 0.0142, lift = 1.00")
    print("  F-LR (sklearn)       : PR-AUC = 0.3501, lift = 30.04")
    print()


if __name__ == "__main__":
    main()
