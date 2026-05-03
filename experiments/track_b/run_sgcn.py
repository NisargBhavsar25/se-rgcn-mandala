"""Track B Week 1 -- SGCN end-to-end runner.

Runs SGCN in three configurations on the conflict-onset task:

  AS_PUBLISHED   -- learnable per-node identity emb + per-node features
  IDENTITY_FREE  -- per-node features only (no learnable identity)
  IDENTITY_ONLY  -- learnable per-node identity emb, features zeroed

For each config: train, evaluate on val + test with PR-AUC + lift + bootstrap CIs,
run identity permutation probe. Output: docs/results/track_b/sgcn_<config>.json.

Week 1 gate: SGCN-IDENTITY_ONLY test PR-AUC must land in [0.12, 0.20]. Outside
that range, the adapter or model has a bug; stop and debug before continuing.
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.common import (  # noqa: E402
    BaselineConfig, build_per_node_features, build_run_manifest, build_signed_graph,
)
from src.baselines.sgcn import SGCN  # noqa: E402
from src.data.loaders import load_atop_alliance_edges  # noqa: E402
from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.data.trade import load_cow_trade_edges  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.evaluation.negative_sampling import time_aware_negatives  # noqa: E402
from src.models.feature_baselines import build_feature_table  # noqa: E402
from src.probes.identity_permutation import identity_permutation_probe  # noqa: E402

logger = logging.getLogger(__name__)

DYAD_FEATURE_COLS = ("log_d_km", "log_trade", "contiguous", "both_major", "rivalry_count")


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
    p.add_argument("--config",
                   choices=[c.value for c in BaselineConfig],
                   default=None,
                   help="If set, run only this single config; else all three.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "docs" / "results" / "track_b")
    return p.parse_args()


def build_data_bundle(args) -> dict:
    logger.info("Loading data layers")
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= args.train_start - 1) & (distance.year <= args.test_end)
    ]
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)
    trade = load_cow_trade_edges(years=range(args.train_start - 1, args.test_end + 1))
    atop = load_atop_alliance_edges(years=range(args.train_start - 1, args.test_end + 1))
    atop = atop[["year", "gwcode_i", "gwcode_j", "edge_present"]]

    logger.info("Building MID-onset labels and feature table")
    mid_table = build_mid_onset_edge_table(prd, onsets)
    candidates = mid_table[mid_table.censored == 0][
        ["year", "gwcode_i", "gwcode_j", "edge_present"]
    ].copy()
    feats = build_feature_table(
        candidates, distance, trade, onsets, rivalry_window=args.rivalry_window,
    )

    # Canonical state vocabulary -- union across all years.
    all_states = sorted(set(prd["gwcode_i"]).union(prd["gwcode_j"]))
    gw_to_node = {gw: i for i, gw in enumerate(all_states)}
    n_nodes = len(all_states)
    logger.info("Vocab: %d unique states", n_nodes)

    # Per-year signed graphs and per-node features.
    by_year: dict[int, dict] = {}
    for y in range(args.train_start - 1, args.test_end + 1):
        prd_y = prd[prd.year == y]
        states_y = sorted(set(prd_y.gwcode_i).union(prd_y.gwcode_j))
        if not states_y:
            continue
        sg = build_signed_graph(y, states_y, atop, onsets, gw_to_node=gw_to_node)
        # Per-node features computed for ALL global states (zero for inactive ones).
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
        "feature_table": feats,  # has dyad-level features for every PRD-uncensored row
        "all_states": all_states,
        "gw_to_node": gw_to_node,
        "n_nodes": n_nodes,
    }


def make_dyad_tensors(
    rows: pd.DataFrame, gw_to_node: dict[int, int], device: torch.device,
) -> dict:
    src = torch.tensor(
        [gw_to_node[int(c)] for c in rows["gwcode_i"]], dtype=torch.long, device=device,
    )
    dst = torch.tensor(
        [gw_to_node[int(c)] for c in rows["gwcode_j"]], dtype=torch.long, device=device,
    )
    dyad_x = torch.tensor(
        rows[list(DYAD_FEATURE_COLS)].to_numpy(dtype=np.float32),
        device=device,
    )
    return {"src": src, "dst": dst, "dyad_x": dyad_x}


def make_year_batch(
    target_year: int,
    bundle: dict,
    args: argparse.Namespace,
    *,
    seed: int,
) -> dict | None:
    """Positives + sampled negatives for a single training-target year."""
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
        n_per_positive=args.neg_per_pos,
        year_window=args.year_window,
        seed=seed,
    )
    if len(sampled) == 0:
        return None

    # Pull the full feature rows for the chosen positives + negatives.
    pos_full = pos.copy()
    pos_full["label"] = 1
    neg_keys = sampled[["year", "gwcode_i", "gwcode_j"]]
    neg_full = neg_keys.merge(feats, on=["year", "gwcode_i", "gwcode_j"], how="left")
    neg_full["label"] = 0
    batch = pd.concat([pos_full, neg_full], ignore_index=True)
    return batch


def train(model: SGCN, optim, bundle: dict, args, device) -> None:
    train_targets = list(range(args.train_start + 1, args.train_end + 1))
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = train_targets.copy()
        rng.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for tgt in order:
            graph_year = tgt - 1
            if graph_year not in bundle["by_year"]:
                continue
            batch = make_year_batch(
                tgt, bundle, args,
                seed=args.seed + epoch * 1000 + tgt,
            )
            if batch is None or len(batch) == 0:
                continue
            g = bundle["by_year"][graph_year]
            edge_pos = g["edge_index_pos"].to(device)
            edge_neg = g["edge_index_neg"].to(device)
            x_features = g["per_node_features"].to(device)

            tensors = make_dyad_tensors(batch, bundle["gw_to_node"], device)
            labels = torch.tensor(
                batch["label"].to_numpy(dtype=np.float32), device=device,
            )

            optim.zero_grad()
            logits = model(
                x_features, edge_pos, edge_neg,
                tensors["src"], tensors["dst"], tensors["dyad_x"],
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        logger.info("[%s] epoch %02d  avg_loss=%.4f", model.config.value, epoch, avg)


@torch.no_grad()
def score_eval_split(
    model: SGCN, bundle: dict, args, device,
    *, year_start: int, year_end: int,
    embed_perm_for_year: dict[int, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every uncensored PRD candidate in [year_start, year_end].

    embed_perm_for_year: optional permutation tensor PER YEAR (the same
    permutation applied at every yearly forward pass for one probe iteration).
    """
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
        edge_pos = g["edge_index_pos"].to(device)
        edge_neg = g["edge_index_neg"].to(device)
        x_features = g["per_node_features"].to(device)
        tensors = make_dyad_tensors(rows, bundle["gw_to_node"], device)
        perm = embed_perm_for_year.get(tgt) if embed_perm_for_year else None
        logits = model(
            x_features, edge_pos, edge_neg,
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
        "n_dyad_years": int(len(y_true)),
        "n_positives": n_pos,
        "base_rate": base_rate,
        "pr_auc": {"point": pr, "lo95": pr_lo, "hi95": pr_hi},
        "recall_at_k": {"point": rec, "lo95": rec_lo, "hi95": rec_hi},
        "lift_at_k": {"point": lift, "lo95": lift_lo, "hi95": lift_hi},
        "brier_positives": brier,
    }


def run_one_config(
    config: BaselineConfig, bundle: dict, args, device, manifest: dict,
) -> dict:
    logger.info("=== Running SGCN with config = %s ===", config.value)
    model = SGCN(
        n_nodes=bundle["n_nodes"],
        n_features=5,
        n_dyad_features=len(DYAD_FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        config=config,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    t0 = time.time()
    train(model, optim, bundle, args, device)
    elapsed = time.time() - t0

    val_yt, val_ys, val_yrs = score_eval_split(
        model, bundle, args, device,
        year_start=args.val_start, year_end=args.val_end,
    )
    test_yt, test_ys, test_yrs = score_eval_split(
        model, bundle, args, device,
        year_start=args.test_start, year_end=args.test_end,
    )
    val_m = report_metrics(val_yt, val_ys, val_yrs, args.bootstrap)
    test_m = report_metrics(test_yt, test_ys, test_yrs, args.bootstrap)

    # Identity probe on test set.
    def score_fn_test(perm):
        if perm is None:
            perm_dict = None
        else:
            perm_dict = {y: perm for y in range(args.test_start, args.test_end + 1)}
        _, ys, _ = score_eval_split(
            model, bundle, args, device,
            year_start=args.test_start, year_end=args.test_end,
            embed_perm_for_year=perm_dict,
        )
        return ys

    probe = identity_permutation_probe(
        score_fn_test,
        test_yt,
        n_nodes=bundle["n_nodes"],
        n_permutations=args.probe_permutations,
        seed=args.seed,
        device=device,
    )

    return {
        "config": config.value,
        "training_seconds": elapsed,
        "val": val_m,
        "test": test_m,
        "identity_probe": probe,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    logger.info("device=%s, seed=%d", device, args.seed)

    bundle = build_data_bundle(args)
    manifest = build_run_manifest(args.seed, vars(args) | {
        "distance": str(args.distance), "out_dir": str(args.out_dir),
    })

    if args.config is not None:
        configs = [BaselineConfig(args.config)]
    else:
        configs = list(BaselineConfig)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {"manifest": manifest, "configs": {}}
    for cfg in configs:
        result = run_one_config(cfg, bundle, args, device, manifest)
        all_results["configs"][cfg.value] = result
        # Also write per-config file for incremental progress.
        with (args.out_dir / f"sgcn_{cfg.value}.json").open("w") as f:
            json.dump({"manifest": manifest, **result}, f, indent=2, default=str)

    with (args.out_dir / "sgcn_summary.json").open("w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print()
    print("=" * 78)
    print("Track B Week 1 -- SGCN three-config comparison (conflict onset)")
    print("=" * 78)
    print(f"  test 2006-2014, n_nodes = {bundle['n_nodes']}")
    print()
    print(f"  {'config':<18}{'PR-AUC (test)':<28}{'lift':<10}{'collapse_ratio':<16}")
    print("  " + "-" * 72)
    for cfg in configs:
        r = all_results["configs"][cfg.value]
        v = r["test"]
        p = r["identity_probe"]
        print(f"  {cfg.value:<18}"
              f"{v['pr_auc']['point']:.4f} [{v['pr_auc']['lo95']:.4f}, {v['pr_auc']['hi95']:.4f}]   "
              f"{v['lift_at_k']['point']:>5.2f}     "
              f"{p['collapse_ratio']:+.4f}")
    print()
    print("Identity-only RGCN reference (existing): PR-AUC 0.1567, lift 14.00")
    print("F-LR (sklearn) reference: PR-AUC 0.3501, lift 30.04")
    print()
    print("Week 1 GATE: SGCN-IDENTITY_ONLY test PR-AUC must land in [0.12, 0.20].")
    if "identity_only" in all_results["configs"]:
        v = all_results["configs"]["identity_only"]["test"]["pr_auc"]["point"]
        if 0.12 <= v <= 0.20:
            print(f"  -> PASS (PR-AUC = {v:.4f})")
        else:
            print(f"  -> FAIL (PR-AUC = {v:.4f}). Adapter or model has a bug; STOP.")


if __name__ == "__main__":
    main()
