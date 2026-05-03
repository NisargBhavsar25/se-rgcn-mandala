"""End-to-end trade-only RGCN baseline on the conflict-onset task.

This is the reviewer's Week 2 second-gate experiment: with only trade
information, can we predict conflict onsets at year t+1 better than the
persistence floor (PR-AUC = base_rate = 0.01415, lift = 1.0)?

Pipeline per year t:
  1. Build trade graph from COW Trade 4.0 at year t.
  2. Look up all PRD dyad-years at year t+1 (uncensored only).
  3. Label them with conflict onsets at year t+1.
  4. Train: positives + uniform time-aware sampled negatives -> BCE loss.
  5. Eval: score every uncensored test-year dyad -> PR-AUC, lift, etc.

Outputs:
  stdout report + JSON at docs/results/trade_only_baseline.json.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.graph_builder import build_yearly_trade_graph, lookup_dyad_trade  # noqa: E402
from src.data.mid import build_mid_onset_edge_table, load_mid_onsets  # noqa: E402
from src.data.prd import prd_dyads_all_years  # noqa: E402
from src.data.trade import load_cow_trade_edges  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    bootstrap_ci, brier_positives, lift_at_k, pr_auc, recall_at_k,
)
from src.evaluation.negative_sampling import time_aware_negatives  # noqa: E402
from src.models.trade_rgcn import TradeOnlyRGCN  # noqa: E402

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
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--neg-per-pos", type=int, default=10)
    p.add_argument("--year-window", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "docs" / "results" / "trade_only_baseline.json")
    p.add_argument(
        "--no-trade", action="store_true",
        help="ABLATION: drop ALL trade signal -- empty trade graph (no message "
             "passing) AND zero raw trade at the MLP head. Tests whether the "
             "trade-only result is dyad-identity memorization or genuine trade "
             "signal. If PR-AUC stays near the with-trade number, the model is "
             "memorizing identities and 'trade-only' is misnamed.",
    )
    return p.parse_args()


def build_full_data(args) -> dict:
    """Load all data layers and assemble the per-year structures the trainer needs."""
    logger.info("Loading distance matrix")
    distance = pd.read_parquet(args.distance)
    distance = distance[
        (distance.year >= args.train_start) & (distance.year <= args.test_end + 1)
    ]

    logger.info("Applying PRD filter")
    prd = prd_dyads_all_years(distance)[["year", "gwcode_i", "gwcode_j"]]

    logger.info("Loading MID onsets")
    onsets = load_mid_onsets(hostility_threshold=args.hostility_threshold)

    logger.info("Building MID-onset edge table over PRD universe")
    mid_table = build_mid_onset_edge_table(prd, onsets)

    logger.info("Loading COW Trade")
    trade = load_cow_trade_edges(
        years=range(args.train_start, args.test_end + 2)
    )

    # Canonical state vocabulary: union of all PRD states across all years.
    all_states = sorted(set(prd["gwcode_i"]).union(prd["gwcode_j"]))
    gw_to_node = {gw: i for i, gw in enumerate(all_states)}
    logger.info("Vocab: %d unique states across all years", len(all_states))

    # Per-year data bundles
    by_year: dict[int, dict] = {}
    for y in range(args.train_start, args.test_end + 1):
        prd_y = prd[prd.year == y]
        states_y = sorted(set(prd_y.gwcode_i).union(prd_y.gwcode_j))
        trade_y = trade[trade.year == y]
        if args.no_trade:
            # Ablation: empty trade graph -> RGCN message passing is a no-op,
            # node embeddings stay at their learned init across all layers.
            global_edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_type_t = torch.empty(0, dtype=torch.long)
        else:
            graph_y = build_yearly_trade_graph(trade_y, states_y)
            global_src = torch.tensor(
                [gw_to_node[graph_y.node_to_gwcode[int(s)]] for s in graph_y.edge_index[0].tolist()],
                dtype=torch.long,
            )
            global_dst = torch.tensor(
                [gw_to_node[graph_y.node_to_gwcode[int(d)]] for d in graph_y.edge_index[1].tolist()],
                dtype=torch.long,
            )
            global_edge_index = torch.stack([global_src, global_dst])
            edge_type_t = graph_y.edge_type

        by_year[y] = {
            "edge_index": global_edge_index,
            "edge_type": edge_type_t,
            "trade_year": trade_y,
            "prd_pairs": prd_y,
        }

    # Per-target-year labels: at year t+1 the model predicts onset.
    # mid_table has [year, gwcode_i, gwcode_j, edge_present, censored]
    label_table = mid_table.copy()

    return {
        "by_year": by_year,
        "label_table": label_table,
        "trade_full": trade,
        "all_states": all_states,
        "gw_to_node": gw_to_node,
    }


def make_year_batch(
    target_year: int,
    bundle: dict,
    *,
    n_neg_per_pos: int,
    year_window: int,
    seed: int,
    no_trade: bool = False,
) -> dict | None:
    """Build a (positives + sampled negatives) training batch for target_year."""
    label_table = bundle["label_table"]
    gw_to_node = bundle["gw_to_node"]
    trade_full = bundle["trade_full"]

    targets = label_table[
        (label_table.year == target_year) & (label_table.censored == 0)
    ]
    pos = targets[targets.edge_present == 1]
    neg_pool = targets[targets.edge_present == 0]

    if len(pos) == 0 or len(neg_pool) == 0:
        return None

    # Time-aware negatives -- but the candidates pool we hand the sampler
    # spans neg years near each positive. Build the candidates from
    # label_table years t +/- window with edge_present=0 and censored=0.
    candidate_years = label_table[
        (label_table.year >= target_year - year_window)
        & (label_table.year <= target_year + year_window)
        & (label_table.censored == 0)
    ]
    sampled_neg = time_aware_negatives(
        positives=pos[["year", "gwcode_i", "gwcode_j"]],
        candidates=candidate_years[
            ["year", "gwcode_i", "gwcode_j", "edge_present"]
        ],
        n_per_positive=n_neg_per_pos,
        year_window=year_window,
        seed=seed,
    )
    if len(sampled_neg) == 0:
        return None

    # Map every positive AND every negative to its local PRD year and look
    # up trade volume from the SAME year (since we predict for year t+1
    # using the GRAPH at year t -> trade volume should be from year t too).
    # We use trade volume FROM target_year - 1 because the prediction is
    # "given year t info, what happens at year t+1". Each row's year here
    # IS the prediction year (= target_year for positives, +/- window for
    # sampled negs). We need trade from row.year - 1.
    rows = []
    for _, r in pos.iterrows():
        rows.append({"year": int(r.year), "gwcode_i": int(r.gwcode_i),
                     "gwcode_j": int(r.gwcode_j), "label": 1})
    for _, r in sampled_neg.iterrows():
        rows.append({"year": int(r.year), "gwcode_i": int(r.gwcode_i),
                     "gwcode_j": int(r.gwcode_j), "label": 0})
    batch = pd.DataFrame(rows)

    # Vectorized trade lookup: one bulk pass per unique graph-year.
    # Under --no-trade ablation, skip lookup and feed zeros to the MLP head.
    batch_graph_years = batch["year"].to_numpy() - 1
    log_trade_per_row = np.zeros(len(batch), dtype=np.float32)
    if not no_trade:
        for graph_year in np.unique(batch_graph_years):
            if int(graph_year) not in bundle["by_year"]:
                continue
            mask = batch_graph_years == graph_year
            sub = batch.loc[mask, ["gwcode_i", "gwcode_j"]]
            trade_y = bundle["by_year"][int(graph_year)]["trade_year"]
            log_trade_per_row[mask] = lookup_dyad_trade(trade_y, sub)

    src_idx = torch.tensor(
        [gw_to_node[c] for c in batch["gwcode_i"]], dtype=torch.long
    )
    dst_idx = torch.tensor(
        [gw_to_node[c] for c in batch["gwcode_j"]], dtype=torch.long
    )

    return {
        "src_idx": src_idx,
        "dst_idx": dst_idx,
        "log_trade": torch.tensor(log_trade_per_row, dtype=torch.float32),
        "labels": torch.tensor(batch["label"].to_numpy(), dtype=torch.float32),
        "row_years": batch["year"].to_numpy(),
    }


def train(model, optim, bundle, args, device) -> None:
    train_targets = list(range(args.train_start + 1, args.train_end + 1))
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = train_targets.copy()
        rng.shuffle(order)
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for tgt in order:
            batch = make_year_batch(
                tgt, bundle,
                n_neg_per_pos=args.neg_per_pos,
                year_window=args.year_window,
                seed=args.seed + epoch * 1000 + tgt,
                no_trade=args.no_trade,
            )
            if batch is None:
                continue
            graph_year = tgt - 1
            if graph_year not in bundle["by_year"]:
                continue
            g = bundle["by_year"][graph_year]
            edge_index = g["edge_index"].to(device)
            edge_type = g["edge_type"].to(device)
            optim.zero_grad()
            logits = model(
                edge_index, edge_type,
                batch["src_idx"].to(device),
                batch["dst_idx"].to(device),
                batch["log_trade"].to(device),
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, batch["labels"].to(device)
            )
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        avg = epoch_loss / max(n_batches, 1)
        logger.info("epoch %02d  avg_loss=%.4f  (%d batches)", epoch, avg, n_batches)


@torch.no_grad()
def evaluate(
    model, bundle, args, device,
    *, year_start: int, year_end: int, no_trade: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every uncensored PRD dyad in [year_start, year_end] (test years).

    Returns (y_true, y_score, years).
    """
    label_table = bundle["label_table"]
    gw_to_node = bundle["gw_to_node"]

    model.eval()
    all_y_true: list[int] = []
    all_y_score: list[float] = []
    all_years: list[int] = []
    for tgt in range(year_start, year_end + 1):
        graph_year = tgt - 1
        if graph_year not in bundle["by_year"]:
            continue
        g = bundle["by_year"][graph_year]
        edge_index = g["edge_index"].to(device)
        edge_type = g["edge_type"].to(device)
        h = model.encode(edge_index, edge_type)

        candidates = label_table[
            (label_table.year == tgt) & (label_table.censored == 0)
        ]
        if len(candidates) == 0:
            continue
        # Look up trade at the GRAPH year (tgt - 1); zeros under ablation.
        if no_trade:
            log_trade = np.zeros(len(candidates), dtype=np.float32)
        else:
            trade_y = bundle["by_year"][graph_year]["trade_year"]
            log_trade = lookup_dyad_trade(
                trade_y,
                candidates[["gwcode_i", "gwcode_j"]],
            )
        src_idx = torch.tensor(
            [gw_to_node[int(c)] for c in candidates["gwcode_i"]],
            dtype=torch.long, device=device,
        )
        dst_idx = torch.tensor(
            [gw_to_node[int(c)] for c in candidates["gwcode_j"]],
            dtype=torch.long, device=device,
        )
        log_trade_t = torch.tensor(log_trade, dtype=torch.float32, device=device)
        logits = model.score_dyads(h, src_idx, dst_idx, log_trade_t)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_y_true.extend(candidates["edge_present"].tolist())
        all_y_score.extend(scores.tolist())
        all_years.extend([tgt] * len(candidates))

    return (
        np.asarray(all_y_true, dtype=int),
        np.asarray(all_y_score, dtype=float),
        np.asarray(all_years, dtype=int),
    )


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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.no_trade and args.out.name == "trade_only_baseline.json":
        args.out = args.out.with_name("identity_only_ablation.json")

    bundle = build_full_data(args)
    n_nodes = len(bundle["all_states"])

    device = torch.device(args.device)
    logger.info("device=%s, n_nodes=%d", device, n_nodes)

    model = TradeOnlyRGCN(
        n_nodes=n_nodes,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    t0 = time.time()
    train(model, optim, bundle, args, device)
    train_secs = time.time() - t0

    logger.info("Evaluating on val %d-%d", args.val_start, args.val_end)
    val_yt, val_ys, val_yrs = evaluate(
        model, bundle, args, device,
        year_start=args.val_start, year_end=args.val_end,
        no_trade=args.no_trade,
    )
    val_metrics = report_metrics(val_yt, val_ys, val_yrs, args.bootstrap)

    logger.info("Evaluating on test %d-%d", args.test_start, args.test_end)
    test_yt, test_ys, test_yrs = evaluate(
        model, bundle, args, device,
        year_start=args.test_start, year_end=args.test_end,
        no_trade=args.no_trade,
    )
    test_metrics = report_metrics(test_yt, test_ys, test_yrs, args.bootstrap)

    results = {
        "config": vars(args) | {"distance": str(args.distance), "out": str(args.out),
                                "device": str(args.device)},
        "n_states_in_vocab": n_nodes,
        "training_seconds": train_secs,
        "val": val_metrics,
        "test": test_metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Wrote results to %s", args.out)

    # Persistence floor for comparison
    persistence_floor = 0.01415

    print()
    print("=" * 76)
    label = "ABLATION: identity-only (no trade)" if args.no_trade else "Trade-only RGCN baseline"
    print(f"{label} -- Conflict onset (hostlev>= {args.hostility_threshold})")
    print(f"  vocab = {n_nodes} states; train {args.train_start}-{args.train_end};")
    print(f"  val {args.val_start}-{args.val_end}; test {args.test_start}-{args.test_end}")
    print(f"  trained in {train_secs:.1f}s on {args.device}")
    print("=" * 76)
    for label, m in (("VAL", val_metrics), ("TEST", test_metrics)):
        print(f"\n[{label}] n_pos = {m['n_positives']}, base_rate = {m['base_rate']:.5f}")
        for metric_name in ("pr_auc", "recall_at_k", "lift_at_k"):
            v = m[metric_name]
            print(f"  {metric_name:14s} = {v['point']:.4f}  "
                  f"[95% CI {v['lo95']:.4f} - {v['hi95']:.4f}]")
        print(f"  brier_positives = {m['brier_positives']:.4f}")

    print()
    print("Reviewer Week-2 gate comparison:")
    print(f"  Persistence PR-AUC floor:  {persistence_floor:.4f}  "
          f"(= base_rate, lift = 1.0)")
    print(f"  Trade-only PR-AUC (test):  {test_metrics['pr_auc']['point']:.4f}  "
          f"(lift = {test_metrics['lift_at_k']['point']:.2f})")
    delta_pct = (test_metrics["pr_auc"]["point"] - persistence_floor) / persistence_floor * 100
    print(f"  Delta over persistence:    {delta_pct:+.1f}%")
    print()


if __name__ == "__main__":
    main()
