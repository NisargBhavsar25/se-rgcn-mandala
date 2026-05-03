"""Identity permutation probe for signed-GNN baselines.

The diagnostic question: does the trained model rely on per-node identity
memorization, or on the features and graph structure?

Mechanism: at inference time, randomly permute the embedding lookup. The
model's learnable embedding for node i (e.g., USA) gets routed to a
randomly chosen node position. Features and graph structure are unchanged.
If the model relied on identity-feature association, that association is
now broken and PR-AUC collapses. If the model relied on
features/structure, performance is invariant.

Headline metric: collapse_ratio = (baseline - mean_permuted) / baseline.
Models that don't use identity should yield collapse_ratio ~ 0.
Models that rely entirely on identity should yield collapse_ratio close to
(baseline - base_rate) / baseline.

Implementation note (the subtle one in the project plan): we permute the
embedding LOOKUP, not the features or edges. This is correct because the
question is "is the embedding for state X attached to state X's
features/structure?" -- which is broken when we randomize the lookup.
Permuting features instead would conflate identity reliance with feature
corruption.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.evaluation.metrics import pr_auc


def identity_permutation_probe(
    score_fn: Callable[[torch.Tensor | None], np.ndarray],
    y_true: np.ndarray,
    *,
    n_nodes: int,
    n_permutations: int = 100,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run the identity permutation probe.

    Args:
      score_fn: callable taking an `embed_perm` (LongTensor (n_nodes,)) or None
        and returning a 1-D np.ndarray of test scores in fixed order matching
        y_true. Must be deterministic in eval mode given a fixed permutation.
      y_true: binary test labels in the same order score_fn returns.
      n_nodes: vocabulary size of node embeddings.
      n_permutations: number of random permutations.
      seed: RNG seed.
      device: torch device for the permutation tensor.

    Returns:
      dict with baseline_pr_auc, permuted_pr_auc_mean / std, collapse_ratio.
    """
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-D")

    baseline_scores = score_fn(None)
    baseline = pr_auc(y_true, baseline_scores)

    rng = np.random.default_rng(seed)
    permuted: list[float] = []
    for _ in range(n_permutations):
        perm_np = rng.permutation(n_nodes)
        perm = torch.from_numpy(perm_np).long().to(device)
        scores = score_fn(perm)
        permuted.append(pr_auc(y_true, scores))

    permuted_arr = np.asarray(permuted, dtype=float)
    permuted_arr = permuted_arr[~np.isnan(permuted_arr)]
    if len(permuted_arr) == 0:
        return {
            "baseline_pr_auc": baseline,
            "permuted_pr_auc_mean": float("nan"),
            "permuted_pr_auc_std": float("nan"),
            "collapse_ratio": float("nan"),
            "n_permutations_valid": 0,
        }

    mean = float(np.mean(permuted_arr))
    std = float(np.std(permuted_arr))
    collapse = (baseline - mean) / baseline if baseline > 0 else float("nan")
    return {
        "baseline_pr_auc": baseline,
        "permuted_pr_auc_mean": mean,
        "permuted_pr_auc_std": std,
        "collapse_ratio": collapse,
        "n_permutations_valid": int(len(permuted_arr)),
    }
