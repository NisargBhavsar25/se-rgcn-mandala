"""Kernel-augmented logistic regression -- the Mandala stepping-stone.

Tests whether the periodic spatial prior $S(d) = \\cos(\\beta d) \\cdot e^{-\\alpha d}$
adds value over raw distance, holding the rest of the standard quantitative-IR
feature set fixed (log_trade, contiguous, both_major, rivalry_count). All four
proposal Tier-2 variants are supported via `kernel_type`:

  raw_distance   -- log(1 + d_km), no learnable kernel params (= F-LR sanity check)
  decay_only     -- exp(-alpha * d_mm), learnable alpha
  periodic_only  -- cos(beta * d_mm), learnable beta
  full_kernel    -- cos(beta * d_mm) * exp(-alpha * d_mm), learnable (alpha, beta)

Distance is in megameters (Mm = d_km / 1000) for gradient conditioning. The
parameterization alpha = exp(log_alpha) keeps alpha positive; beta is
unconstrained (cos is even, sign-invariant).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.models.feature_baselines import FEATURE_COLS

logger = logging.getLogger(__name__)

KERNEL_TYPES = ("raw_distance", "decay_only", "periodic_only", "full_kernel")


class KernelLR(nn.Module):
    def __init__(
        self,
        kernel_type: str = "full_kernel",
        init_alpha: float = 0.5,
        init_beta: float = math.pi,
    ) -> None:
        super().__init__()
        if kernel_type not in KERNEL_TYPES:
            raise ValueError(f"kernel_type must be one of {KERNEL_TYPES}")
        self.kernel_type = kernel_type
        if kernel_type in ("decay_only", "full_kernel"):
            self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        if kernel_type in ("periodic_only", "full_kernel"):
            self.beta = nn.Parameter(torch.tensor(init_beta))
        # 5 features: [distance_term, log_trade, contiguous, both_major, rivalry_count]
        self.linear = nn.Linear(5, 1)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    def kernel_value(self, d_mm: torch.Tensor, d_km: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == "raw_distance":
            return torch.log1p(d_km)
        if self.kernel_type == "decay_only":
            return torch.exp(-self.alpha * d_mm)
        if self.kernel_type == "periodic_only":
            return torch.cos(self.beta * d_mm)
        # full_kernel
        return torch.cos(self.beta * d_mm) * torch.exp(-self.alpha * d_mm)

    def forward(
        self,
        d_mm: torch.Tensor,
        d_km: torch.Tensor,
        log_trade: torch.Tensor,
        contiguous: torch.Tensor,
        both_major: torch.Tensor,
        rivalry_count: torch.Tensor,
    ) -> torch.Tensor:
        dist = self.kernel_value(d_mm, d_km)
        x = torch.stack(
            [dist, log_trade, contiguous, both_major, rivalry_count], dim=-1
        )
        return self.linear(x).squeeze(-1)

    def kernel_params_summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if hasattr(self, "log_alpha"):
            out["alpha"] = float(self.alpha.detach())
        if hasattr(self, "beta"):
            out["beta"] = float(self.beta.detach())
        return out


@dataclass
class TrainConfig:
    epochs: int = 500
    lr: float = 1e-2
    weight_decay: float = 1e-4
    log_every: int = 50
    seed: int = 0


def _features_to_tensors(
    feats: pd.DataFrame, device: torch.device
) -> dict[str, torch.Tensor]:
    d_km = torch.tensor(feats["d_km"].to_numpy(), dtype=torch.float32, device=device)
    return {
        "d_km": d_km,
        "d_mm": d_km / 1000.0,
        "log_trade": torch.tensor(feats["log_trade"].to_numpy(), dtype=torch.float32, device=device),
        "contiguous": torch.tensor(feats["contiguous"].to_numpy(), dtype=torch.float32, device=device),
        "both_major": torch.tensor(feats["both_major"].to_numpy(), dtype=torch.float32, device=device),
        "rivalry_count": torch.tensor(
            feats["rivalry_count"].to_numpy(), dtype=torch.float32, device=device
        ),
    }


def train_kernel_lr(
    model: KernelLR,
    train_features: pd.DataFrame,
    *,
    cfg: TrainConfig = TrainConfig(),
    device: torch.device = torch.device("cpu"),
) -> KernelLR:
    """Full-batch training with class-balanced BCE."""
    torch.manual_seed(cfg.seed)
    model = model.to(device)
    tensors = _features_to_tensors(train_features, device)
    y = torch.tensor(
        train_features["edge_present"].to_numpy().astype(np.float32),
        dtype=torch.float32, device=device,
    )
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.epochs):
        model.train()
        optim.zero_grad()
        logits = model(
            tensors["d_mm"], tensors["d_km"],
            tensors["log_trade"], tensors["contiguous"],
            tensors["both_major"], tensors["rivalry_count"],
        )
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            params = model.kernel_params_summary()
            param_str = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
            logger.info(
                "[%s] epoch %3d  loss=%.4f  %s",
                model.kernel_type, epoch + 1, float(loss.detach()), param_str,
            )
    return model


@torch.no_grad()
def predict_proba(
    model: KernelLR, features: pd.DataFrame, device: torch.device
) -> np.ndarray:
    model.eval()
    tensors = _features_to_tensors(features, device)
    logits = model(
        tensors["d_mm"], tensors["d_km"],
        tensors["log_trade"], tensors["contiguous"],
        tensors["both_major"], tensors["rivalry_count"],
    )
    return torch.sigmoid(logits).cpu().numpy()


_ = FEATURE_COLS  # imported to assert API contract with feature_baselines.
