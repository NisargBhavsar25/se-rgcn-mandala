"""Smoke tests for KernelLR (the Tier-2 kernel ablation grid model)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from src.models.kernel_lr import KERNEL_TYPES, KernelLR, predict_proba, train_kernel_lr


def _features_df() -> pd.DataFrame:
    """Tiny synthetic feature table matching the F-LR schema."""
    return pd.DataFrame({
        "year":          [2010, 2010, 2010, 2011],
        "gwcode_i":      [   2,  100,  140,    2],
        "gwcode_j":      [ 200,  101,  160,  365],
        "edge_present":  [   1,    0,    1,    0],
        "d_km":          [3000.0, 0.0, 0.0, 8000.0],
        "log_trade":     [10.0,   1.0,  3.0,  6.0],
        "contiguous":    [   0,    1,    1,    0],
        "both_major":    [   1,    0,    0,    1],
        "rivalry_count": [   3,    0,    1,    0],
    })


@pytest.mark.parametrize("kernel_type", KERNEL_TYPES)
def test_forward_pass_shape(kernel_type):
    feats = _features_df()
    model = KernelLR(kernel_type=kernel_type)
    d_km = torch.tensor(feats["d_km"].to_numpy(), dtype=torch.float32)
    out = model(
        d_km / 1000.0, d_km,
        torch.tensor(feats["log_trade"].to_numpy(), dtype=torch.float32),
        torch.tensor(feats["contiguous"].to_numpy(), dtype=torch.float32),
        torch.tensor(feats["both_major"].to_numpy(), dtype=torch.float32),
        torch.tensor(feats["rivalry_count"].to_numpy(), dtype=torch.float32),
    )
    assert out.shape == (4,)


def test_full_kernel_at_zero_distance_is_one():
    """S(0) = cos(0) * exp(0) = 1 -- the max-coupling boundary condition."""
    model = KernelLR(kernel_type="full_kernel")
    d_km = torch.tensor([0.0])
    val = model.kernel_value(d_km / 1000.0, d_km)
    assert torch.allclose(val, torch.tensor([1.0]))


def test_decay_only_at_zero_distance_is_one():
    model = KernelLR(kernel_type="decay_only")
    d_km = torch.tensor([0.0])
    val = model.kernel_value(d_km / 1000.0, d_km)
    assert torch.allclose(val, torch.tensor([1.0]))


def test_periodic_only_at_zero_distance_is_one():
    model = KernelLR(kernel_type="periodic_only")
    d_km = torch.tensor([0.0])
    val = model.kernel_value(d_km / 1000.0, d_km)
    assert torch.allclose(val, torch.tensor([1.0]))


def test_alpha_remains_positive_through_training():
    """alpha = exp(log_alpha) -- by construction always > 0."""
    model = KernelLR(kernel_type="full_kernel")
    feats = _features_df()
    train_kernel_lr(model, feats, cfg=__import__("src.models.kernel_lr",
                                                 fromlist=["TrainConfig"]).TrainConfig(epochs=10, log_every=100))
    assert float(model.alpha.detach()) > 0.0


def test_gradient_flows_to_alpha_and_beta():
    model = KernelLR(kernel_type="full_kernel")
    d_km = torch.tensor([1000.0, 2000.0])
    out = model(
        d_km / 1000.0, d_km,
        torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2),
    )
    out.sum().backward()
    assert model.log_alpha.grad is not None
    assert abs(model.log_alpha.grad.item()) > 0
    assert model.beta.grad is not None
    assert abs(model.beta.grad.item()) > 0


def test_raw_distance_variant_has_no_kernel_params():
    model = KernelLR(kernel_type="raw_distance")
    assert not hasattr(model, "log_alpha")
    assert not hasattr(model, "beta")


def test_decay_only_has_alpha_no_beta():
    model = KernelLR(kernel_type="decay_only")
    assert hasattr(model, "log_alpha")
    assert not hasattr(model, "beta")


def test_periodic_only_has_beta_no_alpha():
    model = KernelLR(kernel_type="periodic_only")
    assert not hasattr(model, "log_alpha")
    assert hasattr(model, "beta")


def test_predict_proba_in_unit_interval():
    model = KernelLR(kernel_type="full_kernel")
    feats = _features_df()
    p = predict_proba(model, feats, torch.device("cpu"))
    assert ((p >= 0) & (p <= 1)).all()


def test_invalid_kernel_type_raises():
    with pytest.raises(ValueError, match="kernel_type"):
        KernelLR(kernel_type="bogus")
