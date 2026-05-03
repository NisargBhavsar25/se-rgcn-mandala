"""SGCN smoke tests across the three configurations."""

from __future__ import annotations

import pytest
import torch

from src.baselines.common import BaselineConfig
from src.baselines.sgcn import SGCN, SignedDeepLayer, SignedFirstLayer, _aggregate_neighbors


# ---------- low-level helpers ------------------------------------------------

def test_aggregate_neighbors_empty_edges_returns_zeros():
    x = torch.randn(5, 4)
    out = _aggregate_neighbors(x, torch.empty(2, 0, dtype=torch.long))
    assert torch.allclose(out, torch.zeros_like(x))


def test_aggregate_neighbors_mean_of_pointed_at_neighbors():
    x = torch.tensor([[1.0], [2.0], [10.0]])
    # Edges: 0->2, 1->2 (so node 2 receives mean(x[0], x[1]) = 1.5)
    ei = torch.tensor([[0, 1], [2, 2]])
    out = _aggregate_neighbors(x, ei)
    assert float(out[2]) == pytest.approx(1.5)
    assert float(out[0]) == 0.0
    assert float(out[1]) == 0.0


# ---------- layer shape correctness ------------------------------------------

def test_signed_first_layer_output_shape():
    x = torch.randn(6, 8)
    ei_pos = torch.tensor([[0, 1, 2], [3, 4, 5]])
    ei_neg = torch.tensor([[0, 1], [4, 3]])
    layer = SignedFirstLayer(in_dim=8, out_dim=12)
    h_B, h_U = layer(x, ei_pos, ei_neg)
    assert h_B.shape == (6, 12)
    assert h_U.shape == (6, 12)


def test_signed_deep_layer_output_shape():
    h_B = torch.randn(6, 8)
    h_U = torch.randn(6, 8)
    ei_pos = torch.tensor([[0, 1], [2, 3]])
    ei_neg = torch.tensor([[0, 1], [3, 2]])
    layer = SignedDeepLayer(in_dim=8, out_dim=8)
    new_B, new_U = layer(h_B, h_U, ei_pos, ei_neg)
    assert new_B.shape == (6, 8)
    assert new_U.shape == (6, 8)


# ---------- SGCN three-config smoke tests -----------------------------------

@pytest.mark.parametrize("config", list(BaselineConfig))
def test_sgcn_forward_runs_for_each_config(config):
    n_nodes = 8
    n_features = 5
    n_dyad_features = 5
    model = SGCN(
        n_nodes=n_nodes, n_features=n_features, n_dyad_features=n_dyad_features,
        hidden_dim=16, n_layers=2, config=config,
    )
    x = torch.randn(n_nodes, n_features)
    ei_pos = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    ei_neg = torch.tensor([[0, 2], [5, 7]])
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([4, 5, 6])
    dyad_x = torch.randn(3, n_dyad_features)
    out = model(x, ei_pos, ei_neg, src, dst, dyad_x)
    assert out.shape == (3,)


def test_identity_only_zeros_features_and_dyad_features():
    """Config IDENTITY_ONLY must give same output regardless of input features."""
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.IDENTITY_ONLY,
    )
    model.eval()
    ei_pos = torch.empty(2, 0, dtype=torch.long)
    ei_neg = torch.empty(2, 0, dtype=torch.long)
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    with torch.no_grad():
        out_a = model(torch.randn(8, 5), ei_pos, ei_neg, src, dst, torch.randn(2, 5))
        out_b = model(torch.zeros(8, 5), ei_pos, ei_neg, src, dst, torch.zeros(2, 5))
    # Identity-only must ignore both per-node and dyad features.
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_identity_free_has_no_id_embedding():
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.IDENTITY_FREE,
    )
    assert model.id_embedding is None
    assert model.feature_proj is not None


def test_identity_only_has_no_feature_proj():
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.IDENTITY_ONLY,
    )
    assert model.id_embedding is not None
    assert model.feature_proj is None


def test_as_published_has_both_id_and_features():
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.AS_PUBLISHED,
    )
    assert model.id_embedding is not None
    assert model.feature_proj is not None


# ---------- embed_perm wiring -----------------------------------------------

def test_embed_perm_changes_output_in_identity_modes():
    """Config IDENTITY_ONLY: permuting the embedding lookup must change the
    output (otherwise the probe doesn't actually probe anything)."""
    torch.manual_seed(0)
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.IDENTITY_ONLY,
    )
    model.eval()
    ei_pos = torch.tensor([[0, 1], [2, 3]])
    ei_neg = torch.tensor([[0], [1]])
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    dyad_x = torch.zeros(2, 5)
    with torch.no_grad():
        baseline = model(torch.zeros(8, 5), ei_pos, ei_neg, src, dst, dyad_x)
        perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
        permuted = model(
            torch.zeros(8, 5), ei_pos, ei_neg, src, dst, dyad_x, embed_perm=perm,
        )
    assert not torch.allclose(baseline, permuted)


def test_embed_perm_is_noop_for_identity_free():
    """IDENTITY_FREE has no learnable embedding, so embed_perm should be a no-op."""
    torch.manual_seed(0)
    model = SGCN(
        n_nodes=8, n_features=5, n_dyad_features=5,
        hidden_dim=16, n_layers=1, config=BaselineConfig.IDENTITY_FREE,
    )
    model.eval()
    ei_pos = torch.tensor([[0, 1], [2, 3]])
    ei_neg = torch.tensor([[0], [1]])
    src = torch.tensor([0, 1])
    dst = torch.tensor([2, 3])
    x = torch.randn(8, 5)
    dyad_x = torch.randn(2, 5)
    with torch.no_grad():
        baseline = model(x, ei_pos, ei_neg, src, dst, dyad_x)
        perm = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])
        permuted = model(x, ei_pos, ei_neg, src, dst, dyad_x, embed_perm=perm)
    assert torch.allclose(baseline, permuted)


# ---------- gradient flow ---------------------------------------------------

def test_gradients_flow_in_all_configs():
    for config in BaselineConfig:
        model = SGCN(
            n_nodes=8, n_features=5, n_dyad_features=5,
            hidden_dim=16, n_layers=2, config=config,
        )
        x = torch.randn(8, 5)
        ei_pos = torch.tensor([[0, 1], [2, 3]])
        ei_neg = torch.tensor([[0], [1]])
        src = torch.tensor([0])
        dst = torch.tensor([2])
        dyad_x = torch.randn(1, 5)
        out = model(x, ei_pos, ei_neg, src, dst, dyad_x)
        out.sum().backward()
        n_params_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert n_params_with_grad > 0, f"No gradients flow in config {config.value}"
