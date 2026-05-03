"""Smoke tests for SiGAT, SDGNN, SignedTransformer.

Parameterized across the three new baselines + three configs to keep test
volume tractable. Per-baseline architectural quirks (attention masks for
SiGAT, 4-stream aggregation for SDGNN, sign-bias parameter for
SignedTransformer) are tested in their own targeted tests below.
"""

from __future__ import annotations

import pytest
import torch

from src.baselines.common import BaselineConfig
from src.baselines.sdgnn import SDGNN, SDGNNLayer
from src.baselines.sgformer import SignedTransformer, SignedTransformerLayer
from src.baselines.sigat import SiGAT, SiGATFirstLayer, SignedAttentionHead, _scatter_softmax

BASELINES = [SiGAT, SDGNN, SignedTransformer]


# ---------- parameterized smoke tests across baselines + configs --------------

@pytest.mark.parametrize("baseline_cls", BASELINES)
@pytest.mark.parametrize("config", list(BaselineConfig))
def test_forward_pass_shape(baseline_cls, config):
    n_nodes, n_features, n_dyad_features = 8, 5, 5
    model = baseline_cls(
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


@pytest.mark.parametrize("baseline_cls", BASELINES)
def test_identity_only_zeros_features_and_dyad(baseline_cls):
    """IDENTITY_ONLY must give same output regardless of input features."""
    model = baseline_cls(
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
    assert torch.allclose(out_a, out_b, atol=1e-5)


@pytest.mark.parametrize("baseline_cls", BASELINES)
def test_embed_perm_changes_output_in_identity_modes(baseline_cls):
    torch.manual_seed(0)
    model = baseline_cls(
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
        permuted = model(torch.zeros(8, 5), ei_pos, ei_neg, src, dst, dyad_x, embed_perm=perm)
    assert not torch.allclose(baseline, permuted)


@pytest.mark.parametrize("baseline_cls", BASELINES)
def test_embed_perm_is_noop_for_identity_free(baseline_cls):
    torch.manual_seed(0)
    model = baseline_cls(
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


@pytest.mark.parametrize("baseline_cls", BASELINES)
def test_gradient_flow_in_all_configs(baseline_cls):
    for config in BaselineConfig:
        model = baseline_cls(
            n_nodes=8, n_features=5, n_dyad_features=5,
            hidden_dim=16, n_layers=2, config=config,
        )
        x = torch.randn(8, 5)
        ei_pos = torch.tensor([[0, 1], [2, 3]])
        ei_neg = torch.tensor([[0], [1]])
        out = model(x, ei_pos, ei_neg,
                    torch.tensor([0]), torch.tensor([2]), torch.randn(1, 5))
        out.sum().backward()
        n = sum(1 for p in model.parameters()
                if p.grad is not None and p.grad.abs().sum() > 0)
        assert n > 0, f"No gradient flows in {baseline_cls.__name__}/{config.value}"


# ---------- SiGAT-specific ---------------------------------------------------

def test_scatter_softmax_handles_empty_input():
    out = _scatter_softmax(torch.empty(0), torch.empty(0, dtype=torch.long), num_nodes=5)
    assert out.shape == (0,)


def test_scatter_softmax_normalizes_per_group():
    """Scatter softmax should sum to 1 within each destination group."""
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1, 1])
    out = _scatter_softmax(src, index, num_nodes=2)
    # Group 0: softmax([1,2]); Group 1: softmax([3,4])
    grp0_sum = float(out[0] + out[1])
    grp1_sum = float(out[2] + out[3])
    assert abs(grp0_sum - 1.0) < 1e-6
    assert abs(grp1_sum - 1.0) < 1e-6


def test_signed_attention_head_empty_edges():
    head = SignedAttentionHead(in_dim=4, out_dim=4)
    out = head(torch.randn(5, 4), torch.empty(2, 0, dtype=torch.long))
    assert torch.allclose(out, torch.zeros(5, 4))


# ---------- SDGNN-specific ---------------------------------------------------

def test_sdgnn_layer_uses_four_weight_matrices():
    layer = SDGNNLayer(in_dim=8, out_dim=8)
    # All four projection matrices distinct
    weights = {
        "pos_out": layer.W_pos_out.weight,
        "pos_in": layer.W_pos_in.weight,
        "neg_out": layer.W_neg_out.weight,
        "neg_in": layer.W_neg_in.weight,
    }
    names = list(weights.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not torch.allclose(weights[names[i]], weights[names[j]]), \
                f"{names[i]} and {names[j]} should be independently initialized"


# ---------- SignedTransformer-specific --------------------------------------

def test_sign_bias_parameter_exists_and_learnable():
    layer = SignedTransformerLayer(dim=16, n_heads=4)
    assert layer.sign_bias.shape == (3,)
    assert layer.sign_bias.requires_grad


def test_signed_transformer_layer_shape():
    layer = SignedTransformerLayer(dim=16, n_heads=4)
    x = torch.randn(8, 16)
    ei_pos = torch.tensor([[0, 1], [2, 3]])
    ei_neg = torch.tensor([[0], [1]])
    out = layer(x, ei_pos, ei_neg)
    assert out.shape == (8, 16)


def test_signed_transformer_dim_must_be_divisible_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        SignedTransformerLayer(dim=15, n_heads=4)
