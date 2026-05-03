"""Smoke tests for TradeOnlyRGCN: forward pass shape + gradient flow."""

from __future__ import annotations

import torch

from src.models.trade_rgcn import TradeOnlyRGCN


def test_forward_pass_returns_correct_shape():
    model = TradeOnlyRGCN(n_nodes=10, hidden_dim=16, n_layers=2)
    edge_index = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
    edge_type = torch.zeros(6, dtype=torch.long)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 0])
    log_trade = torch.tensor([2.3, 1.5, 0.8])
    logits = model(edge_index, edge_type, src, dst, log_trade)
    assert logits.shape == (3,)


def test_gradient_flows_through_node_embeddings():
    model = TradeOnlyRGCN(n_nodes=10, hidden_dim=16, n_layers=2)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_type = torch.zeros(2, dtype=torch.long)
    src = torch.tensor([0])
    dst = torch.tensor([1])
    log_trade = torch.tensor([1.0])
    logits = model(edge_index, edge_type, src, dst, log_trade)
    loss = logits.sum()
    loss.backward()
    assert model.node_emb.weight.grad is not None
    assert model.node_emb.weight.grad.abs().sum().item() > 0


def test_eval_mode_disables_dropout():
    model = TradeOnlyRGCN(n_nodes=10, hidden_dim=16, n_layers=2, dropout=0.5)
    model.eval()
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_type = torch.zeros(2, dtype=torch.long)
    src = torch.tensor([0])
    dst = torch.tensor([1])
    log_trade = torch.tensor([1.0])
    with torch.no_grad():
        a = model(edge_index, edge_type, src, dst, log_trade)
        b = model(edge_index, edge_type, src, dst, log_trade)
    assert torch.allclose(a, b)
