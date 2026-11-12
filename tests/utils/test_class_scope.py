"""Tests for the class-scope accuracy split."""

import pytest
import torch

from src.utils.class_scope import ClassScopeAccuracy


def _logits(pred: list[int], n_classes: int = 4) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(pred), n_classes).float()


def test_private_seen_unseen():
    """Private accuracy splits into the agent's own classes and the rest."""
    acc = ClassScopeAccuracy({0: [0, 1]})
    y = torch.tensor([0, 1, 2, 3])
    acc.add_private(
        0, _logits([0, 0, 2, 0]), y
    )  # seen: 1/2 right, unseen: 1/2 right
    logs = acc.logs('test')
    assert logs['test/scope/private_seen_agent_0'] == pytest.approx(0.5)
    assert logs['test/scope/private_unseen'] == pytest.approx(0.5)
    assert logs['test/scope/n_private_seen'] == 2


def test_comm_scopes_pool_over_edges():
    """Comm scopes are pooled over edges, per agent and network-wide."""
    # agent 0 knows {0, 1}, agent 1 knows {1, 2}
    acc = ClassScopeAccuracy({0: [0, 1], 1: [1, 2]})
    y = torch.tensor([0, 1, 2, 3])
    acc.add_comm(
        0, 1, _logits([3, 1, 2, 0]), y
    )  # receiver 1: right on 1 and 2
    acc.add_comm(
        1, 0, _logits([0, 1, 0, 0]), y
    )  # receiver 0: right on 0 and 1
    logs = acc.logs('test')
    # sample 1 is seen by both on both edges: 2/2
    assert logs['test/scope/comm_both_seen'] == pytest.approx(1.0)
    # edge 0->1, class 2 (tx unseen, rx seen): right; edge 1->0, class 0: right
    assert logs['test/scope/comm_tx_unseen_rx_seen'] == pytest.approx(1.0)
    assert logs['test/scope/comm_tx_unseen_rx_seen_edge_0_1'] == pytest.approx(
        1.0
    )
    # class 0 on edge 0->1 and class 2 on edge 1->0: both wrong
    assert logs['test/scope/comm_tx_seen_rx_unseen'] == pytest.approx(0.0)
    assert logs['test/scope/comm_all_rx_1'] == pytest.approx(0.5)
    assert logs['test/scope/comm_all'] == pytest.approx(0.5)
    assert logs['test/scope/n_comm_both_unseen'] == 2
    assert logs['test/scope/n_comm_tx_unseen_rx_seen'] == 2
    assert 'test/scope/n_comm_all_rx_1' not in logs
    assert acc.covers(0, 1) and not acc.covers(2)
