"""Tests for src.utils.simclr and the BaseOrchestrator SimCLR hook."""

import pytest
import torch
import torch.nn as nn

from src.agents.resnet_classifier import ResNetClassifier
from src.orchestrators.non_cooperative import NonCooperativeLearning
from src.utils.simclr import frozen_running_stats, nt_xent, simclr_augment


class MockOptimizer:
    _target_ = 'torch.optim.Adam'
    lr = 0.001


def _agent():
    return ResNetClassifier(
        in_features=3, num_classes=10, encoder_hidden_dims=[8, 16],
        blocks_per_stage=[1, 1], stage_strides=[1, 2], latent_norm=True,
    )


def test_augment_shape_range_and_randomness():
    x = torch.rand(6, 3, 32, 32)
    a, b = simclr_augment(x), simclr_augment(x)
    assert a.shape == x.shape
    assert a.min() >= 0 and a.max() <= 1
    assert not torch.allclose(a, b)


def test_nt_xent_prefers_matched_pairs():
    p = torch.randn(16, 8)
    matched = nt_xent(p, p + 0.01 * torch.randn_like(p), 0.5)
    shuffled = nt_xent(p, p[torch.randperm(16)], 0.5)
    assert matched < shuffled


def test_frozen_running_stats_keeps_buffers():
    agent = _agent().train()
    bufs = {k: v.clone() for k, v in agent.state_dict().items() if 'running' in k}
    with frozen_running_stats(agent):
        agent.encode(torch.rand(4, 3, 32, 32))
    for k, v in agent.state_dict().items():
        if 'running' in k:
            assert torch.equal(v, bufs[k]), k
    assert all(m.momentum != 0.0 for m in agent.modules() if hasattr(m, 'running_mean'))


def test_noncoop_simclr_backward_and_head_grads():
    orch = NonCooperativeLearning(
        agents={0: _agent(), 1: _agent()}, neighbors={0: set(), 1: set()},
        optimizer=MockOptimizer(), simclr_weight=1.0,
    )
    batch = {
        0: (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))),
        1: (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))),
        'global_pilot': (torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))),
    }
    loss = orch.training_step(batch, batch_idx=0)
    loss.backward()
    assert isinstance(orch.simclr_heads['0'], nn.Module)
    assert orch.simclr_heads['0'][0].weight.grad is not None
    assert {id(p) for p in orch.simclr_heads.parameters()} <= {id(p) for p in orch.parameters()}


def test_simclr_off_by_default_and_requires_pilots():
    orch = NonCooperativeLearning(
        agents={0: _agent()}, neighbors={0: set()}, optimizer=MockOptimizer(),
    )
    assert orch.simclr_heads is None
    orch = NonCooperativeLearning(
        agents={0: _agent()}, neighbors={0: set()}, optimizer=MockOptimizer(),
        simclr_weight=1.0,
    )
    with pytest.raises(ValueError, match='pilots'):
        orch._local_simclr_loss(orch.agents['0'], {0: (torch.rand(4, 3, 32, 32),
                                                       torch.zeros(4).long())}, 0)
