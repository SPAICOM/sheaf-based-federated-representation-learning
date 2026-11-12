"""Tests for src.orchestrators.fedproto (per-agent projectors)."""

import warnings

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import PersonalizedClassifier
from src.orchestrators.fedproto import FedProto

OPTIMIZER = {'_target_': 'torch.optim.SGD', 'lr': 0.1}


def _agents(*dims: int) -> dict[int, PersonalizedClassifier]:
    torch.manual_seed(0)
    return {
        i: PersonalizedClassifier(
            encoder=nn.Linear(6, d), latent_dim=d, num_classes=3
        )
        for i, d in enumerate(dims)
    }


def _fedproto(dims, **kwargs) -> FedProto:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return FedProto(
            agents=_agents(*dims),
            neighbors={0: {1}, 1: {0}},
            optimizer=OPTIMIZER,
            **kwargs,
        )


def _batch() -> dict:
    y = torch.tensor([0, 1, 2, 0])
    return {i: (torch.randn(4, 6), y) for i in range(2)}


def _two_steps(orch: FedProto) -> torch.Tensor:
    """Step 0 seeds the global prototypes; step 1 is the first regularised one."""
    for step in range(2):
        orch.zero_grad()
        _, loss = orch._shared_eval(_batch(), step, 'train')
        loss.backward()
    return loss


class TestFedProtoProjectors:
    def test_default_has_no_projector(self):
        orch = _fedproto((5, 5))
        assert orch.projectors is None
        _two_steps(orch)

    def test_different_dims_without_proj_dim_fail_at_init(self):
        with pytest.raises(ValueError, match='set proj_dim'):
            _fedproto((5, 7))

    def test_different_dims_train_in_the_common_space(self):
        orch = _fedproto((5, 7), proj_dim=8)
        _two_steps(orch)
        assert all(p.shape == (8,) for p in orch._global_prototypes.values())
        for idx in ('0', '1'):
            grads = [p.grad for p in orch.projectors[idx].parameters()]
            assert all(g is not None and g.abs().sum() > 0 for g in grads)

    def test_projectors_are_optimised(self):
        orch = _fedproto((5, 7), proj_dim=8)
        own = {id(p) for p in orch.parameters()}
        assert all(id(p) in own for p in orch.projectors.parameters())

    def test_uploads_are_charged_at_proj_dim(self):
        orch = _fedproto((5, 7), proj_dim=8)
        orch._shared_eval(_batch(), 0, 'train')
        # 3 classes x 8 floats x 32 bit, from each of the 2 agents.
        kb = orch._communication_by_split['train']['kilobytes']
        assert kb == pytest.approx(2 * 3 * 8 * 4 / 1024)
