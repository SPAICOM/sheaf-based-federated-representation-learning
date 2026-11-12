"""Tests for src.orchestrators.fedmuscle (per-agent projectors)."""

import warnings

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import PersonalizedClassifier
from src.orchestrators.fedmuscle import FedMuscle

OPTIMIZER = {'_target_': 'torch.optim.SGD', 'lr': 0.1}


def _agents(*dims: int) -> dict[int, PersonalizedClassifier]:
    torch.manual_seed(0)
    return {
        i: PersonalizedClassifier(
            encoder=nn.Linear(6, d), latent_dim=d, num_classes=3
        )
        for i, d in enumerate(dims)
    }


def _fedmuscle(dims, **kwargs) -> FedMuscle:
    # E=0: every epoch is a contrastive epoch, so the Muscle loss runs.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return FedMuscle(
            agents=_agents(*dims),
            neighbors={0: {1}, 1: {0}},
            optimizer=OPTIMIZER,
            E=0,
            T=1,
            **kwargs,
        )


def _batch() -> dict:
    y = torch.tensor([0, 1, 2, 0])
    batch = {i: (torch.randn(4, 6), y) for i in range(2)}
    batch.update(
        {f'global_pilot_{i}': (torch.randn(4, 6), y) for i in range(2)}
    )
    return batch


class TestFedMuscleProjectors:
    def test_default_has_no_projector(self):
        orch = _fedmuscle((5, 5))
        assert orch.projectors is None
        _, loss = orch._shared_eval(_batch(), 0, 'train')
        loss.backward()

    def test_different_dims_without_proj_dim_fail_at_init(self):
        with pytest.raises(ValueError, match='set proj_dim'):
            _fedmuscle((5, 7))

    @pytest.mark.parametrize('mode', ['contrastive', 'alignment'])
    def test_different_dims_train_in_the_common_space(self, mode):
        orch = _fedmuscle((5, 7), proj_dim=8, alignment_mode=mode)
        _, loss = orch._shared_eval(_batch(), 0, 'train')
        loss.backward()
        for idx in ('0', '1'):
            grads = [p.grad for p in orch.projectors[idx].parameters()]
            assert all(g is not None and g.abs().sum() > 0 for g in grads)
            enc = orch.agents[idx].encoder.weight.grad
            assert enc is not None

    def test_exchanged_pilots_are_charged_at_proj_dim(self):
        orch = _fedmuscle((5, 7), proj_dim=8)
        orch._shared_eval(_batch(), 0, 'train')
        # 4 pilots x 8 floats x 32 bit, each agent to its 1 neighbour.
        kb = orch._communication_by_split['train']['kilobytes']
        assert kb == pytest.approx(2 * 4 * 8 * 4 / 1024)
