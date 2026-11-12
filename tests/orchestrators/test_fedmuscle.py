"""Tests for src.orchestrators.fedmuscle (per-agent projectors)."""

import warnings

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def test_different_dims_default_to_the_largest_dim(self):
        orch = _fedmuscle((5, 7))
        assert {k: m.weight.shape[0] for k, m in orch.projectors.items()} == {
            '0': 7,
            '1': 7,
        }

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


def _reference_contrastive_loss(reps: list[torch.Tensor], tau: float):
    """Compute the Muscle loss with the original per-sample loop (before vectorisation)."""
    n, b = len(reps), reps[0].shape[0]
    means = torch.stack(reps).mean(dim=0)
    total = torch.tensor(0.0)
    for k in range(n):
        losses = []
        for i in range(b):
            z = reps[k][i].unsqueeze(0)
            other = torch.stack([reps[j][i] for j in range(n) if j != k]).mean(
                0
            )
            pos = F.cosine_similarity(z, other.unsqueeze(0), dim=1)
            negs = [
                F.cosine_similarity(z, means[j].unsqueeze(0), dim=1)
                for j in range(b)
                if j != i
            ]
            neg = torch.stack(negs).mean(0) if negs else torch.tensor(-1.0)
            pe, ne = torch.exp(pos / tau), torch.exp(neg / tau)
            losses.append(-torch.log(pe / (pe + ne)))
        total = total + torch.stack(losses).mean()
    return total / n


class TestContrastiveLoss:
    @pytest.mark.parametrize('n_agents', [2, 3, 4])
    @pytest.mark.parametrize('batch', [1, 5])
    def test_matches_the_per_sample_loop(self, n_agents, batch):
        orch = _fedmuscle((5, 5))
        torch.manual_seed(1)
        reps = [
            torch.randn(batch, 4, requires_grad=True) for _ in range(n_agents)
        ]
        loss = orch._compute_contrastive_loss(dict(enumerate(reps)))
        grads = torch.autograd.grad(loss, reps)
        ref = _reference_contrastive_loss(reps, orch.hparams.temperature)
        ref_grads = torch.autograd.grad(ref.sum(), reps)
        assert loss.item() == pytest.approx(ref.item(), abs=1e-5)
        for g, r in zip(grads, ref_grads, strict=True):
            assert torch.allclose(g, r, atol=1e-5)

    def test_unequal_batches_give_zero(self):
        orch = _fedmuscle((5, 5))
        loss = orch._compute_contrastive_loss(
            {0: torch.randn(3, 4), 1: torch.randn(2, 4)}
        )
        assert loss.item() == 0.0
