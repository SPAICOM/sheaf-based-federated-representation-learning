"""Tests for src.orchestrators.comfed (common projection dimension)."""

import warnings

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import PersonalizedClassifier
from src.orchestrators.comfed import ComFed

OPTIMIZER = {'_target_': 'torch.optim.SGD', 'lr': 0.1}


def _comfed(*dims: int, **kwargs) -> ComFed:
    torch.manual_seed(0)
    agents = {
        i: PersonalizedClassifier(
            encoder=nn.Linear(6, d), latent_dim=d, num_classes=3
        )
        for i, d in enumerate(dims)
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ComFed(
            agents=agents,
            neighbors={0: {1}, 1: {0}},
            optimizer=OPTIMIZER,
            max_lmb=0.1,
            latent_dims=dict(enumerate(dims)),
            **kwargs,
        )


def _batch() -> dict:
    y = torch.tensor([0, 1, 2, 0])
    return {i: (torch.randn(4, 6), y) for i in range(2)}


class TestComFedProjDim:
    def test_default_is_the_largest_latent_dim(self):
        orch = _comfed(5, 7)
        assert orch.proj_dim == 7
        shapes = {
            k: tuple(P.shape) for k, P in orch.projection_matrices.items()
        }
        assert shapes == {'0': (7, 5), '1': (7, 7)}

    def test_explicit_proj_dim_is_kept(self):
        assert _comfed(5, 7, proj_dim=4).projection_matrices['0'].shape == (
            4,
            5,
        )

    def test_rejects_bad_proj_dim(self):
        with pytest.raises(ValueError, match='proj_dim'):
            _comfed(5, 7, proj_dim=0)

    def test_alignment_trains_encoders_and_projections(self):
        orch = _comfed(5, 7)
        _, loss = orch._shared_eval(_batch(), 0, 'train')
        loss.backward()
        for P in orch.projection_matrices.values():
            assert P.grad is not None and P.grad.abs().sum() > 0
        assert orch.agents['0'].encoder.weight.grad is not None
