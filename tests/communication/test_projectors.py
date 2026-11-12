"""Tests for src.communication.projectors."""

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import PersonalizedClassifier
from src.communication.projectors import (
    build_projectors,
    common_dim,
    latent_dim,
    project,
    retarget_input,
)


def _agent(d: int, hidden: list[int] | None = None) -> PersonalizedClassifier:
    return PersonalizedClassifier(
        encoder=nn.Linear(6, d),
        latent_dim=d,
        num_classes=3,
        decoder_hidden_dims=[4] if hidden is None else hidden,
        use_batchnorm=True,
    )


def _agents(*dims: int) -> dict[str, PersonalizedClassifier]:
    return {str(i): _agent(d) for i, d in enumerate(dims)}


class TestLatentDim:
    def test_reads_decoder_input_width(self):
        assert latent_dim(_agent(5)) == 5
        assert latent_dim(_agent(7, hidden=[])) == 7

    def test_none_without_a_linear_layer(self):
        agent = _agent(5)
        agent._decoder = nn.Identity()
        assert latent_dim(agent) is None


class TestBuildProjectors:
    def test_none_keeps_equal_dims_unprojected(self):
        projectors = build_projectors(_agents(5, 5), None)
        z = torch.randn(2, 5)
        assert projectors is None
        assert project(projectors, 0, z) is z

    def test_none_projects_to_the_largest_dim(self):
        projectors = build_projectors(_agents(5, 7, 3), None)
        assert {k: tuple(m.weight.shape) for k, m in projectors.items()} == {
            '0': (7, 5),
            '1': (7, 7),
            '2': (7, 3),
        }

    def test_maps_every_agent_to_proj_dim(self):
        projectors = build_projectors(_agents(5, 7), 8)
        assert project(projectors, 0, torch.randn(2, 5)).shape == (2, 8)
        assert project(projectors, '1', torch.randn(2, 7)).shape == (2, 8)

    def test_maps_are_plain_bias_free_linears(self):
        proj = build_projectors(_agents(5, 7), None)['0']
        assert type(proj) is nn.Linear
        assert proj.bias is None

    def test_common_dim(self):
        assert common_dim(_agents(4, 4), None) is None
        assert common_dim(_agents(4, 9), None) == 9
        assert common_dim(_agents(4, 4), 6) == 6

    def test_rejects_bad_proj_dim_and_unreadable_dims(self):
        with pytest.raises(ValueError):
            build_projectors(_agents(5), 0)
        agents = _agents(5, 6)
        agents['0']._decoder = nn.Identity()
        with pytest.raises(TypeError, match='latent dim'):
            build_projectors(agents, 4)


class TestRetargetInput:
    def test_replaces_only_the_first_linear(self):
        decoder = _agent(5).decoder
        out = retarget_input(decoder, 9)
        assert out.network[0].in_features == 9
        assert out.network[0].out_features == decoder.network[0].out_features
        for key, value in decoder.network[1:].state_dict().items():
            assert torch.equal(out.network[1:].state_dict()[key], value)
        assert out(torch.randn(3, 9)).shape == (3, 3)
        assert decoder.network[0].in_features == 5  # original untouched

    def test_single_linear_module(self):
        assert retarget_input(nn.Linear(5, 3), 9).in_features == 9
