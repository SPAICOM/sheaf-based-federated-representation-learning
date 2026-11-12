"""Tests for src.communication.projectors."""

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import PersonalizedClassifier
from src.communication.projectors import (
    build_projectors,
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

    def test_none_with_different_dims_raises_early(self):
        with pytest.raises(ValueError, match='set proj_dim'):
            build_projectors(_agents(5, 7), None)

    def test_maps_every_agent_to_proj_dim(self):
        projectors = build_projectors(_agents(5, 7), 8)
        assert project(projectors, 0, torch.randn(2, 5)).shape == (2, 8)
        assert project(projectors, '1', torch.randn(2, 7)).shape == (2, 8)

    @pytest.mark.parametrize('d, p', [(5, 8), (8, 5)])
    def test_orthogonal_maps_are_semi_orthogonal(self, d, p):
        W = build_projectors(_agents(d), p)['0'].weight
        gram = W.T @ W if p >= d else W @ W.T
        assert torch.allclose(gram, torch.eye(min(d, p)), atol=1e-5)

    def test_orthogonal_map_cannot_shrink_under_a_distance_loss(self):
        """The FedProto failure mode: ||P z||^2 driven to zero by P alone."""
        ratio = {}
        for orthogonal in (True, False):
            torch.manual_seed(0)
            proj = build_projectors(_agents(4), 8, orthogonal=orthogonal)['0']
            opt = torch.optim.SGD(proj.parameters(), lr=0.5)
            z = torch.randn(16, 4)
            for _ in range(50):
                opt.zero_grad()
                proj(z).pow(2).sum(dim=1).mean().backward()
                opt.step()
            with torch.no_grad():
                ratio[orthogonal] = float(proj(z).norm() / z.norm())
        assert ratio[True] == pytest.approx(1.0, abs=1e-5)
        assert ratio[False] < 1e-2

    def test_warns_when_proj_dim_is_below_a_latent_dim(self):
        with pytest.warns(UserWarning, match='proj_dim >= 7'):
            build_projectors(_agents(5, 7), 6)

    def test_rejects_bad_proj_dim_and_unreadable_dims(self):
        with pytest.raises(ValueError):
            build_projectors(_agents(5), 0)
        agents = _agents(5)
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
