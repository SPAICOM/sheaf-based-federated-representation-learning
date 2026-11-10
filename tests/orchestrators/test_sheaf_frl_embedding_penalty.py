"""The sheaf residual is X − A·Y: the lower-dim side embedded into the higher-dim one."""

import torch
import torch.nn as nn

from src.communication.whitening import fit_alignment
from src.orchestrators.di_sheaf_frl import DiSheafFRL
from src.orchestrators.sheaf_frl import SheafFRL


class _Agent(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, 2, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def _build(cls, d_high: int, d_low: int, penalty_norm: str = 'dim', **kwargs):
    orchestrator = cls(
        agents={0: _Agent(d_high), 1: _Agent(d_low)},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: d_high, 1: d_low},
        whitening=False,
        penalty_norm=penalty_norm,
        **kwargs,
    )
    orchestrator.eval()  # `self.log` needs a Trainer; only train mode logs
    return orchestrator


def _semi_orthogonal(d_high: int, d_low: int) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(d_high, d_low))
    return Q


def test_rectangular_edge_is_compared_in_the_higher_dimension() -> None:
    orchestrator = _build(SheafFRL, 3, 2)
    V = _semi_orthogonal(3, 2)
    Z_i, Z_j = torch.randn(5, 3), torch.randn(5, 2)

    X, AY = orchestrator._edge_sides(Z_i, Z_j, V)

    assert X.shape == AY.shape == (5, 3)
    assert torch.equal(X, Z_i)
    assert torch.allclose(AY, (V @ Z_j.T).T)


def test_square_orthogonal_edge_matches_the_old_residual() -> None:
    orchestrator = _build(SheafFRL, 3, 3)
    V = _semi_orthogonal(3, 3)
    Z_i, Z_j = torch.randn(5, 3), torch.randn(5, 3)

    X, AY = orchestrator._edge_sides(Z_i, Z_j, V)

    new = (X - AY).pow(2).sum()
    old = (Z_i @ V - Z_j).pow(2).sum()
    assert torch.isclose(new, old)


def test_general_map_is_fitted_and_used_as_an_embedding() -> None:
    orchestrator = _build(SheafFRL, 3, 2, use_general_maps=True)
    true_map = torch.randn(3, 2)
    Z_j = torch.randn(64, 2)
    Z_i = Z_j @ true_map.T

    orchestrator._fit_edge_map('0_1', 0, 1, Z_i, Z_j, update_maps=True)

    V = orchestrator.stiefel_matrices['0_1']
    X, AY = orchestrator._edge_sides(Z_i, Z_j, V)
    assert (X - AY).abs().max() < 1e-2
    assert torch.allclose(
        orchestrator._directed_embedding_map(0, 1), V.T
    )
    assert orchestrator._directed_embedding_map(1, 0) is None


def test_large_to_small_map_inverts_a_general_embedding() -> None:
    orchestrator = _build(SheafFRL, 3, 2, use_general_maps=True)
    V = torch.randn(3, 2)
    Z_j = torch.randn(5, 2)

    back = Z_j @ V.T @ orchestrator._large_to_small_map(V)

    assert torch.allclose(back, Z_j, atol=1e-4)


def test_posthoc_general_embedding_map_is_fitted_small_to_large() -> None:
    Z_r = torch.randn(64, 2)
    Z_s = Z_r @ torch.randn(2, 3)

    E = fit_alignment(Z_r, Z_s).T

    assert E.shape == (2, 3)
    assert (Z_r @ E - Z_s).abs().max() < 1e-2


def test_dim_norm_divides_a_rectangular_edge_by_the_sum_of_dims() -> None:
    orchestrator = _build(SheafFRL, 3, 2)
    diff = torch.ones(4, 3)

    term = orchestrator._edge_penalty_term('0_1', diff)

    assert torch.isclose(term, torch.tensor(3.0 / 5.0))


def test_disheaf_rectangular_penalty_backward_reaches_both_encoders() -> None:
    orchestrator = _build(DiSheafFRL, 3, 2, penalty_norm='energy')
    x = torch.randn(6, 4)
    Z_i = orchestrator.agents['0'].encode(x)
    Z_j = orchestrator.agents['1'].encode(x)
    V = orchestrator.stiefel_matrices['0_1']
    X, AY = orchestrator._edge_sides(Z_i, Z_j, V)
    orchestrator._step_edge_rows['0_1'] = (Z_i, Z_j)

    loss = orchestrator._edge_penalty_term('0_1', X - AY, sides=(X, AY))
    loss.backward()

    for idx in ('0', '1'):
        grad = orchestrator.agents[idx].encoder.weight.grad
        assert grad is not None and grad.abs().sum() > 0
