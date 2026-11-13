"""Tests for src/utils/sheaf_laplacian.py."""

import torch

from src.utils.sheaf_laplacian import (
    coboundary,
    min_eigenvalue,
    sheaf_laplacian,
    total_variation,
)


def _semi_orthogonal(d_low: int, d_high: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d_high, d_low, generator=g))
    return q.T  # d_low x d_high, orthonormal rows


def test_total_variation_matches_edge_sum():
    """x^T L x equals the summed squared edge residuals; L is symmetric PSD."""
    dims = {0: 4, 1: 2, 2: 3}
    maps = {
        (0, 1): _semi_orthogonal(2, 4, 0),
        (0, 2): _semi_orthogonal(3, 4, 1),
        (2, 1): _semi_orthogonal(2, 3, 2),
    }
    Z = {v: torch.randn(5, d) for v, d in dims.items()}
    L = sheaf_laplacian(maps, dims)
    direct = sum(
        ((Z[s].double() - Z[r].double() @ E.double()) ** 2).sum(1)
        for (s, r), E in maps.items()
    ).mean()
    assert abs(total_variation(L, Z) - float(direct)) < 1e-5
    assert torch.allclose(L, L.T)
    assert coboundary(maps, dims).shape == (4 + 4 + 3, 9)
    assert min_eigenvalue(L) > -1e-10


def test_consistent_maps_have_a_global_section():
    """Maps composing consistently on a triangle: λ_min = 0, TV of a section = 0."""
    dims = {0: 3, 1: 3, 2: 3}
    A, B = _semi_orthogonal(3, 3, 0), _semi_orthogonal(3, 3, 1)
    maps = {(0, 1): A, (1, 2): B, (0, 2): B @ A}
    L = sheaf_laplacian(maps, dims)
    assert abs(min_eigenvalue(L)) < 1e-10
    x2 = torch.randn(4, 3).double()
    section = {2: x2, 1: x2 @ B.double(), 0: x2 @ B.double() @ A.double()}
    assert total_variation(L, section) < 1e-10


def test_inconsistent_cycle_has_no_global_section():
    """Identity, identity, 90° rotation: the holonomy fixes no vector, L has no kernel."""
    dims = {0: 2, 1: 2, 2: 2}
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    maps = {(0, 1): torch.eye(2), (1, 2): torch.eye(2), (0, 2): rot}
    assert min_eigenvalue(sheaf_laplacian(maps, dims)) > 1e-3
