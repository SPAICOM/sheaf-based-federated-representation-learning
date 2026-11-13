"""Sheaf Laplacian materialised from one alignment map per edge.

Each undirected edge ``(s, r)`` carries a map ``E`` (``d_r x d_s``) with
``Z_r @ E ≈ Z_s`` in row-vector convention: the receiver ``r`` is embedded
into the sender's space, whose dimension is the edge stalk. This is the
convention of ``_directed_embedding_map`` (post-hoc maps and SheafFRL's
learned restriction maps alike). Restriction maps: ``F_s = I``,
``F_r = E^T``; coboundary ``(δx)_e = x_s - E^T x_r``; ``L = δ^T δ``.

``total_variation`` is ``mean_n x_n^T L x_n = mean_n Σ_e ||x_s - x_r E||²``
over the rows of node signals matched by sample; the smallest eigenvalue of
``L`` is 0 iff a non-zero global section exists (maps consistent around every
cycle), so on a graph with cycles it measures how far the maps are from
admitting one.
"""

from collections.abc import Mapping

import torch


def node_offsets(dims: Mapping[int, int]) -> dict[int, int]:
    """Start row of each node's stalk in the stacked vector (sorted node ids)."""
    out, start = {}, 0
    for node in sorted(dims):
        out[node] = start
        start += int(dims[node])
    return out


def coboundary(
    maps: Mapping[tuple[int, int], torch.Tensor], dims: Mapping[int, int]
) -> torch.Tensor:
    """Dense coboundary ``δ`` (Σ_e d_s x Σ_v d_v), edges in sorted key order."""
    offsets = node_offsets(dims)
    n_cols = sum(int(d) for d in dims.values())
    blocks = []
    for (s, r) in sorted(maps):
        E = maps[(s, r)].detach().double().cpu()
        if E.shape != (dims[r], dims[s]):
            raise ValueError(
                f'edge {s}-{r}: map shape {tuple(E.shape)}, '
                f'expected ({dims[r]}, {dims[s]})'
            )
        block = torch.zeros(dims[s], n_cols, dtype=torch.float64)
        block[:, offsets[s]:offsets[s] + dims[s]] = torch.eye(dims[s])
        block[:, offsets[r]:offsets[r] + dims[r]] = -E.T
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def sheaf_laplacian(
    maps: Mapping[tuple[int, int], torch.Tensor], dims: Mapping[int, int]
) -> torch.Tensor:
    """``L = δ^T δ`` (float64, symmetric PSD)."""
    delta = coboundary(maps, dims)
    return delta.T @ delta


def stack_signals(Z: Mapping[int, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-node signals (n x d_v, rows matched) in sorted node order."""
    return torch.cat([Z[v].detach().double().cpu() for v in sorted(Z)], dim=1)


def total_variation(L: torch.Tensor, Z: Mapping[int, torch.Tensor]) -> float:
    """Mean over rows of ``x^T L x`` for the stacked node signals ``Z``."""
    X = stack_signals(Z)
    return float(((X @ L) * X).sum(dim=1).mean())


def min_eigenvalue(L: torch.Tensor) -> float:
    """Smallest eigenvalue of the (symmetric) Laplacian."""
    return float(torch.linalg.eigvalsh(L)[0])
