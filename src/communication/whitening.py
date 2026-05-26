"""PPFE – Pre-whitening + Pairwise Feature-space alignment Evaluation.

Pipeline (non-cooperative, post-hoc):
  1. Each agent fits a whitening operator on its own training latents.
  2. For every directed edge (i → j), pilot representations (whitened) are
     used to learn a linear map  A_{j←i}  that minimises ||X_j - X_i A^T||²_F.
  3. Cross-accuracy: agent j classifies agent i's test samples by whitening
     them (with W_i), aligning (A_{j←i}), re-colouring (C_j), then decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ── Whitening / colouring ─────────────────────────────────────────────────────


@dataclass
class WhiteningOp:
    """Pre-whitening and colouring operators for one agent's latent space.

    Convention (row vectors, batch dimension first):
        z_white = (z - mean) @ W          shape: (n, d)
        z_orig  = z_white @ C.T + mean    shape: (n, d)
    """

    mean: torch.Tensor  # (d,)
    W: torch.Tensor  # (d, d)  right-multiply whitening
    C: torch.Tensor  # (d, d)  right-multiply colouring  (W^{-1} row-wise)


def fit_whitening(Z: torch.Tensor, eps: float = 1e-6) -> WhiteningOp:
    """Fit whitening/colouring operators from latent matrix Z (n, d).

    Uses eigendecomposition of the sample covariance with a small ridge for
    numerical stability.  All outputs stored on CPU.
    """
    Z = Z.float()
    mean = Z.mean(0)
    Z_c = Z - mean
    n = Z_c.shape[0]
    d = Z_c.shape[1]

    Cov = (Z_c.T @ Z_c) / max(n - 1, 1)
    Cov = Cov + eps * torch.eye(d, dtype=Z_c.dtype)

    eigenvalues, V = torch.linalg.eigh(Cov)  # V: columns are eigenvectors
    eigenvalues = eigenvalues.clamp(min=eps)

    # W: right-multiply whitening  z_white = (z - mean) @ W
    # C: right-multiply colouring  z = z_white @ C.T + mean
    W = V * eigenvalues.pow(-0.5)  # (d, d)
    C = V * eigenvalues.pow(0.5)  # (d, d)

    return WhiteningOp(mean=mean.cpu(), W=W.cpu(), C=C.cpu())


def whiten(Z: torch.Tensor, op: WhiteningOp) -> torch.Tensor:
    dev = Z.device
    return (Z.float() - op.mean.to(dev)) @ op.W.to(dev)


def color(Z_white: torch.Tensor, op: WhiteningOp) -> torch.Tensor:
    dev = Z_white.device
    return Z_white.float() @ op.C.T.to(dev) + op.mean.to(dev)


# ── Alignment ─────────────────────────────────────────────────────────────────


def fit_alignment(
    X_i: torch.Tensor,
    X_j: torch.Tensor,
    lambda_reg: float = 1e-4,
) -> torch.Tensor:
    """Learn A (d, d) s.t.  X_i @ A.T ≈ X_j  (least squares, closed form).

    Both X_i and X_j are (n, d) whitened representations of the *same* pilot
    samples encoded by agents i and j respectively.

    Returns A on CPU.
    """
    X_i = X_i.float()
    X_j = X_j.float()
    d = X_i.shape[1]
    gram = X_i.T @ X_i + lambda_reg * torch.eye(
        d, dtype=X_i.dtype, device=X_i.device
    )
    # A.T = gram^{-1} @ X_i.T @ X_j  →  A = X_j.T @ X_i @ gram^{-1}
    A = X_j.T @ X_i @ torch.linalg.inv(gram)
    return A.cpu()


# ── Latent extraction ─────────────────────────────────────────────────────────


@torch.no_grad()
def extract_latents(
    agent: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all samples in *loader* through *agent*.

    Returns
    -------
    Z : (n, d) float tensor of latent representations.
    y : (n,)   long tensor of labels.
    """
    agent.eval()
    Zs, ys = [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        Zs.append(agent.encode(x).cpu().float())
        ys.append(y.cpu())
    return torch.cat(Zs), torch.cat(ys)


# ── Common-pilot helpers ──────────────────────────────────────────────────────


def common_pilot_indices(
    pilot_i,
    pilot_j,
) -> tuple[list[int], list[int]]:
    """Return local indices into pilot_i and pilot_j for shared global samples.

    If sample_ids are unavailable, assumes the two datasets are already aligned
    and uses the full shorter length.
    """
    ids_i = getattr(pilot_i, 'sample_ids', None)
    ids_j = getattr(pilot_j, 'sample_ids', None)

    if ids_i is None or ids_j is None:
        n = min(len(pilot_i), len(pilot_j))
        return list(range(n)), list(range(n))

    map_i = {sid: loc for loc, sid in enumerate(ids_i)}
    map_j = {sid: loc for loc, sid in enumerate(ids_j)}
    common = sorted(set(map_i) & set(map_j))
    return [map_i[s] for s in common], [map_j[s] for s in common]


# ── Cross-accuracy ─────────────────────────────────────────────────────────────


@torch.no_grad()
def cross_accuracy(
    agent_j: nn.Module,
    Z_i_test: torch.Tensor,
    y_i_test: torch.Tensor,
    op_i: WhiteningOp,
    op_j: WhiteningOp,
    A_ji: torch.Tensor,
    device: torch.device,
) -> float:
    """Classify agent i's test latents using agent j's decoder.

    Steps:
      1. Whiten Z_i_test with agent i's whitening operator.
      2. Align to agent j's whitened space with A_{j←i}.
      3. Re-colour with agent j's colouring operator.
      4. Classify with agent j's decoder.

    Returns top-1 accuracy as a float in [0, 1].
    """
    agent_j.eval()
    Z_white = whiten(Z_i_test, op_i)  # (n, d)
    Z_aligned = Z_white.float() @ A_ji.T.to(Z_white.device)  # (n, d)
    Z_recolored = color(Z_aligned, op_j)  # (n, d)
    logits = agent_j.decoder(Z_recolored.to(device))
    preds = logits.argmax(1).cpu()
    return float((preds == y_i_test).float().mean().item())


@torch.no_grad()
def self_accuracy(
    agent_i: nn.Module,
    Z_i_test: torch.Tensor,
    y_i_test: torch.Tensor,
    device: torch.device,
) -> float:
    """Agent i classifies its own test latents (no alignment)."""
    agent_i.eval()
    logits = agent_i.decoder(Z_i_test.to(device))
    preds = logits.argmax(1).cpu()
    return float((preds == y_i_test).float().mean().item())
