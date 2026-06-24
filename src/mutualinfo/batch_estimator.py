"""Neural (CE-alignment) PID estimator.

Implements the second estimator of Liang et al. (2023).  Three
discriminative critics are trained to approximate

    p(y | z1),    p(y | z2),    p(y | z1, z2)

after which an alignment network ``Q(z2 | z1, y)`` is fitted by
minimising the cross-entropy

    L = E_{p(z1) p(y | z1)} [ -E_{Q(z2 | z1, y)} log p(y' | z1, z2) ]

(approximated through Sinkhorn iterations that enforce the conditional
marginals ``Q(z2|z1) = p(z2)`` and the row marginal ``Q(z2|y) = p(z2|y)``).
Once these networks are trained, the four PID components are read off
as

    R  = I(Y; Z1) + I(Y; Z2) − I_Q(Y; Z1, Z2)
    U1 = I_Q(Y; Z1, Z2) − I(Y; Z2)
    U2 = I_Q(Y; Z1, Z2) − I(Y; Z1)
    S  = I(Y; Z1, Z2) − I_Q(Y; Z1, Z2)

Compared with the reference implementation we (a) made everything
device-agnostic, (b) removed the dependency on ``.cuda()``-only code
paths, and (c) added a thin ``batch_pid`` wrapper that hides the
discriminator-training boilerplate.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.mutualinfo._common import PIDResult, standardize, to_numpy


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    n_hidden: int,
    activation: str = 'relu',
) -> nn.Sequential:
    act = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_hidden):
        layers += [nn.Linear(hidden, hidden), act()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class _Discriminator(nn.Module):
    """Maps one or two flattened views to label logits."""

    def __init__(
        self, in_dim: int, hidden: int, num_classes: int, n_hidden: int = 2
    ) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, num_classes, n_hidden)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x.flatten(1) for x in xs], dim=-1))


def _sinkhorn_step(
    matrix: torch.Tensor,
    x1_probs: torch.Tensor,
    x2_probs: torch.Tensor,
    atol: float = 1e-2,
) -> tuple[torch.Tensor, bool]:
    """One pass of biproportional fitting on a (batch, batch) matrix."""
    matrix = matrix / (matrix.sum(dim=0, keepdim=True) + 1e-8) * x2_probs[None]
    if torch.allclose(matrix.sum(dim=1), x1_probs, rtol=0, atol=atol):
        return matrix, True
    matrix = (
        matrix / (matrix.sum(dim=1, keepdim=True) + 1e-8) * x1_probs[:, None]
    )
    if torch.allclose(matrix.sum(dim=0), x2_probs, rtol=0, atol=atol):
        return matrix, True
    return matrix, False


class _CEAlignment(nn.Module):
    """``Q(z2 | z1, y)`` parameterised by two label-conditioned embeddings."""

    def __init__(
        self,
        z1_dim: int,
        z2_dim: int,
        hidden: int,
        embed: int,
        num_classes: int,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.f1 = _mlp(z1_dim, hidden, embed * num_classes, n_hidden)
        self.f2 = _mlp(z2_dim, hidden, embed * num_classes, n_hidden)

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        p_y_x1: torch.Tensor,
        p_y_x2: torch.Tensor,
        sinkhorn_iters: int = 200,
    ) -> torch.Tensor:
        q1 = self.f1(z1).unflatten(1, (self.num_classes, -1))
        q2 = self.f2(z2).unflatten(1, (self.num_classes, -1))

        # Per-label feature standardisation stabilises the softmax align.
        q1 = (q1 - q1.mean(dim=2, keepdim=True)) / torch.sqrt(
            q1.var(dim=2, keepdim=True) + 1e-8
        )
        q2 = (q2 - q2.mean(dim=2, keepdim=True)) / torch.sqrt(
            q2.var(dim=2, keepdim=True) + 1e-8
        )

        # (a, b, y): unnormalised joint a↔b weight per label.
        align = torch.exp(
            torch.einsum('ahx, bhx -> abh', q1, q2) / math.sqrt(q1.size(-1))
        )

        normalised: list[torch.Tensor] = []
        for y_idx in range(align.size(-1)):
            current = align[..., y_idx]
            for _ in range(sinkhorn_iters):
                current, stop = _sinkhorn_step(
                    current, p_y_x1[:, y_idx], p_y_x2[:, y_idx]
                )
                if stop:
                    break
            normalised.append(current)
        out = torch.stack(normalised, dim=-1)
        if torch.isnan(out).any():
            raise RuntimeError('Alignment produced NaNs (Sinkhorn diverged).')
        return out


class _CEAlignmentInformation(nn.Module):
    """Wrapper that turns the alignment into PID + CE-alignment loss."""

    def __init__(
        self,
        z1_dim: int,
        z2_dim: int,
        hidden: int,
        embed: int,
        num_classes: int,
        n_hidden: int,
        discrim_1: nn.Module,
        discrim_2: nn.Module,
        discrim_12: nn.Module,
        p_y: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.align = _CEAlignment(
            z1_dim, z2_dim, hidden, embed, num_classes, n_hidden
        )
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        for m in (self.discrim_1, self.discrim_2, self.discrim_12):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        self.register_buffer('p_y', p_y)

    def align_parameters(self):
        return self.align.parameters()

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            p_y_x1 = F.softmax(self.discrim_1(z1), dim=-1)
            p_y_x2 = F.softmax(self.discrim_2(z2), dim=-1)
            p_y_x1x2 = F.softmax(self.discrim_12(z1, z2), dim=-1)

        align = self.align(z1.flatten(1), z2.flatten(1), p_y_x1, p_y_x2)
        q_x2_x1y = align / (align.sum(dim=1, keepdim=True) + 1e-8)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(
            torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1) + 1e-8
        )[:, :, None]
        loss = (p_y_x1[:, None, :] * q_x2_x1y * log_term).sum(dim=(-1, -2)).mean()

        p_y = self.p_y
        log_p_y = torch.log(p_y + 1e-8)

        mi_y_z1 = (
            p_y_x1 * (torch.log(p_y_x1 + 1e-8) - log_p_y[None])
        ).sum(dim=-1).mean()
        mi_y_z2 = (
            p_y_x2 * (torch.log(p_y_x2 + 1e-8) - log_p_y[None])
        ).sum(dim=-1).mean()
        mi_y_z1z2 = (
            p_y_x1x2 * (torch.log(p_y_x1x2 + 1e-8) - log_p_y[None])
        ).sum(dim=-1).mean()

        mi_q_y_z1z2 = (
            p_y_x1[:, None, :]
            * q_x2_x1y
            * (
                log_term
                + torch.log(p_y_x1 + 1e-8)[:, None, :]
                - log_p_y[None, None, :]
            )
        ).sum(dim=(-1, -2)).mean()

        redundancy = mi_y_z1 + mi_y_z2 - mi_q_y_z1z2
        unique1 = mi_q_y_z1z2 - mi_y_z2
        unique2 = mi_q_y_z1z2 - mi_y_z1
        synergy = mi_y_z1z2 - mi_q_y_z1z2

        return loss, torch.stack(
            [
                redundancy,
                unique1,
                unique2,
                synergy,
                mi_y_z1,
                mi_y_z2,
                mi_y_z1z2,
            ]
        )


def _train_module(
    module: nn.Module,
    dataset: Dataset,
    forward_fn,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    """Generic SGD training loop used for the three discriminators."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    optim = torch.optim.Adam(module.parameters(), lr=lr)
    module.train()
    for _ in range(epochs):
        for batch in loader:
            optim.zero_grad(set_to_none=True)
            batch = [b.to(device) for b in batch]
            loss = forward_fn(module, batch)
            loss.backward()
            optim.step()
    module.eval()


def batch_pid(
    z1,
    z2,
    y,
    num_classes: int | None = None,
    hidden_dim: int = 32,
    embed_dim: int = 10,
    n_hidden: int = 2,
    discrim_epochs: int = 40,
    align_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    standardize_features: bool = True,
    eval_batches: int = 8,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> PIDResult:
    """Neural CE-alignment PID estimator.

    Parameters
    ----------
    z1, z2 : array-like, shape (N, d1) / (N, d2)
        Continuous representations of the two views.
    y : array-like, shape (N,)
        Discrete labels.
    num_classes : int or None
        If ``None``, inferred from ``y``.
    hidden_dim, embed_dim, n_hidden : int
        Architecture of the MLP critics and the alignment net.
    discrim_epochs : int
        Epochs spent training the three discriminators.
    align_epochs : int
        Epochs spent training the alignment net.
    batch_size : int
        Batch size for all three training loops.
    lr : float
        Adam learning rate.
    standardize_features : bool
        Z-score the inputs before training.
    eval_batches : int
        Number of mini-batches used to average the PID estimate at
        eval time.  Sinkhorn requires a fixed batch size, so we
        average across batches.
    seed : int
        Manual seed (sets ``torch.manual_seed``).
    device : str | torch.device | None
        Defaults to ``cuda`` if available, otherwise ``cpu``.

    Returns
    -------
    PIDResult
    """
    torch.manual_seed(int(seed))

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    z1_np = to_numpy(z1).astype(np.float32)
    z2_np = to_numpy(z2).astype(np.float32)
    y_np = to_numpy(y).astype(np.int64).reshape(-1)

    if z1_np.ndim == 1:
        z1_np = z1_np[:, None]
    if z2_np.ndim == 1:
        z2_np = z2_np[:, None]

    # Remap labels so they live in {0, …, K-1} regardless of input encoding.
    classes, y_compact = np.unique(y_np, return_inverse=True)
    inferred_num_classes = int(len(classes))
    if num_classes is None:
        num_classes = inferred_num_classes
    elif num_classes < inferred_num_classes:
        raise ValueError(
            f'num_classes={num_classes} but observed {inferred_num_classes}.'
        )

    if standardize_features:
        z1_np = standardize(z1_np).astype(np.float32)
        z2_np = standardize(z2_np).astype(np.float32)

    z1_t = torch.from_numpy(z1_np)
    z2_t = torch.from_numpy(z2_np)
    y_t = torch.from_numpy(y_compact.astype(np.int64))

    d1 = int(z1_t.shape[1])
    d2 = int(z2_t.shape[1])

    # ── Train the three label discriminators. ────────────────────────────
    discrim_1 = _Discriminator(d1, hidden_dim, num_classes, n_hidden).to(device)
    discrim_2 = _Discriminator(d2, hidden_dim, num_classes, n_hidden).to(device)
    discrim_12 = _Discriminator(
        d1 + d2, hidden_dim, num_classes, n_hidden
    ).to(device)

    def _fwd_one(module, batch):
        z, lbl = batch
        return F.cross_entropy(module(z), lbl)

    def _fwd_two(module, batch):
        za, zb, lbl = batch
        return F.cross_entropy(module(za, zb), lbl)

    _train_module(
        discrim_1,
        TensorDataset(z1_t, y_t),
        _fwd_one,
        epochs=discrim_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    _train_module(
        discrim_2,
        TensorDataset(z2_t, y_t),
        _fwd_one,
        epochs=discrim_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    _train_module(
        discrim_12,
        TensorDataset(z1_t, z2_t, y_t),
        _fwd_two,
        epochs=discrim_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    # ── Train the alignment network. ─────────────────────────────────────
    p_y = (
        F.one_hot(y_t, num_classes=num_classes).sum(dim=0).float()
        / float(len(y_t))
    ).to(device)

    info_model = _CEAlignmentInformation(
        z1_dim=d1,
        z2_dim=d2,
        hidden=hidden_dim,
        embed=embed_dim,
        num_classes=num_classes,
        n_hidden=n_hidden,
        discrim_1=discrim_1,
        discrim_2=discrim_2,
        discrim_12=discrim_12,
        p_y=p_y,
    ).to(device)

    optim = torch.optim.Adam(info_model.align_parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(z1_t, z2_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    info_model.train()
    for _ in range(align_epochs):
        for z1_b, z2_b, y_b in train_loader:
            optim.zero_grad(set_to_none=True)
            z1_b = z1_b.to(device)
            z2_b = z2_b.to(device)
            y_b = y_b.to(device)
            loss, _ = info_model(z1_b, z2_b, y_b)
            loss.backward()
            optim.step()
    info_model.eval()

    # ── Average PID over several batches for stability. ──────────────────
    eval_loader = DataLoader(
        TensorDataset(z1_t, z2_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    accum = torch.zeros(7, device=device)
    n_seen = 0
    with torch.no_grad():
        for z1_b, z2_b, y_b in eval_loader:
            if n_seen >= eval_batches:
                break
            _, stats = info_model(
                z1_b.to(device), z2_b.to(device), y_b.to(device)
            )
            accum += stats
            n_seen += 1
    if n_seen == 0:
        raise RuntimeError(
            'No evaluation batch was processed — increase the sample count '
            f'(N={len(y_t)}) or lower batch_size (={batch_size}).'
        )
    accum = (accum / n_seen).cpu()

    redundancy = float(max(accum[0].item(), 0.0))
    unique1 = float(max(accum[1].item(), 0.0))
    unique2 = float(max(accum[2].item(), 0.0))
    synergy = float(max(accum[3].item(), 0.0))
    mi_y_z1 = float(accum[4].item())
    mi_y_z2 = float(accum[5].item())
    mi_y_z1z2 = float(accum[6].item())

    return PIDResult(
        redundancy=redundancy,
        unique1=unique1,
        unique2=unique2,
        synergy=synergy,
        mi_y_z1=mi_y_z1,
        mi_y_z2=mi_y_z2,
        mi_y_z1z2=mi_y_z1z2,
        method='batch',
    )
