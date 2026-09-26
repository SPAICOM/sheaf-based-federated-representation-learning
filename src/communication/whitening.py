"""Whitening and pairwise alignment operators for cross-agent evaluation.

Post-hoc alignment pipeline (non-cooperative setting):
  1. Each agent fits a whitening operator on its own training latents.
  2. For every directed edge (i → j), pilot representations (whitened) are
     used to learn an alignment map  A_{j←i}  (general or Procrustes).
  3. Cross-accuracy: agent j classifies agent i's test samples by whitening
     them (with W_i), aligning (A_{j←i}), re-colouring (C_j), then decoding.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

# ── Whitening / colouring ─────────────────────────────────────────────────────

#: Absolute floor below which an eigenvalue is treated as numerically zero;
#: guards 1/sqrt(lambda) against a fully collapsed latent space.
_MIN_EIG = 1e-12


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
    # Diagnostics from the fit (see `whitening_from_moments`): fraction of the
    # d directions whose eigenvalue was raised by the ridge, and the condition
    # number of the ridged spectrum.  A `clamped_fraction` near 1 means the
    # ridge — not the data — is setting the scale of almost every direction,
    # i.e. the latents are barely whitened at all.
    clamped_fraction: float = 0.0
    cond: float = 1.0


def latent_moments(
    Z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Mean and sample covariance of a latent matrix Z (n, d), in float64.

    Split out of :func:`fit_whitening` so the sufficient statistics can be
    smoothed across refits (an EMA of the covariance keeps the whitened frame
    from jumping between epochs) before the operators are built by
    :func:`whitening_from_moments`.

    Accumulation is done in float64: the covariance squares the dynamic range
    of the latents, and a float32 sum over tens of thousands of rows loses the
    small eigenvalues that whitening then has to invert.
    """
    Z = Z.double()
    # Sanitize non-finite values that can arise from training instability so
    # that the eigendecomposition never receives a corrupt matrix.
    if not torch.isfinite(Z).all():
        warnings.warn(
            f'latent_moments: input Z contains non-finite values '
            f'({(~torch.isfinite(Z)).sum().item()} entries). '
            'Replacing with 0.0 — check for training instability.',
            RuntimeWarning,
            stacklevel=2,
        )
        Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    n = Z.shape[0]
    mean = Z.mean(0)
    Z_c = Z - mean
    cov = (Z_c.T @ Z_c) / max(n - 1, 1)
    return mean, cov, int(n)


def whitening_from_moments(
    mean: torch.Tensor,
    cov: torch.Tensor,
    n: int,
    eps_rel: float = 1e-4,
    eps_abs: float = 0.0,
    eps_null: float = 1e-2,
    shrinkage: float = 0.0,
) -> WhiteningOp:
    """Build whitening/colouring operators from (mean, covariance).

    The ridge is **relative** to the spectrum: an observed eigenvalue is
    floored at ``eps_rel * lambda_max`` rather than at an absolute constant.
    An absolute floor is scale-dependent, and latent scales vary by orders of
    magnitude across architectures — with the old absolute ``eps=1e-2`` on a
    trained MNIST CNN (``lambda_max ~ 0.5``, median ``~7e-4``) 86% of the 64
    directions were pinned to the ridge, so they were multiplied by a constant
    ``1/sqrt(eps)`` instead of their own ``1/sqrt(lambda)`` and the "whitened"
    covariance was nowhere near the identity.  A relative floor caps the
    condition number of the whitened spectrum at ``1/eps_rel`` regardless of
    how the latents happen to be scaled.

    The structurally unobserved directions (when ``n - 1 < d``) keep their
    own **absolute** floor ``eps_null``: they carry no data at all, so giving
    them the (much smaller) relative floor would amplify pure noise by
    ``1/sqrt(eps_rel * lambda_max)`` on every sample seen after the fit.

    ``shrinkage`` optionally applies Ledoit-Wolf-style shrinkage towards a
    scaled identity, ``cov <- (1 - g) cov + g (tr cov / d) I``, before the
    decomposition.  Unlike a floor this conditions the whole spectrum
    smoothly and is scale-equivariant; 0.0 (the default) disables it.

    The spectrum is extracted with the SVD of ``cov`` rather than
    ``linalg.eigh``: for a symmetric (PSD) matrix the singular values are the
    eigenvalues and the left singular vectors equal the eigenvectors up to
    column signs — which cancel in ``W @ C.T`` — so it yields the identical
    operator.  The divide-and-conquer eigensolver fails to converge on
    ill-conditioned or heavily clustered spectra (``linalg.eigh: ... the
    input matrix is ill-conditioned or has too many repeated eigenvalues``),
    whereas the SVD never does on finite input; latent covariances are
    typically near-degenerate, so this removes an entire class of crashes.
    A non-finite ``cov`` (e.g. from latent blow-up) degrades to identity
    whitening instead of raising.

    Parameters
    ----------
    mean, cov : torch.Tensor
        Latent mean (d,) and sample covariance (d, d), e.g. from
        :func:`latent_moments`.
    n : int
        Number of samples the moments were estimated from — only used to
        locate the structurally unobserved directions (``n - 1 < d``).
    eps_rel : float
        Relative eigenvalue floor for observed directions, as a fraction of
        ``lambda_max``.
    eps_abs : float
        Additional absolute floor for observed directions; the effective floor
        is ``max(eps_rel * lambda_max, eps_abs)``.  ``eps_rel=0.0,
        eps_abs=1e-2`` reproduces the pre-change behaviour exactly.
    eps_null : float
        Absolute eigenvalue floor for the unobserved directions.
    shrinkage : float
        Shrinkage intensity in [0, 1) towards ``(tr cov / d) I``.
    """
    cov = cov.double()
    mean = mean.double()
    d = cov.shape[0]
    device = cov.device

    if not 0.0 <= float(shrinkage) < 1.0:
        raise ValueError(f'shrinkage must lie in [0, 1), got {shrinkage}')
    for name, value in (
        ('eps_rel', eps_rel),
        ('eps_abs', eps_abs),
        ('eps_null', eps_null),
    ):
        if float(value) < 0.0:
            raise ValueError(f'{name} must be non-negative, got {value}')

    # Symmetrise (guards against asymmetry accumulated by an EMA in float32)
    # and optionally shrink towards a scaled identity.
    cov = 0.5 * (cov + cov.T)
    if shrinkage > 0.0:
        target = (torch.diagonal(cov).sum() / d) * torch.eye(
            d, dtype=cov.dtype, device=device
        )
        cov = (1.0 - shrinkage) * cov + shrinkage * target

    # Guard a non-finite covariance (latent blow-up, EMA corruption): SVD on
    # NaN/Inf itself fails, and there is nothing sensible to whiten with.
    if not torch.isfinite(cov).all():
        warnings.warn(
            f'whitening_from_moments: covariance contains non-finite values '
            f'({(~torch.isfinite(cov)).sum().item()} entries); '
            'falling back to identity whitening — check for training '
            'instability.',
            RuntimeWarning,
            stacklevel=2,
        )
        return WhiteningOp(
            mean=mean.float(),
            W=torch.eye(d, dtype=torch.float32, device=device),
            C=torch.eye(d, dtype=torch.float32, device=device),
            clamped_fraction=1.0,
            cond=1.0,
        )
    try:
        _, S, Vh = torch.linalg.svd(cov)
    except RuntimeError:
        _, S, Vh = torch.linalg.svd(cov.cpu())
        S = S.to(device)
        Vh = Vh.to(device)
    # SVD returns singular values in descending order (== the eigenvalues of
    # this PSD covariance), so index i < rank picks out the directions the
    # data actually spans.
    eigenvalues = S.clamp(min=0.0)
    V = Vh.T  # (d, d) — eigenvectors as columns

    # Per-direction ridge: relative for observed directions, absolute for the
    # structurally unobserved tail (n - 1 < d).
    rank = max(min(int(n) - 1, d), 0)
    obs_floor = max(
        float(eps_rel) * float(eigenvalues[0]), float(eps_abs), _MIN_EIG
    )
    floor = torch.full(
        (d,), float(eps_null), dtype=eigenvalues.dtype, device=device
    )
    floor[:rank] = obs_floor
    clamped = eigenvalues < floor
    eigenvalues = torch.maximum(eigenvalues, floor)

    W = V * eigenvalues.pow(-0.5)  # z_white = (z - mean) @ W
    C = V * eigenvalues.pow(0.5)  # z = z_white @ C.T + mean

    return WhiteningOp(
        mean=mean.float(),
        W=W.float(),
        C=C.float(),
        clamped_fraction=float(clamped.float().mean()),
        cond=float(eigenvalues[0] / eigenvalues[-1].clamp(min=_MIN_EIG)),
    )


def fit_whitening(
    Z: torch.Tensor,
    eps_rel: float = 1e-4,
    eps_abs: float = 0.0,
    eps_null: float = 1e-2,
    shrinkage: float = 0.0,
) -> WhiteningOp:
    """Fit whitening/colouring operators from latent matrix Z (n, d).

    Thin wrapper over :func:`latent_moments` + :func:`whitening_from_moments`;
    see the latter for how the ridge (``eps_rel`` / ``eps_abs`` / ``eps_null``
    / ``shrinkage``) is applied.  Both stages run in float64, so routing through the covariance
    (rather than an SVD of the centred data matrix) does not cost accuracy
    even though it squares the condition number: the relative ridge caps that
    at ``1/eps_rel``, many orders below float64 resolution.
    """
    mean, cov, n = latent_moments(Z)
    return whitening_from_moments(
        mean,
        cov,
        n,
        eps_rel=eps_rel,
        eps_abs=eps_abs,
        eps_null=eps_null,
        shrinkage=shrinkage,
    )


def whiten(Z: torch.Tensor, op: WhiteningOp) -> torch.Tensor:
    dev = Z.device
    return (Z.float() - op.mean.to(dev)) @ op.W.to(dev)


def color(Z_white: torch.Tensor, op: WhiteningOp) -> torch.Tensor:
    dev = Z_white.device
    return Z_white.float() @ op.C.T.to(dev) + op.mean.to(dev)


# ── Learnable whitening (SWBN) ─────────────────────────────────────────────────


class SWBNWhiteningLayer(nn.Module):
    """Learnable ZCA-whitening layer (SWBN; Zhang et al., CVPR 2021).

    Owns the full whitening parameter tuple ``phi_i = (running_mean,
    running_var, W, gamma, beta)`` for a single agent's ``d``-dimensional latent
    space.  It is the parameterised whitening map ``g_{phi_i}`` of Section 5.2.

    The key SWBN idea is to **decouple** the whitening matrix from the task
    backward graph:

    * ``W`` (the ZCA whitening matrix, initialised to ``I``) is updated *online*
      in the forward pass by a single stochastic step on a whitening criterion,
      fully **detached** from autograd — exactly as in SWBN.
    * ``gamma`` / ``beta`` are ordinary leaf parameters trained by backprop.
    * ``running_mean`` / ``running_var`` are EMA buffers used at inference,
      analogously to BatchNorm.

    Convention (row vectors, batch dimension first).  For input ``Z`` of shape
    ``(K, d)``::

        Z_s = (Z - mu) / sqrt(v + eps)  # standardise (batch stats in train)
        Z_w = Z_s @ W.T  # whiten   (W symmetric -> ZCA)
        out = Z_w * gamma + beta  # affine rescale

    During training ``mu`` / ``v`` are the per-feature batch statistics (and the
    running buffers are updated from them); at inference the frozen running
    statistics are used.  The paired :class:`SWBNColouringLayer` reads this
    layer's ``phi_i`` and inverts these operations analytically, so it owns no
    parameters of its own.
    """

    def __init__(
        self,
        d: int,
        criterion: str = 'fro',
        alpha: float = 1e-5,
        momentum: float = 0.95,
        eps: float = 1e-8,
    ):
        super().__init__()
        if criterion not in ('fro', 'kl'):
            raise ValueError(
                f"Unknown whitening criterion {criterion!r}; expected 'fro' or 'kl'."
            )
        self.d = int(d)
        self.criterion = criterion
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.eps = float(eps)

        # Task parameters — trained by backprop.
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta = nn.Parameter(torch.zeros(d))
        # Whitening matrix + running statistics — updated outside autograd.
        self.register_buffer('W', torch.eye(d))
        self.register_buffer('running_mean', torch.zeros(d))
        self.register_buffer('running_var', torch.ones(d))

    @torch.no_grad()
    def _update_W(self, Z_s: torch.Tensor) -> None:
        """One detached stochastic step on the whitening criterion.

        ``Z_s`` is the (detached) standardised batch.  Updates ``self.W`` in
        place and re-symmetrises it (ZCA), all under ``no_grad`` so nothing
        enters the task backward graph.
        """
        n = Z_s.shape[0]
        Sigma = (Z_s.T @ Z_s) / max(
            n, 1
        )  # sample correlation, entries in [-1, 1]
        eye = torch.eye(self.d, device=Z_s.device, dtype=Z_s.dtype)
        WSWt = self.W @ Sigma @ self.W.T
        residual = WSWt - eye
        if self.criterion == 'kl':
            dW = residual @ self.W
        else:  # 'fro'
            dW = (residual @ self.W @ Sigma) / residual.norm().clamp(
                min=self.eps
            )
        W_new = self.W - self.alpha * dW
        # Enforce symmetry (a ZCA whitening matrix is symmetric) and *reassign*
        # the buffer rather than copy_ in place: the previous W tensor may be
        # saved in an autograd graph from an earlier forward on this same layer
        # in the same step (a node with >1 incident edge), and mutating it in
        # place would trigger a version-counter error at backward.
        W_sym = 0.5 * (W_new + W_new.T)
        # Never commit a non-finite update: the W buffer persists across steps,
        # so a single corrupted update would break all future whitening/colouring.
        # Keep the last good W instead (the standardised batch was already
        # sanitized in `forward`, so this is a final safety net).
        if torch.isfinite(W_sym).all():
            self.W = W_sym

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # Z: (K, d)
        Z = Z.float()
        # Sanitize non-finite latents *before* they touch the running stats or
        # the detached W update.  W is a persistent buffer carried across steps,
        # so a single Inf/NaN here would poison it permanently and break every
        # subsequent colouring solve.  Mirrors the guard in `fit_whitening`.
        if not torch.isfinite(Z).all():
            warnings.warn(
                f'SWBNWhiteningLayer: input contains '
                f'{(~torch.isfinite(Z)).sum().item()} non-finite entries; '
                'replacing with 0.0 — check for training instability.',
                RuntimeWarning,
                stacklevel=2,
            )
            Z = torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training and Z.shape[0] > 1:
            mu = Z.mean(0)
            v = Z.var(0, unbiased=True).clamp(min=0.0)
            self.running_mean.mul_(self.momentum).add_(
                (1.0 - self.momentum) * mu.detach()
            )
            self.running_var.mul_(self.momentum).add_(
                (1.0 - self.momentum) * v.detach()
            )
        else:
            mu, v = self.running_mean, self.running_var

        Z_s = (Z - mu) / (v + self.eps).sqrt()  # standardise (differentiable)
        if self.training and Z.shape[0] > 1:
            # W is refined from the *detached* standardised batch; the task loss
            # never sees this update (SWBN decoupling).
            self._update_W(Z_s.detach())
        # W is symmetric, so W.T == W; keep .T for an explicit right-multiply.
        return Z_s @ self.W.T * self.gamma + self.beta


class SWBNColouringLayer(nn.Module):
    """Closed-form inverse (colouring map ``g*_{phi_i}``) of a whitening layer.

    Owns **no parameters of its own**: it reads ``phi_i`` directly from the
    paired :class:`SWBNWhiteningLayer` and inverts each whitening step
    analytically, guaranteeing ``g*_{phi_i} ∘ g_{phi_i} = id`` whenever both use
    the same statistics (exactly at inference; approximately during training,
    where the forward pass standardises with batch statistics while colouring
    uses the running statistics — the standard SWBN/BatchNorm trade-off).

    The whitening layer is stored as a *non-registered* reference so its
    parameters/buffers are not double-counted in this module's ``parameters()``
    or ``state_dict()``.
    """

    def __init__(self, whitening_layer: SWBNWhiteningLayer):
        super().__init__()
        # Bypass nn.Module.__setattr__ so the shared whitening layer is NOT
        # registered as a submodule here (it already lives on the orchestrator).
        object.__setattr__(self, 'W_layer', whitening_layer)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:  # Z: (K, d)
        wl: SWBNWhiteningLayer = self.W_layer
        Z = Z.float()
        # 1. invert affine rescale:  out = Z_w * gamma + beta  ->  Z_w = (out - beta) / gamma
        Z = (Z - wl.beta) / wl.gamma
        # 2. invert whitening:  Z_w = Z_s @ W.T  ->  Z_s = solve(W.T, Z_w.T).T
        #    W is symmetric, so W.T == W; solve avoids forming W^{-1} explicitly.
        W = wl.W.to(Z.dtype)
        if not torch.isfinite(W).all():
            # A non-finite W (upstream divergence) would make the solve raise or
            # silently emit NaN.  Degrade gracefully to identity de-whitening for
            # this call rather than crashing/poisoning the step.
            warnings.warn(
                'SWBNColouringLayer: whitening matrix W is non-finite; '
                'skipping de-whitening (identity fallback) — check training '
                'stability.',
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            try:
                Z = torch.linalg.solve(W.T, Z.T).T
            except RuntimeError:
                # Near-singular W (should not happen near identity): ridge, then
                # fall back to a pseudo-inverse if even the ridged solve fails.
                ridge = wl.eps * torch.eye(
                    wl.d, device=W.device, dtype=W.dtype
                )
                try:
                    Z = torch.linalg.solve(W.T + ridge, Z.T).T
                except RuntimeError:
                    Z = Z @ torch.linalg.pinv(W.T)
        # 3. invert standardisation:  Z_s = (z - mu) / sqrt(v + eps)
        #    ->  z = Z_s * sqrt(v + eps) + mu        (running stats, as in SWBN)
        return Z * (wl.running_var + wl.eps).sqrt() + wl.running_mean


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


def fit_procrustes(
    X_i: torch.Tensor,
    X_j: torch.Tensor,
) -> torch.Tensor:
    """Learn orthogonal/semi-orthogonal V s.t. X_i @ V ≈ X_j (Procrustes).

    Solves the orthogonal Procrustes problem via SVD of the cross-covariance:
    V = U W^T where U, W come from SVD(X_i^T X_j).  When d_i ≠ d_j the map
    is semi-orthogonal (V^T V = I).

    Both X_i and X_j are (n, d) whitened pilot representations.

    Returns V on CPU, shape (d_i, d_j).
    """
    X_i = X_i.float()
    X_j = X_j.float()
    C = X_i.T @ X_j
    C = C + torch.randn_like(C) * 1e-6
    try:
        U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
    except RuntimeError:
        C = C.cpu()
        U, _S, W_T = torch.linalg.svd(C, full_matrices=False)
    return (U @ W_T).cpu()


def fit_relative_alignment(
    A_i: torch.Tensor,
    A_j: torch.Tensor,
) -> torch.Tensor:
    """Zero-shot map M (d_i, d_j) s.t. Z_i @ M ≈ Z_j via relative representations.

    ``A_i``/``A_j`` are (K, d_i)/(K, d_j) whitened anchor matrices encoding
    the *same* K pilot samples on the sender/receiver side. Unlike
    :func:`fit_alignment`/:func:`fit_procrustes`, nothing is regressed — the
    anchors themselves are the operators (Fiorellino et al. 2025,
    arXiv:2507.17835): the sender *analyses* its latent against its own
    anchors, ``r = z @ A_i.T`` (the relative representation, invariant across
    the two whitened spaces up to the approximately orthogonal transform
    relating them), and the receiver *synthesises* from those shared
    coordinates with the canonical dual of its own anchor frame,
    ``z_hat = r @ pinv(A_j).T``.  The composition is the single right-multiply
    map

        M = A_i.T @ pinv(A_j).T

    which exists as soon as the anchors are exchanged.  With K < d_j the
    synthesis reconstructs only within span(A_j) — the paper's compression
    regime, not a failure mode.

    Returns M on CPU.
    """
    A_i = A_i.float()
    A_j = A_j.float()
    return (A_i.T @ torch.linalg.pinv(A_j).T).cpu()


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
