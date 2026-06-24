"""Discrete (cluster-then-CVX) PID estimator.

Implements the first estimator of Liang et al. (2023).  The continuous
representations ``Z1`` and ``Z2`` are first turned into discrete codes
via K-Means clustering, after which the empirical joint distribution
``P(C1, C2, Y)`` is built.  The Bertschinger–Rauh–Olbrich operational
PID is then computed by solving the convex programme

    Q* = argmin_{Q : Q(C1,Y)=P(C1,Y), Q(C2,Y)=P(C2,Y)}  D_KL(Q || Q̃)

where ``Q̃(c1,c2,y) = (1/|Y|) · Σ_y' Q(c1,c2,y')`` is the product-
factorised reference.  The four PID components are then read off from
``P`` and ``Q*`` as

    R  = CoI(Q*)            (redundancy)
    U1 = I_{Q*}(Y; C1 | C2) (unique to view 1)
    U2 = I_{Q*}(Y; C2 | C1) (unique to view 2)
    S  = I_P(Y; C1, C2) − I_{Q*}(Y; C1, C2)   (synergy)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mutualinfo._common import (
    PIDResult,
    coinfo,
    conditional_mi,
    joint_pmf_from_codes,
    kmeans_cluster,
    mi_2d,
    standardize,
    to_numpy,
)


def _solve_q_marginal_preserving(p: np.ndarray) -> np.ndarray:
    """Solve the Bertschinger convex programme via CVXPY.

    Parameters
    ----------
    p : ndarray, shape (|X1|, |X2|, |Y|)
        Empirical joint pmf.

    Returns
    -------
    q : ndarray, shape (|X1|, |X2|, |Y|)
        The marginal-preserving distribution closest (in KL) to the
        ``Y``-product mixture.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            'cvxpy is required for cvxpy_pid. Install it with '
            "'uv add cvxpy' or 'pip install cvxpy'."
        ) from exc

    n_x1, n_x2, n_y = p.shape
    p_x1y = p.sum(axis=1)
    p_x2y = p.sum(axis=0)

    # Per-Y slices of Q and the corresponding mixture.
    q_slices = [cp.Variable((n_x1, n_x2), nonneg=True) for _ in range(n_y)]
    q_mix = [cp.Variable((n_x1, n_x2), nonneg=True) for _ in range(n_y)]

    constraints: list[Any] = []
    # Total mass.
    constraints.append(cp.sum([cp.sum(qy) for qy in q_slices]) == 1)

    # Marginal-preserving constraints: Q(X1,Y)=P(X1,Y) and Q(X2,Y)=P(X2,Y).
    constraints.extend(
        cp.sum([q_slices[y][x1, x2] for x2 in range(n_x2)]) == p_x1y[x1, y]
        for x1 in range(n_x1)
        for y in range(n_y)
    )
    constraints.extend(
        cp.sum([q_slices[y][x1, x2] for x1 in range(n_x1)]) == p_x2y[x2, y]
        for x2 in range(n_x2)
        for y in range(n_y)
    )

    mixture = cp.sum(q_slices) / n_y
    constraints += [mixture == q_mix[i] for i in range(n_y)]

    objective = cp.sum(
        [cp.sum(cp.rel_entr(q_slices[i], q_mix[i])) for i in range(n_y)]
    )
    problem = cp.Problem(cp.Minimize(objective), constraints)
    try:
        problem.solve(verbose=False, max_iters=10000)
    except (cp.error.SolverError, Exception):
        problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(
            f'CVXPY problem did not solve to optimality: {problem.status}'
        )

    q = np.stack([qy.value for qy in q_slices], axis=2)
    q = np.clip(q, 0.0, None)
    q /= q.sum()
    return q


def cvxpy_pid(
    z1,
    z2,
    y,
    n_clusters: int = 10,
    standardize_features: bool = True,
    seed: int = 0,
    max_samples: int | None = None,
) -> PIDResult:
    """Cluster-then-CVX PID estimator.

    Parameters
    ----------
    z1, z2 : array-like, shape (N, d1) / (N, d2)
        Continuous representations of the two views.
    y : array-like, shape (N,)
        Discrete labels (will be remapped to ``{0, …, |Y|−1}``).
    n_clusters : int
        Number of K-Means clusters per view.  Joint table size grows
        as ``n_clusters² · |Y|``; values above ~16 quickly become
        infeasible for CVXPY.
    standardize_features : bool
        Z-score each feature before clustering.
    seed : int
        K-Means / RNG seed.
    max_samples : int or None
        If set, subsample this many points to keep clustering fast.

    Returns
    -------
    PIDResult
    """
    z1 = to_numpy(z1).astype(np.float64)
    z2 = to_numpy(z2).astype(np.float64)
    y = to_numpy(y).astype(np.int64).reshape(-1)

    if z1.ndim == 1:
        z1 = z1[:, None]
    if z2.ndim == 1:
        z2 = z2[:, None]

    if not (len(z1) == len(z2) == len(y)):
        raise ValueError(
            'z1, z2, y must share the leading dimension '
            f'(got {len(z1)}, {len(z2)}, {len(y)}).'
        )

    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), size=int(max_samples), replace=False)
        z1, z2, y = z1[idx], z2[idx], y[idx]

    if standardize_features:
        z1 = standardize(z1)
        z2 = standardize(z2)

    # Remap labels to a contiguous integer range so the pmf has no holes.
    _, y_compact = np.unique(y, return_inverse=True)

    c1 = kmeans_cluster(z1, n_clusters=n_clusters, seed=seed)
    c2 = kmeans_cluster(z2, n_clusters=n_clusters, seed=seed + 1)

    p = joint_pmf_from_codes(c1, c2, y_compact.astype(np.int64))
    q = _solve_q_marginal_preserving(p)

    # I_P(Y; C1, C2) and I_P(Y; C1), I_P(Y; C2).
    p_x12y = p.transpose(2, 0, 1).reshape(p.shape[2], -1).T
    mi_y_z1z2 = mi_2d(p_x12y)
    mi_y_z1 = mi_2d(p.sum(axis=1))
    mi_y_z2 = mi_2d(p.sum(axis=0))

    redundancy = coinfo(q)
    unique1 = conditional_mi(q, condition_on=1)
    unique2 = conditional_mi(q, condition_on=0)
    q_x12y = q.transpose(2, 0, 1).reshape(q.shape[2], -1).T
    synergy = mi_y_z1z2 - mi_2d(q_x12y)

    # Numerical floor — KL-based optimisations can produce tiny negative drift.
    redundancy = max(redundancy, 0.0)
    unique1 = max(unique1, 0.0)
    unique2 = max(unique2, 0.0)
    synergy = max(synergy, 0.0)

    return PIDResult(
        redundancy=redundancy,
        unique1=unique1,
        unique2=unique2,
        synergy=synergy,
        mi_y_z1=mi_y_z1,
        mi_y_z2=mi_y_z2,
        mi_y_z1z2=mi_y_z1z2,
        method='cvxpy',
    )
