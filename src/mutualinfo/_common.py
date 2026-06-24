"""Shared utilities for the PID estimators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class PIDResult:
    """Container for PID outputs.

    Fields
    ------
    redundancy, unique1, unique2, synergy : float
        The four non-negative PID components (in nats).
    mi_y_z1, mi_y_z2, mi_y_z1z2 : float
        Marginal / joint mutual informations between ``Y`` and the
        views, returned for sanity checks (``R + U1 + U2 + S`` should
        equal ``I(Z1, Z2; Y)``).
    method : str
        Which estimator produced the result.
    """

    redundancy: float
    unique1: float
    unique2: float
    synergy: float
    mi_y_z1: float
    mi_y_z2: float
    mi_y_z1z2: float
    method: str

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d.pop('method', None)
        return d

    def total_information(self) -> float:
        return self.redundancy + self.unique1 + self.unique2 + self.synergy


def to_numpy(x) -> np.ndarray:
    """Convert tensor/list/ndarray to a contiguous ``float32`` numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def standardize(x: np.ndarray) -> np.ndarray:
    """Per-feature zero-mean / unit-variance scaling."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd


def kmeans_cluster(
    x: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    max_iter: int = 100,
) -> np.ndarray:
    """Return cluster labels of ``x`` (shape ``(N,)``, dtype ``int64``).

    Uses ``sklearn.cluster.MiniBatchKMeans`` when available; falls back
    to a NumPy implementation otherwise.  Single-feature inputs are
    binned via quantiles, which is more stable than K-Means in 1-D.
    """
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    n_clusters = max(2, min(int(n_clusters), n))

    if x.shape[1] == 1:
        flat = x[:, 0]
        uniq = np.unique(flat)
        if len(uniq) <= n_clusters:
            # Few distinct values → map them directly to {0, …, K'-1}.
            idx_map = {float(v): i for i, v in enumerate(uniq.tolist())}
            return np.array(
                [idx_map[float(v)] for v in flat.tolist()], dtype=np.int64
            )
        # Quantile binning preserves order and avoids degenerate K-Means.
        # ``side='right'`` ensures values that fall on an edge land in the
        # next bin rather than collapsing the leftmost bin.
        qs = np.linspace(0.0, 1.0, n_clusters + 1)[1:-1]
        edges = np.quantile(flat, qs)
        labels = np.searchsorted(edges, flat, side='right')
        return labels.astype(np.int64)

    try:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=int(seed),
            batch_size=max(256, n_clusters * 4),
            n_init=3,
            max_iter=max_iter,
        )
        return km.fit_predict(x).astype(np.int64)
    except ImportError:
        return _numpy_kmeans(x, n_clusters, seed=seed, max_iter=max_iter)


def _numpy_kmeans(
    x: np.ndarray, k: int, *, seed: int = 0, max_iter: int = 100
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centers[c] = x[mask].mean(axis=0)
            else:
                centers[c] = x[rng.integers(n)]
    return labels


def joint_pmf_from_codes(
    code1: np.ndarray, code2: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Build the empirical 3-way pmf ``P[x1, x2, y]`` from integer codes."""
    n_x1 = int(code1.max()) + 1
    n_x2 = int(code2.max()) + 1
    n_y = int(y.max()) + 1
    p = np.zeros((n_x1, n_x2, n_y), dtype=np.float64)
    for a, b, c in zip(code1, code2, y, strict=True):
        p[int(a), int(b), int(c)] += 1.0
    s = p.sum()
    if s <= 0:
        raise ValueError('Empty joint distribution.')
    return p / s


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-30, None))


def mi_2d(p_xy: np.ndarray) -> float:
    """Mutual information of a 2-D joint pmf, in nats."""
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    outer = p_x @ p_y
    mask = p_xy > 0
    return float(np.sum(p_xy[mask] * (_safe_log(p_xy[mask]) - _safe_log(outer[mask]))))


def coinfo(p_xyz: np.ndarray) -> float:
    """Co-information ``I(Y;X1) + I(Y;X2) - I(Y;X1,X2)``."""
    p_x1y = p_xyz.sum(axis=1)
    p_x2y = p_xyz.sum(axis=0)
    p_x12y = p_xyz.transpose(2, 0, 1).reshape(
        p_xyz.shape[2], p_xyz.shape[0] * p_xyz.shape[1]
    ).T
    return mi_2d(p_x1y) + mi_2d(p_x2y) - mi_2d(p_x12y)


def conditional_mi(p_xyz: np.ndarray, condition_on: int) -> float:
    """``I(Y; X_a | X_b)`` where ``condition_on`` is the index of ``X_b``."""
    if condition_on == 0:
        marg = p_xyz.sum(axis=(1, 2))
        slabs = [p_xyz[i, :, :] for i in range(p_xyz.shape[0])]
    elif condition_on == 1:
        marg = p_xyz.sum(axis=(0, 2))
        slabs = [p_xyz[:, j, :] for j in range(p_xyz.shape[1])]
    else:
        raise ValueError('condition_on must be 0 (X1) or 1 (X2).')

    total = 0.0
    for w, slab in zip(marg, slabs, strict=True):
        s = slab.sum()
        if s > 0:
            total += float(w * mi_2d(slab / s))
    return total
