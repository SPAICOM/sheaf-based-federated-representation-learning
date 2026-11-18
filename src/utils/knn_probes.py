"""Cosine k-NN probes of the label information stored in a latent space.

CenterNorm latents lie on the unit sphere, so every probe uses the cosine
distance. All probes score on the test split cut into two stratified halves:
fit on one half, score on the other, swap, and pool the two scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def _knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)


def stratified_halves(
    y: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split the rows in two halves with the same class proportions.

    Parameters
    ----------
    y : np.ndarray
        Integer labels, shape ``(n,)``.
    seed : int, optional
        Seed of the shuffle inside each class; the default makes the split
        identical for every run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Disjoint row indices ``(a, b)`` covering all rows.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a.append(idx[: len(idx) // 2])
        b.append(idx[len(idx) // 2 :])
    return np.concatenate(a), np.concatenate(b)


def knn_probe(Z: np.ndarray, y: np.ndarray, ks: list[int]) -> dict[int, float]:
    """Two-fold k-NN accuracy of ``y`` from the latents ``Z``.

    Parameters
    ----------
    Z : np.ndarray
        Latents, shape ``(n, d)``.
    y : np.ndarray
        Integer labels, shape ``(n,)``.
    ks : list[int]
        Numbers of neighbours.

    Returns
    -------
    dict[int, float]
        Accuracy (mean of the two folds) for each ``k``.
    """
    a, b = stratified_halves(y)
    out = {}
    for k in ks:
        clf = _knn(k)
        scores = [
            clf.fit(Z[f], y[f]).score(Z[s], y[s]) for f, s in ((a, b), (b, a))
        ]
        out[k] = float(np.mean(scores))
    return out


def within_parity(
    Z: np.ndarray, y: np.ndarray, ks: list[int]
) -> dict[int, float]:
    """k-NN accuracy of the digit among images of the same parity.

    Separates the even digits among the even rows and the odd digits among the
    odd rows (chance 0.2 for 10 digits). The score pools both parity groups
    and both folds.

    Parameters
    ----------
    Z : np.ndarray
        Latents, shape ``(n, d)``.
    y : np.ndarray
        Digit labels in ``0..9``, shape ``(n,)``.
    ks : list[int]
        Numbers of neighbours.

    Returns
    -------
    dict[int, float]
        Pooled accuracy for each ``k``.
    """
    hits = dict.fromkeys(ks, 0)
    total = 0
    for p in (0, 1):
        m = y % 2 == p
        Zp, yp = Z[m], y[m]
        a, b = stratified_halves(yp)
        for f, s in ((a, b), (b, a)):
            total += len(s)
            for k in ks:
                pred = _knn(k).fit(Zp[f], yp[f]).predict(Zp[s])
                hits[k] += int((pred == yp[s]).sum())
    return {k: hits[k] / total for k in ks}


def few_label(
    Z: np.ndarray,
    y: np.ndarray,
    ns: list[int],
    ks: list[int],
    repeats: int,
    seed: int = 0,
) -> dict[tuple[int, int], tuple[float, float]]:
    """k-NN accuracy with only ``n`` labelled references per class.

    References are drawn from one half of the rows and scored on the other
    (then swapped), ``repeats`` random draws each. Pairs with ``k > n`` are
    skipped: the vote degenerates when there are fewer references per class
    than neighbours.

    Parameters
    ----------
    Z : np.ndarray
        Latents, shape ``(n_rows, d)``.
    y : np.ndarray
        Integer labels, shape ``(n_rows,)``.
    ns : list[int]
        References per class.
    ks : list[int]
        Numbers of neighbours.
    repeats : int
        Random reference draws per ``n`` and fold.
    seed : int, optional
        Seed of the reference draws (the same for every run by default).

    Returns
    -------
    dict[tuple[int, int], tuple[float, float]]
        ``(n, k) -> (mean, std)`` of the accuracy over folds and draws.
    """
    a, b = stratified_halves(y)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    scores: dict[tuple[int, int], list[float]] = {}
    for f, s in ((a, b), (b, a)):
        by_class = {c: f[y[f] == c] for c in classes}
        for n in ns:
            for _ in range(repeats):
                ref = np.concatenate(
                    [
                        rng.choice(by_class[c], n, replace=False)
                        for c in classes
                    ]
                )
                for k in ks:
                    if k > n:
                        continue
                    pred = _knn(k).fit(Z[ref], y[ref]).predict(Z[s])
                    scores.setdefault((n, k), []).append(
                        float((pred == y[s]).mean())
                    )
    return {
        key: (float(np.mean(v)), float(np.std(v))) for key, v in scores.items()
    }
