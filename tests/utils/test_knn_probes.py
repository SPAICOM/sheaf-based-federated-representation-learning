"""Tests for the cosine k-NN probes."""

import numpy as np
import pytest

from src.utils.knn_probes import (
    few_label,
    knn_probe,
    stratified_halves,
    within_parity,
)


def _clusters(n_per_class=20, d=16, noise=0.05, seed=0):
    """10 well separated classes on the sphere (orthogonal centres)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(10), n_per_class)
    centres = np.eye(d)[:10]
    Z = centres[y] + noise * rng.normal(size=(len(y), d))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True), y


def test_stratified_halves_are_disjoint_balanced_and_cover():
    """Halves are disjoint, class-balanced, exhaustive and reproducible."""
    _, y = _clusters()
    a, b = stratified_halves(y)
    assert set(a).isdisjoint(b)
    assert len(a) + len(b) == len(y)
    assert np.array_equal(np.bincount(y[a]), np.bincount(y[b]))
    a2, b2 = stratified_halves(y)
    assert np.array_equal(a, a2)
    assert np.array_equal(b, b2)


def test_knn_probe_separable_vs_noise():
    """Separable clusters score 1, pure noise stays near chance."""
    Z, y = _clusters()
    acc = knn_probe(Z, y, [1, 5])
    assert acc[1] == pytest.approx(1.0)
    assert acc[5] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    noise = rng.normal(size=Z.shape)
    assert knn_probe(noise, y, [5])[5] < 0.3


def test_within_parity_separable_and_chance():
    """Within-parity probe: 1 on clusters, near 0.2 on noise."""
    Z, y = _clusters()
    assert within_parity(Z, y, [3])[3] == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    acc = within_parity(rng.normal(size=Z.shape), y, [5])[5]
    assert acc < 0.5  # chance is 0.2


def test_few_label_skips_k_above_n_and_is_deterministic():
    """Pairs with k > n are skipped and the draws are seeded."""
    Z, y = _clusters()
    out = few_label(Z, y, ns=[1, 5], ks=[1, 5], repeats=2)
    assert set(out) == {(1, 1), (5, 1), (5, 5)}
    assert out[(5, 5)][0] > 0.9
    assert out == few_label(Z, y, ns=[1, 5], ks=[1, 5], repeats=2)
