"""Tests for src.utils.graph_generator."""

import pytest

from src.utils.graph_generator import generate_neighbors


class TestGenerateNeighbors:
    """Tests for generate_neighbors function."""

    def test_manual_mode(self):
        """Test manual neighbor dictionary."""
        neighbors = generate_neighbors(
            mode='manual', n_agents=3, manual={0: [1, 2], 1: [0], 2: [0]}
        )
        assert neighbors == {0: {1, 2}, 1: {0}, 2: {0}}

    def test_fully_connected(self):
        """Test fully connected graph."""
        neighbors = generate_neighbors(mode='fully_connected', n_agents=4)
        for i in range(4):
            assert len(neighbors[i]) == 3

    def test_erdos_renyi(self):
        """Test Erdos-Renyi random graph."""
        neighbors = generate_neighbors(
            mode='erdos_renyi', n_agents=10, p=0.5, seed=42
        )
        assert len(neighbors) == 10

    def test_barabasi_albert(self):
        """Test Barabasi-Albert scale-free graph."""
        neighbors = generate_neighbors(
            mode='barabasi', n_agents=10, m=2, seed=42
        )
        assert len(neighbors) == 10

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            generate_neighbors(mode='invalid', n_agents=5)


class TestClassCover:
    """Edge iff the two agents' classes cover all classes."""

    # 5-agent ring 0-1-2-3-4-0: each agent misses 3 of 10 classes; neighbours
    # miss disjoint sets, non-neighbours share one missing class.
    MISSING = {0: {0, 3, 5}, 1: {1, 4, 6}, 2: {0, 2, 7}, 3: {1, 3, 8}, 4: {2, 4, 9}}

    def test_ring(self):
        tc = {i: set(range(10)) - m for i, m in self.MISSING.items()}
        nb = generate_neighbors(mode='class_cover', n_agents=5, target_classes=tc)
        assert nb == {i: {(i - 1) % 5, (i + 1) % 5} for i in range(5)}

    def test_disconnected_raises(self):
        tc = {0: {0, 1}, 1: {2, 3}, 2: {0, 2}, 3: {1, 3}}  # edges 0-1, 2-3 only
        with pytest.raises(ValueError, match='disconnected'):
            generate_neighbors(mode='class_cover', n_agents=4, target_classes=tc)
