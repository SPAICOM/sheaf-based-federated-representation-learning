"""Tests for src.utils.data_partitioner."""

from collections import Counter

import pytest
import torch

from src.utils.data_partitioner import (
    _generate_skewed_proportions,
)


class TestGenerateSkewedProportions:
    """Tests for the alpha router helper."""

    def test_positive_alpha_returns_dirichlet_like_probabilities(self):
        probs = _generate_skewed_proportions(
            n_parts=4,
            alpha=0.5,
            generator=torch.Generator().manual_seed(7),
        )

        assert probs.shape == (4,)
        assert torch.all(probs > 0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))

    def test_negative_alpha_returns_lognormal_probabilities(self):
        probs = _generate_skewed_proportions(
            n_parts=4,
            alpha=-1.0,
            generator=torch.Generator().manual_seed(7),
        )

        assert probs.shape == (4,)
        assert torch.all(probs > 0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
