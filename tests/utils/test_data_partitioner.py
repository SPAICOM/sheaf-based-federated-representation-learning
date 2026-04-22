"""Tests for src.utils.data_partitioner."""

from collections import Counter

import pytest
import torch

from src.utils.data_partitioner import (
    _generate_skewed_proportions,
    partition_non_iid_fair,
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


class TestPartitionNonIddFair:
    """Tests for the fair per-agent-skew non-IID partitioner."""

    def test_enforces_uniform_volume_and_exact_k_classes(self):
        labels = [class_label for class_label in range(10) for _ in range(60)]

        partition, agent_classes = partition_non_iid_fair(
            labels=labels,
            n_agents=6,
            classes_per_agent=5,
            seed=13,
            alpha=-1.0,
            return_agent_classes=True,
            safety_margin=5,
        )

        flattened = [idx for indices in partition.values() for idx in indices]
        assert sorted(flattened) == list(range(len(labels)))
        assert {len(indices) for indices in partition.values()} == {100}
        assert all(len(classes) == 5 for classes in agent_classes.values())
        assert set().union(*agent_classes.values()) == set(range(10))

        for agent_id, indices in partition.items():
            local_counts = Counter(labels[index] for index in indices)
            assert set(local_counts).issubset(set(agent_classes[agent_id]))
            for class_label in agent_classes[agent_id]:
                assert local_counts[class_label] >= 5

    def test_all_classes_per_agent_keeps_every_label_local(self):
        labels = [class_label for class_label in range(10) for _ in range(100)]

        partition, agent_classes = partition_non_iid_fair(
            labels=labels,
            n_agents=5,
            classes_per_agent=10,
            seed=5,
            alpha=0.3,
            return_agent_classes=True,
            safety_margin=5,
        )

        assert {len(indices) for indices in partition.values()} == {200}
        for agent_id, indices in partition.items():
            local_labels = {labels[index] for index in indices}
            assert set(agent_classes[agent_id]) == set(range(10))
            assert local_labels == set(range(10))

    def test_requires_divisible_total_sample_count(self):
        labels = [class_label for class_label in range(3) for _ in range(5)]

        with pytest.raises(ValueError, match='divisible by n_agents'):
            partition_non_iid_fair(
                labels=labels,
                n_agents=4,
                classes_per_agent=2,
                seed=3,
                alpha=0.5,
                safety_margin=0,
            )

    def test_negative_alpha_returns_lognormal_probabilities(self):
        probs = _generate_skewed_proportions(
            n_parts=4,
            alpha=-1.0,
            generator=torch.Generator().manual_seed(7),
        )

        assert probs.shape == (4,)
        assert torch.all(probs > 0)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
