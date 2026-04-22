"""Tests for src.datamodules.utils."""

from collections import Counter

from src.datamodules.utils import assign_agents_to_random_groups


class TestAssignAgentsToRandomGroups:
    """Tests for deterministic seeded group assignment."""

    def test_assignment_is_deterministic_for_same_seed(self):
        first = assign_agents_to_random_groups(
            n_agents=40,
            group_values=[0, 90, 180, 270],
            seed=42,
        )
        second = assign_agents_to_random_groups(
            n_agents=40,
            group_values=[0, 90, 180, 270],
            seed=42,
        )

        assert first == second

    def test_assignment_is_even_when_agents_divide_groups(self):
        assignments = assign_agents_to_random_groups(
            n_agents=40,
            group_values=[0, 90, 180, 270],
            seed=42,
        )

        assert sorted(assignments.keys()) == list(range(40))
        counts = Counter(assignments.values())
        assert counts == Counter({0: 10, 90: 10, 180: 10, 270: 10})

    def test_assignment_balances_remainder_across_first_groups(self):
        assignments = assign_agents_to_random_groups(
            n_agents=10,
            group_values=[0, 90, 180],
            seed=7,
        )

        counts = Counter(assignments.values())
        assert counts == Counter({0: 4, 90: 3, 180: 3})
