"""Tests for src.utils.anchors."""

import torch

from src.utils.anchors import shared_anchor_rows


class TestSharedAnchorRows:
    """Tests for cross-agent anchor matching."""

    def test_exact_matching_uses_full_semantic_key(self):
        """Exact matching should keep only identical semantic keys."""
        A_i = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        A_j = torch.tensor([[3.0, 0.0], [4.0, 0.0]])

        shared_rows = shared_anchor_rows(
            A_i=A_i,
            keys_i=[(0, 0), (1, 0)],
            A_j=A_j,
            keys_j=[(1, 0), (2, 0)],
            match_by='exact',
        )

        assert shared_rows is not None
        matched_i, matched_j = shared_rows
        assert torch.equal(matched_i, torch.tensor([[2.0, 0.0]]))
        assert torch.equal(matched_j, torch.tensor([[3.0, 0.0]]))

    def test_class_matching_ignores_local_slot_ids(self):
        """Class-based matching should not assume slot ids are globally shared."""
        A_i = torch.tensor(
            [
                [0.0, 0.0],
                [10.0, 0.0],
            ]
        )
        A_j = torch.tensor(
            [
                [10.2, 0.0],
                [0.1, 0.0],
            ]
        )

        shared_rows = shared_anchor_rows(
            A_i=A_i,
            keys_i=[(7, 0), (7, 1)],
            A_j=A_j,
            keys_j=[(7, 0), (7, 1)],
            match_by='class',
            projected_A_j=A_j,
        )

        assert shared_rows is not None
        matched_i, matched_j = shared_rows
        assert torch.allclose(matched_i, A_i)
        assert torch.allclose(matched_j, A_j.flip(0))
