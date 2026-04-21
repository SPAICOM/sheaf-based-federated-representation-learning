"""Tests for src.utils.anchors."""

import torch

from src.utils.anchors import AnchorConfig, shared_anchor_rows


class TestSharedAnchorRows:
    """Tests for cross-agent anchor matching."""

    def test_prototype_matching_uses_both_label_sets(self):
        """Prototype alignment should match by shared class labels."""
        A_i = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
        labels_i = torch.tensor([0, 1, 1])
        A_j = torch.tensor([[10.0, 0.0], [20.0, 0.0], [40.0, 0.0]])
        labels_j = torch.tensor([1, 2, 1])
        config = AnchorConfig(
            parseval_normalization=False,
            l2_normalization=False,
            filter_unseen_classes=False,
            use_prototypes=True,
        )

        shared_rows = shared_anchor_rows(
            A_i=A_i,
            A_j=A_j,
            labels_i=labels_i,
            labels_j=labels_j,
            seen_i={0, 1},
            seen_j={1, 2},
            config=config,
        )

        assert shared_rows is not None
        matched_i, matched_j = shared_rows
        assert torch.allclose(matched_i, torch.tensor([[4.0, 0.0]]))
        assert torch.allclose(matched_j, torch.tensor([[25.0, 0.0]]))

    def test_filter_unseen_classes_uses_seen_and_batch_intersection(self):
        """Filtering should restrict alignment to mutually seen classes."""
        A_i = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        labels_i = torch.tensor([0, 1])
        A_j = torch.tensor([[10.0, 0.0], [20.0, 0.0]])
        labels_j = torch.tensor([1, 2])
        config = AnchorConfig(
            parseval_normalization=False,
            l2_normalization=False,
            filter_unseen_classes=True,
            use_prototypes=True,
        )

        shared_rows = shared_anchor_rows(
            A_i=A_i,
            A_j=A_j,
            labels_i=labels_i,
            labels_j=labels_j,
            seen_i={0, 1},
            seen_j={0, 1, 2},
            config=config,
        )

        assert shared_rows is not None
        matched_i, matched_j = shared_rows
        assert torch.equal(matched_i, torch.tensor([[2.0, 0.0]]))
        assert torch.equal(matched_j, torch.tensor([[10.0, 0.0]]))

    def test_raw_row_matching_requires_aligned_selected_labels(self):
        """Non-prototype matching should only keep already aligned rows."""
        A_i = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        labels_i = torch.tensor([0, 1])
        A_j = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
        labels_j = torch.tensor([1, 0])
        config = AnchorConfig(
            parseval_normalization=False,
            l2_normalization=False,
            filter_unseen_classes=False,
            use_prototypes=False,
        )

        shared_rows = shared_anchor_rows(
            A_i=A_i,
            A_j=A_j,
            labels_i=labels_i,
            labels_j=labels_j,
            seen_i={0, 1},
            seen_j={0, 1},
            config=config,
        )

        assert shared_rows is None
