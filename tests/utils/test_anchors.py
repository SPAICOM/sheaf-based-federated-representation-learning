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


class TestStratifiedSelection:
    """``stratified`` raw-row selection biased towards the focus classes."""

    @staticmethod
    def _labels() -> torch.Tensor:
        return torch.arange(10).repeat_interleave(50)  # 500 rows, 50 / class

    def test_focus_share_and_equal_per_class(self):
        """Focus classes get round(bias*K) rows, equally split."""
        from src.utils.anchors import stratified_row_indices

        labels = self._labels()
        idx = stratified_row_indices(labels, 100, {2, 7}, 0.7, seed=3)
        assert idx.numel() == 100 and idx.unique().numel() == 100
        picked = labels[idx]
        assert int(torch.isin(picked, torch.tensor([2, 7])).sum()) == 70
        assert int((picked == 2).sum()) == int((picked == 7).sum()) == 35

    def test_deterministic_and_order_independent(self):
        """Same inputs give the same rows; row order does not matter."""
        from src.utils.anchors import stratified_row_indices

        labels = self._labels()
        keys = torch.arange(500) * 3 + 1
        a = stratified_row_indices(
            labels, 64, {0, 1, 9}, 0.7, keys=keys, seed=5
        )
        b = stratified_row_indices(
            labels, 64, {0, 1, 9}, 0.7, keys=keys, seed=5
        )
        assert torch.equal(a, b)
        perm = torch.randperm(500)
        c = stratified_row_indices(
            labels[perm], 64, {0, 1, 9}, 0.7, keys=keys[perm], seed=5
        )
        assert torch.equal(keys[a].sort().values, keys[perm][c].sort().values)
        d = stratified_row_indices(
            labels, 64, {0, 1, 9}, 0.7, keys=keys, seed=6
        )
        assert not torch.equal(a, d)

    def test_shortfalls_are_filled_to_k(self):
        """Missing focus rows are replaced so K rows come out."""
        from src.utils.anchors import stratified_row_indices

        labels = torch.tensor([0] * 5 + [1] * 60)
        # focus class 0 has only 5 rows: its share cannot be met, K is still met.
        idx = stratified_row_indices(labels, 20, {0}, 0.7, seed=0)
        assert idx.numel() == 20
        assert int((labels[idx] == 0).sum()) == 5
        # no focus classes / K >= n degrade gracefully
        assert (
            stratified_row_indices(labels, 20, None, 0.7, seed=0).numel() == 20
        )
        assert (
            stratified_row_indices(labels, 99, {0}, 0.7, seed=0).numel() == 65
        )

    def test_select_paired_anchors_keeps_pairs(self):
        """Both sides keep the same row positions."""
        from src.utils.anchors import select_paired_anchors

        labels = self._labels()
        X_i = torch.randn(500, 3)
        X_j = X_i * 2
        A_i, A_j = select_paired_anchors(
            X_i,
            X_j,
            labels=labels,
            selection='stratified',
            num_anchors=32,
            parseval=False,
            focus_classes={4},
            seed=1,
        )
        assert A_i.shape == (32, 3)
        assert torch.allclose(A_j, A_i * 2)

    def test_focus_classes_is_symmetric_difference(self):
        """Focus classes are C_i xor C_j."""
        from src.utils.anchors import stratified_focus_classes

        tc = {0: {0, 1, 2}, 1: {2, 3}}
        assert stratified_focus_classes(tc, 0, 1) == frozenset({0, 1, 3})
        assert stratified_focus_classes(None, 0, 1) == frozenset()
