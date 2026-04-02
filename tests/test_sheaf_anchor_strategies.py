import unittest

import torch
import torch.nn as nn

from src.orchestrators.sheaf_frl import SheafFRL
from src.utils.anchors import (
    AnchorConfig,
    build_anchor_bundles,
    build_semantic_pilot_bundles,
    shared_anchor_rows,
)


class IdentityAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Identity()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((), dtype=y_hat.dtype, device=y_hat.device)

    def task_performance(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros((), dtype=y_hat.dtype, device=y_hat.device)


def make_orchestrator(
    anchor_strategy: str,
    num_anchors: int = 12,
) -> SheafFRL:
    return SheafFRL(
        agents={0: IdentityAgent(), 1: IdentityAgent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        lambda_sheaf=1.0,
        latent_dims={0: 2, 1: 2},
        anchor_strategy=anchor_strategy,
        num_anchors=num_anchors,
        parseval_normalization=False,
        l2_normalization=False,
    )


class SheafAnchorStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.latents = {
            0: torch.tensor(
                [
                    [10.0, 0.0],
                    [11.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                ]
            ),
            1: torch.tensor(
                [
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [0.0, 10.0],
                    [0.0, 11.0],
                ]
            ),
        }
        self.labels = {
            0: torch.tensor([0, 0, 1, 1, 1]),
            1: torch.tensor([1, 1, 1, 2, 2]),
        }

    def _assert_shared_keys_are_class_consistent(self, strategy: str) -> None:
        anchor_config = AnchorConfig(
            strategy=strategy,
            num_anchors=12,
            parseval_normalization=False,
            l2_normalization=False,
        )
        anchors, keys = build_anchor_bundles(
            self.latents,
            self.labels,
            anchor_config,
        )
        shared_keys = sorted(set(keys[0]) & set(keys[1]))

        self.assertTrue(shared_keys)
        self.assertTrue(all(class_label == 1 for class_label, _ in shared_keys))

        shared_rows = shared_anchor_rows(
            anchors[0],
            keys[0],
            anchors[1],
            keys[1],
        )
        self.assertIsNotNone(shared_rows)
        A_i_shared, A_j_shared = shared_rows
        self.assertTrue(torch.allclose(A_i_shared, A_j_shared))

    def _assert_semantic_pilot_keys_are_sample_consistent(self) -> None:
        anchor_config = AnchorConfig(
            strategy='semantic_pilots',
            num_anchors=12,
            parseval_normalization=False,
            l2_normalization=False,
        )
        pilot_latents = {
            0: torch.tensor(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                ]
            ),
            1: torch.tensor(
                [
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [4.0, 4.0],
                ]
            ),
        }
        pilot_ids = {
            0: torch.tensor([10, 11, 12]),
            1: torch.tensor([11, 12, 13]),
        }

        anchors, keys = build_semantic_pilot_bundles(
            pilot_latents,
            pilot_ids,
            anchor_config,
        )
        shared_keys = sorted(set(keys[0]) & set(keys[1]))

        self.assertEqual(shared_keys, [(11, 0), (12, 0)])
        shared_rows = shared_anchor_rows(
            anchors[0],
            keys[0],
            anchors[1],
            keys[1],
        )
        self.assertIsNotNone(shared_rows)
        A_i_shared, A_j_shared = shared_rows
        self.assertTrue(torch.allclose(A_i_shared, A_j_shared))

    def test_prototype_alignment_uses_only_shared_classes(self):
        self._assert_shared_keys_are_class_consistent('prototype')

    def test_random_alignment_uses_only_shared_classes(self):
        self._assert_shared_keys_are_class_consistent('random')

    def test_balanced_alignment_uses_only_shared_classes(self):
        self._assert_shared_keys_are_class_consistent('balanced')

    def test_semantic_pilots_alignment_uses_only_shared_pilot_ids(self):
        self._assert_semantic_pilot_keys_are_sample_consistent()

    def test_semantic_pilots_shared_eval_uses_pilot_batches(self):
        orchestrator = make_orchestrator('semantic_pilots')
        batch = {
            0: (
                torch.tensor([[10.0, 0.0], [11.0, 0.0]]),
                torch.tensor([0, 0]),
            ),
            1: (
                torch.tensor([[0.0, 10.0], [0.0, 11.0]]),
                torch.tensor([2, 2]),
            ),
            'pilot_0': (
                torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                torch.tensor([9, 9]),
                torch.tensor([100, 101]),
            ),
            'pilot_1': (
                torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
                torch.tensor([9, 9]),
                torch.tensor([100, 101]),
            ),
        }

        _outputs, total_loss = orchestrator._shared_eval(
            batch=batch,
            batch_idx=0,
            prefix='validation',
        )
        self.assertTrue(torch.allclose(total_loss, torch.tensor(0.0)))

    def test_clustered_pilots_alignment_uses_only_shared_classes(self):
        self._assert_shared_keys_are_class_consistent('clustered_pilots')

    def test_dynamic_alignment_uses_only_shared_classes(self):
        self._assert_shared_keys_are_class_consistent('dynamic')


if __name__ == '__main__':
    unittest.main()
