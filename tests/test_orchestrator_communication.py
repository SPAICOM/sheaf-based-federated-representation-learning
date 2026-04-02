import unittest

import torch
import torch.nn as nn

from src.orchestrators import FederatedLearning, NonCooperativeLearning


class _TinyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    @property
    def decoder(self) -> nn.Module:
        return self.linear

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.encode(x))

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(y_hat, y)

    def task_performance(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return (y_hat.argmax(dim=1) == y).float().mean()


class OrchestratorCommunicationTests(unittest.TestCase):
    def test_federated_records_epoch_level_state_communication(self):
        orchestrator = FederatedLearning(
            agents={0: _TinyAgent(), 1: _TinyAgent()},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        )

        orchestrator.on_train_epoch_end()

        self.assertGreater(orchestrator._communication_totals['bits'], 0.0)
        self.assertEqual(orchestrator._communication_rounds, 2)

    def test_non_cooperative_keeps_communication_at_zero(self):
        orchestrator = NonCooperativeLearning(
            agents={0: _TinyAgent(), 1: _TinyAgent()},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        )
        batch = {
            0: (
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]),
            ),
            1: (
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]),
            ),
        }

        _outputs, total_loss = orchestrator._shared_eval(
            batch=batch,
            batch_idx=0,
            prefix='test',
        )

        self.assertGreaterEqual(float(total_loss.detach()), 0.0)
        self.assertEqual(orchestrator._communication_totals['bits'], 0.0)
        self.assertEqual(orchestrator._communication_rounds, 0)


if __name__ == '__main__':
    unittest.main()
