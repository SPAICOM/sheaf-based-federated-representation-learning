"""Tests for src.orchestrators.base_orchestrator."""

import pytest
import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator


class DummyAgent(nn.Module):
    """Dummy agent for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        return self.fc(x)

    def encode(self, x):
        return x

    def compute_loss(self, y_hat, y):
        return nn.functional.cross_entropy(y_hat, y)

    def task_performance(self, y_hat, y):
        return (torch.argmax(y_hat, dim=1) == y).float().mean()


class ReconstructionAgent(nn.Module):
    task_type = 'reconstruction'

    def __init__(self):
        super().__init__()
        self.decoder = nn.Unflatten(1, (1, 2, 2))

    def forward(self, x):
        return x

    def encode(self, x):
        return x.flatten(1)

    def compute_loss(self, y_hat, y):
        return nn.functional.mse_loss(y_hat, y)

    def task_performance(self, y_hat, y):
        mse = nn.functional.mse_loss(y_hat, y).clamp_min(1e-10)
        return 10.0 * torch.log10(1.0 / mse)


class ReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, value: float):
        self.x = torch.full((1, 2, 2), value)

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.x, self.x


class ReconstructionDataModule:
    def __init__(self):
        self.test_datasets = {
            0: ReconstructionDataset(0.25),
            1: ReconstructionDataset(0.25),
        }


class ZeroDecoder(nn.Module):
    def forward(self, z):
        return torch.zeros(z.shape[0], 1, 2, 2, device=z.device)


class ZeroDecoderReconstructionAgent(nn.Module):
    task_type = 'reconstruction'

    def __init__(self):
        super().__init__()
        self.decoder = ZeroDecoder()

    def encode(self, x):
        return x.flatten(1)


class DirectionalMaskReconstructionDataset(torch.utils.data.Dataset):
    def __init__(self, target: torch.Tensor, mask: torch.Tensor):
        self.x = torch.zeros_like(target)
        self.target = target
        self.mask = mask

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.x, self.target, self.mask


class DirectionalMaskReconstructionDataModule:
    def __init__(self):
        sender_target = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        sender_mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        receiver_target = torch.zeros(1, 2, 2)
        receiver_mask = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        self.test_datasets = {
            0: DirectionalMaskReconstructionDataset(
                sender_target, sender_mask
            ),
            1: DirectionalMaskReconstructionDataset(
                receiver_target, receiver_mask
            ),
        }


class ConcreteOrchestrator(BaseOrchestrator):
    """Concrete implementation for testing."""

    def __init__(self, agents, neighbors, optimizer):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
            log_latent_diagnostics=False,
        )

    def on_train_epoch_end(self):
        pass

    def send_message(self, sender_idx, receiver_idx, Z_sender):
        return Z_sender

    def _shared_eval(self, batch, batch_idx, prefix):
        outputs = self(batch)
        total_loss = 0
        for idx in outputs:
            y_hat, y = outputs[idx]
            loss = self.agents[idx].compute_loss(y_hat, y)
            total_loss += loss
        return outputs, total_loss


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestBaseOrchestrator:
    """Tests for BaseOrchestrator class."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = DummyAgent()
        agents = {0: agent}
        neighbors = {0: set()}

        orchestrator = ConcreteOrchestrator(
            agents=agents,
            neighbors=neighbors,
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None

    def test_empty_agents_raises(self):
        """Test that empty agents raises assertion."""
        with pytest.raises(AssertionError):
            ConcreteOrchestrator(
                agents={},
                neighbors={},
                optimizer=MockOptimizer(),
            )

    def test_forward(self):
        """Test forward pass."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 128)
        y = torch.randint(0, 10, (8,))
        batch = {'0': (x, y)}
        outputs = orchestrator(batch)
        assert '0' in outputs

    def test_training_step(self):
        """Test training step returns loss."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 128)
        y = torch.randint(0, 10, (8,))
        batch = {'0': (x, y)}
        loss = orchestrator.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    def test_communication_accounting_is_split_by_stage(self):
        """Train/test communication counters should be tracked separately."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        orchestrator.on_train_start()
        orchestrator._record_communication_round(prefix='train')
        orchestrator._record_communication(
            torch.ones(4),
            n_transmissions=2,
            prefix='train',
        )

        orchestrator.on_test_start()
        orchestrator._record_communication_round(prefix='test')
        orchestrator._record_communication(
            torch.ones(2),
            n_transmissions=1,
            prefix='test',
        )

        train_metrics = orchestrator._communication_metrics('train')
        test_metrics = orchestrator._communication_metrics('test')

        assert train_metrics['train/communication_rounds'] == 1.0
        assert test_metrics['test/communication_rounds'] == 1.0
        assert (
            train_metrics['train/communication_kilobytes']
            > test_metrics['test/communication_kilobytes']
        )

    def test_eval_logs_include_cumulative_train_communication(self):
        """Test-monitor/test logs should include cumulative train budget."""
        agent = DummyAgent()
        orchestrator = ConcreteOrchestrator(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        logged_metrics = []

        def capture_log_dict(metrics, **kwargs):
            logged_metrics.append(dict(metrics))

        orchestrator.log_dict = capture_log_dict

        orchestrator.on_train_start()
        orchestrator._record_communication_round(prefix='train')
        orchestrator._record_communication(
            torch.ones(4),
            n_transmissions=2,
            prefix='train',
        )

        train_metrics = orchestrator._communication_metrics('train')

        orchestrator.on_validation_start()
        orchestrator._validation_prefixes_seen.add('test_monitor')
        orchestrator.on_validation_epoch_end()

        orchestrator.on_test_start()
        orchestrator.on_test_epoch_end()

        assert {
            'test_monitor/train_communication_kilobytes_cumulative': (
                train_metrics['train/communication_kilobytes']
            ),
            'test_monitor/train_communication_rounds_cumulative': 1.0,
        } in logged_metrics
        assert {
            'test/train_communication_kilobytes_cumulative': (
                train_metrics['train/communication_kilobytes']
            ),
            'test/train_communication_rounds_cumulative': 1.0,
        } in logged_metrics

    def test_evaluate_communication_reports_reconstruction_psnr(self):
        orchestrator = ConcreteOrchestrator(
            agents={0: ReconstructionAgent(), 1: ReconstructionAgent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )

        logs = orchestrator.evaluate_communication_accuracy(
            ReconstructionDataModule(),
            prefix='validation',
        )

        assert logs['validation/avg_private_task_perf'] > 90.0
        assert logs['validation/avg_comm_task_perf'] > 90.0

    def test_reconstruction_comm_performance_uses_full_image_mse(self):
        orchestrator = ConcreteOrchestrator(
            agents={
                0: ZeroDecoderReconstructionAgent(),
                1: ZeroDecoderReconstructionAgent(),
            },
            neighbors={0: set(), 1: {0}},
            optimizer=MockOptimizer(),
        )

        logs = orchestrator.evaluate_communication_accuracy(
            DirectionalMaskReconstructionDataModule(),
            prefix='validation',
        )

        expected_psnr = 10.0 * torch.log10(torch.tensor(2.0)).item()
        assert logs['validation/private_task_perf_agent_0'] == pytest.approx(
            expected_psnr
        )
        assert logs['validation/private_mse_full_agent_0'] == pytest.approx(
            0.5
        )
        assert logs['validation/private_mse_visible_agent_0'] == pytest.approx(
            1e-10
        )
        assert logs['validation/private_mse_missing_agent_0'] == pytest.approx(
            1.0
        )
        assert logs['validation/comm_task_perf_agent_1'] == pytest.approx(
            expected_psnr
        )
        assert logs['validation/comm_mse_full_agent_1'] == pytest.approx(0.5)
        assert logs[
            'validation/comm_mse_tx_missing_rx_visible_agent_1'
        ] == pytest.approx(0.5)
