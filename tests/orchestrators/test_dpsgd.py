"""Tests for src.orchestrators.dpsgd."""

import pytest
import torch
from lightning.pytorch import Trainer

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.dpsgd import DPSGD


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestDPSGD:
    """Tests for DPSGD class."""

    def test_initialization_single(self):
        """Test initialization with single agent, no neighbors."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None
        assert len(orchestrator.mixing_weights) == 1
        assert orchestrator.mixing_weights[(0, 0)] == 1.0

    def test_initialization_dual(self):
        """Test initialization with two agents, same architecture."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None
        assert len(orchestrator.mixing_weights) == 4

    def test_mixing_weights_created(self):
        """Test Metropolis-Hastings mixing weights are computed."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )

        assert orchestrator.mixing_weights[(0, 0)] > 0
        assert orchestrator.mixing_weights[(1, 1)] > 0
        assert orchestrator.mixing_weights[(0, 1)] > 0
        assert orchestrator.mixing_weights[(1, 0)] > 0

        total_0 = (
            orchestrator.mixing_weights[(0, 0)]
            + orchestrator.mixing_weights[(0, 1)]
        )
        total_1 = (
            orchestrator.mixing_weights[(1, 0)]
            + orchestrator.mixing_weights[(1, 1)]
        )
        assert abs(total_0 - 1.0) < 1e-6
        assert abs(total_1 - 1.0) < 1e-6

    def test_validate_agents_passes(self):
        """Test that identical architectures pass validation."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None

    def test_validate_agents_raises_different_in_features(self):
        """Test that different in_features raises error."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=256, num_classes=10, latent_dim=64
        )
        with pytest.raises(ValueError):
            DPSGD(
                agents={0: agent1, 1: agent2},
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
            )

    def test_validate_agents_raises_different_latent_dim(self):
        """Test that different latent_dim raises error."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=128
        )
        with pytest.raises(ValueError):
            DPSGD(
                agents={0: agent1, 1: agent2},
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
            )

    def test_on_before_optimizer_step(self):
        """Test parameter mixing between neighbors."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )

        for p in agent1.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)
        for p in agent2.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)

        orchestrator.on_before_optimizer_step(None)

    def test_on_before_optimizer_step_without_neighbors_records_no_rounds(self):
        """Isolated agents should not accrue communication rounds."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        for p in agent.parameters():
            if p.requires_grad:
                p.grad = torch.randn_like(p)

        orchestrator.on_before_optimizer_step(None)

        metrics = orchestrator._communication_metrics('train')
        assert metrics['train/communication_rounds'] == 0.0
        assert metrics['train/communication_kilobytes'] == 0.0

    def test_shared_eval(self):
        """Test _shared_eval returns loss."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DPSGD(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 128)
        y = torch.randint(0, 10, (8,))
        batch = {0: [x, y]}

        outputs, total_loss = orchestrator._shared_eval(batch, 0, 'train')
        assert isinstance(total_loss, torch.Tensor)

    @pytest.mark.slow
    def test_train_one_epoch_with_cnn(self):
        """Test training for one epoch with CNNClassifier on CIFAR10."""
        agent1 = CNNClassifier(
            in_features=3,
            num_classes=10,
            encoder_hidden_dims=[16, 32, 64],
            decoder_hidden_dims=[256],
        )
        agent2 = CNNClassifier(
            in_features=3,
            num_classes=10,
            encoder_hidden_dims=[16, 32, 64],
            decoder_hidden_dims=[256],
        )

        orchestrator = DPSGD(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
        )

        datamodule = ClassificationDataModule(
            repo='uoft-cs',
            name='cifar10',
            data_key='img',
            attributes=['label'],
            n_agents=2,
            split_strategy='uniform',
            batch_size=4,
            num_workers=0,
            mode='min_size',
            val_split=0.1,
            test_split=0.1,
            seed=42,
        )
        datamodule.prepare_data()
        datamodule.setup('train')

        trainer = Trainer(
            max_epochs=3,
            accelerator='cuda',
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(orchestrator, datamodule)
