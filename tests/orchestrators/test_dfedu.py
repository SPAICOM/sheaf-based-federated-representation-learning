"""Tests for src.orchestrators.dfedu."""

import pytest
import torch
from lightning.pytorch import Trainer

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.dfedu import DFedU


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestDFedU:
    """Tests for DFedU class."""

    def test_initialization_single(self):
        """Test initialization with single agent, no neighbors."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DFedU(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            eta=0.05,
        )
        assert orchestrator is not None

    def test_initialization_dual(self):
        """Test initialization with two agents, same architecture."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DFedU(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            eta=0.05,
        )
        assert orchestrator is not None

    def test_validate_agents_passes(self):
        """Test that identical architectures pass validation."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DFedU(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            eta=0.05,
        )
        assert orchestrator is not None

    def test_validate_agents_raises_different_architecture(self):
        """Test that different architectures raise error."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=256, num_classes=10, latent_dim=128
        )
        with pytest.raises(ValueError):
            DFedU(
                agents={0: agent1, 1: agent2},
                neighbors={0: {1}, 1: {0}},
                optimizer=MockOptimizer(),
                eta=0.05,
            )

    def test_on_train_epoch_end_laplacian(self):
        """Test Laplacian penalty is applied at epoch end."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DFedU(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            eta=0.05,
        )

        orig1 = {k: v.clone() for k, v in agent1.state_dict().items()}
        orig2 = {k: v.clone() for k, v in agent2.state_dict().items()}

        orchestrator.on_train_epoch_end()

        params1 = agent1.state_dict()
        params2 = agent2.state_dict()
        updated = False
        for k in orig1:
            if not torch.equal(params1[k], orig1[k]) or not torch.equal(
                params2[k], orig2[k]
            ):
                updated = True
                break
        assert updated

    def test_shared_eval(self):
        """Test _shared_eval returns loss."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = DFedU(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            eta=0.05,
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

        orchestrator = DFedU(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
            eta=0.05,
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
