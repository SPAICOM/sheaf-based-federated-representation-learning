"""Tests for src.orchestrators.federated."""

import pytest
import torch
from lightning.pytorch import Trainer

from src.agents.cnn_classifier import CNNClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.federated import FederatedLearning


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestFederatedLearning:
    """Tests for FederatedLearning class."""

    def test_initialization(self):
        """Test basic initialization."""
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
        orchestrator = FederatedLearning(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        assert orchestrator is not None

    def test_on_train_epoch_end_aggregation(self):
        """Test epoch-end parameter aggregation."""
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
        orchestrator = FederatedLearning(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )

        # Store original params
        orig1 = {k: v.clone() for k, v in agent1.state_dict().items()}

        # Run aggregation
        orchestrator.on_train_epoch_end()

        # Params should be updated
        assert not all(
            torch.equal(orchestrator.agents['0'].state_dict()[k], orig1[k])
            for k in orig1
        )

    def test_shared_eval(self):
        """Test _shared_eval returns loss."""
        agent = CNNClassifier(
            in_features=3,
            num_classes=10,
            encoder_hidden_dims=[32, 64, 128],
            decoder_hidden_dims=[256],
        )
        orchestrator = FederatedLearning(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )

        x = torch.randn(8, 3, 32, 32)
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

        orchestrator = FederatedLearning(
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
