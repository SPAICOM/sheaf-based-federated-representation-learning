"""Tests for src.orchestrators.sheaf_frl."""

import pytest
import torch
from lightning.pytorch import Trainer

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.sheaf_frl import SheafFRL


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestSheafFRL:
    """Tests for SheafFRL class."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFRL(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            lambda_sheaf=0.1,
            latent_dims={0: 64},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
        )
        assert orchestrator is not None
        assert orchestrator.hparams.lambda_sheaf == 0.1

    def test_stiefel_matrices_created(self):
        """Test Stiefel matrices are created for edges."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFRL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            lambda_sheaf=0.1,
            latent_dims={0: 64, 1: 64},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
        )
        assert len(orchestrator.stiefel_matrices) > 0

    def test_on_train_epoch_end_with_anchors(self):
        """Test epoch end updates Stiefel matrices."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFRL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            lambda_sheaf=0.1,
            latent_dims={0: 64, 1: 64},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
        )

        # Initialize and collect anchors
        orchestrator.on_train_epoch_start()
        orchestrator.epoch_anchors[0].append(torch.randn(16, 64))
        orchestrator.epoch_anchors[1].append(torch.randn(16, 64))
        orchestrator.epoch_anchor_ids[0].append(torch.randint(0, 10, (16,)))
        orchestrator.epoch_anchor_ids[1].append(torch.randint(0, 10, (16,)))

        orig = {k: v.clone() for k, v in orchestrator.stiefel_matrices.items()}
        orchestrator.on_train_epoch_end()

        # Stiefel should be updated
        for k in orig:
            assert not torch.equal(orchestrator.stiefel_matrices[k], orig[k])

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
            encoder_hidden_dims=[32, 64, 128],
            decoder_hidden_dims=[256],
        )

        orchestrator = SheafFRL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
            lambda_sheaf=0.1,
            latent_dims={0: 64, 1: 128},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
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
