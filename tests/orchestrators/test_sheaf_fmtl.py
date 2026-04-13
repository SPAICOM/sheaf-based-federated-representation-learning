"""Tests for src.orchestrators.sheaf_fmtl."""

import pytest
import torch
from lightning.pytorch import Trainer
from torch.nn.utils import parameters_to_vector

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.sheaf_fmtl import SheafFMTL
from src.utils import calculate_communication_cost


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class TestSheafFMTL:
    """Tests for SheafFMTL class."""

    def test_initialization_single(self):
        """Test initialization with single agent, no neighbors."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )
        assert orchestrator is not None
        assert len(orchestrator.projection_matrices) == 0

    def test_initialization_dual(self):
        """Test initialization with two agents, bidirectional neighbors."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )
        assert orchestrator is not None
        assert len(orchestrator.projection_matrices) == 2
        assert '0_1' in orchestrator.projection_matrices
        assert '1_0' in orchestrator.projection_matrices

    def test_projection_matrices_created(self):
        """Test projection matrices are created for edges."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )
        d1 = sum(p.numel() for p in agent1.parameters() if p.requires_grad)
        d2 = sum(p.numel() for p in agent2.parameters() if p.requires_grad)
        d_ij = max(1, int(0.01 * min(d1, d2)))

        assert orchestrator.projection_matrices['0_1'].shape == (d_ij, d1)
        assert orchestrator.projection_matrices['1_0'].shape == (d_ij, d2)

    def test_on_train_epoch_end_updates_agent_parameters(self):
        """Test epoch-end synchronization updates agent weights once."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )

        original_theta_1 = parameters_to_vector(
            [p for p in agent1.parameters() if p.requires_grad]
        ).clone()
        original_theta_2 = parameters_to_vector(
            [p for p in agent2.parameters() if p.requires_grad]
        ).clone()

        orchestrator.on_train_epoch_end()

        updated_theta_1 = parameters_to_vector(
            [p for p in agent1.parameters() if p.requires_grad]
        )
        updated_theta_2 = parameters_to_vector(
            [p for p in agent2.parameters() if p.requires_grad]
        )

        assert not torch.equal(updated_theta_1, original_theta_1)
        assert not torch.equal(updated_theta_2, original_theta_2)

        projected_dim = orchestrator.projection_matrices['0_1'].shape[0]
        expected_kilobytes = calculate_communication_cost(
            projected_dim,
            n_transmissions=4,
        )['kilobytes']
        metrics = orchestrator._communication_metrics('train')
        assert metrics['train/communication_rounds'] == 2.0
        assert metrics['train/communication_kilobytes'] == pytest.approx(
            expected_kilobytes
        )

    def test_on_train_epoch_end_updates_projection_matrices(self):
        """Test epoch-end synchronization updates restriction maps."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )

        orig_P_01 = orchestrator.projection_matrices['0_1'].clone()
        orig_P_10 = orchestrator.projection_matrices['1_0'].clone()

        orchestrator.on_train_epoch_end()

        assert not torch.equal(
            orchestrator.projection_matrices['0_1'], orig_P_01
        )
        assert not torch.equal(
            orchestrator.projection_matrices['1_0'], orig_P_10
        )
        projected_dim = orchestrator.projection_matrices['0_1'].shape[0]
        expected_kilobytes = calculate_communication_cost(
            projected_dim,
            n_transmissions=4,
        )['kilobytes']
        metrics = orchestrator._communication_metrics('train')
        assert metrics['train/communication_rounds'] == 2.0
        assert metrics['train/communication_kilobytes'] == pytest.approx(
            expected_kilobytes
        )

    def test_epoch_level_sync_records_two_projected_vector_rounds(self):
        """Test epoch-end synchronization logs two projected-vector rounds."""
        agent1 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent2 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
        )

        orchestrator.on_train_epoch_end()

        projected_dim = orchestrator.projection_matrices['0_1'].shape[0]
        expected_kilobytes = calculate_communication_cost(
            projected_dim,
            n_transmissions=4,
        )['kilobytes']
        metrics = orchestrator._communication_metrics('train')
        assert metrics['train/communication_rounds'] == 2.0
        assert metrics['train/communication_kilobytes'] == pytest.approx(
            expected_kilobytes
        )

    def test_shared_eval(self):
        """Test _shared_eval returns loss."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        orchestrator = SheafFMTL(
            agents={0: agent},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
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
            encoder_hidden_dims=[32, 64, 128],
            decoder_hidden_dims=[256],
        )

        orchestrator = SheafFMTL(
            agents={0: agent1, 1: agent2},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
            gamma=0.01,
            lambda_reg=0.001,
            eta=0.01,
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
