"""Tests for src.orchestrators.sheaf_frl."""

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.datamodules.classification_datamodule import ClassificationDataModule
from src.orchestrators.sheaf_frl import SheafFRL


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class _ToySheafDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.train_datasets = {
            0: TensorDataset(
                torch.randn(16, 8),
                torch.randint(0, 3, (16,)),
            ),
            1: TensorDataset(
                torch.randn(16, 8),
                torch.randint(0, 3, (16,)),
            ),
        }

    def train_dataloader(self):
        return {
            agent_idx: DataLoader(dataset, batch_size=4)
            for agent_idx, dataset in self.train_datasets.items()
        }


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
            max_lmb=0.1,
            latent_dims={0: 64},
            #anchor_strategy='prototype',
            #num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
        )
        assert orchestrator is not None
        assert orchestrator.hparams.max_lmb == 0.1

    def test_dynamic_anchor_strategy_is_not_supported(self):
        """Dynamic anchors should fail fast at construction time."""
        agent = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )

        with pytest.raises(ValueError, match='Unknown anchor_strategy'):
            SheafFRL(
                agents={0: agent},
                neighbors={0: set()},
                optimizer=MockOptimizer(),
                max_lmb=0.1,
                latent_dims={0: 64},
                anchor_strategy='dynamic',
                num_anchors=10,
                parseval_normalization=False,
                l2_normalization=False,
            )

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
            max_lmb=0.1,
            latent_dims={0: 64, 1: 64},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
            filter_unseen_classes=False,
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
            max_lmb=0.1,
            latent_dims={0: 64, 1: 64},
            anchor_strategy='prototype',
            num_anchors=10,
            parseval_normalization=False,
            l2_normalization=False,
            filter_unseen_classes=False,
        )

        # Initialize and collect anchors
        orchestrator.on_train_epoch_start()
        orchestrator.epoch_anchors[0].append(torch.randn(16, 64))
        orchestrator.epoch_anchors[1].append(torch.randn(16, 64))
        shared_labels = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        )
        orchestrator.epoch_anchor_ids[0].append(shared_labels.clone())
        orchestrator.epoch_anchor_ids[1].append(shared_labels.clone())

        orig = {k: v.clone() for k, v in orchestrator.stiefel_matrices.items()}
        orchestrator.on_train_epoch_end()

        # Stiefel should be updated
        for k in orig:
            assert not torch.equal(orchestrator.stiefel_matrices[k], orig[k])

    def test_training_step_adds_sheaf_penalty_and_records_communication(self):
        """Train batches optimize task loss plus sheaf penalty."""
        orchestrator = SheafFRL(
            agents={
                0: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
                1: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
            max_lmb=0.1,
            latent_dims={0: 4, 1: 4},
            anchor_strategy='prototype',
            num_anchors=4,
            parseval_normalization=False,
            l2_normalization=False,
        )

        batch = {
            0: [torch.randn(8, 8), torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])],
            1: [torch.randn(8, 8), torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])],
        }

        expected_loss = 0.0
        for idx, agent in orchestrator.agents.items():
            x, y = batch[int(idx)]
            y_hat = agent(x)
            expected_loss += agent.compute_loss(y_hat, y)

        orchestrator.on_train_start()
        orchestrator.on_train_epoch_start()
        actual_loss = orchestrator.training_step(batch, 0)
        train_metrics = orchestrator._communication_metrics('train')

        assert actual_loss >= expected_loss
        assert train_metrics['train/communication_rounds'] == 1.0
        assert train_metrics['train/communication_kilobytes'] > 0.0

    def test_shared_eval_records_split_specific_anchor_communication(self):
        """Train and test anchor exchanges are both recorded per batch."""
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
            max_lmb=0.1,
            latent_dims={0: 64, 1: 64},
            anchor_strategy='prototype',
            num_anchors=4,
            parseval_normalization=False,
            l2_normalization=False,
        )

        batch = {
            0: [torch.randn(8, 128), torch.randint(0, 10, (8,))],
            1: [torch.randn(8, 128), torch.randint(0, 10, (8,))],
        }

        orchestrator.on_train_start()
        orchestrator.on_train_epoch_start()
        orchestrator._shared_eval(batch, 0, 'train')
        train_metrics = orchestrator._communication_metrics('train')

        orchestrator.on_test_start()
        orchestrator._shared_eval(batch, 0, 'test')
        test_metrics = orchestrator._communication_metrics('test')

        assert train_metrics['train/communication_rounds'] == 1.0
        assert train_metrics['train/communication_kilobytes'] > 0.0
        assert test_metrics['test/communication_rounds'] == 1.0
        assert test_metrics['test/communication_kilobytes'] > 0.0

    def test_trainer_logs_nonzero_train_communication_metrics(self):
        """Epoch-end anchor exchanges should emit positive train communication."""
        orchestrator = SheafFRL(
            agents={
                0: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
                1: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
            max_lmb=0.1,
            latent_dims={0: 4, 1: 4},
            anchor_strategy='prototype',
            num_anchors=4,
            parseval_normalization=False,
            l2_normalization=False,
        )

        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )

        trainer.fit(orchestrator, datamodule=_ToySheafDataModule())

        assert (
            float(trainer.callback_metrics['train/communication_kilobytes'])
            > 0.0
        )
        assert (
            float(trainer.callback_metrics['train/communication_rounds'])
            > 0.0
        )

    def test_train_communication_is_recorded_per_step_and_stiefel_updates_at_epoch_end(self):
        """Train communication happens per step; Stiefel refresh happens at epoch end."""
        orchestrator = SheafFRL(
            agents={
                0: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
                1: LatentClassifier(
                    in_features=8,
                    num_classes=3,
                    latent_dim=4,
                    encoder_hidden_dims=[6],
                ),
            },
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
            max_lmb=0.1,
            latent_dims={0: 4, 1: 4},
            anchor_strategy='prototype',
            num_anchors=4,
            parseval_normalization=False,
            l2_normalization=False,
        )
        orchestrator.log = lambda *args, **kwargs: None

        batch = {
            0: [torch.randn(8, 8), torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])],
            1: [torch.randn(8, 8), torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])],
        }

        orchestrator.on_train_start()
        orchestrator.on_train_epoch_start()
        stiefel_before = {
            name: param.detach().clone()
            for name, param in orchestrator.stiefel_matrices.items()
        }

        orchestrator.training_step(batch, 0)
        train_metrics = orchestrator._communication_metrics('train')

        assert train_metrics['train/communication_rounds'] == 1.0
        assert train_metrics['train/communication_kilobytes'] > 0.0

        orchestrator.on_train_epoch_end()
        epoch_metrics = orchestrator._communication_metrics('train')

        assert epoch_metrics['train/communication_rounds'] == 1.0
        assert epoch_metrics['train/communication_kilobytes'] > 0.0
        assert any(
            not torch.allclose(param.detach(), stiefel_before[name])
            for name, param in orchestrator.stiefel_matrices.items()
        )

    @pytest.mark.slow
    def test_train_one_epoch_with_cnn(self):
        """Test training for one epoch with CNNClassifier on CIFAR10."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA is required for this slow integration test.')

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
            max_lmb=0.1,
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
        try:
            datamodule.prepare_data()
        except OSError as exc:
            pytest.skip(f'Dataset cache is not writable in this environment: {exc}')
        datamodule.setup('train')

        trainer = Trainer(
            max_epochs=3,
            accelerator='cuda',
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(orchestrator, datamodule)
