"""Tests for src.orchestrators.non_cooperative."""

import pytest
import torch
from lightning.pytorch import Trainer

from src.agents.cnn_classifier import CNNClassifier
from src.agents.latent_classifier import LatentClassifier
from src.orchestrators.non_cooperative import NonCooperativeLearning


class MockOptimizer:
    _target_ = 'torch.optim.Adam'
    lr = 0.001


def _make_agent():
    return LatentClassifier(in_features=128, num_classes=10, latent_dim=64)


def _batch(n_agents: int = 2) -> dict:
    return {
        str(i): (torch.randn(8, 128), torch.randint(0, 10, (8,)))
        for i in range(n_agents)
    }


class TestNonCooperativeLearning:
    def test_initialization_single_agent(self):
        orch = NonCooperativeLearning(
            agents={0: _make_agent()},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )
        assert orch is not None

    def test_initialization_multi_agent(self):
        orch = NonCooperativeLearning(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        assert orch is not None

    def test_on_train_epoch_end_is_noop(self):
        """Epoch end must not raise and returns None."""
        orch = NonCooperativeLearning(
            agents={0: _make_agent()},
            neighbors={0: set()},
            optimizer=MockOptimizer(),
        )
        assert orch.on_train_epoch_end() is None

    def test_training_step_returns_loss(self):
        orch = NonCooperativeLearning(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: set(), 1: set()},
            optimizer=MockOptimizer(),
        )
        loss = orch.training_step(_batch(), batch_idx=0)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_communication_stays_zero(self):
        """No communication should be recorded during training."""
        orch = NonCooperativeLearning(
            agents={0: _make_agent(), 1: _make_agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer=MockOptimizer(),
        )
        orch.training_step(_batch(), batch_idx=0)
        assert orch._communication_rounds == 0
        assert orch._communication_totals['bits'] == 0.0

    def test_heterogeneous_agents_allowed(self):
        """NonCooperative imposes no architecture constraints."""
        agent0 = LatentClassifier(
            in_features=128, num_classes=10, latent_dim=64
        )
        agent1 = CNNClassifier(
            in_features=3, num_classes=10, encoder_hidden_dims=[16, 32]
        )
        orch = NonCooperativeLearning(
            agents={0: agent0, 1: agent1},
            neighbors={0: set(), 1: set()},
            optimizer=MockOptimizer(),
        )
        # Each agent gets its own matching batch
        batch = {
            '0': (torch.randn(4, 128), torch.randint(0, 10, (4,))),
            '1': (torch.randn(4, 3, 16, 16), torch.randint(0, 10, (4,))),
        }
        loss = orch.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

    @pytest.mark.slow
    def test_one_epoch_with_trainer(self):
        """Smoke test: full training epoch via Lightning Trainer."""
        from src.datamodules.classification_datamodule import (
            ClassificationDataModule,
        )

        agents = {
            i: CNNClassifier(
                in_features=3, num_classes=10, encoder_hidden_dims=[8, 16]
            )
            for i in range(2)
        }
        orch = NonCooperativeLearning(
            agents=agents,
            neighbors={0: set(), 1: set()},
            optimizer={'_target_': 'torch.optim.Adam', 'lr': 0.001},
        )
        dm = ClassificationDataModule(
            repo='uoft-cs',
            name='cifar10',
            data_key='img',
            attributes=['label'],
            n_agents=2,
            split_strategy='uniform',
            batch_size=16,
            num_workers=0,
            mode='min_size',
            val_split=0.1,
            test_split=0.1,
            seed=42,
        )
        dm.prepare_data()
        dm.setup('fit')
        trainer = Trainer(
            max_epochs=1,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(orch, datamodule=dm)
