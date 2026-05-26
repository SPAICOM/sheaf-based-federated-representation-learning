"""Tests for src.orchestrators.epochend_dpsgd."""

import torch
import torch.nn as nn
from lightning.pytorch import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

from src.agents.latent_classifier import LatentClassifier
from src.orchestrators.dpsgd import DPSGD
from src.orchestrators.epochend_dpsgd import EpochEndDPSGD


class MockOptimizer:
    """Mock optimizer config for testing."""

    _target_ = 'torch.optim.Adam'
    lr = 0.001


class _ToyEpochEndDPSGDDataModule(LightningDataModule):
    def setup(self, stage=None):
        self.train_datasets = {
            0: TensorDataset(
                torch.randn(16, 8),
                torch.randint(0, 10, (16,)),
            ),
            1: TensorDataset(
                torch.randn(16, 8),
                torch.randint(0, 10, (16,)),
            ),
        }

    def train_dataloader(self):
        return {
            agent_idx: DataLoader(dataset, batch_size=4)
            for agent_idx, dataset in self.train_datasets.items()
        }


class _TinyBatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.linear = nn.Linear(2, 2, bias=False)


def _set_agent_state(
    agent: _TinyBatchNormNet,
    *,
    linear_weight: torch.Tensor,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> None:
    with torch.no_grad():
        agent.linear.weight.copy_(linear_weight)
        agent.bn.weight.copy_(bn_weight)
        agent.bn.bias.copy_(bn_bias)
        agent.bn.running_mean.copy_(running_mean)
        agent.bn.running_var.copy_(running_var)


def test_epochend_matches_dpsgd_parameter_mixing_and_preserves_buffers():
    """Epoch-end variant should reuse D-PSGD's parameter-only mixing rule."""
    step_agents = {0: _TinyBatchNormNet(), 1: _TinyBatchNormNet()}
    epoch_agents = {0: _TinyBatchNormNet(), 1: _TinyBatchNormNet()}

    _set_agent_state(
        step_agents[0],
        linear_weight=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        bn_weight=torch.tensor([1.0, 2.0]),
        bn_bias=torch.tensor([0.5, -0.5]),
        running_mean=torch.tensor([10.0, 20.0]),
        running_var=torch.tensor([30.0, 40.0]),
    )
    _set_agent_state(
        step_agents[1],
        linear_weight=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        bn_weight=torch.tensor([3.0, 4.0]),
        bn_bias=torch.tensor([1.5, -1.5]),
        running_mean=torch.tensor([100.0, 200.0]),
        running_var=torch.tensor([300.0, 400.0]),
    )
    epoch_agents[0].load_state_dict(step_agents[0].state_dict())
    epoch_agents[1].load_state_dict(step_agents[1].state_dict())

    initial_epoch_buffers = {
        idx: {
            name: buf.detach().clone() for name, buf in agent.named_buffers()
        }
        for idx, agent in epoch_agents.items()
    }

    neighbors = {0: {1}, 1: {0}}
    stepwise = DPSGD(
        agents=step_agents,
        neighbors=neighbors,
        optimizer=MockOptimizer(),
    )
    epochend = EpochEndDPSGD(
        agents=epoch_agents,
        neighbors=neighbors,
        optimizer=MockOptimizer(),
    )

    stepwise.on_before_optimizer_step(None)
    epochend.on_train_epoch_end()

    for idx in step_agents:
        step_params = dict(step_agents[idx].named_parameters())
        epoch_params = dict(epoch_agents[idx].named_parameters())
        assert step_params.keys() == epoch_params.keys()
        for name in step_params:
            assert torch.allclose(step_params[name], epoch_params[name])

    for idx, agent in epoch_agents.items():
        for name, buf in agent.named_buffers():
            assert torch.equal(buf, initial_epoch_buffers[idx][name])


def test_epochend_disables_stepwise_hook():
    """Calling the Lightning step hook should not alter parameters."""
    agent0 = _TinyBatchNormNet()
    agent1 = _TinyBatchNormNet()
    before = {
        idx: {
            name: param.detach().clone()
            for name, param in agent.named_parameters()
        }
        for idx, agent in {0: agent0, 1: agent1}.items()
    }

    orchestrator = EpochEndDPSGD(
        agents={0: agent0, 1: agent1},
        neighbors={0: {1}, 1: {0}},
        optimizer=MockOptimizer(),
    )
    orchestrator.on_before_optimizer_step(None)

    for idx, agent in orchestrator.agents.items():
        for name, param in agent.named_parameters():
            assert torch.equal(param, before[int(idx)][name])


def test_trainer_logs_one_epoch_end_communication_round():
    """Epoch-end D-PSGD should communicate once per epoch, not per step."""
    orchestrator = EpochEndDPSGD(
        agents={
            0: LatentClassifier(
                in_features=8,
                num_classes=10,
                latent_dim=4,
                encoder_hidden_dims=[6],
            ),
            1: LatentClassifier(
                in_features=8,
                num_classes=10,
                latent_dim=4,
                encoder_hidden_dims=[6],
            ),
        },
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
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

    trainer.fit(orchestrator, datamodule=_ToyEpochEndDPSGDDataModule())

    assert float(trainer.callback_metrics['train/communication_rounds']) == 1.0
    assert (
        float(trainer.callback_metrics['train/communication_kilobytes']) > 0.0
    )
