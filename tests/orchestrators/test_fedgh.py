"""Tests for src.orchestrators.fedgh."""

import copy

import lightning as l
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer

from src.agents.personalized_classifier import PersonalizedClassifier
from src.orchestrators.fedgh import FedGH
from src.utils import calculate_communication_cost

OPTIMIZER = {'_target_': 'torch.optim.SGD', 'lr': 0.01}
N_CLASSES = 4


class MockOptimizer:
    _target_ = 'torch.optim.SGD'
    lr = 0.01


class TinyDataset(torch.utils.data.Dataset):
    """Items are ``(x, y)``; a set ``transform`` breaks encoding on purpose."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]


class TinyDataModule(l.LightningDataModule):
    def __init__(self, transform=None):
        super().__init__()
        torch.manual_seed(0)
        # Agent 0 holds classes {0, 1}, agent 1 holds {1, 2, 3}.
        self.train_datasets = {
            0: TinyDataset(
                torch.randn(12, 6), torch.tensor([0, 1] * 6), transform
            ),
            1: TinyDataset(
                torch.randn(12, 6), torch.tensor([1, 2, 3] * 4), transform
            ),
        }

    def train_dataloader(self):
        def collate(items):
            return {
                idx: (
                    torch.stack([x for x, _ in chunk]),
                    torch.stack([y for _, y in chunk]),
                )
                for idx, chunk in enumerate(zip(*items))
            }

        paired = list(zip(self.train_datasets[0], self.train_datasets[1]))
        return torch.utils.data.DataLoader(
            paired, batch_size=4, collate_fn=collate
        )


def _agent(latent_dim: int = 5, hidden: list[int] | None = None):
    return PersonalizedClassifier(
        encoder=nn.Linear(6, latent_dim),
        latent_dim=latent_dim,
        num_classes=N_CLASSES,
        decoder_hidden_dims=[3] if hidden is None else hidden,
        use_batchnorm=True,
    )


def _fedgh(**kwargs) -> FedGH:
    defaults = {
        'agents': {0: _agent(5), 1: _agent(7)},
        'neighbors': {0: {1}, 1: {0}},
        'optimizer': MockOptimizer(),
    }
    return FedGH(**{**defaults, **kwargs})


def _kilobytes(payload, n_transmissions: int = 1) -> float:
    return calculate_communication_cost(
        payload, n_transmissions=n_transmissions
    )['kilobytes']


class TestConstruction:
    def test_header_is_the_decoder_output_layer(self):
        agent = _agent()
        rep = torch.randn(2, 3)
        head = FedGH._shared_head(agent)
        assert head.in_features == 3
        assert torch.equal(agent.decoder.network[-1](rep), head(rep))

    def test_heterogeneous_extractors_share_one_header(self):
        orch = _fedgh()
        assert orch.global_header.in_features == 3
        assert orch.hparams.header_lr == 0.01

    def test_empty_hidden_dims_makes_the_decoder_the_header(self):
        orch = _fedgh(
            agents={0: _agent(5, hidden=[]), 1: _agent(5, hidden=[])}
        )
        assert orch.global_header.in_features == 5

    def test_rejects_mismatched_header_shapes(self):
        with pytest.raises(ValueError, match='one header shape'):
            _fedgh(agents={0: _agent(hidden=[3]), 1: _agent(hidden=[4])})

    def test_rejects_non_classification_agents(self):
        agent = _agent()
        agent.task_type = 'reconstruction'
        with pytest.raises(TypeError, match='classification'):
            _fedgh(agents={0: agent, 1: _agent()})

    @pytest.mark.parametrize(
        'bad', [{'header_lr': 0.0}, {'local_epochs': 0}]
    )
    def test_rejects_bad_hyperparameters(self, bad):
        with pytest.raises(ValueError):
            _fedgh(**bad)


class TestLARs:
    def test_lars_are_class_means_of_the_header_input(self):
        """Eq. 3, over the classes each agent actually holds."""
        orch = _fedgh()
        orch.eval()
        dm = TinyDataModule()
        lars = orch._compute_lars(dm)

        assert lars[0][1].tolist() == [0, 1]
        assert lars[1][1].tolist() == [1, 2, 3]

        agent = orch.agents['1']
        data = dm.train_datasets[1]
        with torch.no_grad():
            reps = agent.decoder.network[:-1](agent.encode(data.x))
        for rep, c in zip(*lars[1]):
            expected = reps[data.y == c].mean(dim=0)
            assert torch.allclose(rep, expected, atol=1e-6)

    def test_keeps_batchnorm_stats_and_train_mode(self):
        orch = _fedgh()
        orch.train()
        before = copy.deepcopy(orch.state_dict())
        orch._compute_lars(TinyDataModule())
        for key, value in orch.state_dict().items():
            assert torch.equal(value, before[key]), key
        assert orch.agents['0'].training

    def test_augmentation_is_off_and_restored(self):
        def augment(_x):
            raise AssertionError('LARs must use un-augmented images')

        dm = TinyDataModule(transform=augment)
        _fedgh()._compute_lars(dm)
        assert all(d.transform is augment for d in dm.train_datasets.values())

    def test_missing_train_data_raises(self):
        class Empty:
            train_datasets = {}

        with pytest.raises(RuntimeError, match='train_datasets'):
            _fedgh()._compute_lars(Empty())


class TestServerUpdate:
    def test_sequential_per_lar_sgd(self):
        """Eq. 4: agent-id order, one step per LAR, no sample weighting."""
        orch = _fedgh(header_lr=0.1)
        expected = copy.deepcopy(orch.global_header)
        reps = torch.randn(3, 3)
        lars = {
            1: (reps[1:], torch.tensor([2, 3])),
            0: (reps[:1], torch.tensor([0])),
        }
        orch._update_global_header(lars)

        opt = torch.optim.SGD(expected.parameters(), lr=0.1)
        for rep, target in zip(reps, [0, 2, 3]):
            opt.zero_grad()
            F.cross_entropy(
                expected(rep.unsqueeze(0)), torch.tensor([target])
            ).backward()
            opt.step()
        for got, want in zip(
            orch.global_header.parameters(), expected.parameters()
        ):
            assert torch.allclose(got, want)

    def test_leaves_no_grad_for_the_agent_optimizer(self):
        orch = _fedgh()
        orch._update_global_header(
            {0: (torch.randn(1, 3), torch.tensor([0]))}
        )
        assert all(p.grad is None for p in orch.global_header.parameters())


class TestBroadcast:
    def test_replaces_only_the_header(self):
        orch = _fedgh()
        with torch.no_grad():
            orch.global_header.weight.fill_(1.0)
        hidden = copy.deepcopy(orch.agents['0'].decoder.network[0].state_dict())
        batchnorm = copy.deepcopy(
            orch.agents['0'].decoder.network[1].state_dict()
        )

        orch._broadcast_header()

        for agent in orch.agents.values():
            assert (FedGH._shared_head(agent).weight == 1.0).all()
        decoder = orch.agents['0'].decoder.network
        for key, value in decoder[0].state_dict().items():
            assert torch.equal(value, hidden[key])
        for key, value in decoder[1].state_dict().items():
            assert torch.equal(value, batchnorm[key])


class TestTraining:
    def _fit(self, max_epochs: int, local_epochs: int):
        torch.manual_seed(0)
        orch = _fedgh(optimizer=OPTIMIZER, local_epochs=local_epochs)
        Trainer(
            max_epochs=max_epochs,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        ).fit(orch, datamodule=TinyDataModule())
        return orch

    def test_rounds_and_communication_accounting(self):
        """Epochs 0, 2 start a round and 1, 3 end one (E=2, real backward)."""
        orch = self._fit(max_epochs=4, local_epochs=2)
        state = orch._communication_by_split['train']
        assert state['rounds'] == 2

        header = orch.global_header.state_dict()
        downlink = 2 * _kilobytes(header, n_transmissions=2)
        # S LARs of d_R=3 floats + S int32 labels, S = 2 and 3 classes.
        uplink = 2 * _kilobytes((2 + 3) * (3 + 1))
        assert state['kilobytes'] == pytest.approx(downlink + uplink)

    def test_final_broadcast_is_not_sent(self):
        """The scored header is the locally trained one (omega_k^{T-1})."""
        orch = self._fit(max_epochs=1, local_epochs=1)
        for agent in orch.agents.values():
            head = FedGH._shared_head(agent)
            assert not torch.equal(head.weight, orch.global_header.weight)

    def test_local_training_keeps_clear_of_the_global_header(self):
        """Only eq. 4 moves theta; the agents' optimizer steps must not."""
        calls = []

        class Spy(FedGH):
            def _update_global_header(self, lars):
                calls.append(self.global_header.weight.detach().clone())
                return super()._update_global_header(lars)

        torch.manual_seed(0)
        orch = Spy(
            agents={0: _agent(5), 1: _agent(7)},
            neighbors={0: {1}, 1: {0}},
            optimizer=OPTIMIZER,
        )
        initial = orch.global_header.weight.detach().clone()
        Trainer(
            max_epochs=1,
            accelerator='cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            limit_val_batches=0,
        ).fit(orch, datamodule=TinyDataModule())
        assert torch.equal(calls[0], initial)
