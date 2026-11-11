"""Which forwards may move CenterNorm's running mean: task batches always, SheafFRL pilots per pilot_bn_mode."""

import pytest
import torch
import torch.nn as nn

from src.agents.personalized_classifier import (
    CenterNorm,
    PersonalizedClassifier,
    frozen_center_norm,
)
from src.orchestrators.di_sheaf_frl import DiSheafFRL
from src.orchestrators.non_cooperative import NonCooperativeLearning
from src.orchestrators.sheaf_frl import SheafFRL

D, B = 6, 16


def _agent(batchnorm: bool = True, pilot_loss_weight: float = 0.0) -> PersonalizedClassifier:
    norm = [nn.BatchNorm1d(16)] if batchnorm else []
    encoder = nn.Sequential(nn.Linear(4, 16), *norm, nn.ReLU(), nn.Linear(16, D))
    return PersonalizedClassifier(
        encoder, latent_dim=D, num_classes=2, latent_norm=True, pilot_loss_weight=pilot_loss_weight
    )


def _batch() -> dict:
    y = torch.randint(0, 2, (B,))
    pilots = [torch.randn(B, 4) + 8.0, y.clone(), torch.arange(B)]
    return {0: [torch.randn(B, 4), y], 1: [torch.randn(B, 4), y.clone()], 'global_pilot': pilots}


def _record_inputs(agent: PersonalizedClassifier) -> list[torch.Tensor]:
    inputs: list[torch.Tensor] = []
    agent._latent_norm.register_forward_pre_hook(lambda _m, a: inputs.append(a[0].detach().clone()))
    return inputs


def test_task_forward_updates_running_mean() -> None:
    layer = CenterNorm(D)
    z = torch.randn(B, D) + 3.0
    layer(z)
    assert torch.allclose(layer.running_mean, 0.1 * z.mean(0))


def test_frozen_forward_centres_on_the_batch_without_writing() -> None:
    layer = CenterNorm(D)
    layer.running_mean.fill_(5.0)
    z = (torch.randn(B, D) + 3.0).requires_grad_()
    centred = z.detach() - z.detach().mean(0)

    with frozen_center_norm(layer):
        out = layer(z)

    assert torch.allclose(out, centred / centred.norm(dim=1, keepdim=True))
    assert torch.equal(layer.running_mean, torch.full((D,), 5.0))
    assert layer.momentum == 0.1
    out.sum().backward()
    assert z.grad is not None


@pytest.mark.parametrize('cls', [SheafFRL, DiSheafFRL])
@pytest.mark.parametrize('bn_mode', ['running', 'frozen_batch', 'update'])
@pytest.mark.parametrize('batchnorm', [True, False])
def test_pilot_writes_to_running_mean_follow_pilot_bn_mode(cls, bn_mode, batchnorm) -> None:
    orchestrator = cls(
        agents={0: _agent(batchnorm), 1: _agent(batchnorm)},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: D, 1: D},
        whitening=False,
        posthoc_whitening=False,
        pilot_bn_mode=bn_mode,
    )
    orchestrator.log = lambda *a, **k: None
    orchestrator.log_dict = lambda *a, **k: None
    orchestrator.train()
    orchestrator._task_latent_buffer = {}
    agent = orchestrator.agents['0']
    inputs = _record_inputs(agent)
    bn = next((m for m in agent.modules() if isinstance(m, nn.BatchNorm1d)), None)
    bn_after_task: list[torch.Tensor] = []
    if bn is not None:  # CenterNorm runs after BN, so pass 0 sees BN updated by the task batch only
        agent._latent_norm.register_forward_pre_hook(
            lambda _m, _a: bn_after_task.append(bn.running_mean.clone())
        )

    _, loss = orchestrator._shared_eval(batch=_batch(), batch_idx=0, prefix='train')
    loss.backward()

    assert len(inputs) == 2  # the task forward, then the pilot forward
    task_only = 0.1 * inputs[0].mean(0)
    expected = 0.9 * task_only + 0.1 * inputs[1].mean(0) if bn_mode == 'update' else task_only
    assert torch.allclose(agent._latent_norm.running_mean, expected)
    if bn is not None:
        pilots_moved_bn = not torch.equal(bn.running_mean, bn_after_task[0])
        assert pilots_moved_bn == (bn_mode == 'update')


def test_pilots_update_both_statistics_by_default() -> None:
    orchestrator = SheafFRL(
        agents={0: _agent(), 1: _agent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: D, 1: D},
    )
    assert orchestrator.hparams.pilot_bn_mode == 'update'


def test_running_center_mode_rejects_updating_pilots() -> None:
    with pytest.raises(ValueError, match='pilot_center_mode'):
        SheafFRL(
            agents={0: _agent(), 1: _agent()},
            neighbors={0: {1}, 1: {0}},
            optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
            max_lmb=0.1,
            latent_dims={0: D, 1: D},
            pilot_bn_mode='update',
            pilot_center_mode='running',
        )


def test_running_center_mode_centres_pilots_on_the_task_only_mean() -> None:
    orchestrator = SheafFRL(
        agents={0: _agent(), 1: _agent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
        max_lmb=0.1,
        latent_dims={0: D, 1: D},
        whitening=False,
        posthoc_whitening=False,
        pilot_bn_mode='running',
        pilot_center_mode='running',
    )
    orchestrator.log = lambda *a, **k: None
    orchestrator.log_dict = lambda *a, **k: None
    orchestrator.train()
    orchestrator._task_latent_buffer = {}
    agent = orchestrator.agents['0']
    inputs = _record_inputs(agent)
    outputs: list[torch.Tensor] = []
    agent._latent_norm.register_forward_hook(lambda _m, _a, o: outputs.append(o.detach().clone()))

    _, loss = orchestrator._shared_eval(batch=_batch(), batch_idx=0, prefix='train')
    loss.backward()

    mu = agent._latent_norm.running_mean
    assert torch.allclose(mu, 0.1 * inputs[0].mean(0))
    centred = inputs[1] - mu
    assert torch.allclose(outputs[1], centred / centred.norm(dim=1, keepdim=True))
    assert agent._latent_norm.training


def test_local_pilot_supervision_does_not_write_running_mean() -> None:
    agent = _agent(pilot_loss_weight=1.0)
    orchestrator = NonCooperativeLearning(
        agents={0: agent, 1: _agent()},
        neighbors={0: {1}, 1: {0}},
        optimizer={'_target_': 'torch.optim.SGD', 'lr': 0.1},
    )
    orchestrator.train()

    orchestrator._local_pilot_loss(orchestrator.agents['0'], _batch(), 0).backward()

    assert torch.equal(orchestrator.agents['0']._latent_norm.running_mean, torch.zeros(D))
