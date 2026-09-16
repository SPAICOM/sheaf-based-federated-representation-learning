"""Regression tests for the matched reconstruction experimental protocol."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from datasets import Dataset, DatasetDict
from hydra import compose, initialize_config_dir
from lightning import Trainer
from PIL import Image

from scripts.plot_reconstruction_overlap_metrics import _aggregate
from scripts.reconstruction_experiment import (
    ReconstructionMaskEpochCallback,
    _build_agents,
    _build_orchestrator,
)
from src.agents.cnn_variationalae import CNNVariationalAE
from src.datamodules.masked_cifar_datamodule import MaskedCIFARDataModule


def make_agent(**kwargs):
    return CNNVariationalAE(
        in_features=4,
        out_features=3,
        img_size=32,
        encoder_hidden_dims=[8, 16],
        latent_dim=4,
        use_batchnorm=False,
        **kwargs,
    )


def test_regional_mse_and_posterior_gradients():
    agent = make_agent(missing_region_loss=True, beta=0)
    target = torch.zeros(2, 3, 32, 32)
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, :, :8] = 1
    prediction = torch.ones_like(target) * 2
    prediction[:, :, :, :8] = 1
    assert agent.reconstruction_mse(prediction, target, mask).item() == 4.25
    x = torch.rand(2, 4, 32, 32)
    mu = agent.encode(x)
    agent.compute_loss(agent.decode_posterior(mu), target, mask).backward()
    grad = agent.encoder.fc_logvar.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    agent.eval()
    assert torch.equal(agent(x), agent(x))


def test_deterministic_control_and_normalized_kl():
    agent = make_agent(sample_posterior=False, normalize_kl=True, free_bits=0)
    x = torch.rand(2, 4, 32, 32)
    assert torch.equal(agent(x), agent(x))
    agent._last_mu = torch.ones(2, 4)
    agent._last_logvar = torch.zeros(2, 4)
    assert agent.kl_penalty().item() == 0.5


def make_dm():
    data = DatasetDict(
        train=Dataset.from_dict(
            {
                'img': [
                    Image.new(
                        'RGB',
                        (32, 32),
                        (i * 3 % 255, i * 7 % 255, i * 11 % 255),
                    )
                    for i in range(80)
                ]
            }
        )
    )
    dm = MaskedCIFARDataModule(
        n_agents=2,
        mask_mode='constant_visible_shared',
        constant_visible_fraction=0.5,
        constant_shared_visible_probability=0.5,
        random_focus_regions=['full', 'full'],
        pilot_split=0.2,
        batch_size=4,
        pilot_batch_size=8,
        num_workers=0,
        return_mask=True,
        include_mask_in_input=True,
    )
    return dm, data


def test_epoch_masks_keep_budget_pairing_and_fixed_evaluation():
    dm, data = make_dm()
    with patch(
        'src.datamodules.masked_cifar_datamodule.load_dataset',
        return_value=data,
    ):
        dm.setup()
    cb = ReconstructionMaskEpochCallback()
    trainer = SimpleNamespace(datamodule=dm, current_epoch=0)
    initial = dm.pilot_datasets[0]._mask(0).clone()
    test_mask = dm.test_datasets[0]._mask(0).clone()
    cb.on_train_epoch_start(trainer, None)
    a, b = [dm.pilot_datasets[i]._mask(0) for i in range(2)]
    assert a.sum() == b.sum() == 512
    assert (a * b).sum() == 256
    assert not torch.equal(a, initial)
    cb.on_validation_start(trainer, None)
    assert torch.equal(dm.pilot_datasets[0]._mask(0), initial)
    cb.on_validation_end(trainer, None)
    assert torch.equal(dm.pilot_datasets[0]._mask(0), a)
    assert torch.equal(dm.test_datasets[0]._mask(0), test_mask)


@pytest.mark.parametrize('orchestrator', ['sheaf_frl', 'non_cooperative'])
def test_real_training_and_communication(orchestrator):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(root / 'config/hydra'), version_base='1.3'
    ):
        cfg = compose(
            config_name='masked_cifar10_vae',
            overrides=[
                f'orchestrator={orchestrator}',
                'model.latent_dim=4',
                'model.encoder_hidden_dims=[8,16]',
                'model.use_batchnorm=false',
                'model.use_residual=false',
                'orchestrator.warmup_epochs=0',
            ],
        )
    dm, data = make_dm()
    with patch(
        'src.datamodules.masked_cifar_datamodule.load_dataset',
        return_value=data,
    ):
        dm.setup()
        agents, dims = _build_agents(cfg, dm, {})
        orch = _build_orchestrator(cfg, agents, {0: {1}, 1: {0}}, dims)
        initial = agents[0].encoder.fc_logvar.weight.detach().clone()
        trainer = Trainer(
            max_epochs=2,
            limit_train_batches=2,
            limit_val_batches=1,
            limit_test_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='cpu',
            callbacks=[ReconstructionMaskEpochCallback()],
        )
        trainer.fit(orch, datamodule=dm)
        assert not torch.equal(agents[0].encoder.fc_logvar.weight, initial)
        # Two local loss calls (private + pilot), two batches, two epochs.
        assert agents[0]._loss_step.item() == 8
        trainer.test(orch, datamodule=dm)
        for metric in (
            'private_mse_missing',
            'comm_mse_full',
            'comm_mse_sender_missing',
        ):
            assert torch.isfinite(
                trainer.callback_metrics[f'test/{metric}_agent_0']
            )
        assert cfg.orchestrator.comm_task_coeff == 0


def test_plot_variability_is_across_seeds_not_agents():
    rows = pd.DataFrame(
        [
            {
                'orchestrator': 'SheafFRL',
                'x_value': 0.5,
                'seed': seed,
                'private_mse_missing': value,
                'comm_mse': value,
            }
            for seed, values in [(1, [0.0, 2.0]), (2, [2.0, 4.0])]
            for value in values
        ]
    )
    result = _aggregate(rows).iloc[0]
    assert result.private_mse_missing_mean == 2
    assert result.private_mse_missing_std == pytest.approx(2**0.5)


def test_result_protocol_keeps_seeds_and_separates_setups(
    tmp_path, monkeypatch
):
    from scripts.plot_reconstruction_overlap_metrics import _load_results
    from scripts.reconstruction_experiment import _comparison_id, _save_results

    monkeypatch.chdir(tmp_path)
    root = Path(__file__).resolve().parents[2]
    configs = []
    for name in ('sheaf_frl', 'non_cooperative'):
        with initialize_config_dir(
            config_dir=str(root / 'config/hydra'), version_base='1.3'
        ):
            configs.append(
                compose(
                    config_name='masked_cifar10_vae',
                    overrides=[f'orchestrator={name}', 'logger.name=test'],
                )
            )
    assert _comparison_id(configs[0]) == _comparison_id(configs[1])
    for seed in (42, 43):
        for cfg, name in zip(configs, ('SheafFRL', 'NonCooperativeLearning')):
            cfg.seed = seed
            metrics = {
                f'test/private_mse_missing_agent_{i}': 0.1 for i in range(2)
            }
            _save_results(cfg, metrics, 2, name)
    args = SimpleNamespace(
        results_dir=tmp_path / 'results/reconstruction',
        mask_mode='constant_visible_shared',
        seed=None,
        comparison_id=None,
        min_value=0,
        max_value=1,
    )
    data = _load_results(args)
    assert len(data) == 8
    configs[0].model.beta = 0.2
    _save_results(configs[0], {}, 2, 'SheafFRL')
    with pytest.raises(SystemExit, match='Multiple setups'):
        _load_results(args)


def test_beta_warmup_and_empty_region():
    agent = make_agent(
        missing_region_loss=True, beta=0.01, beta_warmup_steps=2
    )
    target = torch.zeros(2, 3, 32, 32)
    x = torch.rand(2, 4, 32, 32)
    for expected in (0, 0.005, 0.01):
        loss = agent.compute_loss(agent(x), target, torch.ones(2, 1, 32, 32))
        assert torch.isfinite(loss)
        assert agent.last_loss_components['mse_missing'] == 0
        assert agent.last_loss_components['beta'].item() == pytest.approx(
            expected
        )
    agent.eval()
    agent.compute_loss(agent(x), target, torch.ones(2, 1, 32, 32))
    assert agent._loss_step.item() == 3
