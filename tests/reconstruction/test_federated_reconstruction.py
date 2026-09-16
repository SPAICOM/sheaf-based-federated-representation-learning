"""FedAvg reconstruction: common initialization, local pilots, identity transport."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from hydra import compose, initialize_config_dir
from lightning import Callback, Trainer
from PIL import Image

from scripts.reconstruction_experiment import (
    ReconstructionMaskEpochCallback,
    _build_agents,
    _build_orchestrator,
)
from src.agents.cnn_variationalae import CNNVariationalAE
from src.datamodules.masked_cifar_datamodule import MaskedCIFARDataModule
from src.orchestrators.federated import FederatedLearning


def assert_same_agents(orch):
    states = [a.state_dict() for a in orch.agents.values()]
    for key, value in states[0].items():
        assert torch.equal(value, states[1][key]), key


@pytest.mark.parametrize('regional_loss', [False, True])
def test_federated_training_reconstruction_and_raw_latent_transport(
    regional_loss,
):
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        config_dir=str(root / 'config/hydra'), version_base='1.3'
    ):
        cfg = compose(
            config_name='masked_cifar10_vae',
            overrides=[
                'orchestrator=federated',
                'model.latent_dim=4',
                'model.encoder_hidden_dims=[8,16]',
                'model.use_residual=false',
                'model.perceptual_weight=0',
                'model.ssim_weight=0',
                'model.reconstruction_l1_weight=0',
                'model.pilot_loss_weight=1',
                'model.beta=0',
                'model.sample_posterior=false',
                'model.sigma_vae=false',
                f'model.missing_region_loss={str(regional_loss).lower()}',
                'orchestrator.log_latent_diagnostics=false',
            ],
        )
    images = [Image.new('RGB', (32, 32), (i * 3, i * 2, i)) for i in range(64)]
    data = DatasetDict(train=Dataset.from_dict({'img': images}))
    dm = MaskedCIFARDataModule(
        n_agents=2,
        mask_mode='constant_visible_shared',
        constant_visible_fraction=0.5,
        constant_shared_visible_probability=0.5,
        random_focus_regions=['left', 'right'],
        pilot_split=0.25,
        pilot_batch_size=4,
        batch_size=4,
        num_workers=0,
        return_mask=True,
        include_mask_in_input=True,
    )

    class CheckInitialization(Callback):
        checked = False

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if trainer.global_step == 0:
                assert_same_agents(pl_module)
                self.checked = True

    check = CheckInitialization()
    with patch(
        'src.datamodules.masked_cifar_datamodule.load_dataset',
        return_value=data,
    ):
        dm.setup()
        agents, dims = _build_agents(cfg, dm, {})
        orch = _build_orchestrator(cfg, agents, {0: {1}, 1: {0}}, dims)
        assert cfg.orchestrator.alignment_method is None
        assert orch.hparams.alignment_method is None

        def reject_map_fit(*args, **kwargs):
            raise AssertionError('FedAvg must not fit a post-hoc map')

        orch._fit_alignment_maps = reject_map_fit
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
            callbacks=[check, ReconstructionMaskEpochCallback()],
        )
        trainer.fit(orch, datamodule=dm)
        assert check.checked
        assert_same_agents(orch)
        assert agents[0]._loss_step.item() == 8  # private + pilot per batch
        assert not trainer.optimizers[0].state  # reset after final aggregation
        z = torch.randn(2, 4)
        assert orch.send_message(0, 1, z) is z
        trainer.test(orch, datamodule=dm)
        metrics = trainer.callback_metrics
        assert torch.isfinite(metrics['test/private_mse_missing_agent_0'])
        # Identical decoders: receiver's communication MSE equals sender's self MSE.
        assert metrics['test/comm_mse_full_agent_0'] == pytest.approx(
            metrics['test/private_mse_full_agent_1'], abs=1e-6
        )
        assert torch.isfinite(metrics['test/avg_private_lpips_full'])
        assert metrics['test/comm_lpips_full_agent_0'] == pytest.approx(
            metrics['test/private_lpips_full_agent_1'], abs=1e-6
        )


def make_orchestrator(**kwargs):
    agents = {
        i: CNNVariationalAE(
            in_features=3,
            img_size=16,
            encoder_hidden_dims=[4, 8],
            latent_dim=4,
        )
        for i in range(2)
    }
    return FederatedLearning(
        agents,
        {0: {1}, 1: {0}},
        {'_target_': 'torch.optim.Adam', 'lr': 1e-3},
        **kwargs,
    )


def test_resume_does_not_overwrite_client_states():
    orch = make_orchestrator()
    previous = {k: v.clone() for k, v in orch.agents['1'].state_dict().items()}
    orch._trainer = SimpleNamespace(global_step=4, current_epoch=1)
    orch.on_train_start()
    for key, value in previous.items():
        assert torch.equal(value, orch.agents['1'].state_dict()[key])


def test_optimizer_state_retention_is_explicitly_configurable():
    orch = make_orchestrator(reset_optimizer_on_aggregation=False)
    optimizer = torch.optim.Adam(orch.parameters())
    parameter = next(orch.parameters())
    optimizer.state[parameter]['sentinel'] = torch.tensor(1.0)
    orch._trainer = SimpleNamespace(
        optimizers=[optimizer], global_step=0, current_epoch=0
    )
    orch.log_dict = lambda *args, **kwargs: None
    orch._log_train_comm_task_perf = lambda: None
    orch.on_train_epoch_end()
    assert optimizer.state[parameter]['sentinel'].item() == 1
    assert_same_agents(orch)
