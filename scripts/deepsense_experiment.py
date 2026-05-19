"""Experiment script for DeepSense multimodal federated learning.

Instantiates the DeepSense datamodule and orchestrator from Hydra configs,
then runs training with a Lightning Trainer.

Usage
-----
    uv run scripts/deepsense_experiment.py
    uv run scripts/deepsense_experiment.py dataset.n_agents=2
    uv run scripts/deepsense_experiment.py orchestrator=federated
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.agents import (
    DeepSenseLiDARClassifier,
    DeepSenseMMWaveClassifier,
    DeepSenseRGBClassifier,
)
from src.utils import (
    _finish_active_wandb_run,
    remove_non_empty_dir,
)
from src.utils.graph_generator import generate_neighbors

MODALITY_CLASSIFIER = {
    0: DeepSenseRGBClassifier,  # (3, 64, 64)
    1: DeepSenseMMWaveClassifier,  # (1, 16, 64)
    2: DeepSenseLiDARClassifier,  # (2, 32, 32)
}


@hydra.main(
    config_path='../config/hydra/',
    config_name='deepsense_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    # --- Datamodule ---
    datamodule = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    datamodule.prepare_data()
    datamodule.setup()

    print('[deepsense] Datamodule ready')
    print(f'  n_agents     : {datamodule.n_agents}')
    print(f'  num_classes  : {datamodule.num_classes}')
    print(f'  agent_modalities: {datamodule.agent_modalities}')
    for agent_id, mods in datamodule.agent_modalities.items():
        shapes = [datamodule.MODALITY_SHAPES[m] for m in mods]
        print(f'  agent {agent_id} modalities {mods} -> shapes {shapes}')

    num_classes = datamodule.num_classes['label']

    # --- Agents ---
    # Each agent is instantiated with the classifier class that matches its
    # assigned primary modality. MODALITY_CLASSIFIER maps modality index to
    # the right DeepSense*Classifier subclass of PersonalizedClassifier.
    agents = {}
    latent_dims = {}

    global_output_dim = cfg.model.get('output_dim', 64)
    global_decoder_hidden_dims = cfg.model.get('decoder_hidden_dims', [256])
    global_dropout = cfg.model.get('dropout', 0.0)
    global_use_batchnorm = cfg.model.get('use_batchnorm', False)
    global_l1_reg = cfg.model.get('l1_reg', 0.0)
    global_sparsity_type = cfg.model.get('sparsity_type', 'l1')
    per_modality_cfg = cfg.model.get('per_modality', {})

    for agent_id, mod_list in datamodule.agent_modalities.items():
        primary_modality = mod_list[0]
        cls = MODALITY_CLASSIFIER[primary_modality]
        shape = datamodule.MODALITY_SHAPES[primary_modality]

        mod_cfg = per_modality_cfg.get(primary_modality, {})
        output_dim = mod_cfg.get('output_dim', global_output_dim)
        decoder_hidden_dims = list(mod_cfg.get('decoder_hidden_dims', global_decoder_hidden_dims))
        dropout = mod_cfg.get('dropout', global_dropout)
        use_batchnorm = mod_cfg.get('use_batchnorm', global_use_batchnorm)
        l1_reg = mod_cfg.get('l1_reg', global_l1_reg)
        sparsity_type = mod_cfg.get('sparsity_type', global_sparsity_type)

        agent = cls(
            num_classes=num_classes,
            input_channels=shape[0],
            output_dim=output_dim,
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )
        agents[agent_id] = agent
        latent_dims[agent_id] = agent.encoder.output_dim  # BaseEncoder attr
        print(
            f'[deepsense] Agent {agent_id}: '
            f'modality={primary_modality} ({cls.__name__}), '
            f'input_channels={shape[0]}, '
            f'output_dim={output_dim}, '
            f'decoder_hidden_dims={decoder_hidden_dims}, '
            f'latent_dim={latent_dims[agent_id]}'
        )

    # --- Neighbor graph ---
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=datamodule.n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )
    print(f'[deepsense] Neighbors: {neighbors}')

    # --- Orchestrator ---
    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )
    print(f'[deepsense] Orchestrator: {type(orchestrator).__name__}')

    # --- Training ---
    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    run_name = f'{cfg.logger.name}__{type(orchestrator).__name__}'
    logger = instantiate(cfg.logger, name=run_name)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(orchestrator, datamodule=datamodule)
    trainer.test(orchestrator, datamodule=datamodule)

    _finish_active_wandb_run()

    # Clean up temporary directories created by Hydra, WandB, and Lightning
    # These directories can accumulate over multiple experiment runs
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir(cfg.logger.project)


if __name__ == '__main__':
    main()
