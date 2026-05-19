"""Experiment script for MFeat multi-view federated learning.

Instantiates the MFeat datamodule and orchestrator from Hydra configs,
then runs training with a Lightning Trainer.

Usage
-----
    uv run scripts/mfeat_experiment.py
    uv run scripts/mfeat_experiment.py dataset.n_agents=5
    uv run scripts/mfeat_experiment.py orchestrator=non_cooperative
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.agents import MFeatMLPClassifier
from src.utils import (
    _finish_active_wandb_run,
    remove_non_empty_dir,
)
from src.utils.graph_generator import generate_neighbors


@hydra.main(
    config_path='../config/hydra/',
    config_name='mfeat_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    datamodule.prepare_data()
    datamodule.setup()

    print('[mfeat] Datamodule ready')
    print(f'  n_agents        : {datamodule.n_agents}')
    print(f'  num_classes     : {datamodule.num_classes}')
    print(f'  input_shape     : {datamodule.input_shape}')
    print(f'  input_dims      : {datamodule.input_dims}')
    print(f'  agent_modalities: {datamodule.agent_modalities}')

    num_classes = datamodule.num_classes['label']

    decoder_hidden_dims = list(cfg.model.decoder_hidden_dims)
    dropout = cfg.model.dropout
    use_batchnorm = cfg.model.use_batchnorm
    l1_reg = cfg.model.l1_reg
    sparsity_type = cfg.model.sparsity_type

    agents = {}
    latent_dims = {}

    for agent_id, modality in datamodule.agent_modalities.items():
        agent_cfg = cfg.model.agents[agent_id]
        hidden_dims = list(agent_cfg.hidden_dims) if agent_cfg.get('hidden_dims') else None
        agent = MFeatMLPClassifier(
            input_dim=agent_cfg.input_dim,
            num_classes=num_classes,
            output_dim=agent_cfg.get('output_dim', 64),
            encoder_hidden_dims=hidden_dims,
            encoder_dropout=agent_cfg.get('encoder_dropout', 0.3),
            decoder_hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            l1_reg=l1_reg,
            sparsity_type=sparsity_type,
        )
        agents[agent_id] = agent
        latent_dims[agent_id] = agent.encoder.output_dim
        print(
            f'[mfeat] Agent {agent_id}: modality={modality}, '
            f'input_dim={agent.encoder.input_dim}, '
            f'latent_dim={latent_dims[agent_id]}'
        )

    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=datamodule.n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )
    print(f'[mfeat] Neighbors: {neighbors}')

    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )
    print(f'[mfeat] Orchestrator: {type(orchestrator).__name__}')

    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    run_name = f'{cfg.logger.name}__{type(orchestrator).__name__}'
    logger = instantiate(cfg.logger, name=run_name)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(orchestrator, datamodule=datamodule)
    trainer.test(orchestrator, datamodule=datamodule)

    _finish_active_wandb_run()

    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir(cfg.logger.project)


if __name__ == '__main__':
    main()
