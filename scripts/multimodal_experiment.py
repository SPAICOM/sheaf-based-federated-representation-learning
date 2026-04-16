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

    output_dim = cfg.model.get('output_dim', 64)
    decoder_hidden_dims = cfg.model.get('decoder_hidden_dims', [256])

    for agent_id, mod_list in datamodule.agent_modalities.items():
        primary_modality = mod_list[0]
        cls = MODALITY_CLASSIFIER[primary_modality]
        shape = datamodule.MODALITY_SHAPES[primary_modality]

        agent = cls(
            num_classes=num_classes,
            input_channels=shape[0],
            output_dim=output_dim,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        agents[agent_id] = agent
        latent_dims[agent_id] = agent.encoder.output_dim  # BaseEncoder attr
        print(
            f'[deepsense] Agent {agent_id}: '
            f'modality={primary_modality} ({cls.__name__}), '
            f'input_channels={shape[0]}, '
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
    logger = instantiate(cfg.logger)
    logger.name = f'{cfg.logger.name}__{type(orchestrator).__name__}'
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(orchestrator, datamodule=datamodule)
    trainer.test(orchestrator, datamodule=datamodule)


if __name__ == '__main__':
    main()
