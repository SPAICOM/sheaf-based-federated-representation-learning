"""Experiment script for mHealth multimodal federated learning.

Instantiates the mHealth datamodule and orchestrator from Hydra configs,
then runs training with a Lightning Trainer.

Usage
-----
    uv run scripts/mhealth_experiment.py
    uv run scripts/mhealth_experiment.py dataset.n_agents=2
    uv run scripts/mhealth_experiment.py orchestrator=federated
"""

import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.agents import (
    MHealthAccelerometerClassifier,
    MHealthECGClassifier,
    MHealthGyroscopeClassifier,
    MHealthMagnetometerClassifier,
)
from src.utils.graph_generator import generate_neighbors

MODALITY_CLASSIFIER = {
    'accelerometer': MHealthAccelerometerClassifier,
    'gyroscope': MHealthGyroscopeClassifier,
    'magnetometer': MHealthMagnetometerClassifier,
    'ecg': MHealthECGClassifier,
}

MODALITY_CHANNELS = {
    'accelerometer': 3,
    'gyroscope': 3,
    'magnetometer': 3,
    'ecg': 2,
}


def _infer_agent_modalities(datamodule) -> dict[int, list[str]]:
    """Infer agent modalities from the datamodule's feature columns.

    Groups feature columns by sensor type and assigns each agent a primary
    modality based on its feature columns.
    """
    feature_cols = datamodule.feature_cols

    modality_features = {
        'accelerometer': [c for c in feature_cols if 'acc_' in c],
        'gyroscope': [c for c in feature_cols if 'gyro_' in c],
        'magnetometer': [c for c in feature_cols if 'mag_' in c],
        'ecg': [c for c in feature_cols if 'ecg_' in c],
    }

    agent_modalities = {}
    for agent_id in range(datamodule.n_agents):
        agent_features = [c for c in feature_cols]

        for mod_name, mod_features in modality_features.items():
            if any(f in agent_features for f in mod_features):
                agent_modalities[agent_id] = [mod_name]
                break
        else:
            agent_modalities[agent_id] = ['accelerometer']

    return agent_modalities


@hydra.main(
    config_path='../config/hydra/',
    config_name='mhealth_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    datamodule.prepare_data()
    datamodule.setup()

    print('[mhealth] Datamodule ready')
    print(f'  n_agents     : {datamodule.n_agents}')
    print(f'  num_classes  : {datamodule.num_classes}')
    print(f'  feature_cols : {datamodule.feature_cols}')
    print(f'  input_shape  : {datamodule.input_shape}')

    agent_modalities = _infer_agent_modalities(datamodule)
    print(f'  agent_modalities: {agent_modalities}')

    num_classes = datamodule.num_classes['label']

    agents = {}
    latent_dims = {}

    output_dim = cfg.model.get('output_dim', 64)
    decoder_hidden_dims = cfg.model.get('decoder_hidden_dims', [256])

    for agent_id, mod_list in agent_modalities.items():
        primary_modality = mod_list[0]
        cls = MODALITY_CLASSIFIER[primary_modality]
        input_channels = MODALITY_CHANNELS[primary_modality]

        agent = cls(
            num_classes=num_classes,
            output_dim=output_dim,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        agents[agent_id] = agent
        latent_dims[agent_id] = agent.encoder.output_dim
        print(
            f'[mhealth] Agent {agent_id}: '
            f'modality={primary_modality} ({cls.__name__}), '
            f'input_channels={input_channels}, '
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
    print(f'[mhealth] Neighbors: {neighbors}')

    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )
    print(f'[mhealth] Orchestrator: {type(orchestrator).__name__}')

    callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]
    run_name = f'{cfg.logger.name}__{type(orchestrator).__name__}'
    logger = instantiate(cfg.logger, name=run_name)
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(orchestrator, datamodule=datamodule)
    trainer.test(orchestrator, datamodule=datamodule)


if __name__ == '__main__':
    main()
