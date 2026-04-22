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
from src.datamodules.mhealth_datamodule import MHEALTH_SENSOR_MODALITIES
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

# Maps each MHEALTH_SENSOR_MODALITIES key to the classifier type key above.
_SENSOR_MOD_TO_CLASSIFIER_TYPE: dict[str, str] = {
    'acc_chest': 'accelerometer',
    'acc_ankle': 'accelerometer',
    'acc_wrist': 'accelerometer',
    'gyro_ankle': 'gyroscope',
    'gyro_wrist': 'gyroscope',
    'mag_ankle': 'magnetometer',
    'mag_wrist': 'magnetometer',
    'ecg': 'ecg',
}


def _resolve_agent_classifier_types(datamodule) -> dict[int, str]:
    """Return the classifier type (e.g. 'accelerometer') for each agent.

    When *agent_modalities* is configured on the datamodule, each agent has
    already been sliced to its own 3-D (or 2-D) sensor columns, so we read
    the assignment directly from there.

    As a fallback (no *agent_modalities* set), the modality is inferred from
    each agent's actual dataset feature columns: the sensor type that
    contributes the most columns wins.  This correctly handles both single-
    modality configs (all columns from one sensor type) and multi-modality
    configs (majority vote).
    """
    if datamodule.agent_modalities:
        return {
            i: _SENSOR_MOD_TO_CLASSIFIER_TYPE[mod]
            for i, mod in datamodule.agent_modalities.items()
        }

    # Build reverse map: column_name → classifier type
    col_to_type: dict[str, str] = {}
    for sensor_mod, cols in MHEALTH_SENSOR_MODALITIES.items():
        cls_type = _SENSOR_MOD_TO_CLASSIFIER_TYPE[sensor_mod]
        for col in cols:
            col_to_type[col] = cls_type

    result: dict[int, str] = {}
    for agent_id in range(datamodule.n_agents):
        agent_cols = datamodule.train_datasets[agent_id].feature_cols
        counts: dict[str, int] = {}
        for col in agent_cols:
            t = col_to_type.get(col)
            if t:
                counts[t] = counts.get(t, 0) + 1
        result[agent_id] = max(counts, key=counts.__getitem__) if counts else 'accelerometer'
    return result


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

    agent_classifier_types = _resolve_agent_classifier_types(datamodule)
    print(f'  agent_classifier_types: {agent_classifier_types}')

    num_classes = datamodule.num_classes['label']

    agents = {}
    latent_dims = {}

    output_dim = cfg.model.get('output_dim', 64)
    decoder_hidden_dims = cfg.model.get('decoder_hidden_dims', [256])

    for agent_id, classifier_type in agent_classifier_types.items():
        cls = MODALITY_CLASSIFIER[classifier_type]
        input_channels = MODALITY_CHANNELS[classifier_type]

        agent = cls(
            num_classes=num_classes,
            output_dim=output_dim,
            decoder_hidden_dims=decoder_hidden_dims,
        )
        agents[agent_id] = agent
        latent_dims[agent_id] = agent.encoder.output_dim
        print(
            f'[mhealth] Agent {agent_id}: '
            f'modality={classifier_type} ({cls.__name__}), '
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
