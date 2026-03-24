"""Script for mixture-of-experts experiments with latent representations.

This script orchestrates the complete training pipeline for federated learning
experiments with multiple expert models, including data loading, agent
instantiation, orchestrator setup, and training execution.
"""

import copy
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.utils import remove_non_empty_dir
from src.utils.graph_generator import generate_neighbors


@hydra.main(
    config_path='../config/hydra/',
    config_name='mixture_experts_latents',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """Run the mixture-of-experts latent representation experiment.

    This main loop orchestrates the complete training pipeline:
    1. Initialize random seed for reproducibility.
    2. Set up WandB logging.
    3. Configure the Lightning Trainer with callbacks.
    4. Instantiate the data module and load datasets.
    5. Create agent models for each data modality.
    6. Generate neighbor graph and instantiate the orchestrator.
    7. Execute training with the orchestrator.
    8. Clean up temporary directories.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing all experiment parameters.

    Raises
    ------
    ValueError
        If the 'label' attribute is not categorical in the dataset.
    """
    seed_everything(cfg.seed, workers=True)

    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'

    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    logger = instantiate(cfg.logger)

    if logger is not None:
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    datamodule = instantiate(cfg.dataset)

    datamodule.prepare_data()
    datamodule.setup()

    num_classes = datamodule.num_classes.get('label')

    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    idx_to_name = {}
    agents = {}

    for i, (model_name, in_features) in enumerate(
        datamodule.input_dims.items()
    ):
        idx_to_name[i] = model_name

        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.in_features = in_features
        model_cfg.num_classes = num_classes

        agents[i] = instantiate(model_cfg)

    n_agents = len(datamodule.models)
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )

    trainer.fit(orchestrator, datamodule=datamodule)

    remove_non_empty_dir('./wandb/')
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)


if __name__ == '__main__':
    main()
