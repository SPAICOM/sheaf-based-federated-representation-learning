""""""

# Add root to the path
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import copy

import hydra
from hydra.utils import instantiate

# import omegaconf
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.utils import remove_non_empty_dir

# torch.serialization.safe_globals([DictConfig, OmegaConf])
# torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])


@hydra.main(
    config_path='../config/hydra/',
    config_name='mixture_experts_latents',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main simulation loop."""

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    # Define some usefull paths
    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'

    # Create directories
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # ===================================================
    #                  Wandb Logger
    # ===================================================
    # Instantiate logger (WandB)
    logger = instantiate(cfg.logger)

    # Log full Hydra config to WandB
    if logger is not None:
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    # ===================================================
    #             Define the Trainer
    # ===================================================
    # Instantiate callbacks
    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    # Instantiate Trainer
    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # ===================================================
    #             Define the DataModule
    # ===================================================
    datamodule = instantiate(cfg.dataset)

    # Prepare and setup the data
    datamodule.prepare_data()
    datamodule.setup()

    # ===================================================
    #                Define the Agents
    # ===================================================
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

    # ===================================================
    #                Define the Orchestrator
    # ===================================================
    neighbors = {int(k): set(v) for k, v in cfg.orchestrator.neighbors.items()}

    print('hdjshdjshdjshdjhdjds')
    # Instantiate orchestrator
    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )

    # -------------------------
    # Train
    # -------------------------
    trainer.fit(orchestrator, datamodule=datamodule)

    # Cleaning the working space
    remove_non_empty_dir('./wandb/')
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)

    return None


if __name__ == '__main__':
    main()
