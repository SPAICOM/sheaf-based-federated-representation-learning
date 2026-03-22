# Add root to the path
import sys
from pathlib import Path
import copy

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.utils import remove_non_empty_dir

@hydra.main(
    config_path='../config/hydra/',
    config_name='sheaf_frl_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """The main simulation loop."""

    # Setting the seed
    seed_everything(cfg.seed, workers=True)

    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    # ===================================================
    #                  Wandb Logger
    # ===================================================
    logger = instantiate(cfg.logger)
    if logger is not None:
        logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True)
        )

    # ===================================================
    #              Define the Trainer
    # ===================================================
    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # ===================================================
    #              Define the DataModule
    # ===================================================
    datamodule = instantiate(cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    # ===================================================
    #                 Define the Agents
    # ===================================================
    num_classes = datamodule.num_classes.get('label')
    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    idx_to_name = {}
    agents = {}
    latent_dims = {}  # dictionary to track the latent sizes

    for i, (model_name, in_features) in enumerate(
        datamodule.input_dims.items()
    ):
        idx_to_name[i] = model_name

        model_cfg = copy.deepcopy(cfg.model)
        model_cfg.in_features = in_features
        model_cfg.num_classes = num_classes

        per_agent_dims = getattr(cfg.orchestrator, "per_agent_hidden_dims", {})
        per_agent_dims = {int(k): v for k, v in per_agent_dims.items()}

        if i in per_agent_dims:
            model_cfg.hidden_dims = per_agent_dims[i]

        if model_cfg.get('hidden_dims'):
            latent_dims[i] = model_cfg.hidden_dims[-1]
        else:
            latent_dims[i] = in_features

        agents[i] = instantiate(model_cfg)

    # ===================================================
    #               Define the Orchestrator
    # ===================================================
    neighbors = {int(k): set(v) for k, v in cfg.orchestrator.neighbors.items()}

    print('blibublbiublbiu')
    
    # Instantiate orchestrator
    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims, 
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