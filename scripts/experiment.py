"""Script for Sheaf-based Federated Representation Learning experiments.

This script orchestrates the complete training pipeline for federated learning
experiments with Sheaf regularization, including data loading, agent
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
    config_name='timm_agents_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    """Run the Sheaf-based Federated Representation Learning experiment.

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

    # Convert Hydra config to plain dict for instantiation
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Pass agent_rotations to datamodule for data augmentation (e.g., rotation)
    if 'agent_rotations' in cfg:
        dataset_cfg['agent_rotations'] = OmegaConf.to_container(
            cfg.agent_rotations, resolve=True
        )

    # Pass agent_classes to datamodule for class-partitioned data splits
    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(
            cfg.agent_classes, resolve=True
        )

    # Pass 'agents' config to datamodule only for SemanticDataModule
    # SemanticDataModule loads pre-computed embeddings from HuggingFace
    # and needs per-agent model configs (e.g., which embedding model to use).
    # Identified by missing 'data_key' which ClassificationDataModule uses.
    if 'agents' in cfg and 'data_key' not in dataset_cfg:
        dataset_cfg['agents'] = OmegaConf.to_container(
            cfg.agents, resolve=True
        )

    datamodule = instantiate(dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup()

    num_classes = datamodule.num_classes.get('label')
    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    agents = {}
    latent_dims = {}

    # Get per-agent configuration from config (if exists)
    per_agents_cfg = getattr(cfg, 'agents', None)
    n_agents = len(datamodule.models)

    # Instantiate agent models for each data modality
    # Each agent can have different model architectures (in cfg.agents)
    for i in range(n_agents):
        # Create a deep copy of the base model config to avoid
        # modifying shared config
        model_cfg = copy.deepcopy(cfg.model)

        # Apply per-agent overrides from config (e.g., different model_name)
        # Only set keys that exist in the model config to avoid Hydra errors
        if per_agents_cfg is not None and i in per_agents_cfg:
            agent_override = per_agents_cfg[i]
            for key, value in agent_override.items():
                if hasattr(model_cfg, key) or key in model_cfg:
                    setattr(model_cfg, key, value)

        # Classification labels remain in the global label space even when an
        # agent only observes a subset of classes, so model heads must keep
        # the global output dimension.
        model_cfg.num_classes = num_classes

        # Set in_features from datamodule input dimensions
        # Required for LatentClassifier which needs explicit input dimension
        if hasattr(model_cfg, 'in_features') or 'in_features' in model_cfg:
            model_cfg.in_features = datamodule.input_dims[str(i)]

        # Apply per-agent encoder_hidden_dims for MLP decoder config
        per_agent_hidden_dims = getattr(cfg.model, 'encoder_hidden_dims', None)
        if per_agent_hidden_dims is not None and hasattr(
            per_agent_hidden_dims, 'items'
        ):
            per_agent_hidden_dims = {
                int(k): v for k, v in per_agent_hidden_dims.items()
            }
            if i in per_agent_hidden_dims:
                model_cfg.encoder_hidden_dims = per_agent_hidden_dims[i]

        # Set latent_dim from config for LatentClassifier
        if model_cfg.get('latent_dim'):
            latent_dims[i] = model_cfg.latent_dim

        # Instantiate the agent model
        agents[i] = instantiate(model_cfg)

        # For TimmClassifier/CNNClassifier: infer latent_dims after
        # instantiation.
        agent_type = str(type(agents[i]))
        if 'TimmClassifier' in agent_type or 'CNNClassifier' in agent_type:
            latent_dims[i] = agents[i].encoder.out_features

    # Generate neighbor graph for federated learning communication
    # Used by SheafFRL to create cross-covariance matrices between agents
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    # Instantiate the orchestrator (e.g., SheafFRL for federated
    # learning with Sheaf regularization)
    # Required parameters:
    # - agents: dictionary of instantiated model agents
    # - neighbors: graph defining which agents communicate
    # - latent_dims: encoder output dims (used for Stiefel matrix shapes)
    # - optimizer: optimizer configuration from Hydra config
    orchestrator = instantiate(
        cfg.orchestrator,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        _convert_='all',
        _recursive_=False,
    )

    # Run training
    trainer.fit(orchestrator, datamodule=datamodule)
    trainer.test(orchestrator, datamodule=datamodule)

    # Clean up temporary directories created by Hydra, WandB, and Lightning
    # These directories can accumulate over multiple experiment runs
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)


if __name__ == '__main__':
    main()
