"""Script for Sheaf-based Federated Representation Learning experiments.

This script orchestrates the complete training pipeline for federated learning
experiments with Sheaf regularization, including data loading, agent
instantiation, orchestrator setup, and training execution.
"""

import copy
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.utils import remove_non_empty_dir
from src.utils.graph_generator import generate_neighbors


def _finish_active_wandb_run() -> None:
    """Close any active WandB run before starting a new trial."""
    try:
        import wandb
    except ImportError:
        return None

    if getattr(wandb, 'run', None) is not None:
        wandb.finish()
    return None


def _update_logger_config(logger: Any, payload: dict[str, Any]) -> None:
    """Update WandB config while allowing resolved metadata overrides."""
    if logger is None:
        return None
    sanitized_payload = _json_ready(payload)
    try:
        logger.experiment.config.update(
            sanitized_payload,
            allow_val_change=True,
        )
    except TypeError:
        logger.experiment.config.update(sanitized_payload)
    return None


def _resolve_agent_overrides(
    cfg: DictConfig,
    *,
    n_agents: int,
) -> dict[int, dict[str, Any]]:
    """Normalize per-agent model override config for the resolved agent set."""
    per_agents_cfg = getattr(cfg, 'agents', None)
    if per_agents_cfg is None:
        return {agent_idx: {} for agent_idx in range(n_agents)}

    overrides = {agent_idx: {} for agent_idx in range(n_agents)}
    for agent_idx, values in per_agents_cfg.items():
        overrides[int(agent_idx)] = (
            {} if values is None else OmegaConf.to_container(values, resolve=True)
        )
    return overrides


def _sanitize_instantiation_config(config: Any) -> Any:
    """Drop unsupported config keys for targets that do not accept kwargs."""
    if not isinstance(config, (dict, DictConfig)):
        return config

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return config

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return config

    allowed_keys = {
        name
        for name in signature.parameters
        if name != 'self'
    }
    allowed_keys.update({'_target_', '_recursive_', '_convert_', '_partial_'})
    sanitized = {
        key: value
        for key, value in config_dict.items()
        if key in allowed_keys
    }
    return OmegaConf.create(sanitized)


def _filter_supported_init_kwargs(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep only kwargs accepted by the target constructor."""
    if not isinstance(config, (dict, DictConfig)):
        return kwargs

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return kwargs

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    allowed_keys = {
        name
        for name in signature.parameters
        if name != 'self'
    }
    return {
        key: value
        for key, value in kwargs.items()
        if key in allowed_keys
    }


def _extract_objective_metric(
    trainer: Trainer,
    *,
    metric_name: str,
) -> float:
    """Extract a scalar objective metric from Lightning callback metrics."""
    if metric_name not in trainer.callback_metrics:
        available = ', '.join(sorted(trainer.callback_metrics.keys()))
        raise ValueError(
            f'Objective metric {metric_name!r} not found. '
            f'Available metrics: {available}'
        )

    metric = trainer.callback_metrics[metric_name]
    if hasattr(metric, 'detach'):
        metric = metric.detach()
    if hasattr(metric, 'cpu'):
        metric = metric.cpu()
    if hasattr(metric, 'item'):
        metric = metric.item()
    return float(metric)


def _json_ready(value: Any) -> Any:
    """Convert nested experiment metadata into JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _persist_run_results(
    *,
    results_path: Path,
    cfg: DictConfig,
    objective_metric_name: str,
    objective_value: float,
    test_results: list[dict[str, Any]] | None,
    datamodule: Any,
) -> Path:
    """Persist run metadata and metrics under ``results/``."""
    try:
        hydra_cfg = HydraConfig.get()
    except ValueError:
        hydra_cfg = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    job_name = 'experiment'
    job_num = None
    override_dirname = ''
    run_dir_value = None
    if hydra_cfg is not None:
        job_name = hydra_cfg.job.name
        job_num = hydra_cfg.job.get('num', None)
        override_dirname = hydra_cfg.job.get('override_dirname', '')
        run_dir_value = hydra_cfg.runtime.output_dir

    run_dir = results_path / job_name
    run_dir.mkdir(exist_ok=True, parents=True)

    file_stem_parts = [timestamp]
    if job_num is not None:
        file_stem_parts.append(f'job_{job_num}')
    if override_dirname:
        file_stem_parts.append('trial')
    result_file = run_dir / f'{"__".join(file_stem_parts)}.json'

    payload = {
        'saved_at': timestamp,
        'hydra': {
            'job_name': job_name,
            'job_num': job_num,
            'override_dirname': override_dirname,
            'run_dir': run_dir_value,
        },
        'objective_metric': objective_metric_name,
        'objective_value': objective_value,
        'test_results': _json_ready(test_results),
        'resolved_agent_classes': _json_ready(
            getattr(datamodule, 'agent_classes', {})
        ),
        'resolved_num_classes_per_agent': _json_ready(
            {
                int(agent_idx): int(num_agent_classes)
                for agent_idx, num_agent_classes in datamodule.num_classes.items()
                if isinstance(agent_idx, int)
            }
        ),
        'config': _json_ready(OmegaConf.to_container(cfg, resolve=True)),
    }

    result_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return result_file


@hydra.main(
    config_path='../config/hydra/',
    config_name='timm_agents_experiment',
    version_base='1.3',
)
def main(cfg: DictConfig) -> float:
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

    objective_metric_name = str(
        cfg.get('optimization', {}).get(
            'objective_metric', 'validation/global_task_performance_epoch'
        )
    )
    run_test = bool(cfg.get('optimization', {}).get('run_test', True))

    CURRENT: Path = Path('.')
    RESULTS_PATH: Path = CURRENT / 'results/'
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    _finish_active_wandb_run()
    logger = instantiate(cfg.logger)
    _update_logger_config(
        logger,
        OmegaConf.to_container(cfg, resolve=True),
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

    resolved_agent_classes = getattr(datamodule, 'agent_classes', {})
    resolved_num_classes_per_agent = {
        int(agent_idx): int(num_agent_classes)
        for agent_idx, num_agent_classes in datamodule.num_classes.items()
        if isinstance(agent_idx, int)
    }
    _update_logger_config(
        logger,
        {
            'resolved_agent_classes': resolved_agent_classes,
            'resolved_num_classes_per_agent': resolved_num_classes_per_agent,
        },
    )

    num_classes = datamodule.num_classes.get('label')
    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    agents = {}
    latent_dims = {}

    n_agents = len(datamodule.models)
    per_agents_cfg = _resolve_agent_overrides(cfg, n_agents=n_agents)

    # Instantiate agent models for each data modality
    # Each agent can have different model architectures (in cfg.agents)
    for i in range(n_agents):
        # Create a deep copy of the base model config to avoid
        # modifying shared config
        model_cfg = copy.deepcopy(cfg.model)

        # Apply per-agent overrides from config (e.g., different model_name)
        # Only set keys that exist in the model config to avoid Hydra errors
        if i in per_agents_cfg:
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
    orchestrator_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    orchestrator_kwargs = _filter_supported_init_kwargs(
        orchestrator_cfg,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
    )
    orchestrator = instantiate(
        orchestrator_cfg,
        **orchestrator_kwargs,
        _convert_='all',
        _recursive_=False,
    )

    # Run training
    trainer.fit(orchestrator, datamodule=datamodule)
    objective_value = _extract_objective_metric(
        trainer,
        metric_name=objective_metric_name,
    )
    test_results = None
    if run_test:
        test_results = trainer.test(orchestrator, datamodule=datamodule)

    result_file = _persist_run_results(
        results_path=RESULTS_PATH,
        cfg=cfg,
        objective_metric_name=objective_metric_name,
        objective_value=objective_value,
        test_results=test_results,
        datamodule=datamodule,
    )
    _update_logger_config(logger, {'results_file': str(result_file)})

    # Clean up temporary directories created by Hydra, WandB, and Lightning
    # These directories can accumulate over multiple experiment runs
    #remove_non_empty_dir('./multirun/')
    #remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()
    return objective_value


if __name__ == '__main__':
    main()
