"""Heterogeneous-width Federated Representation Learning experiment.

Runs 30 agents split into three complexity tiers by rate-based width scaling:
  - rate=1.00  → agents  0-9   (full-width HeteroCNNClassifier + HeteroMLP)
  - rate=0.50  → agents 10-19  (half-width)
  - rate=0.25  → agents 20-29  (quarter-width)

Runs HeteroFL on MNIST with a 5-layer CNN encoder [64,128,256,512],
batch size 10, 5 local epochs per communication round, 200 rounds total.
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


def _extract_agent_rates(
    per_agents_cfg: dict[int, dict[str, Any]],
    *,
    n_agents: int,
    default_rate: float = 1.0,
) -> dict[int, float]:
    """Build the rates dict required by HeteroFL from per-agent overrides."""
    return {
        i: float(per_agents_cfg[i].get('rate', default_rate))
        for i in range(n_agents)
    }


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


def _log_communication_summary(stats: dict[str, float]) -> None:
    """Print a concise communication cost summary to stdout."""
    rounds = stats.get('total_rounds', 0.0)
    total_mb = stats.get('total_kilobytes', 0.0) / 1024.0
    bytes_per_round = stats.get('bytes_per_round', 0.0)
    kb_per_round = bytes_per_round / 1024.0
    print(
        f'\n[communication] rounds={int(rounds)}  '
        f'total={total_mb:.2f} MB  '
        f'per_round={kb_per_round:.2f} KB ({bytes_per_round:.0f} B)\n'
    )


def _extract_communication_stats(orchestrator: Any) -> dict[str, float]:
    """Extract total and per-round communication cost from the orchestrator."""
    state = getattr(orchestrator, '_communication_by_split', {}).get('train', {})
    total_kb = float(state.get('kilobytes', 0.0))
    total_rounds = float(state.get('rounds', 0.0))
    kb_per_round = total_kb / total_rounds if total_rounds > 0 else 0.0
    return {
        'total_kilobytes': total_kb,
        'total_rounds': total_rounds,
        'kilobytes_per_round': kb_per_round,
        'bytes_per_round': kb_per_round * 1024.0,
        'megabytes_per_round': kb_per_round / 1024.0,
    }


def _persist_run_results(
    *,
    results_path: Path,
    cfg: DictConfig,
    objective_metric_name: str,
    objective_value: float,
    test_results: list[dict[str, Any]] | None,
    datamodule: Any,
    agent_rates: dict[int, float],
    communication_stats: dict[str, float] | None = None,
) -> Path:
    """Persist run metadata and metrics under ``results/``."""
    try:
        hydra_cfg = HydraConfig.get()
    except ValueError:
        hydra_cfg = None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    job_name = 'heterogenous_experiment'
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
        'agent_rates': _json_ready(agent_rates),
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
        'communication': _json_ready(communication_stats or {}),
    }

    result_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return result_file


def _define_wandb_custom_axes(logger: Any) -> None:
    """Register custom WandB x-axes for communication-efficiency charts."""
    try:
        import wandb
        _ = logger.experiment  # trigger wandb.init() if not yet called
        wandb.define_metric(
            'validation/avg_task_performance_epoch',
            step_metric='train/communication_bytes',
        )
        wandb.define_metric(
            'train/communication_kb_epoch',
            step_metric='trainer/global_step',
        )
    except Exception:
        pass


@hydra.main(
    config_path='../config/hydra/',
    config_name='hetero_rate_30agents_mnist',
    version_base='1.3',
)
def main(cfg: DictConfig) -> float:
    """Run the HeteroFL experiment on MNIST with heterogeneous-width agents."""
    seed_everything(cfg.seed, workers=True)

    objective_metric_name = str(
        cfg.get('optimization', {}).get(
            'objective_metric', 'validation/avg_task_performance_epoch'
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
    _define_wandb_custom_axes(logger)

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Convert Hydra config to plain dict for instantiation
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    # Pass agent_classes to datamodule for class-partitioned data splits
    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(
            cfg.agent_classes, resolve=True
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
    agent_rates = _extract_agent_rates(per_agents_cfg, n_agents=n_agents)

    # Instantiate agent models for each data modality
    # Each agent can have different model architectures (in cfg.agents)
    for i in range(n_agents):
        # Create a deep copy of the base model config to avoid
        # modifying shared config
        model_cfg = copy.deepcopy(cfg.model)

        # Apply per-agent overrides from config (e.g., rate, encoder_hidden_dims)
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
        # instantiation (rate-scaled for HeteroCNN).
        agent_type = str(type(agents[i]))
        if 'TimmClassifier' in agent_type or 'CNNClassifier' in agent_type:
            latent_dims[i] = agents[i].encoder.out_features

    _update_logger_config(logger, {'agent_rates': _json_ready(agent_rates)})

    # Generate neighbor graph for federated learning communication
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    # Instantiate the orchestrator (e.g., HeteroFL, SheafFRL, SheafFMTL).
    # rates is consumed by HeteroFL; silently dropped for SheafFMTL/SheafFRL.
    orchestrator_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    orchestrator_kwargs = _filter_supported_init_kwargs(
        orchestrator_cfg,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        rates=agent_rates,
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

    communication_stats = _extract_communication_stats(orchestrator)
    _log_communication_summary(communication_stats)

    result_file = _persist_run_results(
        results_path=RESULTS_PATH,
        cfg=cfg,
        objective_metric_name=objective_metric_name,
        objective_value=objective_value,
        test_results=test_results,
        datamodule=datamodule,
        agent_rates=agent_rates,
        communication_stats=communication_stats,
    )
    _update_logger_config(
        logger,
        {'results_file': str(result_file), 'communication': communication_stats},
    )

    # Clean up temporary directories created by Hydra, WandB, and Lightning
    # These directories can accumulate over multiple experiment runs
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    _finish_active_wandb_run()
    return objective_value


if __name__ == '__main__':
    main()
