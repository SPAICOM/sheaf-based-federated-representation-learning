"""Heterogeneous-width Federated Representation Learning experiment.

Runs 30 agents split into three complexity tiers by rate-based width scaling:
  - rate=1.00  → agents  0-9   (full-width HeteroCNNClassifier + HeteroMLP)
  - rate=0.50  → agents 10-19  (half-width)
  - rate=0.25  → agents 20-29  (quarter-width)

Supports multirun comparison of sheaf_fmtl, sheaf_frl, and heterofl via:
    python scripts/heterogenous_experiment.py \\
        --multirun orchestrator=sheaf_fmtl,sheaf_frl,heterofl
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
    try:
        import wandb
    except ImportError:
        return None
    if getattr(wandb, 'run', None) is not None:
        wandb.finish()
    return None


def _update_logger_config(logger: Any, payload: dict[str, Any]) -> None:
    if logger is None:
        return None
    sanitized_payload = _json_ready(payload)
    try:
        logger.experiment.config.update(
            sanitized_payload, allow_val_change=True
        )
    except TypeError:
        logger.experiment.config.update(sanitized_payload)
    return None


def _resolve_agent_overrides(
    cfg: DictConfig,
    *,
    n_agents: int,
) -> dict[int, dict[str, Any]]:
    per_agents_cfg = getattr(cfg, 'agents', None)
    if per_agents_cfg is None:
        return {agent_idx: {} for agent_idx in range(n_agents)}

    overrides = {agent_idx: {} for agent_idx in range(n_agents)}
    for agent_idx, values in per_agents_cfg.items():
        overrides[int(agent_idx)] = (
            {}
            if values is None
            else OmegaConf.to_container(values, resolve=True)
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
    if not isinstance(config, (dict, DictConfig)):
        return config

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return config

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return config

    allowed_keys = {n for n in signature.parameters if n != 'self'}
    allowed_keys.update({'_target_', '_recursive_', '_convert_', '_partial_'})
    sanitized = {k: v for k, v in config_dict.items() if k in allowed_keys}
    return OmegaConf.create(sanitized)


def _filter_supported_init_kwargs(
    config: Any, **kwargs: Any
) -> dict[str, Any]:
    """Keep only kwargs accepted by the orchestrator constructor."""
    if not isinstance(config, (dict, DictConfig)):
        return kwargs

    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return kwargs

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    allowed_keys = {n for n in signature.parameters if n != 'self'}
    return {k: v for k, v in kwargs.items() if k in allowed_keys}


def _extract_objective_metric(trainer: Trainer, *, metric_name: str) -> float:
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
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
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
    agent_rates: dict[int, float],
) -> Path:
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
                int(agent_idx): int(nc)
                for agent_idx, nc in datamodule.num_classes.items()
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
    config_name='hetero_rate_30agents_cifar10',
    version_base='1.3',
)
def main(cfg: DictConfig) -> float:
    """Run the heterogeneous-width federated learning experiment.

    Supports three orchestrators via Hydra multirun:
        orchestrator=sheaf_fmtl,sheaf_frl,heterofl

    The ``rates`` dict (required by HeteroFL) is extracted from
    ``cfg.agents`` and injected into the orchestrator automatically;
    orchestrators that do not accept ``rates`` (SheafFMTL, SheafFRL)
    receive it silently dropped by ``_filter_supported_init_kwargs``.
    """
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
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)

    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(
            cfg.agent_classes, resolve=True
        )

    datamodule = instantiate(dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup()

    num_classes = datamodule.num_classes.get('label')
    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    resolved_agent_classes = getattr(datamodule, 'agent_classes', {})
    resolved_num_classes_per_agent = {
        int(agent_idx): int(nc)
        for agent_idx, nc in datamodule.num_classes.items()
        if isinstance(agent_idx, int)
    }
    _update_logger_config(
        logger,
        {
            'resolved_agent_classes': resolved_agent_classes,
            'resolved_num_classes_per_agent': resolved_num_classes_per_agent,
        },
    )

    n_agents = len(datamodule.models)
    per_agents_cfg = _resolve_agent_overrides(cfg, n_agents=n_agents)
    agent_rates = _extract_agent_rates(per_agents_cfg, n_agents=n_agents)

    agents: dict[int, Any] = {}
    latent_dims: dict[int, int] = {}

    for i in range(n_agents):
        model_cfg = copy.deepcopy(cfg.model)
        OmegaConf.set_struct(model_cfg, False)

        # Apply per-agent overrides (rate, encoder_hidden_dims, etc.).
        if i in per_agents_cfg:
            for key, value in per_agents_cfg[i].items():
                setattr(model_cfg, key, value)

        model_cfg.num_classes = num_classes
        model_cfg.in_features = datamodule.input_dims[str(i)]

        agents[i] = instantiate(model_cfg)

        # Infer latent_dim post-instantiation (rate-scaled for HeteroCNN).
        agent_type = str(type(agents[i]))
        if 'CNNClassifier' in agent_type or 'TimmClassifier' in agent_type:
            latent_dims[i] = agents[i].encoder.out_features
        elif model_cfg.get('latent_dim'):
            latent_dims[i] = model_cfg.latent_dim

    _update_logger_config(logger, {'agent_rates': _json_ready(agent_rates)})

    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    orchestrator_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    # rates is consumed by HeteroFL; silently dropped for SheafFMTL/SheafFRL.
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

    trainer.fit(orchestrator, datamodule=datamodule)
    objective_value = _extract_objective_metric(
        trainer, metric_name=objective_metric_name
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
        agent_rates=agent_rates,
    )
    _update_logger_config(logger, {'results_file': str(result_file)})

    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()
    return objective_value


if __name__ == '__main__':
    main()
