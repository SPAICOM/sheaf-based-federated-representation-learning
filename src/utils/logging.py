import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf


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
            {}
            if values is None
            else OmegaConf.to_container(values, resolve=True)
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

    allowed_keys = {name for name in signature.parameters if name != 'self'}
    allowed_keys.update({'_target_', '_recursive_', '_convert_', '_partial_'})
    sanitized = {
        key: value for key, value in config_dict.items() if key in allowed_keys
    }
    return OmegaConf.create(sanitized)


def _filter_supported_init_kwargs(
    config: Any, **kwargs: Any
) -> dict[str, Any]:
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

    allowed_keys = {name for name in signature.parameters if name != 'self'}
    return {key: value for key, value in kwargs.items() if key in allowed_keys}


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
