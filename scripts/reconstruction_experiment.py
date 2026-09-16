"""Multi-agent masked-image reconstruction experiment.

Dedicated entrypoint for CIFAR-10 masked reconstruction/inpainting. It mirrors
the high-level multi-agent flow, but drops classification-only machinery:
target-class groups, class-overlap graphs, PID over discrete labels, and Optuna
studies.

Usage:
    python scripts/reconstruction_experiment.py
"""

from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
import torch
from hydra.utils import get_class, instantiate
from lightning import Callback, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid

from src.utils.graph_generator import generate_neighbors


def _finish_active_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, 'run', None) is not None:
        wandb.finish()


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
        except (RuntimeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _update_logger_config(logger: Any, payload: dict[str, Any]) -> None:
    if logger is None:
        return
    sanitized = _json_ready(payload)
    try:
        logger.experiment.config.update(sanitized, allow_val_change=True)
    except TypeError:
        logger.experiment.config.update(sanitized)


def _sanitize_instantiation_config(config: Any) -> Any:
    if not isinstance(config, (dict, DictConfig)):
        return config
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return config

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        return config

    allowed = {name for name in signature.parameters if name != 'self'}
    allowed.update({'_target_', '_recursive_', '_convert_', '_partial_'})
    return OmegaConf.create(
        {key: value for key, value in config_dict.items() if key in allowed}
    )


def _filter_supported_init_kwargs(
    config: Any, **kwargs: Any
) -> dict[str, Any]:
    if not isinstance(config, (dict, DictConfig)):
        return kwargs
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict) or '_target_' not in config_dict:
        return kwargs

    target = get_class(config_dict['_target_'])
    signature = inspect.signature(target.__init__)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        return kwargs

    allowed = {name for name in signature.parameters if name != 'self'}
    return {key: value for key, value in kwargs.items() if key in allowed}


class ReconstructionMaskEpochCallback(Callback):
    """Resample training views jointly; validation/test use fixed pilot views."""

    @staticmethod
    def _set(dm, groups, epoch):
        for group in groups:
            for dataset in getattr(dm, group, {}).values():
                dataset.mask_epoch.fill_(epoch)

    def on_train_epoch_start(self, trainer, pl_module):
        self._set(
            trainer.datamodule,
            ('train_datasets', 'pilot_datasets'),
            trainer.current_epoch + 1,
        )

    def on_validation_start(self, trainer, pl_module):
        self._set(trainer.datamodule, ('pilot_datasets',), 0)

    def on_validation_end(self, trainer, pl_module):
        self._set(
            trainer.datamodule, ('pilot_datasets',), trainer.current_epoch + 1
        )

    def on_test_start(self, trainer, pl_module):
        self._set(trainer.datamodule, ('pilot_datasets',), 0)


class MaskedReconstructionDiagnosticsCallback(Callback):
    """Log masks, masked inputs, targets, and reconstructions to W&B."""

    def __init__(
        self,
        enabled: bool = True,
        num_samples: int = 4,
        every_n_epochs: int = 5,
        log_train_examples: bool = True,
        log_mask_stats: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.num_samples = max(1, int(num_samples))
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.log_train_examples = bool(log_train_examples)
        self.log_mask_stats = bool(log_mask_stats)

    def _active(self, trainer: Trainer) -> bool:
        dm = getattr(trainer, 'datamodule', None)
        return (
            self.enabled
            and dm is not None
            and hasattr(dm, 'train_datasets')
            and hasattr(dm, 'models')
        )

    @staticmethod
    def _mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
        return mask.repeat(3, 1, 1) if mask.shape[0] == 1 else mask

    @staticmethod
    def _input_to_rgb(x: torch.Tensor) -> torch.Tensor:
        return x[:3] if x.shape[0] > 3 else x

    def _wandb_experiments(self, trainer: Trainer) -> list[tuple[Any, Any]]:
        try:
            import wandb
        except ImportError:
            return []

        loggers = getattr(trainer, 'loggers', None)
        if not loggers:
            logger = getattr(trainer, 'logger', None)
            loggers = [logger] if logger is not None else []

        experiments = []
        for logger in loggers:
            experiment = getattr(logger, 'experiment', None)
            if experiment is not None and hasattr(experiment, 'log'):
                experiments.append((experiment, wandb))
        return experiments

    def _log_metrics(
        self, trainer: Trainer, metrics: dict[str, float]
    ) -> None:
        loggers = getattr(trainer, 'loggers', None)
        if not loggers:
            logger = getattr(trainer, 'logger', None)
            loggers = [logger] if logger is not None else []
        for logger in loggers:
            if logger is not None and hasattr(logger, 'log_metrics'):
                logger.log_metrics(metrics, step=trainer.global_step)

    def _log_image(
        self,
        trainer: Trainer,
        key: str,
        image: torch.Tensor,
        caption: str,
    ) -> None:
        for experiment, wandb in self._wandb_experiments(trainer):
            experiment.log({key: wandb.Image(image, caption=caption)})

    def _sample_rows(
        self,
        dataset,
        agent,
        device: torch.device,
        include_reconstruction: bool,
    ) -> list[torch.Tensor]:
        rows = []
        was_training = agent.training
        agent.eval()
        with torch.no_grad():
            for sample_idx in range(min(self.num_samples, len(dataset))):
                item = dataset[sample_idx]
                masked, target = item[0], item[1]
                mask = dataset._mask(sample_idx).to(dtype=masked.dtype)
                rows.extend(
                    [
                        self._input_to_rgb(masked).clamp(0.0, 1.0),
                        self._mask_to_rgb(mask).clamp(0.0, 1.0),
                        target.clamp(0.0, 1.0),
                    ]
                )
                if include_reconstruction:
                    recon = (
                        agent(masked.unsqueeze(0).to(device)).squeeze(0).cpu()
                    )
                    rows.append(recon.clamp(0.0, 1.0))
                    completed = target * mask + recon * (1.0 - mask)
                    rows.append(completed.clamp(0.0, 1.0))
        agent.train(was_training)
        return rows

    def _log_split_examples(
        self,
        trainer: Trainer,
        pl_module,
        split: str,
        include_reconstruction: bool,
    ) -> None:
        datasets = getattr(trainer.datamodule, f'{split}_datasets', None)
        if not datasets:
            return

        columns = 5 if include_reconstruction else 3
        caption = (
            'per sample: masked | visible mask | target | reconstruction | '
            'completed (original visible pixels + reconstructed missing pixels)'
            if include_reconstruction
            else 'per sample: masked | mask | target'
        )
        for agent_idx in getattr(trainer.datamodule, 'models', []):
            agent_key = str(agent_idx)
            if agent_key not in pl_module.agents:
                continue
            dataset = datasets.get(agent_idx)
            if dataset is None:
                continue
            rows = self._sample_rows(
                dataset,
                pl_module.agents[agent_key],
                pl_module.device,
                include_reconstruction=include_reconstruction,
            )
            if rows:
                self._log_image(
                    trainer,
                    f'{split}/masked_reconstruction_examples_agent_{agent_idx}',
                    make_grid(rows, nrow=columns, padding=2),
                    caption,
                )

    def _mask_stats_for_split(self, dm, split: str) -> dict[str, float]:
        datasets = getattr(dm, f'{split}_datasets', None)
        if not datasets:
            return {}

        masks = {
            int(agent_idx): [
                dataset._mask(sample_idx).bool()
                for sample_idx in range(min(self.num_samples, len(dataset)))
            ]
            for agent_idx, dataset in datasets.items()
        }

        logs: dict[str, float] = {}
        for agent_idx, agent_masks in masks.items():
            if agent_masks:
                visible = torch.stack(
                    [mask.float().mean() for mask in agent_masks]
                )
                logs[f'diagnostics/{split}_mask_visible_agent_{agent_idx}'] = (
                    float(visible.mean())
                )

        agent_ids = sorted(masks)
        for pos, i in enumerate(agent_ids):
            for j in agent_ids[pos + 1 :]:
                pair_count = min(len(masks[i]), len(masks[j]))
                if pair_count == 0:
                    continue
                intersections, ious = [], []
                for sample_idx in range(pair_count):
                    mi, mj = masks[i][sample_idx], masks[j][sample_idx]
                    intersection = torch.logical_and(mi, mj).float().mean()
                    union = (
                        torch.logical_or(mi, mj).float().mean().clamp_min(1e-8)
                    )
                    intersections.append(intersection)
                    ious.append(intersection / union)
                logs[f'diagnostics/{split}_mask_overlap_agent_{i}_{j}'] = (
                    float(torch.stack(intersections).mean())
                )
                logs[f'diagnostics/{split}_mask_iou_agent_{i}_{j}'] = float(
                    torch.stack(ious).mean()
                )
        return logs

    def on_fit_start(self, trainer: Trainer, pl_module) -> None:
        if not self._active(trainer):
            return
        if self.log_mask_stats:
            logs = {}
            for split in ('train', 'val', 'pilot'):
                logs.update(
                    self._mask_stats_for_split(trainer.datamodule, split)
                )
            if logs:
                self._log_metrics(trainer, logs)
        if self.log_train_examples:
            self._log_split_examples(
                trainer,
                pl_module,
                split='train',
                include_reconstruction=False,
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        if (
            self._active(trainer)
            and trainer.current_epoch % self.every_n_epochs == 0
        ):
            self._log_split_examples(
                trainer,
                pl_module,
                split='val',
                include_reconstruction=True,
            )

    def log_communication_reconstruction(
        self, trainer, pl_module, dm, sender_idx, receiver_idx, prediction
    ) -> None:
        """Log the exact cross-decoder predictions used by test metrics."""
        if not self._active(trainer):
            return
        dataset = dm.test_datasets.get(sender_idx)
        if dataset is None:
            return
        rows = []
        for idx in range(min(self.num_samples, len(dataset), len(prediction))):
            masked, target = dataset[idx][:2]
            mask = dataset._mask(idx).to(dtype=masked.dtype)
            recon = prediction[idx].detach().cpu()
            rows.extend([
                self._input_to_rgb(masked).clamp(0, 1),
                self._mask_to_rgb(mask).clamp(0, 1),
                target.clamp(0, 1),
                recon.clamp(0, 1),
                (target * mask + recon * (1 - mask)).clamp(0, 1),
            ])
        if rows:
            self._log_image(
                trainer,
                f'test/communication_reconstruction_sender_{sender_idx}_receiver_{receiver_idx}',
                make_grid(rows, nrow=5, padding=2),
                'per sample: sender masked input | sender visible mask | target | '
                'receiver reconstruction from sender latent | completed using '
                'sender visible pixels (display only; metrics use raw reconstruction)',
            )

    def on_test_end(self, trainer: Trainer, pl_module) -> None:
        """Always log the final model on fixed test views, regardless of cadence."""
        if self._active(trainer):
            self._log_split_examples(
                trainer, pl_module, split='test', include_reconstruction=True
            )


def _parse_per_agent_cfg(cfg: DictConfig) -> dict[int, dict]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_dict = cfg_dict.get('model', {})
    if isinstance(model_dict, dict) and 'agents' in model_dict:
        raw = model_dict['agents']
    elif 'agents' in cfg_dict:
        raw = cfg_dict['agents']
    else:
        raw = {}
    return {int(key): (value or {}) for key, value in raw.items()}


def _build_agents(
    cfg: DictConfig,
    datamodule: Any,
    per_agents_cfg: dict[int, dict],
) -> tuple[dict[int, Any], dict[int, int]]:
    seed_everything(cfg.seed, workers=True)
    num_classes = getattr(datamodule, 'num_classes', {}).get('label')
    agents: dict[int, Any] = {}
    latent_dims: dict[int, int] = {}

    for agent_idx in datamodule.models:
        model_cfg = copy.deepcopy(cfg.model)
        OmegaConf.set_struct(model_cfg, False)
        if 'agents' in model_cfg:
            del model_cfg['agents']
        for key, value in per_agents_cfg.get(agent_idx, {}).items():
            setattr(model_cfg, key, value)

        model_cfg.num_classes = num_classes
        model_cfg.in_features = datamodule.input_dims[str(agent_idx)]
        if (
            hasattr(datamodule, 'target_dims')
            and OmegaConf.select(model_cfg, 'out_features') is None
        ):
            model_cfg.out_features = datamodule.target_dims[str(agent_idx)]
        if OmegaConf.select(model_cfg, 'img_size') is None:
            model_cfg.img_size = datamodule.input_shape[-1]

        agent = instantiate(_sanitize_instantiation_config(model_cfg))
        if getattr(agent, 'task_type', None) != 'reconstruction':
            raise ValueError(
                'reconstruction_experiment.py expects reconstruction agents, '
                f'got {type(agent).__name__} for agent {agent_idx}.'
            )

        agents[agent_idx] = agent
        latent_dims[agent_idx] = int(getattr(agent, 'latent_dim'))
        print(
            f'[agent {agent_idx}] {type(agent).__name__} '
            f'latent_dim={latent_dims[agent_idx]}'
        )

    return agents, latent_dims


def _build_orchestrator(
    cfg: DictConfig,
    agents: dict[int, Any],
    neighbors: dict[int, set[int]],
    latent_dims: dict[int, int],
) -> Any:
    from src.orchestrators.federated import FederatedLearning

    if issubclass(get_class(cfg.orchestrator._target_), FederatedLearning):
        # The shared reconstruction config otherwise injects Procrustes.
        # FedAvg communicates raw latents using its synchronized model space.
        cfg.orchestrator.alignment_method = None
    orch_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    kwargs = _filter_supported_init_kwargs(
        orch_cfg,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
    )
    return instantiate(orch_cfg, **kwargs, _convert_='all', _recursive_=False)


def _build_callbacks(cfg: DictConfig) -> list[Callback]:
    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]
    if OmegaConf.select(
        cfg, 'mask_augmentation.resample_each_epoch', default=False
    ):
        callbacks.insert(0, ReconstructionMaskEpochCallback())
    diag_cfg = OmegaConf.select(cfg, 'diagnostics.masked_reconstruction')
    if diag_cfg is not None and diag_cfg.get('enabled', False):
        callbacks.append(
            MaskedReconstructionDiagnosticsCallback(
                enabled=diag_cfg.get('enabled', True),
                num_samples=diag_cfg.get('num_samples', 4),
                every_n_epochs=diag_cfg.get('every_n_epochs', 5),
                log_train_examples=diag_cfg.get('log_train_examples', True),
                log_mask_stats=diag_cfg.get('log_mask_stats', True),
            )
        )
    return callbacks


def _run_name(cfg: DictConfig, orch_name: str) -> str:
    try:
        from hydra.core.hydra_config import HydraConfig

        job_num = HydraConfig.get().job.num
    except Exception:
        job_num = 0

    mask_mode = OmegaConf.select(cfg, 'dataset.mask_mode', default='unknown')
    parts = [orch_name, f'mask_{mask_mode}']
    variant = OmegaConf.select(cfg, 'loss_variant', default=None)
    if variant is not None:
        parts.insert(0, str(variant))
    if OmegaConf.select(cfg, 'orchestrator.max_lmb', default=None) is not None:
        parts.append(f'lmb_{float(cfg.orchestrator.max_lmb):.4e}')
    if (
        OmegaConf.select(cfg, 'orchestrator.comm_task_coeff', default=None)
        is not None
    ):
        parts.append(f'ctc_{cfg.orchestrator.comm_task_coeff}')
    parts.append(str(job_num))
    return '_'.join(parts)


def _comparison_id(cfg: DictConfig) -> str:
    """Prevent plots from pooling different objectives, budgets or map families."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    common = {
        key: payload.get(key)
        for key in (
            'model',
            'dataset',
            'optimizer',
            'trainer',
            'mask_augmentation',
        )
    }
    common['dataset'] = dict(common['dataset'])
    for key in (
        'seed',
        'constant_shared_visible_probability',
        'random_shared_visible_probability',
        'region_overlap_fraction',
    ):
        common['dataset'].pop(key, None)
    common['alignment'] = {
        key: payload['orchestrator'].get(key)
        for key in (
            'alignment_method',
            'use_general_maps',
            'anchor_selection',
            'comm_task_coeff',
            'max_lmb',
            'warmup_epochs',
        )
    }
    return hashlib.sha256(
        json.dumps(common, sort_keys=True).encode()
    ).hexdigest()[:12]


def _save_results(
    cfg: DictConfig,
    callback_metrics: dict[str, float],
    n_agents: int,
    orch_name: str,
) -> Path:
    rows = [
        {
            'agent': agent_idx,
            'private_lpips_full': callback_metrics.get(
                f'test/private_lpips_full_agent_{agent_idx}', float('nan')
            ),
            'comm_lpips_full': callback_metrics.get(
                f'test/comm_lpips_full_agent_{agent_idx}', float('nan')
            ),
            'lpips_backbone': 'alex',
            'lpips_version': '0.1',
            'loss_variant': OmegaConf.select(cfg, 'loss_variant', default=None),
            'private_psnr_full': callback_metrics.get(
                f'test/private_task_perf_agent_{agent_idx}', float('nan')
            ),
            'private_mse_full': callback_metrics.get(
                f'test/private_mse_full_agent_{agent_idx}',
                callback_metrics.get(
                    f'test/loss_mse_agent_{agent_idx}', float('nan')
                ),
            ),
            'private_mse_visible': callback_metrics.get(
                f'test/private_mse_visible_agent_{agent_idx}', float('nan')
            ),
            'private_mse_missing': callback_metrics.get(
                f'test/private_mse_missing_agent_{agent_idx}', float('nan')
            ),
            'comm_psnr': callback_metrics.get(
                f'test/comm_task_perf_agent_{agent_idx}', float('nan')
            ),
            'comm_mse': callback_metrics.get(
                f'test/comm_mse_full_agent_{agent_idx}',
                callback_metrics.get(
                    f'test/comm_mse_tx_missing_rx_visible_agent_{agent_idx}',
                    callback_metrics.get(
                        f'test/comm_mse_visible_agent_{agent_idx}',
                        float('nan'),
                    ),
                ),
            ),
            'comm_mse_sender_missing': callback_metrics.get(
                f'test/comm_mse_sender_missing_agent_{agent_idx}', float('nan')
            ),
            'task_fidelity': callback_metrics.get(
                f'test/task_fidelity_agent_{agent_idx}', float('nan')
            ),
            'orchestrator': orch_name,
            'mask_mode': OmegaConf.select(
                cfg, 'dataset.mask_mode', default=None
            ),
            'region_visible_fraction': OmegaConf.select(
                cfg,
                'dataset.region_visible_fraction',
                default=None,
            ),
            'region_overlap_fraction': OmegaConf.select(
                cfg,
                'dataset.region_overlap_fraction',
                default=None,
            ),
            'inactive_random_focus_regions': str(
                OmegaConf.select(
                    cfg, 'dataset.random_focus_regions', default=None
                )
            ),
            'inactive_random_private_visible_probability': OmegaConf.select(
                cfg,
                'dataset.random_private_visible_probability',
                default=None,
            ),
            'inactive_random_off_focus_visible_probability': OmegaConf.select(
                cfg,
                'dataset.random_off_focus_visible_probability',
                default=None,
            ),
            'inactive_random_shared_visible_probability': OmegaConf.select(
                cfg,
                'dataset.random_shared_visible_probability',
                default=None,
            ),
            'constant_visible_fraction': OmegaConf.select(
                cfg,
                'dataset.constant_visible_fraction',
                default=None,
            ),
            'constant_shared_visible_probability': OmegaConf.select(
                cfg,
                'dataset.constant_shared_visible_probability',
                default=None,
            ),
            'inactive_fixed_overlap_px': OmegaConf.select(
                cfg,
                'dataset.fixed_overlap_px',
                default=None,
            ),
            'seed': int(cfg.seed),
            'protocol': 'inpainting_v2',
            'comparison_id': _comparison_id(cfg),
            'latent_dim': int(cfg.model.latent_dim),
            'beta': float(cfg.model.beta),
            'comm_task_coeff': float(
                cfg.orchestrator.get('comm_task_coeff', 0.0)
            ),
            'config_yaml': OmegaConf.to_yaml(cfg, resolve=True),
        }
        for agent_idx in range(n_agents)
    ]

    df = pd.DataFrame(rows)
    results_dir = Path('results') / 'reconstruction'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    mask_mode = OmegaConf.select(cfg, 'dataset.mask_mode', default='unknown')
    out_path = results_dir / (
        f'reconstruction__{orch_name}__mask_{mask_mode}'
        f'__seed{cfg.seed}__{timestamp}.parquet'
    )
    df.to_parquet(out_path, index=False)
    print(f'\nResults saved -> {out_path}')
    print(
        df[
            [
                'agent',
                'private_psnr_full',
                'private_lpips_full',
                'comm_lpips_full',
                'private_mse_full',
                'private_mse_visible',
                'private_mse_missing',
                'comm_psnr',
                'comm_mse',
                'task_fidelity',
            ]
        ].to_string(index=False)
    )
    return out_path


@hydra.main(
    config_path='../config/hydra/',
    config_name='masked_cifar10_vae',
    version_base='1.3',
)
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, workers=True)
    _finish_active_wandb_run()

    datamodule = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    datamodule.prepare_data()
    datamodule.setup()
    n_agents = len(datamodule.models)

    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.get('seed', 42),
        p=cfg.graph.get('p', 0.3),
        m=cfg.graph.get('m', 3),
        manual=cfg.graph.get('neighbors', {}),
    )
    n_edges = sum(len(v) for v in neighbors.values()) // 2
    possible = n_agents * (n_agents - 1) // 2
    print(
        f'\nCommunication graph ({cfg.graph.neighbors_mode}): '
        f'{n_edges}/{possible} edges'
    )

    per_agents_cfg = _parse_per_agent_cfg(cfg)
    agents, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)

    orch_target = OmegaConf.select(cfg, 'orchestrator._target_', default='')
    orch_name = orch_target.split('.')[-1] if orch_target else 'unknown'
    orchestrator = _build_orchestrator(cfg, agents, neighbors, latent_dims)

    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    logger_cfg['name'] = _run_name(cfg, orch_name)
    logger = instantiate(logger_cfg)
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))
    _update_logger_config(
        logger,
        {
            'orchestrator_name': orch_name,
            'latent_dim': min(latent_dims.values()),
            'task_type': 'reconstruction',
        },
    )

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not orchestrator.automatic_optimization:
        for key in ('gradient_clip_val', 'gradient_clip_algorithm'):
            trainer_kwargs.pop(key, None)
    trainer = Trainer(
        **trainer_kwargs,
        callbacks=_build_callbacks(cfg),
        logger=logger,
    )

    trainer.fit(orchestrator, datamodule=datamodule)
    run_test = cfg.optimization.get('run_test', True)
    if run_test:
        trainer.test(orchestrator, datamodule=datamodule)

    metrics = {
        key: float(value) for key, value in trainer.callback_metrics.items()
    }
    # Primary objective: mean private inpainting error (lower is better).
    missing = [
        metrics.get(f'test/private_mse_missing_agent_{i}', float('nan'))
        for i in range(n_agents)
    ]
    objective = sum(missing) / n_agents

    out_path = _save_results(cfg, metrics, n_agents, orch_name)
    _update_logger_config(logger, {'results_file': str(out_path)})

    _finish_active_wandb_run()

    del trainer, orchestrator, datamodule, logger
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return objective


if __name__ == '__main__':
    main()
