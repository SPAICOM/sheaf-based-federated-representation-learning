"""Rank study script (single-agent).

Trains an agent and evaluates its test accuracy. No communication or
alignment evaluation is performed. Input effective rank is computed on
the validation split before training and saved alongside task accuracy.

Usage:
    python scripts/rank_study.py

Sweep over shift_strength:
    python scripts/rank_study.py \\
        --multirun dataset.shift_strength=0.0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import copy
import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from hydra.utils import get_class, instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf

from src.utils import remove_non_empty_dir
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
        except (ValueError, RuntimeError):
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
    sig = inspect.signature(target.__init__)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return config
    allowed = {n for n in sig.parameters if n != 'self'}
    allowed.update({'_target_', '_recursive_', '_convert_', '_partial_'})
    return OmegaConf.create(
        {k: v for k, v in config_dict.items() if k in allowed}
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
    sig = inspect.signature(target.__init__)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return kwargs
    allowed = {n for n in sig.parameters if n != 'self'}
    return {k: v for k, v in kwargs.items() if k in allowed}


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    config_path='../config/hydra/',
    config_name='hetero_rate_2agents_mnist',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    shift_strength = float(
        OmegaConf.to_container(cfg.dataset, resolve=True).get(
            'shift_strength', 0.0
        )
    )

    # ── Setup ─────────────────────────────────────────────────────────────────
    _finish_active_wandb_run()
    logger = instantiate(cfg.logger)
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))

    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

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

    n_agents = len(datamodule.models)

    _model_agents = OmegaConf.select(cfg.model, 'agents')
    if _model_agents is not None:
        per_agents_cfg_raw = OmegaConf.to_container(
            _model_agents, resolve=True
        )
    else:
        _top_agents = OmegaConf.select(cfg, 'agents')
        per_agents_cfg_raw = (
            OmegaConf.to_container(_top_agents, resolve=True)
            if _top_agents is not None
            else {}
        )
    per_agents_cfg = {int(k): (v or {}) for k, v in per_agents_cfg_raw.items()}

    agents: dict[int, Any] = {}
    latent_dims: dict[int, int] = {}
    for i in range(n_agents):
        model_cfg = copy.deepcopy(cfg.model)
        OmegaConf.set_struct(model_cfg, False)
        if 'agents' in model_cfg:
            del model_cfg['agents']
        if i in per_agents_cfg:
            for key, value in per_agents_cfg[i].items():
                setattr(model_cfg, key, value)
        model_cfg.num_classes = num_classes
        model_cfg.in_features = datamodule.input_dims[str(i)]
        if OmegaConf.select(model_cfg, 'img_size') is None and hasattr(
            datamodule, 'input_shape'
        ):
            model_cfg.img_size = datamodule.input_shape[-1]
        model_cfg = _sanitize_instantiation_config(model_cfg)
        agents[i] = instantiate(model_cfg)
        agent_type = str(type(agents[i]))
        if (
            'CNNClassifier' in agent_type
            or 'TimmClassifier' in agent_type
            or 'TransformerClassifier' in agent_type
        ):
            latent_dims[i] = agents[i].encoder.out_features
        elif model_cfg.get('latent_dim'):
            latent_dims[i] = model_cfg.latent_dim

    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    orch_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    orch_kwargs = _filter_supported_init_kwargs(
        orch_cfg,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        rates={
            i: float(per_agents_cfg.get(i, {}).get('rate', 1.0))
            for i in range(n_agents)
        },
    )
    orchestrator = instantiate(
        orch_cfg, **orch_kwargs, _convert_='all', _recursive_=False
    )

    # ── Input effective rank (pre-training) ───────────────────────────────────
    print(
        '\nInput effective rank per agent (validation split, before training):'
    )
    input_er: dict[int, float] = {}
    for i in range(n_agents):
        er = datamodule.input_effective_rank(i, split='val')
        input_er[i] = er
        print(f'  agent {i}: input ER = {er:.2f}')

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(orchestrator, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────────────
    trainer.test(orchestrator, datamodule=datamodule)

    cb = {k: float(v) for k, v in trainer.callback_metrics.items()}

    rows = []
    for i in range(n_agents):
        rows.append(
            {
                'agent': i,
                'input_effective_rank': input_er[i],
                'self_accuracy': cb.get(
                    f'test/task_performance_agent_{i}', float('nan')
                ),
                'shift_strength': shift_strength,
                'seed': int(cfg.seed),
            }
        )

    df = pd.DataFrame(rows)

    # ── Save parquet ──────────────────────────────────────────────────────────
    results_dir = Path('.') / 'results' / 'rank_study'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'rank_study__shift{shift_strength:.3f}__seed{cfg.seed}__{timestamp}.parquet'
    out_path = results_dir / fname
    df.to_parquet(out_path, index=False)
    print(f'\nResults saved → {out_path}')
    print(
        df[['agent', 'input_effective_rank', 'self_accuracy']].to_string(
            index=False
        )
    )

    # Cleanup.
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()


if __name__ == '__main__':
    main()
