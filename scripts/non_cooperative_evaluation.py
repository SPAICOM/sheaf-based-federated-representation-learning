"""Non-cooperative evaluation script.

Trains the model with the standard heterogeneous experiment setup, then
evaluates communication accuracy via the orchestrator's built-in test-time
alignment pipeline (triggered by trainer.test):

  1. Whitening: each agent fits W_i / C_i on its training latents.
  2. Alignment: for every directed edge (i→j), fit A_{j←i} on shared
                pilot latents (whitened).
  3. Evaluation: self-accuracy, cross-agent communication accuracy, and
                 task fidelity are logged per agent and saved to parquet.

Usage (single run):
    python scripts/non_cooperative_evaluation.py

Sweep over shift_strength:
    python scripts/non_cooperative_evaluation.py \\
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

# ── Reuse helpers from the main experiment script ─────────────────────────────


def _finish_active_wandb_run() -> None:
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, 'run', None) is not None:
        wandb.finish()


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


def _parse_groups(cfg: DictConfig) -> dict[int, list[int]]:
    """Return {group_id: [agent_ids]} from dataset.groups config."""
    raw = OmegaConf.to_container(cfg.dataset.groups, resolve=True)
    groups: dict[int, list[int]] = {}
    for k, v in raw.items():
        gid = int(k)
        if isinstance(v, (list, tuple)):
            groups[gid] = [int(a) for a in v]
        else:
            groups[gid] = [int(a) for a in v['agents']]
    return groups


def _agent_to_group(groups: dict[int, list[int]]) -> dict[int, int]:
    return {a: g for g, agents in groups.items() for a in agents}


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    config_path='../config/hydra/',
    config_name='hetero_rate_30agents_cifar10',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    shift_strength = float(
        OmegaConf.to_container(cfg.dataset, resolve=True).get(
            'shift_strength', 0.0
        )
    )

    # ── Setup (mirrors heterogenous_experiment.py) ────────────────────────────
    _finish_active_wandb_run()
    logger = instantiate(cfg.logger)

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
    per_agents_cfg_raw = OmegaConf.to_container(
        getattr(cfg, 'agents', {}), resolve=True
    )
    per_agents_cfg = {int(k): (v or {}) for k, v in per_agents_cfg_raw.items()}

    agents: dict[int, Any] = {}
    latent_dims: dict[int, int] = {}
    for i in range(n_agents):
        model_cfg = copy.deepcopy(cfg.model)
        OmegaConf.set_struct(model_cfg, False)
        if i in per_agents_cfg:
            for key, value in per_agents_cfg[i].items():
                setattr(model_cfg, key, value)
        model_cfg.num_classes = num_classes
        model_cfg.in_features = datamodule.input_dims[str(i)]
        agents[i] = instantiate(model_cfg)
        agent_type = str(type(agents[i]))
        if 'CNNClassifier' in agent_type or 'TimmClassifier' in agent_type:
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

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(orchestrator, datamodule=datamodule)

    # ── Test (triggers orchestrator alignment evaluation) ─────────────────────
    groups = _parse_groups(cfg)
    a2g = _agent_to_group(groups)

    trainer.test(orchestrator, datamodule=datamodule)

    # ── Collect per-agent metrics logged by the orchestrator ──────────────────
    cb = {k: float(v) for k, v in trainer.callback_metrics.items()}

    rows = []
    for i in range(n_agents):
        rows.append(
            {
                'agent': i,
                'group': a2g[i],
                'self_accuracy': cb.get(f'test/private_task_perf_agent_{i}', float('nan')),
                'comm_accuracy': cb.get(f'test/comm_task_perf_agent_{i}', float('nan')),
                'task_fidelity': cb.get(f'test/task_fidelity_agent_{i}', float('nan')),
                'shift_strength': shift_strength,
                'seed': int(cfg.seed),
            }
        )

    df = pd.DataFrame(rows)

    # ── Save parquet ──────────────────────────────────────────────────────────
    results_dir = Path('.') / 'results' / 'non_cooperative'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'non_cooperative__shift{shift_strength:.3f}__seed{cfg.seed}__{timestamp}.parquet'
    out_path = results_dir / fname
    df.to_parquet(out_path, index=False)
    print(f'\nResults saved → {out_path}')
    print(df[['agent', 'group', 'self_accuracy', 'comm_accuracy', 'task_fidelity']].to_string(index=False))

    # Cleanup.
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()


if __name__ == '__main__':
    from src.utils.hydra_outputs import disable_hydra_outputs

    disable_hydra_outputs()
    main()
