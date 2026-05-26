"""Non-cooperative evaluation script.

Trains the model with the standard heterogeneous experiment setup, then
runs a non-cooperative evaluation via pre-whitening + pairwise alignment:

  1. Pre-whitening: each agent learns W_i / C_i on its training latents.
  2. Alignment:     for every directed edge (i→j), learn A_{j←i} on shared
                    pilot latents (whitened).
  3. Evaluation:    compute top-1 accuracy for every (source, target) pair.
  4. Save:          one parquet per run, filename encodes shift_strength + seed.

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
import torch
from hydra.utils import get_class, instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.communication.whitening import (
    WhiteningOp,
    common_pilot_indices,
    cross_accuracy,
    extract_latents,
    fit_alignment,
    fit_whitening,
    self_accuracy,
    whiten,
)
from src.datamodules.classification_datamodule import _collate_fn
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


# ── PPFE helpers ──────────────────────────────────────────────────────────────


def _agent_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # ── PPFE evaluation ───────────────────────────────────────────────────────
    groups = _parse_groups(cfg)
    a2g = _agent_to_group(groups)

    batch_size = int(cfg.dataset.get('batch_size', 256))

    print('\n[PPFE] Extracting training latents for whitening...')
    train_ops: dict[int, WhiteningOp] = {}
    for i in range(n_agents):
        agent = orchestrator.agents[str(i)].to(device)
        loader = _agent_loader(datamodule.train_datasets[i], batch_size)
        Z_train, _ = extract_latents(agent, loader, device)
        train_ops[i] = fit_whitening(Z_train)

    print('[PPFE] Extracting pilot latents for alignment...')
    # Whitened pilot representations per agent.
    pilot_Z: dict[int, torch.Tensor] = {}
    for i in range(n_agents):
        if (
            i not in datamodule.pilot_datasets
            or not datamodule.pilot_datasets[i]
        ):
            pilot_Z[i] = torch.empty(0, list(train_ops[i].W.shape)[0])
            continue
        agent = orchestrator.agents[str(i)].to(device)
        loader = _agent_loader(datamodule.pilot_datasets[i], batch_size)
        Z_pilot, _ = extract_latents(agent, loader, device)
        pilot_Z[i] = whiten(Z_pilot, train_ops[i])  # (n_pilot_i, d)

    print('[PPFE] Fitting pairwise alignment matrices...')
    # A[i][j] = A_{j←i}: maps from i's whitened space to j's whitened space.
    A: dict[int, dict[int, torch.Tensor]] = {i: {} for i in range(n_agents)}
    for i in range(n_agents):
        for j in neighbors[i]:
            if j == i:
                continue
            pi = datamodule.pilot_datasets.get(i)
            pj = datamodule.pilot_datasets.get(j)
            if pi is None or pj is None:
                continue
            idx_i, idx_j = common_pilot_indices(pi, pj)
            if len(idx_i) < 2:
                print(
                    f'  [warn] agents ({i},{j}): only {len(idx_i)} common pilot samples'
                )
                continue
            X_i = pilot_Z[i][idx_i]
            X_j = pilot_Z[j][idx_j]
            A[i][j] = fit_alignment(X_i, X_j)  # A_{j←i}

    print('[PPFE] Extracting test latents...')
    test_Z: dict[int, torch.Tensor] = {}
    test_y: dict[int, torch.Tensor] = {}
    for i in range(n_agents):
        agent = orchestrator.agents[str(i)].to(device)
        loader = _agent_loader(datamodule.test_datasets[i], batch_size)
        test_Z[i], test_y[i] = extract_latents(agent, loader, device)

    print('[PPFE] Computing accuracies...')
    rows = []

    for i in range(n_agents):
        # Self accuracy.
        agent_i = orchestrator.agents[str(i)].to(device)
        acc_self = self_accuracy(agent_i, test_Z[i], test_y[i], device)
        rows.append(
            {
                'source_agent': i,
                'target_agent': i,
                'source_group': a2g[i],
                'target_group': a2g[i],
                'is_self': True,
                'homophilous': True,
                'accuracy': acc_self,
                'n_test_samples': len(test_y[i]),
                'shift_strength': shift_strength,
                'seed': int(cfg.seed),
            }
        )

        # Cross accuracy from i to each neighbour j.
        for j in neighbors[i]:
            if j == i:
                continue
            if j not in A[i]:
                continue
            agent_j = orchestrator.agents[str(j)].to(device)
            acc = cross_accuracy(
                agent_j,
                test_Z[i],
                test_y[i],
                train_ops[i],
                train_ops[j],
                A[i][j],
                device,
            )
            rows.append(
                {
                    'source_agent': i,
                    'target_agent': j,
                    'source_group': a2g[i],
                    'target_group': a2g[j],
                    'is_self': False,
                    'homophilous': a2g[i] == a2g[j],
                    'accuracy': acc,
                    'n_test_samples': len(test_y[i]),
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
    print(f'\n[PPFE] Results saved → {out_path}')
    print(
        df.groupby(['is_self', 'homophilous'])['accuracy'].describe().round(3)
    )

    # Cleanup.
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()


if __name__ == '__main__':
    main()
