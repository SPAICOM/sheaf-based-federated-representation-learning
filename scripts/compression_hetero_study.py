"""Compression x model-heterogeneity study (2-agent MNIST): ComFed vs SheafCFRL.

Companion to ``compression_study.py``, but instead of sweeping the
compression bottleneck across a *distribution shift* grid, this script fixes
the shift and instead crosses the compression grid with how architecturally
different the two agents are:

  * Agent ``study.fixed_agent`` (default: 0) keeps the architecture defined
    under ``model.agents.<fixed_agent>`` for the entire run.
  * Agent ``study.variable_agent`` (default: 1) is re-instantiated at each
    sweep step from ``study.heterogeneity_levels``, a list ordered from
    "identical to the fixed agent" to "a completely different model family"
    (e.g. shrinking a shared CNN, then dropping batchnorm, then switching to
    a Vision Transformer entirely).

Communication accuracy is evaluated the same way as in ``compression_study.py``
via the orchestrator's built-in test-time alignment pipeline (triggered by
``trainer.test``):

  1. Whitening: each agent fits W_i / C_i on its training latents.
  2. Alignment: for every directed edge (i→j), fit A_{j←i} on shared
                pilot latents (whitened).
  3. Evaluation: self-accuracy, cross-agent communication accuracy, and
                 task fidelity are logged per agent and saved to parquet.

Unlike ``compression_study.py``, there is **no** λ (max_lmb) study here —
``orchestrator.max_lmb`` is read as-is from the config and never tuned.

Sweep structure (all handled *inside* this script, so every combination gets
its own, independently-finished wandb run — see ``_run_full_experiment``):

  * The orchestrator is the only Hydra multirun dimension (sheaf_cfrl, comfed).
  * For each orchestrator we loop over ``study.heterogeneity_levels`` and,
    for every level, over ``study.compression_factors``:
      - Level loop (outer): agent ``study.variable_agent``'s architecture is
        swapped in from that level's ``model`` block and latent dims are
        re-discovered (they can change as the architecture changes).
      - Compression loop (inner): the shared bottleneck is re-derived from
        each factor at the level's (possibly new) latent dim — so the
        *relative* bottleneck stays comparable across levels even though the
        absolute latent dims drift.
  * ``study.shift_strength`` is a single fixed value — there is no shift grid.

Usage (compare both orchestrators):
    python scripts/compression_hetero_study.py --multirun orchestrator=sheaf_cfrl,comfed

Single orchestrator:
    python scripts/compression_hetero_study.py orchestrator=comfed
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

# Reuse every helper that doesn't touch the (removed) λ study or the
# compression-factor/shift sweep loop from compression_study.py.
from scripts.compression_study import (  # noqa: E402
    ORCHESTRATORS_WITH_COMPRESSION_FACTOR,
    ORCHESTRATORS_WITH_LMB,
    ORCHESTRATORS_WITH_PROJ_DIM,
    _agent_to_group,
    _apply_compression,
    _build_agents,
    _build_datamodule,
    _build_orchestrator,
    _finish_active_wandb_run,
    _new_wandb_run_logger,
    _parse_groups,
    _parse_per_agent_cfg,
    _update_logger_config,
)
from src.utils import remove_non_empty_dir
from src.utils.graph_generator import generate_neighbors


def _agent_arch_name(cfg: DictConfig, agent_id: int) -> str:
    """Short class name of ``model.agents.<agent_id>._target_``.

    ``model.agents`` keys are loaded from YAML as ints, and
    ``OmegaConf.select(cfg, f'model.agents.{agent_id}._target_')`` silently
    fails for int keys (dotted-path lookup treats path segments as strings) —
    so this indexes directly instead of going through a dotted path.
    """
    agent_cfg = OmegaConf.select(cfg, 'model.agents', default={}).get(
        agent_id, {}
    )
    target = OmegaConf.select(agent_cfg, '_target_', default='?')
    return str(target).split('.')[-1]


def _run_full_experiment(
    cfg: DictConfig,
    datamodule: Any,
    neighbors: dict[int, set[int]],
    per_agents_cfg: dict[int, dict],
    orch_name: str,
    a2g: dict[int, int],
    compression_factor: float,
    proj_dim: int,
    shift_strength: float,
    heterogeneity_level: int,
    heterogeneity_label: str,
    fixed_agent: int,
    variable_agent: int,
    job_num: int,
) -> pd.DataFrame:
    """Train + test one heterogeneity-level configuration; return rows.

    ``cfg.orchestrator`` is expected to already carry the resolved compression
    hyperparameter (via :func:`_apply_compression`) and the fixed ``max_lmb``.
    """
    n_agents = len(datamodule.models)

    # ── Build run name ────────────────────────────────────────────────────────
    name_parts = [
        orch_name,
        f'het_{heterogeneity_level}_{heterogeneity_label}',
    ]
    if orch_name in ORCHESTRATORS_WITH_COMPRESSION_FACTOR:
        name_parts.append(f'cf_{compression_factor}')
    if orch_name in ORCHESTRATORS_WITH_PROJ_DIM:
        name_parts.append(f'pd_{proj_dim}')
    if orch_name in ORCHESTRATORS_WITH_LMB:
        lmb = float(OmegaConf.select(cfg, 'orchestrator.max_lmb'))
        name_parts.append(f'lmb_{lmb:.4e}')
    name_parts.append(str(job_num))
    run_name = '_'.join(name_parts)

    # ── Logger ────────────────────────────────────────────────────────────────
    # Namespaced with a "sweep_" prefix so they can never collide with an
    # orchestrator's own hparams (see compression_study.py for the full story
    # on why an unprefixed key gets silently overwritten by Lightning's
    # automatic log_hyperparams() call).
    sweep_metadata = {
        'orchestrator_name': orch_name,
        'sweep_compression_factor': float(compression_factor),
        'sweep_proj_dim': int(proj_dim),
        'sweep_shift_strength': float(shift_strength),
        'sweep_heterogeneity_level': int(heterogeneity_level),
        'sweep_heterogeneity_label': str(heterogeneity_label),
        'sweep_fixed_agent': int(fixed_agent),
        'sweep_variable_agent': int(variable_agent),
        'sweep_fixed_agent_arch': _agent_arch_name(cfg, fixed_agent),
        'sweep_variable_agent_arch': _agent_arch_name(cfg, variable_agent),
    }
    logger = _new_wandb_run_logger(cfg, run_name)
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))
    _update_logger_config(logger, sweep_metadata)

    # ── Full training run ─────────────────────────────────────────────────────
    callbacks = [
        hydra.utils.instantiate(cb_conf) for cb_conf in cfg.callbacks.values()
    ]
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    agents, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)
    _update_logger_config(logger, {'latent_dim': min(latent_dims.values())})
    orchestrator = _build_orchestrator(
        cfg, agents, neighbors, latent_dims, per_agents_cfg
    )

    trainer.fit(orchestrator, datamodule=datamodule)
    # Re-assert after fit(): see compression_study.py for why this is needed.
    _update_logger_config(logger, sweep_metadata)
    trainer.test(orchestrator, datamodule=datamodule)

    # ── Collect per-agent metrics logged by the orchestrator ──────────────────
    cb = {k: float(v) for k, v in trainer.callback_metrics.items()}
    max_lmb = (
        float(OmegaConf.select(cfg, 'orchestrator.max_lmb'))
        if orch_name in ORCHESTRATORS_WITH_LMB
        else float('nan')
    )

    rows = []
    for i in range(n_agents):
        rows.append(
            {
                'orchestrator': orch_name,
                'agent': i,
                'group': a2g[i],
                'self_accuracy': cb.get(
                    f'test/private_task_perf_agent_{i}', float('nan')
                ),
                'comm_accuracy': cb.get(
                    f'test/comm_task_perf_agent_{i}', float('nan')
                ),
                'task_fidelity': cb.get(
                    f'test/task_fidelity_agent_{i}', float('nan')
                ),
                'compression_factor': float(compression_factor),
                'proj_dim': int(proj_dim),
                'max_lmb': max_lmb,
                'shift_strength': float(shift_strength),
                'heterogeneity_level': int(heterogeneity_level),
                'heterogeneity_label': str(heterogeneity_label),
                'agent_arch': _agent_arch_name(cfg, i),
                'is_variable_agent': bool(i == variable_agent),
                'seed': int(cfg.seed),
            }
        )

    # Finish THIS run through its own handle before the next level — see
    # compression_study.py for why this matters (wandb run-reuse footgun).
    try:
        if logger is not None:
            logger.experiment.finish()
    except Exception as exc:
        print(f'WARNING: failed to finish wandb run {run_name!r}: {exc}')
    _finish_active_wandb_run()
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    config_path='../config/hydra/',
    config_name='hetero_rate_2agents_mnist_compression_hetero',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    _finish_active_wandb_run()

    orch_target = OmegaConf.select(cfg, 'orchestrator._target_', default='')
    orch_name = orch_target.split('.')[-1] if orch_target else 'unknown'

    # ── Study grid ────────────────────────────────────────────────────────────
    study_cfg = cfg.get('study', {})
    shift_strength = float(study_cfg.get('shift_strength', 0.7))
    compression_factors = [
        float(x) for x in study_cfg.get('compression_factors', [0.5])
    ]
    fixed_agent = int(study_cfg.get('fixed_agent', 0))
    variable_agent = int(study_cfg.get('variable_agent', 1))
    levels = OmegaConf.to_container(
        study_cfg.get('heterogeneity_levels', []), resolve=True
    )
    if not levels:
        raise ValueError(
            'study.heterogeneity_levels must define at least one level'
        )

    groups = _parse_groups(cfg)
    a2g = _agent_to_group(groups)

    try:
        from hydra.core.hydra_config import HydraConfig

        job_num = HydraConfig.get().job.num
    except Exception:
        job_num = 0

    # ── Datamodule (fixed shift — built once, reused across every level) ──────
    with open_dict(cfg):
        cfg.dataset.shift_strength = float(shift_strength)

    print(
        f'\n[{orch_name}] building datamodule (shift_strength={shift_strength}) ...'
    )
    datamodule = _build_datamodule(cfg, shift_strength)
    n_agents = len(datamodule.models)

    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.seed,
        p=cfg.graph.p,
        m=cfg.graph.m,
        manual=cfg.graph.get('neighbors', {}),
    )

    print(
        '\nInput effective rank per agent (validation split, before training):'
    )
    for i in range(n_agents):
        er = datamodule.input_effective_rank(i, split='val')
        print(f'  agent {i}: input ER = {er:.2f}')

    results_dir = Path('.') / 'results' / 'compression_hetero'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_frames: list[pd.DataFrame] = []
    for level_idx, level in enumerate(levels):
        label = str(level.get('label', f'level_{level_idx}'))
        variable_model_cfg = level['model']

        # Swap in this level's architecture for the variable agent only; the
        # fixed agent's `model.agents.<fixed_agent>` block never changes.
        with open_dict(cfg):
            cfg.model.agents[variable_agent] = variable_model_cfg
        per_agents_cfg = _parse_per_agent_cfg(cfg)

        # Latent dims can shift with the architecture → re-discover per level,
        # once, then reuse across every compression_factor at this level.
        _, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)
        latent_dim = min(latent_dims.values())

        fixed_arch = _agent_arch_name(cfg, fixed_agent)
        variable_arch = _agent_arch_name(cfg, variable_agent)

        # ── Compression sweep at this heterogeneity level ──────────────────────
        for compression_factor in compression_factors:
            proj_dim = _apply_compression(
                cfg, orch_name, compression_factor, latent_dim
            )
            print(
                f'\n=== {orch_name} | level {level_idx} ({label}) — '
                f'agent {fixed_agent}: {fixed_arch} (fixed) vs '
                f'agent {variable_agent}: {variable_arch} (variable) '
                f'| latent_dim(min)={latent_dim} | compression_factor='
                f'{compression_factor} → bottleneck dim={proj_dim} ==='
            )

            df = _run_full_experiment(
                cfg,
                datamodule,
                neighbors,
                per_agents_cfg,
                orch_name,
                a2g,
                compression_factor,
                proj_dim,
                shift_strength,
                level_idx,
                label,
                fixed_agent,
                variable_agent,
                job_num,
            )
            all_frames.append(df)

            # Incremental save (robust against a crash mid-grid).
            fname = (
                f'{orch_name}__het{level_idx}_{label}__cf{compression_factor:.3f}'
                f'__shift{shift_strength:.3f}__seed{cfg.seed}__{timestamp}.parquet'
            )
            df.to_parquet(results_dir / fname, index=False)
            print(
                df[
                    [
                        'agent',
                        'group',
                        'self_accuracy',
                        'comm_accuracy',
                        'task_fidelity',
                    ]
                ].to_string(index=False)
            )

    # ── Combined parquet for this orchestrator ────────────────────────────────
    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined_path = (
            results_dir
            / f'{orch_name}__combined__seed{cfg.seed}__{timestamp}.parquet'
        )
        combined.to_parquet(combined_path, index=False)
        print(f'\nCombined results saved → {combined_path}')

    # Cleanup.
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()


if __name__ == '__main__':
    main()
