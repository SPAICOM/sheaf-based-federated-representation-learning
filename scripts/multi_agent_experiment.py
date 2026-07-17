"""Multi-agent experiment script (heterogeneous multi-agent MNIST).

Same pipeline as ``toy_experiment.py`` but defaults to the multi-agent config
(``hetero_rate_multiagent_mnist.yaml``).  Trains the model with the
heterogeneous experiment setup, then evaluates communication accuracy via the
orchestrator's built-in test-time alignment pipeline (triggered by
trainer.test):

  1. Whitening: each agent fits W_i / C_i on its training latents.
  2. Alignment: for every directed edge (i→j), fit A_{j←i} on shared
                pilot latents (whitened).
  3. Evaluation: self-accuracy, cross-agent communication accuracy, and
                 task fidelity are logged per agent and saved to parquet.

When the orchestrator is one of SheafFRL / SheafFMTL / ComFed / FedProto /
FedMuscle, an Optuna study first searches for the best max_lmb over a short
number of epochs, then uses that value for the full training run.

``lmb_study.efficient`` (boolean):
    With many agents the Optuna search becomes expensive because every trial
    trains the whole network for a few epochs.  When ``efficient: true`` the
    max_lmb search is run on only the **first two agents** of the network
    (a single sheaf edge), which is much cheaper; the best lambda found is
    then applied to the full multi-agent training run.

Usage (single run):
    python scripts/multi_agent_experiment.py

Sweep over shift_strength:
    python scripts/multi_agent_experiment.py \\
        --multirun dataset.shift_strength=0.0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import copy
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import hydra
import pandas as pd
from hydra.utils import get_class, instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils import remove_non_empty_dir
from src.utils.graph_generator import generate_neighbors

# ── Orchestrators that support max_lmb and/or comm_task_coeff ─────────────────
ORCHESTRATORS_WITH_LMB = {
    'SheafFRL',
    'SheafCFRL',
    'SheafFMTL',
    'CESheafFRL',
    'ComFed',
    'FedProto',
    'FedMuscle',
}
ORCHESTRATORS_WITH_CTC = {'SheafFRL'}

# Number of agents used for the lmb study when lmb_study.efficient is True.
EFFICIENT_LMB_AGENTS = 2

# ── Reuse helpers from the main experiment script ─────────────────────────────


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


def _parse_agent_target_classes(
    cfg: DictConfig,
) -> dict[int, set[int]] | None:
    """Return {agent_id: set(target_classes)} from dataset.groups, or None.

    Used to build the ``class_overlap`` communication graph. Agents whose group
    declares no ``target_classes`` map to an empty set (they share nothing).
    """
    groups_cfg = OmegaConf.select(cfg, 'dataset.groups')
    if groups_cfg is None:
        return None
    raw = OmegaConf.to_container(groups_cfg, resolve=True)
    out: dict[int, set[int]] = {}
    for v in raw.values():
        if isinstance(v, dict):
            agents = v.get('agents', [])
            targets = v.get('target_classes', []) or []
        else:
            agents, targets = v, []
        for a in agents:
            out[int(a)] = {int(c) for c in targets}
    return out or None


# ── Model-building helpers ─────────────────────────────────────────────────────


def _parse_per_agent_cfg(cfg: DictConfig) -> dict[int, dict]:
    """Extract per-agent config overrides from model.agents or top-level agents."""
    # Resolve from the full cfg root so cross-config interpolations
    # (e.g. ${model.encoder_hidden_dims}) are resolvable.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    model_dict = cfg_dict.get('model', {})
    if isinstance(model_dict, dict) and 'agents' in model_dict:
        raw = model_dict['agents']
    elif 'agents' in cfg_dict:
        raw = cfg_dict['agents']
    else:
        raw = {}
    return {int(k): (v or {}) for k, v in raw.items()}


def _build_agents(
    cfg: DictConfig,
    datamodule: Any,
    per_agents_cfg: dict[int, dict],
) -> tuple[dict[int, Any], dict[int, int]]:
    """Instantiate per-agent models; returns (agents, latent_dims)."""
    num_classes = datamodule.num_classes.get('label')
    if num_classes is None:
        raise ValueError('Attribute "label" is not categorical')

    n_agents = len(datamodule.models)
    seed_everything(cfg.seed, workers=True)

    agents: dict[int, Any] = {}
    latent_dims: dict[int, int] = {}
    for i in datamodule.models:
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
        elif hasattr(agents[i], 'latent_dim'):
            latent_dims[i] = int(agents[i].latent_dim)
        elif model_cfg.get('latent_dim'):
            latent_dims[i] = model_cfg.latent_dim

    return agents, latent_dims


def _build_orchestrator(
    cfg: DictConfig,
    agents: dict[int, Any],
    neighbors: dict[int, set[int]],
    latent_dims: dict[int, int],
    per_agents_cfg: dict[int, dict],
) -> Any:
    """Instantiate the orchestrator from config."""
    orch_cfg = _sanitize_instantiation_config(cfg.orchestrator)
    orch_kwargs = _filter_supported_init_kwargs(
        orch_cfg,
        agents=agents,
        neighbors=neighbors,
        latent_dims=latent_dims,
        optimizer=cfg.optimizer,
        rates={
            i: float(per_agents_cfg.get(i, {}).get('rate', 1.0))
            for i in agents
        },
    )
    return instantiate(
        orch_cfg, **orch_kwargs, _convert_='all', _recursive_=False
    )


# ── Agent-subsetting helpers (for the efficient lmb study) ────────────────────


def _subset_neighbors(
    neighbors: dict[int, set[int]], keep: list[int]
) -> dict[int, set[int]]:
    """Restrict the neighbour graph to the ``keep`` agents (and their mutual edges)."""
    keep_set = set(keep)
    return {
        i: {n for n in neighbors.get(i, set()) if n in keep_set} for i in keep
    }


@contextmanager
def _restrict_datamodule_to_agents(
    datamodule: Any, keep: list[int]
) -> Iterator[Any]:
    """Temporarily restrict the datamodule to a subset of agents.

    The datamodule's own loader/builder code keys everything off the per-agent
    dataset dicts and ``models`` list, so saving those, swapping in the
    ``keep``-only subset, and restoring on exit yields a faithful view of a
    smaller network without rebuilding anything.
    """
    keep_set = set(keep)
    dataset_attrs = (
        'train_datasets',
        'val_datasets',
        'test_datasets',
        'pilot_datasets',
    )
    saved: dict[str, Any] = {}
    for attr in (*dataset_attrs, 'models', 'n_agents'):
        if hasattr(datamodule, attr):
            saved[attr] = getattr(datamodule, attr)
    try:
        for attr in dataset_attrs:
            orig = saved.get(attr)
            if isinstance(orig, dict):
                setattr(
                    datamodule,
                    attr,
                    {i: v for i, v in orig.items() if i in keep_set},
                )
        datamodule.models = list(keep)
        datamodule.n_agents = len(keep)
        yield datamodule
    finally:
        for attr, value in saved.items():
            setattr(datamodule, attr, value)


# ── Optuna tuning for max_lmb ─────────────────────────────────────────────────


class _ValAsTestWrapper:
    """Thin dm wrapper that substitutes val_datasets for test_datasets.

    Passed to evaluate_communication_accuracy during Optuna trials so that
    comm accuracy is measured on the validation split, keeping the test set
    unseen during hyperparameter search.
    """

    def __init__(self, dm: Any) -> None:
        self._dm = dm
        self.test_datasets = dm.val_datasets

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dm, name)


def tune_max_lmb(
    cfg: DictConfig,
    datamodule: Any,
    neighbors: dict[int, set[int]],
    per_agents_cfg: dict[int, dict],
    n_trials: int = 20,
    n_tune_epochs: int = 5,
    lmb_low: float = 1e-4,
    lmb_high: float = 10.0,
    agent_subset: list[int] | None = None,
) -> float:
    """Search for the best max_lmb using Optuna.

    Runs ``n_trials`` short training runs of ``n_tune_epochs`` epochs each,
    maximising the harmonic mean of ``validation/avg_task_performance`` and
    validation-split communication accuracy.  Using the harmonic mean prevents
    either metric from dominating: a lambda that buys high task performance at
    the cost of near-zero comm accuracy (or vice-versa) scores poorly.

    Communication accuracy is evaluated on the validation split (via
    _ValAsTestWrapper) to keep the test set unseen during tuning.

    When ``agent_subset`` is given, every trial only builds and trains those
    agents (with the graph and per-agent configs restricted to match), which
    makes the search dramatically cheaper for large networks.

    Returns the best max_lmb found.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Restrict the graph / per-agent configs to the subset once up front.
    if agent_subset is not None:
        study_neighbors = _subset_neighbors(neighbors, agent_subset)
        study_per_agents_cfg = {
            i: per_agents_cfg[i] for i in agent_subset if i in per_agents_cfg
        }
    else:
        study_neighbors = neighbors
        study_per_agents_cfg = per_agents_cfg

    scope = (
        f'first {len(agent_subset)} agents'
        if agent_subset is not None
        else 'all agents'
    )
    print(
        f'\nTuning max_lmb in [{lmb_low:.0e}, {lmb_high:.0e}] on {scope} '
        f'— {n_trials} trials × {n_tune_epochs} epochs ...'
    )

    def objective(trial: optuna.Trial) -> float:
        lmb = trial.suggest_float('max_lmb', lmb_low, lmb_high, log=True)

        # Build a plain-dict copy of cfg so we can override max_lmb freely.
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(trial_cfg, 'orchestrator.max_lmb', lmb)

        restrict = (
            _restrict_datamodule_to_agents(datamodule, agent_subset)
            if agent_subset is not None
            else nullcontext(datamodule)
        )
        with restrict as dm_view:
            # Fresh agent weights for every trial (same seed → same init).
            agents, latent_dims = _build_agents(
                trial_cfg, dm_view, study_per_agents_cfg
            )
            orchestrator = _build_orchestrator(
                trial_cfg,
                agents,
                study_neighbors,
                latent_dims,
                study_per_agents_cfg,
            )

            trainer = Trainer(
                max_epochs=n_tune_epochs,
                accelerator=cfg.trainer.accelerator,
                devices=cfg.trainer.devices,
                deterministic=cfg.trainer.deterministic,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False,
            )
            # Pass pre-built loaders instead of the datamodule to avoid
            # Lightning calling setup() (and load_dataset()) on every trial.
            # Fresh DataLoader objects are created each time so worker
            # processes are not shared across trainers.
            trainer.fit(
                orchestrator,
                train_dataloaders=dm_view.train_dataloader(),
                val_dataloaders=dm_view.val_dataloader(),
            )

            task_perf = float(
                trainer.callback_metrics.get(
                    'validation/avg_task_performance_epoch', 0.0
                )
            )

            # Evaluate comm accuracy on the validation split (no test leakage).
            val_dm = _ValAsTestWrapper(dm_view)
            comm_logs = orchestrator.evaluate_communication_accuracy(val_dm)

        comm_perf = float(comm_logs.get('test/avg_comm_task_perf', 0.0))

        denom = task_perf + comm_perf
        return (2.0 * task_perf * comm_perf / denom) if denom > 0.0 else 0.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_lmb = study.best_params['max_lmb']
    print(
        f'Tuning done  →  best max_lmb = {best_lmb:.4e}'
        f'  (harmonic mean = {study.best_value:.4f})\n'
    )
    return best_lmb


# ── Main ──────────────────────────────────────────────────────────────────────


@hydra.main(
    config_path='../config/hydra/',
    config_name='hetero_rate_multiagent_mnist',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    shift_strength = float(
        OmegaConf.to_container(cfg.dataset, resolve=True).get(
            'shift_strength', 0.0
        )
    )
    _finish_active_wandb_run()

    orch_target = OmegaConf.select(cfg, 'orchestrator._target_', default='')
    orch_name = orch_target.split('.')[-1] if orch_target else 'unknown'

    # ── Datamodule ────────────────────────────────────────────────────────────
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(
            cfg.agent_classes, resolve=True
        )
    datamodule = instantiate(dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup()

    n_agents = len(datamodule.models)

    # ── Graph ─────────────────────────────────────────────────────────────────
    agent_target_classes = _parse_agent_target_classes(cfg)
    if agent_target_classes is not None:
        # Guarantee every agent is a node even if some group omits it.
        for i in range(n_agents):
            agent_target_classes.setdefault(i, set())
    # seed/p/m are only consumed by the random-graph modes (erdos_renyi,
    # barabasi); read them defensively so configs using class_overlap /
    # fully_connected / manual don't have to declare them.
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n_agents,
        seed=cfg.graph.get('seed', 42),
        p=cfg.graph.get('p', 0.3),
        m=cfg.graph.get('m', 3),
        manual=cfg.graph.get('neighbors', {}),
        target_classes=agent_target_classes,
        max_edge_frac=cfg.graph.get('max_edge_frac', 0.4),
        similarity=cfg.graph.get('similarity', 'intersection'),
    )

    n_edges = sum(len(v) for v in neighbors.values()) // 2
    possible = n_agents * (n_agents - 1) // 2
    degrees = [len(neighbors[i]) for i in range(n_agents)]
    print(
        f'\nCommunication graph ({cfg.graph.neighbors_mode}): '
        f'{n_edges}/{possible} edges '
        f'({100 * n_edges / possible:.1f}% of fully connected)'
    )
    print(
        f'  degree: min={min(degrees)}, max={max(degrees)}, '
        f'mean={sum(degrees) / n_agents:.1f}  '
        f'(min>0 → no isolated agents)'
    )

    per_agents_cfg = _parse_per_agent_cfg(cfg)

    # ── Input effective rank (validation set, pre-training) ───────────────────
    # print(
    #     '\nInput effective rank per agent (validation split, before training):'
    # )
    # for i in range(n_agents):
    #     er = datamodule.input_effective_rank(i, split='val')
    #     print(f'  agent {i}: input ER = {er:.2f}')

    # ── Tune max_lmb (Optuna) if requested and the orchestrator supports it ──────
    # With learn_lmb the per-edge multipliers are learned by dual ascent during
    # training itself, so max_lmb is unused and the search would be wasted.
    learn_lmb = bool(
        OmegaConf.select(cfg, 'orchestrator.learn_lmb', default=False)
    )
    lmb_study_cfg = cfg.get('lmb_study', {})
    if learn_lmb and lmb_study_cfg.get('enabled', False):
        print('\n[lmb_study] skipped: orchestrator.learn_lmb=True (dual ascent).')
    if (
        lmb_study_cfg.get('enabled', False)
        and orch_name in ORCHESTRATORS_WITH_LMB
        and not learn_lmb
    ):
        agent_subset: list[int] | None = None
        if lmb_study_cfg.get('efficient', False):
            n_lmb_agents = min(EFFICIENT_LMB_AGENTS, n_agents)
            agent_subset = list(range(n_lmb_agents))
            print(
                f'\n[lmb_study] efficient mode → tuning on the first '
                f'{n_lmb_agents} agents only.'
            )
        best_lmb = tune_max_lmb(
            cfg,
            datamodule,
            neighbors,
            per_agents_cfg,
            n_trials=lmb_study_cfg.get('n_trials', 20),
            n_tune_epochs=lmb_study_cfg.get('n_tune_epochs', 5),
            lmb_low=lmb_study_cfg.get('lmb_low', 1e-4),
            lmb_high=lmb_study_cfg.get('lmb_high', 10.0),
            agent_subset=agent_subset,
        )
        with open_dict(cfg):
            cfg.orchestrator.max_lmb = best_lmb

    # ── Build run name ────────────────────────────────────────────────────────
    try:
        from hydra.core.hydra_config import HydraConfig

        job_num = HydraConfig.get().job.num
    except Exception:
        job_num = 0

    name_parts = [orch_name]
    if orch_name in ORCHESTRATORS_WITH_LMB:
        if learn_lmb:
            rho = float(
                OmegaConf.select(cfg, 'orchestrator.dual_rho', default=0.1)
            )
            name_parts.append(f'lmb_dual_rho{rho:g}')
        else:
            lmb = float(OmegaConf.select(cfg, 'orchestrator.max_lmb'))
            name_parts.append(f'lmb_{lmb:.4e}')
    if orch_name in ORCHESTRATORS_WITH_CTC:
        ctc = OmegaConf.select(
            cfg, 'orchestrator.comm_task_coeff', default=None
        )
        name_parts.append(f'ctc_{ctc}')
    shift = OmegaConf.select(cfg, 'dataset.shift_strength')
    name_parts.append(f'shift_{shift}_')
    name_parts.append(str(job_num))
    run_name = '_'.join(name_parts)

    # ── Logger ────────────────────────────────────────────────────────────────
    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    logger_cfg['name'] = run_name
    logger = instantiate(logger_cfg)

    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))
    _update_logger_config(logger, {'orchestrator_name': orch_name})

    # ── Build agents + orchestrator ───────────────────────────────────────────
    agents, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)
    _update_logger_config(logger, {'latent_dim': min(latent_dims.values())})
    orchestrator = _build_orchestrator(
        cfg, agents, neighbors, latent_dims, per_agents_cfg
    )

    # ── Trainer ─────────────────────────────────────────────────────────────────
    # Lightning forbids automatic gradient clipping under manual optimization, so
    # strip the clip keys for manual-opt orchestrators (e.g. ComFed) — they clip
    # their gradients manually inside their own training_step instead.
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not orchestrator.automatic_optimization:
        for key in ('gradient_clip_val', 'gradient_clip_algorithm'):
            trainer_kwargs.pop(key, None)
    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]
    trainer = Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger)

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
                'self_accuracy': cb.get(
                    f'test/private_task_perf_agent_{i}', float('nan')
                ),
                'comm_accuracy': cb.get(
                    f'test/comm_task_perf_agent_{i}', float('nan')
                ),
                'task_fidelity': cb.get(
                    f'test/task_fidelity_agent_{i}', float('nan')
                ),
                'shift_strength': shift_strength,
                'seed': int(cfg.seed),
            }
        )

    df = pd.DataFrame(rows)

    # ── Save parquet ──────────────────────────────────────────────────────────
    results_dir = Path('.') / 'results' / 'multi_agent'
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = (
        f'multi_agent__shift{shift_strength:.3f}__seed{cfg.seed}'
        f'__{timestamp}.parquet'
    )
    out_path = results_dir / fname
    df.to_parquet(out_path, index=False)
    print(f'\nResults saved → {out_path}')
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

    # Cleanup.
    remove_non_empty_dir('./multirun/')
    remove_non_empty_dir('./outputs/')
    remove_non_empty_dir('~/.cache/wandb/')
    remove_non_empty_dir(cfg.logger.project)
    _finish_active_wandb_run()


if __name__ == '__main__':
    main()
