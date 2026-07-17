"""Compression study (2-agent MNIST): ComFed vs SheafCFRL.

Compares the two communication-efficient orchestrators across a grid of
compression factors and distribution shifts, evaluating communication accuracy
via the orchestrator's built-in test-time alignment pipeline (triggered by
trainer.test):

  1. Whitening: each agent fits W_i / C_i on its training latents.
  2. Alignment: for every directed edge (i→j), fit A_{j←i} on shared
                pilot latents (whitened).
  3. Evaluation: self-accuracy, cross-agent communication accuracy, and
                 task fidelity are logged per agent and saved to parquet.

Sweep structure (all handled *inside* this script so that λ can be shared):

  * The orchestrator is the only Hydra multirun dimension (sheaf_cfrl, comfed).
  * For each orchestrator we loop over ``study.compression_factors`` and, for
    every factor, run **one** Optuna ``max_lmb`` study at a single reference
    shift.  The tuned ``max_lmb`` is then held **fixed across all
    ``study.shift_strengths``** — so a factor's λ never changes with the shift.
  * Compression is compared coherently: SheafCFRL keeps an edge stalk of
    ``c = round(factor · min_d)`` dimensions while ComFed projects onto
    ``proj_dim = round(factor · latent_dim)``.  With equal agent latent dims
    these coincide, giving a shared bottleneck-dimension x-axis for plotting.

Usage (compare both orchestrators):
    python scripts/compression_study.py --multirun orchestrator=sheaf_cfrl,comfed

Single orchestrator:
    python scripts/compression_study.py orchestrator=comfed
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

# Orchestrators whose compression bottleneck is a *fraction* of the edge stalk
# (SheafCFRL: c_ij = round(factor · min_d)) vs. those parameterised by an
# *absolute* projected/shared dimension (ComFed: proj_dim).  We drive both from
# a single ``compression_factor`` so the two are compared at the same effective
# bottleneck (see ``_apply_compression``).
ORCHESTRATORS_WITH_COMPRESSION_FACTOR = {'SheafCFRL'}
ORCHESTRATORS_WITH_PROJ_DIM = {'ComFed'}

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
    n_agents = len(agents)
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
    return instantiate(
        orch_cfg, **orch_kwargs, _convert_='all', _recursive_=False
    )


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
) -> float:
    """Search for the best max_lmb using Optuna.

    Runs ``n_trials`` short training runs of ``n_tune_epochs`` epochs each,
    maximising the harmonic mean of ``validation/avg_task_performance`` and
    validation-split communication accuracy.  Using the harmonic mean prevents
    either metric from dominating: a lambda that buys high task performance at
    the cost of near-zero comm accuracy (or vice-versa) scores poorly.

    Communication accuracy is evaluated on the validation split (via
    _ValAsTestWrapper) to keep the test set unseen during tuning.

    Returns the best max_lmb found.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(
        f'\nTuning max_lmb in [{lmb_low:.0e}, {lmb_high:.0e}] '
        f'— {n_trials} trials × {n_tune_epochs} epochs ...'
    )

    val_dm = _ValAsTestWrapper(datamodule)

    def objective(trial: optuna.Trial) -> float:
        lmb = trial.suggest_float('max_lmb', lmb_low, lmb_high, log=True)

        # Fresh agent weights for every trial (same seed → same init).
        agents, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)

        # Build a plain-dict copy of cfg so we can override max_lmb freely.
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.update(trial_cfg, 'orchestrator.max_lmb', lmb)

        orchestrator = _build_orchestrator(
            trial_cfg, agents, neighbors, latent_dims, per_agents_cfg
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
        # Fresh DataLoader objects are created each time so worker processes
        # are not shared across trainers.
        trainer.fit(
            orchestrator,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
        )

        task_perf = float(
            trainer.callback_metrics.get(
                'validation/avg_task_performance_epoch', 0.0
            )
        )

        # Evaluate comm accuracy on the validation split (no test leakage).
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


# ── Study helpers (compression grid + shared-λ full runs) ─────────────────────


def _build_datamodule(cfg: DictConfig, shift_strength: float) -> Any:
    """Instantiate + set up the datamodule at a given distribution shift."""
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg['shift_strength'] = float(shift_strength)
    if 'agent_classes' in cfg:
        dataset_cfg['agent_classes'] = OmegaConf.to_container(
            cfg.agent_classes, resolve=True
        )
    datamodule = instantiate(dataset_cfg)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def _apply_compression(
    cfg: DictConfig,
    orch_name: str,
    compression_factor: float,
    latent_dim: int,
) -> int:
    """Set the orchestrator's compression hyperparameter for this factor.

    The two orchestrators expose compression differently — SheafCFRL via the
    fraction ``compression_factor`` (edge stalk ``c = round(factor · min_d)``)
    and ComFed via the absolute ``proj_dim`` (shared-space dimension).  To
    compare them at the *same* effective bottleneck we map the fraction to a
    projection dimension ``proj_dim = round(factor · latent_dim)`` — which, when
    both agents share ``latent_dim = min_d``, equals SheafCFRL's edge-stalk size.

    Mutates ``cfg.orchestrator`` in place and returns the resolved bottleneck
    dimension (recorded on every row as ``proj_dim`` for a common x-axis).
    """
    bottleneck_dim = max(1, round(float(compression_factor) * int(latent_dim)))
    with open_dict(cfg):
        if orch_name in ORCHESTRATORS_WITH_PROJ_DIM:
            cfg.orchestrator.proj_dim = bottleneck_dim
        elif orch_name in ORCHESTRATORS_WITH_COMPRESSION_FACTOR:
            cfg.orchestrator.compression_factor = float(compression_factor)
    return bottleneck_dim


def _new_wandb_run_logger(cfg: DictConfig, run_name: str) -> Any:
    """Instantiate ``cfg.logger`` backed by an *explicitly* fresh wandb run.

    Lightning's ``WandbLogger.experiment`` property is lazy: the first time it
    is accessed it checks ``if wandb.run is not None`` and, if so, **silently
    reuses that run** — it never even calls ``wandb.init()`` in that branch, so
    kwargs like ``reinit``/``id``/``name`` passed to the logger are irrelevant.
    Any small gap between finishing the previous run and constructing the next
    logger (e.g. a slow network flush online) falls into that reuse branch,
    which is exactly what was collapsing every shift into one wandb run.

    We sidestep the lazy property entirely: finish any active run ourselves,
    call ``wandb.init()`` directly, and hand the resulting ``Run`` to
    ``WandbLogger`` via its ``experiment=`` constructor argument — at that
    point ``self._experiment`` is already set, so the reuse-check code path is
    never consulted.
    """
    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    target = str(logger_cfg.get('_target_', ''))
    logger_cfg['name'] = run_name

    if 'Wandb' not in target:
        return instantiate(logger_cfg)

    import wandb

    _finish_active_wandb_run()
    run = wandb.init(
        project=logger_cfg.get('project'),
        group=logger_cfg.get('group'),
        name=run_name,
        dir=logger_cfg.get('save_dir') or logger_cfg.get('dir') or '.',
        reinit=True,
    )
    return instantiate(logger_cfg, experiment=run)


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
    job_num: int,
) -> pd.DataFrame:
    """Train + test one (compression_factor, shift) configuration; return rows.

    ``cfg.orchestrator`` is expected to already carry the resolved compression
    hyperparameter (via :func:`_apply_compression`) and the shared ``max_lmb``.
    """
    n_agents = len(datamodule.models)

    # ── Build run name ────────────────────────────────────────────────────────
    name_parts = [orch_name]
    if orch_name in ORCHESTRATORS_WITH_COMPRESSION_FACTOR:
        name_parts.append(f'cf_{compression_factor}')
    if orch_name in ORCHESTRATORS_WITH_PROJ_DIM:
        name_parts.append(f'pd_{proj_dim}')
    if orch_name in ORCHESTRATORS_WITH_LMB:
        lmb = float(OmegaConf.select(cfg, 'orchestrator.max_lmb'))
        name_parts.append(f'lmb_{lmb:.4e}')
    name_parts.append(f'shift_{shift_strength}')
    name_parts.append(str(job_num))
    run_name = '_'.join(name_parts)

    # ── Logger ────────────────────────────────────────────────────────────────
    # These three are the ground truth for this iteration of the study grid.
    # They're namespaced with a "sweep_" prefix so they can never collide with
    # an orchestrator's own hparams: ComFed absorbs the sheaf-only
    # `orchestrator.compression_factor` override into its **kwargs (it doesn't
    # use it — proj_dim is what actually drives its bottleneck), and
    # `self.save_hyperparameters()` captures it as an hparam. Lightning
    # auto-logs those hparams when `trainer.fit()` starts, which — with an
    # unprefixed key — silently overwrote this exact metadata with that
    # stale, never-varying value (proj_dim was unaffected only because it's a
    # real, correctly-mutated ComFed constructor argument).
    sweep_metadata = {
        'orchestrator_name': orch_name,
        'sweep_compression_factor': float(compression_factor),
        'sweep_proj_dim': int(proj_dim),
        'sweep_shift_strength': float(shift_strength),
    }
    logger = _new_wandb_run_logger(cfg, run_name)
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))
    _update_logger_config(logger, sweep_metadata)

    # ── Full training run ─────────────────────────────────────────────────────
    callbacks = [instantiate(cb_conf) for cb_conf in cfg.callbacks.values()]
    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    agents, latent_dims = _build_agents(cfg, datamodule, per_agents_cfg)
    _update_logger_config(logger, {'latent_dim': min(latent_dims.values())})
    orchestrator = _build_orchestrator(
        cfg, agents, neighbors, latent_dims, per_agents_cfg
    )

    trainer.fit(orchestrator, datamodule=datamodule)
    # Re-assert after fit(): Lightning's automatic log_hyperparams() call
    # (triggered inside fit) can overwrite same-named keys with the
    # orchestrator's own (possibly stale/unused) hparams — see note above.
    # The "sweep_" prefix already prevents any collision, this is belt-and-
    # suspenders in case a future orchestrator absorbs one of these exact names.
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
                'seed': int(cfg.seed),
            }
        )

    # Finish THIS run through its own handle so wandb.run is cleared before the
    # next (factor, shift) — otherwise Lightning's WandbLogger silently reuses
    # the still-open run and every shift collapses into one wandb run. Any
    # failure here is printed rather than swallowed: silently ignoring it would
    # reproduce exactly that collapse without any visible signal.
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
    config_name='hetero_rate_2agents_mnist_compression',
    version_base='1.3',
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)
    _finish_active_wandb_run()

    orch_target = OmegaConf.select(cfg, 'orchestrator._target_', default='')
    orch_name = orch_target.split('.')[-1] if orch_target else 'unknown'

    # ── Study grid ────────────────────────────────────────────────────────────
    study_cfg = cfg.get('study', {})
    compression_factors = [
        float(x) for x in study_cfg.get('compression_factors', [0.5])
    ]
    shift_strengths = [
        float(x) for x in study_cfg.get('shift_strengths', [0.7])
    ]

    lmb_study_cfg = cfg.get('lmb_study', {})
    tune_enabled = (
        bool(lmb_study_cfg.get('enabled', False))
        and orch_name in ORCHESTRATORS_WITH_LMB
    )
    # The λ study is run once per (orchestrator, compression_factor) at a single
    # reference shift; that λ is then held fixed across every shift strength.
    ref_shift = lmb_study_cfg.get('ref_shift_strength', None)
    if ref_shift is None:
        ref_shift = shift_strengths[len(shift_strengths) // 2]
    ref_shift = float(ref_shift)

    per_agents_cfg = _parse_per_agent_cfg(cfg)
    groups = _parse_groups(cfg)
    a2g = _agent_to_group(groups)

    try:
        from hydra.core.hydra_config import HydraConfig

        job_num = HydraConfig.get().job.num
    except Exception:
        job_num = 0

    # ── Reference datamodule (λ tuning + latent-dim discovery) ────────────────
    print(
        f'\n[{orch_name}] building reference datamodule '
        f'(shift_strength={ref_shift}) ...'
    )
    ref_dm = _build_datamodule(cfg, ref_shift)
    n_agents = len(ref_dm.models)

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
        er = ref_dm.input_effective_rank(i, split='val')
        print(f'  agent {i}: input ER = {er:.2f}')

    # Latent dim is fixed by the encoder config (independent of shift/factor);
    # discover it once so ComFed's proj_dim can be derived per factor.
    _, latent_dims = _build_agents(cfg, ref_dm, per_agents_cfg)
    latent_dim = min(latent_dims.values())
    print(f'\nLatent dim (min over agents): {latent_dim}')

    # Datamodules are deterministic per shift → build once, reuse across factors.
    dm_cache: dict[float, Any] = {ref_shift: ref_dm}

    results_dir = Path('.') / 'results' / 'compression'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_frames: list[pd.DataFrame] = []
    for compression_factor in compression_factors:
        proj_dim = _apply_compression(
            cfg, orch_name, compression_factor, latent_dim
        )
        print(
            f'\n=== {orch_name} | compression_factor={compression_factor} '
            f'→ bottleneck dim={proj_dim} ==='
        )

        # ── One λ study per compression factor, shared across all shifts ──────
        if tune_enabled:
            best_lmb = tune_max_lmb(
                cfg,
                ref_dm,
                neighbors,
                per_agents_cfg,
                n_trials=lmb_study_cfg.get('n_trials', 20),
                n_tune_epochs=lmb_study_cfg.get('n_tune_epochs', 5),
                lmb_low=lmb_study_cfg.get('lmb_low', 1e-4),
                lmb_high=lmb_study_cfg.get('lmb_high', 10.0),
            )
            with open_dict(cfg):
                cfg.orchestrator.max_lmb = best_lmb

        # ── Full training run per distribution shift (fixed λ) ────────────────
        for shift in shift_strengths:
            # Reflect the active shift in cfg so the *logged* config (and thus
            # the wandb run) reports the true shift instead of the file default.
            with open_dict(cfg):
                cfg.dataset.shift_strength = float(shift)

            datamodule = dm_cache.get(shift)
            if datamodule is None:
                datamodule = _build_datamodule(cfg, shift)
                dm_cache[shift] = datamodule

            df = _run_full_experiment(
                cfg,
                datamodule,
                neighbors,
                per_agents_cfg,
                orch_name,
                a2g,
                compression_factor,
                proj_dim,
                shift,
                job_num,
            )
            all_frames.append(df)

            # Incremental save (robust against a crash mid-grid).
            fname = (
                f'{orch_name}__cf{compression_factor:.3f}'
                f'__shift{shift:.3f}__seed{cfg.seed}__{timestamp}.parquet'
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
