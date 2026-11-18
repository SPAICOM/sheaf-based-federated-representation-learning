"""Does 10-class information emerge in the latent space of an even/odd agent.

Two MNIST agents (config ``mnist_2agents_parity_d40x8``): agent 0 learns the
10 digit classes (d=40), agent 1 only even vs odd (d=8), both with CenterNorm.
Each method is trained once per seed:

  noncoop   independent training, post-hoc Procrustes on the pilots at test time
  disheaf   DiSheafFRL (directed sheaf penalty on the shared unlabelled pilots),
            once per --lmb value

The 10-class k-NN probe runs on agent 1's TEST latents with the true digit
labels (src/utils/knn_probes.py: two stratified halves, cosine distance). The
same probe on agent 0's latents is the 10-class reference.

Also reported per run: agent 1 parity accuracy, agent 0 10-class accuracy, and
comm accuracy 1 -> 0 (agent 1's test latents mapped into agent 0's space and
decoded by agent 0's 10-class head). Post-hoc alignment only affects that last
number; it never changes agent 1's own latents.

Outputs: every number goes to the wandb project named after the config (set
WANDB_MODE=disabled for smoke runs, see `just mnist-parity-smoke`). Locally
each run only leaves <out>/latents/<run>.npz (test latents + scalars), the
offline cache that the tables, the plot and scripts/mnist_parity_probes.py are
recomputed from. A run whose npz exists for the same composed config is
skipped, so an interrupted sweep restarts with the same command.

Usage:
    uv run python scripts/mnist_parity_knn.py                      # seed 42, λ=4e-3
    uv run python scripts/mnist_parity_knn.py --seeds 1 2 3 --lmb 4e-3 3e-2
    uv run python scripts/mnist_parity_knn.py --plot-only
    uv run python scripts/mnist_parity_knn.py --seeds 1 2 3 --train-only  # parallel jobs
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)

from src.utils import (  # noqa: E402
    _finish_active_wandb_run,
    _update_logger_config,
    generate_neighbors,
    knn_probe,
)
from src.utils.plotting import use_style  # noqa: E402
from src.utils.progress import (  # noqa: E402
    EpochProgressBar,
    StatusFileCallback,
)

CONFIG_NAME = 'mnist_2agents_parity_d40x8'
FINE, PARITY = 0, 1  # agent ids
METHODS = {'noncoop': 'non_cooperative', 'disheaf': 'di_sheaf_frl'}
LABELS = {'noncoop': 'Non-coop + post-hoc alignment', 'disheaf': 'DiSheafFRL'}
COLORS = {'noncoop': '#eb6834', 'disheaf': '#2a78d6'}
# blue shades (light -> dark with λ) for DiSheafFRL; non-coop is always orange
LMB_CMAP = 'Blues'
# per-run scalars cached next to the latents
SCALARS = (
    'parity_acc_agent1',
    'fine_acc_agent0',
    'comm_acc_1_to_0',
    'gate_mean',
)
RUN_NAME = re.compile(
    r'(?P<method>noncoop|disheaf)(_lmb(?P<lmb>[^_]+))?_s(?P<seed>\d+)'
)


def apply_style() -> None:
    """House style at paper size (as the other plot scripts), no LaTeX."""
    use_style()
    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.size': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'lines.markersize': 5,
            'lines.linewidth': 1.2,
        }
    )


def run_color(method: str, lmb: float, lmbs: list[float]) -> object:
    """Colour of a (method, λ) curve: orange non-coop, blue shades by λ."""
    if method == 'noncoop':
        return COLORS['noncoop']
    if len(lmbs) == 1:
        return COLORS['disheaf']
    pos = list(lmbs).index(lmb) / (len(lmbs) - 1)
    return plt.get_cmap(LMB_CMAP)(0.35 + 0.6 * pos)


def legend_below(ax_or_fig, handles=None, labels=None, ncol=2) -> None:
    """Legend under the panels so it never covers the curves."""
    kw = {'frameon': False, 'ncol': ncol}
    if hasattr(ax_or_fig, 'savefig'):  # a Figure: one legend for all panels
        ax_or_fig.legend(handles, labels, loc='outside lower center', **kw)
    else:
        ax_or_fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), **kw)


def save_fig(fig, out: Path, name: str) -> None:
    """Save ``<out>/<name>.{pdf,png}`` (paper PDF + preview PNG)."""
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'plot -> {out / (name + ".png")}')


def run_stem(r: dict) -> str:
    """File stem of a cached run (``noncoop_s42``, ``disheaf_lmb0.1_s42``)."""
    tag = r['method'] + (
        f'_lmb{r["lmb"]:g}' if r['method'] == 'disheaf' else ''
    )
    return f'{tag}_s{r["seed"]}'


def cached_probe(out: Path, r: dict, kind: str, params: dict, compute):
    """Probe table of one run, computed once and cached in ``<out>/probes/``.

    The file name carries a hash of the setup, the probe kind and its
    parameters, so changing ``k``/``n``/``repeats`` recomputes and the same
    request is read back instantly.

    Parameters
    ----------
    out : Path
        Study folder.
    r : dict
        A run from ``load_runs``.
    kind : str
        Probe name (``knn``, ``wp``, ``fl``).
    params : dict
        JSON-serialisable probe parameters (part of the cache key).
    compute : callable
        Zero-argument function returning the run's probe ``DataFrame``.

    Returns
    -------
    pd.DataFrame
        The probe rows of the run.
    """
    key = hashlib.sha1(
        json.dumps([r['setup'], kind, params], sort_keys=True).encode()
    ).hexdigest()[:10]
    path = out / 'probes' / f'{run_stem(r)}_{kind}_{key}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def _compose(method: str, seed: int, lmb: float | None, args) -> object:
    overrides = [
        f'orchestrator={METHODS[method]}',
        f'logger.group={METHODS[method]}',
        f'seed={seed}',
        f'trainer.accelerator={args.accelerator}',
        f'dataset.pilot_num_samples={args.pilots}',
        f'dataset.agent_train_samples={args.train_samples}',
        *args.overrides,
    ]
    if args.epochs is not None:
        overrides.append(f'trainer.max_epochs={args.epochs}')
    if lmb is not None:
        overrides.append(f'orchestrator.max_lmb={lmb}')
    with initialize_config_dir(
        config_dir=str(ROOT / 'config/hydra'), version_base='1.3'
    ):
        return compose(config_name=args.config, overrides=overrides)


def setup_id(args) -> str:
    """Hash of the composed config minus method, λ and seed.

    Stored in every npz: a cached run is reused only under the same setup, and
    the tables refuse to mix setups (use one --out folder per setup).
    """
    cfg = OmegaConf.to_container(
        _compose('noncoop', 0, None, args), resolve=True
    )
    for key in ('seed', 'logger', 'orchestrator'):
        cfg.pop(key)
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[
        :10
    ]


def latents_path(args, method: str, lmb: float | None, seed: int) -> Path:
    """Offline cache of one run (read back by ``load_runs``)."""
    tag = method + (f'_lmb{lmb:g}' if lmb is not None else '')
    return args.out / 'latents' / f'{tag}_s{seed}.npz'


def _test_labels(ds) -> np.ndarray:
    return np.asarray(ds.dataset[ds.label_key], dtype=np.int64)


@torch.no_grad()
def _decode_argmax(orch, agent, Z: torch.Tensor) -> np.ndarray:
    was = agent.training
    agent.eval()
    preds = torch.cat(
        [
            agent.decoder(z.to(orch.device)).argmax(1).cpu()
            for z in Z.split(1024)
        ]
    )
    agent.train(was)
    return preds.numpy()


def run_one(method: str, seed: int, lmb: float | None, args) -> None:
    """Train one (method, λ, seed) cell and cache its test latents."""
    cfg = _compose(method, seed, lmb, args)
    seed_everything(cfg.seed, workers=True)
    dm = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    dm.prepare_data()
    dm.setup()
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode, n_agents=len(dm.models)
    )
    per_agents_cfg = _parse_per_agent_cfg(cfg)
    agents, latent_dims = _build_agents(cfg, dm, per_agents_cfg)
    name = method + (f'_lmb{lmb:g}' if lmb is not None else '') + f'_s{seed}'
    print(
        f'\n=== {name}: latent dims {latent_dims}, '
        f'{len(dm.train_datasets[0])} train rows/agent, '
        f'{len(dm.pilot_datasets[0])} pilots ===',
        flush=True,
    )
    orch = _build_orchestrator(
        cfg, agents, neighbors, latent_dims, per_agents_cfg
    )

    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    logger_cfg['name'] = name
    logger = instantiate(logger_cfg)
    _update_logger_config(logger, OmegaConf.to_container(cfg, resolve=True))
    trainer = Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=logger,
        callbacks=[StatusFileCallback(), EpochProgressBar()],
        enable_progress_bar=args.progress,
    )
    trainer.fit(orch, datamodule=dm)
    gate = trainer.callback_metrics.get(f'train/gate_mean_{FINE}_{PARITY}')
    trainer.test(orch, datamodule=dm, verbose=False)
    cb = {k: float(v) for k, v in trainer.callback_metrics.items()}

    y = _test_labels(dm.test_datasets[PARITY])
    assert np.array_equal(y, _test_labels(dm.test_datasets[FINE])), (
        'agents must share the test split'
    )
    Z = {
        i: orch._encode_dataset(orch.agents[str(i)], dm.test_datasets[i])
        for i in (FINE, PARITY)
    }
    scalars = {
        'parity_acc_agent1': float(
            (
                _decode_argmax(orch, orch.agents[str(PARITY)], Z[PARITY])
                == y % 2
            ).mean()
        ),
        'fine_acc_agent0': float(
            (_decode_argmax(orch, orch.agents[str(FINE)], Z[FINE]) == y).mean()
        ),
        'comm_acc_1_to_0': cb.get(
            f'test/comm_task_perf_agent_{FINE}', float('nan')
        ),
        'gate_mean': float(gate) if gate is not None else float('nan'),
    }
    knn = {i: knn_probe(Z[i].numpy(), y, args.k) for i in (FINE, PARITY)}

    # numbers -> wandb (summary of the training run)
    summary = {f'probe/{key}': v for key, v in scalars.items()}
    for k in args.k:
        summary[f'probe/knn_parity_agent_k{k}'] = knn[PARITY][k]
        summary[f'probe/knn_fine_agent_k{k}'] = knn[FINE][k]
    logger.experiment.summary.update(summary)

    # offline re-scoring cache
    path = latents_path(args, method, lmb, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        z_fine=Z[FINE].numpy(),
        z_parity=Z[PARITY].numpy(),
        y=y,
        setup=args.setup,
        **scalars,
    )

    print(
        f'parity acc (agent 1) {scalars["parity_acc_agent1"]:.4f} | '
        f'10-class acc (agent 0) {scalars["fine_acc_agent0"]:.4f} | '
        f'comm 1->0 {scalars["comm_acc_1_to_0"]:.4f}'
    )
    print(
        'k-NN 10-class on agent 1 latents: '
        + '  '.join(f'k={k}: {knn[PARITY][k]:.4f}' for k in args.k),
        flush=True,
    )

    _finish_active_wandb_run()
    del trainer, orch, agents, dm
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_runs(out: Path) -> list[dict]:
    """Read the cached runs of ``out`` (latents + scalars).

    Parameters
    ----------
    out : Path
        Study folder holding ``latents/*.npz``.

    Returns
    -------
    list[dict]
        One dict per run: ``method``, ``lmb``, ``seed``, the latents of both
        agents, the labels and the scalar metrics.
    """
    runs = []
    for f in sorted((out / 'latents').glob('*.npz')):
        m = RUN_NAME.fullmatch(f.stem)
        if m is None:
            continue
        d = np.load(f)
        runs.append(
            {
                'method': m['method'],
                'lmb': float(m['lmb'] or 0.0),
                'seed': int(m['seed']),
                'setup': str(d['setup']),
                **{k: d[k] for k in ('z_fine', 'z_parity', 'y')},
                **{k: float(d[k]) for k in SCALARS},
            }
        )
    if not runs:
        sys.exit(f'no cached runs in {out / "latents"}: run the script first')
    if len({r['setup'] for r in runs}) > 1:
        sys.exit(
            f'{out / "latents"} mixes runs of different setups (config, '
            'epochs, pilots, overrides); use one --out folder per setup'
        )
    return runs


def knn_table(runs: list[dict], ks: list[int], out: Path) -> pd.DataFrame:
    """One row per (run, k): both agents' 10-class k-NN plus the scalars.

    The k-NN scores are cached per run in ``<out>/probes/``.
    """
    rows = []
    for r in runs:

        def compute(r=r):
            return pd.DataFrame(
                [
                    {'agent': ag, 'k': k, 'acc': a}
                    for ag, key in (('parity', 'z_parity'), ('fine', 'z_fine'))
                    for k, a in knn_probe(r[key], r['y'], ks).items()
                ]
            )

        t = cached_probe(out, r, 'knn', {'ks': ks}, compute)
        par = t[t.agent == 'parity'].set_index('k')['acc']
        fine = t[t.agent == 'fine'].set_index('k')['acc']
        rows += [
            {
                'method': r['method'],
                'lmb': r['lmb'],
                'seed': r['seed'],
                'k': k,
                'knn_parity_agent': float(par[k]),
                'knn_fine_agent': float(fine[k]),
                **{s: r[s] for s in SCALARS},
            }
            for k in ks
        ]
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.grid(alpha=0.25, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _curves(df: pd.DataFrame):
    """List (method, λ, group), non-coop first and then λ ascending."""
    return sorted(
        ((m, lmb, g) for (m, lmb), g in df.groupby(['method', 'lmb'])),
        key=lambda t: (t[0] != 'noncoop', t[1]),
    )


def plot(df: pd.DataFrame, out: Path, dims: str) -> None:
    """10-class k-NN of the even/odd agent against k (and against λ)."""
    apply_style()
    lmbs = sorted(df.loc[df.method == 'disheaf', 'lmb'].unique())
    sweep = len(lmbs) > 1
    fig, axes = plt.subplots(
        1,
        2 if sweep else 1,
        figsize=(7.0 if sweep else 3.6, 3.0),
        constrained_layout=True,
        squeeze=False,
    )
    ax = axes[0, 0]
    for method, lmb, g in _curves(df):
        stats = g.groupby('k')['knn_parity_agent'].agg(['mean', 'std'])
        ax.errorbar(
            stats.index,
            stats['mean'],
            yerr=stats['std'].fillna(0.0),
            marker='o',
            capsize=2,
            elinewidth=0.8,
            color=run_color(method, lmb, lmbs),
            label=LABELS[method]
            + (f' (λ={lmb:g})' if method == 'disheaf' else ''),
        )
    ref = df.groupby('k')['knn_fine_agent'].mean()
    ax.plot(
        ref.index,
        ref.values,
        ls='--',
        lw=1.0,
        color='#6b6a63',
        label='Reference: 10-class agent',
    )
    ax.set_xscale('log')
    ax.set_xticks(sorted(df['k'].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('k (neighbours)')
    ax.set_ylabel('10-class k-NN accuracy')
    ax.set_title(f'(a) Even/odd agent, d = {dims.replace("x", " / ")}')
    _style(ax)
    legend_below(ax)
    if sweep:
        _plot_vs_lmb(df, axes[0, 1])
    save_fig(fig, out, 'knn_vs_k')


def _plot_vs_lmb(df: pd.DataFrame, ax, k: int = 10) -> None:
    """k-NN probe (one k) and comm accuracy 1 -> 0 against DiSheafFRL's λ."""
    ks = sorted(set(df['k']))
    k = k if k in ks else ks[len(ks) // 2]
    d = df[(df.method == 'disheaf') & (df.k == k)]
    nc = df[(df.method == 'noncoop') & (df.k == k)]
    # blue = DiSheafFRL, orange = non-coop; shade/marker tell the metrics apart
    style = {
        'knn_parity_agent': ('#1f5fb0', '#eb6834', 'o', f'{k}-NN 10-class'),
        'comm_acc_1_to_0': ('#7fb2ee', '#f5a57a', 's', 'comm. accuracy 1→0'),
    }
    for col, (blue, orange, marker, name) in style.items():
        s = d.groupby('lmb')[col].agg(['mean', 'std'])
        ax.errorbar(
            s.index,
            s['mean'],
            yerr=s['std'].fillna(0.0),
            marker=marker,
            capsize=2,
            elinewidth=0.8,
            color=blue,
            label=f'DiSheafFRL, {name}',
        )
        if not nc.empty:
            ax.axhline(
                nc[col].mean(),
                color=orange,
                ls=':',
                lw=1.2,
                label=f'Non-coop, {name}',
            )
    ax.set_xscale('log')
    ax.set_xlabel('DiSheafFRL λ')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'(b) Sweep over λ (k = {k})')
    _style(ax)
    legend_below(ax, ncol=1)


def summary(df: pd.DataFrame) -> None:
    """Print the k-NN table and the task accuracies (mean over seeds)."""
    piv = df.pivot_table(
        index=['method', 'lmb'],
        columns='k',
        values='knn_parity_agent',
        aggfunc='mean',
    )
    other = (
        df.groupby(['method', 'lmb'])
        .agg({**dict.fromkeys(SCALARS, 'mean'), 'seed': 'nunique'})
        .rename(columns={'seed': 'n_seeds'})
    )
    pd.set_option('display.width', 200)
    print("\n10-class k-NN accuracy on the even/odd agent's test latents:")
    print(piv.round(4).to_string())
    print('\nTask accuracies (mean over seeds):')
    print(other.round(4).to_string())


def main() -> None:
    """Train the missing cells, then print the tables and redraw the plot."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--config', default=CONFIG_NAME)
    p.add_argument(
        '--methods', nargs='+', default=list(METHODS), choices=list(METHODS)
    )
    p.add_argument(
        '--lmb',
        nargs='+',
        type=float,
        default=[4e-3],
        help='DiSheafFRL λ values',
    )
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument(
        '--k', nargs='+', type=int, default=[1, 3, 5, 10, 20, 50, 100]
    )
    p.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='default: trainer.max_epochs of the config (100)',
    )
    p.add_argument('--pilots', type=int, default=512)
    p.add_argument(
        '--train-samples',
        type=int,
        default=20000,
        help='training rows per agent',
    )
    p.add_argument(
        '--accelerator',
        default='auto',
        help="'auto' picks MPS on Apple silicon; 'cpu' also works",
    )
    p.add_argument(
        '--progress', action='store_true', help='show Lightning progress bars'
    )
    p.add_argument(
        '--out', type=Path, default=None, help='default: results/<config>'
    )
    p.add_argument('--plot-only', action='store_true')
    p.add_argument(
        '--train-only',
        action='store_true',
        help='train the missing cells and stop (parallel jobs; plot afterwards)',
    )
    p.add_argument(
        '--overrides',
        nargs='*',
        default=[],
        help='extra Hydra overrides, e.g. model.agents.1.encoder_hidden_dims=[16,64]',
    )
    args = p.parse_args()
    args.out = args.out or ROOT / 'results' / args.config

    warnings.filterwarnings('ignore', category=UserWarning)
    args.setup = setup_id(args)
    if not args.plot_only:
        cells = [
            (m, lm, s)
            for s in args.seeds
            for m in args.methods
            for lm in ([None] if m == 'noncoop' else args.lmb)
        ]
        for method, lmb, seed in cells:
            path = latents_path(args, method, lmb, seed)
            if path.exists() and str(np.load(path)['setup']) == args.setup:
                print(f'skip {method} λ={lmb} seed={seed}: cached in {path}')
                continue
            run_one(method, seed, lmb, args)

    if args.train_only:
        return
    runs = load_runs(args.out)
    df = knn_table(runs, args.k, args.out)
    summary(df)
    dims = f'{runs[0]["z_fine"].shape[1]}x{runs[0]["z_parity"].shape[1]}'
    plot(df, args.out, dims)


if __name__ == '__main__':
    main()
