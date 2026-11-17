"""Does 10-class information emerge in the latent space of an even/odd agent?

Two MNIST agents (config ``mnist_2agents_parity``): agent 0 learns the 10 digit
classes (d=40), agent 1 only even vs odd (d=8), both with CenterNorm. Each
method is trained once per seed:

  noncoop   independent training, post-hoc Procrustes on the pilots at test time
  disheaf   DiSheafFRL (directed sheaf penalty on the shared unlabelled pilots),
            once per --lmb value

The 10-class k-NN probe runs on agent 1's TEST latents with the true digit
labels: the test split is cut in two stratified halves, the probe is fitted on
one and scored on the other, then the halves swap and the two scores are
averaged. Cosine distance (the CenterNorm latents lie on the unit sphere). The
same probe on agent 0's latents is reported as a 10-class reference.

Also reported per run: agent 1 parity accuracy, agent 0 10-class accuracy, and
comm accuracy 1 -> 0 (agent 1's test latents mapped into agent 0's space and
decoded by agent 0's 10-class head). Post-hoc alignment only affects that last
number; it never changes agent 1's own latents.

Rows are appended to <out>/runs.csv as runs finish, and finished (method, λ,
seed, epochs, pilots, latent dims) cells are skipped on a re-run; the latent
dims (e.g. '40x8') are read from the config, so other sizes live side by side. The plot is redrawn from the CSV at the end.

Usage:
    uv run python scripts/mnist_parity_knn.py                      # seed 42, λ=4e-3
    uv run python scripts/mnist_parity_knn.py --seeds 1 2 3 --lmb 4e-3 3e-2
    uv run python scripts/mnist_parity_knn.py --epochs 2 --out results/tmp_smoke  # quick check
    uv run python scripts/mnist_parity_knn.py --plot-only
"""

from __future__ import annotations

import argparse
import gc
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
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from multi_agent_experiment import (
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)

from src.utils.graph_generator import generate_neighbors

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = 'mnist_2agents_parity'
FINE, PARITY = 0, 1   # agent ids
METHODS = {'noncoop': 'non_cooperative', 'disheaf': 'di_sheaf_frl'}
LABELS = {'noncoop': 'Non-coop + post-hoc alignment', 'disheaf': 'DiSheafFRL'}
COLORS = {'noncoop': '#2a78d6', 'disheaf': '#eb6834'}   # categorical slots 1, 2


def _compose(method: str, seed: int, lmb: float | None, args) -> object:
    overrides = [f'orchestrator={METHODS[method]}', f'seed={seed}',
                 f'trainer.max_epochs={args.epochs}',
                 f'trainer.accelerator={args.accelerator}',
                 f'dataset.pilot_num_samples={args.pilots}',
                 f'dataset.agent_train_samples={args.train_samples}',
                 *args.overrides]
    if lmb is not None:
        overrides.append(f'orchestrator.max_lmb={lmb}')
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        return compose(config_name=CONFIG_NAME, overrides=overrides)


def latents_path(args, method: str, lmb: float | None, seed: int) -> Path:
    """Test latents of one run (read by scripts/mnist_parity_probes.py)."""
    tag = method + (f'_lmb{lmb:g}' if lmb is not None else '')
    return args.out / 'latents' / f'{tag}_s{seed}_e{args.epochs}_p{args.pilots}_d{args.dims}.npz'


def dims_tag(args) -> str:
    """'<d_fine>x<d_parity>' as configured (overrides included)."""
    cfg = _compose('noncoop', 0, None, args)
    return 'x'.join(str(cfg.model.agents[i].encoder_hidden_dims[-1]) for i in (FINE, PARITY))


def _test_labels(ds) -> np.ndarray:
    return np.asarray(ds.dataset[ds.label_key], dtype=np.int64)


def _halves(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two stratified halves of the test split (the same for every run)."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a.append(idx[: len(idx) // 2])
        b.append(idx[len(idx) // 2:])
    return np.concatenate(a), np.concatenate(b)


def knn_probe(Z: np.ndarray, y: np.ndarray, ks: list[int]) -> dict[int, float]:
    a, b = _halves(y)
    out = {}
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)
        scores = [clf.fit(Z[f], y[f]).score(Z[s], y[s]) for f, s in ((a, b), (b, a))]
        out[k] = float(np.mean(scores))
    return out


@torch.no_grad()
def _decode_argmax(orch, agent, Z: torch.Tensor) -> np.ndarray:
    was = agent.training
    agent.eval()
    preds = torch.cat([agent.decoder(z.to(orch.device)).argmax(1).cpu() for z in Z.split(1024)])
    agent.train(was)
    return preds.numpy()


def run_one(method: str, seed: int, lmb: float | None, args) -> list[dict]:
    cfg = _compose(method, seed, lmb, args)
    seed_everything(cfg.seed, workers=True)
    dm = instantiate(OmegaConf.to_container(cfg.dataset, resolve=True))
    dm.prepare_data()
    dm.setup()
    neighbors = generate_neighbors(mode=cfg.graph.neighbors_mode, n_agents=len(dm.models))
    per_agents_cfg = _parse_per_agent_cfg(cfg)
    agents, latent_dims = _build_agents(cfg, dm, per_agents_cfg)
    print(f'\n=== {method} λ={lmb} seed={seed}: latent dims {latent_dims}, '
          f'{len(dm.train_datasets[0])} train rows/agent, {len(dm.pilot_datasets[0])} pilots ===',
          flush=True)
    orch = _build_orchestrator(cfg, agents, neighbors, latent_dims, per_agents_cfg)

    logger = False
    if args.wandb:
        logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
        logger_cfg['name'] = f'{method}' + (f'_lmb{lmb:g}' if lmb is not None else '') + f'_s{seed}'
        logger = instantiate(logger_cfg)
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer = Trainer(**trainer_kwargs, logger=logger, enable_progress_bar=args.progress)
    trainer.fit(orch, datamodule=dm)
    gate = trainer.callback_metrics.get(f'train/gate_mean_{FINE}_{PARITY}')
    trainer.test(orch, datamodule=dm, verbose=False)
    cb = {k: float(v) for k, v in trainer.callback_metrics.items()}

    ds = dm.test_datasets[PARITY]
    y = _test_labels(ds)
    assert np.array_equal(y, _test_labels(dm.test_datasets[FINE])), 'agents must share the test split'
    Z = {i: orch._encode_dataset(orch.agents[str(i)], dm.test_datasets[i]) for i in (FINE, PARITY)}
    parity_acc = float((_decode_argmax(orch, orch.agents[str(PARITY)], Z[PARITY]) == y % 2).mean())
    fine_acc = float((_decode_argmax(orch, orch.agents[str(FINE)], Z[FINE]) == y).mean())
    knn = {i: knn_probe(Z[i].numpy(), y, args.k) for i in (FINE, PARITY)}
    path = latents_path(args, method, lmb, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, z_fine=Z[FINE].numpy(), z_parity=Z[PARITY].numpy(), y=y)

    rows = []
    for k in args.k:
        rows.append({
            'method': method, 'lmb': 0.0 if lmb is None else lmb, 'seed': seed, 'k': k,
            'knn_parity_agent': knn[PARITY][k], 'knn_fine_agent': knn[FINE][k],
            'parity_acc_agent1': parity_acc, 'fine_acc_agent0': fine_acc,
            'comm_acc_1_to_0': cb.get(f'test/comm_task_perf_agent_{FINE}', float('nan')),
            'gate_mean': float(gate) if gate is not None else float('nan'),
            'epochs': args.epochs, 'pilots': args.pilots, 'train_samples': args.train_samples,
            'dims': args.dims,
        })
    print(f'parity acc (agent 1) {parity_acc:.4f} | 10-class acc (agent 0) {fine_acc:.4f} | '
          f'comm 1->0 {rows[0]["comm_acc_1_to_0"]:.4f}')
    print('k-NN 10-class on agent 1 latents: '
          + '  '.join(f'k={k}: {knn[PARITY][k]:.4f}' for k in args.k), flush=True)

    del trainer, orch, agents, dm
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return rows


def plot(df: pd.DataFrame, out: Path) -> None:
    lmbs = sorted(df.loc[df.method == 'disheaf', 'lmb'].unique())
    sweep = len(lmbs) > 1
    fig, axes = plt.subplots(1, 2 if sweep else 1, figsize=(13 if sweep else 7, 4.8),
                             constrained_layout=True, squeeze=False)
    ax = axes[0, 0]
    # one hue, light -> dark with λ, when several DiSheafFRL λ are on the plot
    shades = dict(zip(lmbs, plt.get_cmap('Oranges')(np.linspace(0.4, 0.95, len(lmbs)))))
    seeds = df.groupby(['method', 'lmb'])['seed'].nunique()
    for (method, lmb), g in df.groupby(['method', 'lmb']):
        stats = g.groupby('k')['knn_parity_agent'].agg(['mean', 'std']).reset_index()
        label = LABELS[method] + (f' (λ={lmb:g})' if method == 'disheaf' else '')
        color = shades[lmb] if method == 'disheaf' and sweep else COLORS[method]
        ax.errorbar(stats['k'], stats['mean'], yerr=stats['std'].fillna(0.0), marker='o',
                    ms=5, lw=2, capsize=3, color=color, label=label)
    if sweep:
        _plot_vs_lmb(df, axes[0, 1])
    ref = df.groupby('k')['knn_fine_agent'].mean()
    ax.plot(ref.index, ref.values, ls='--', lw=1.5, color='#6b6a63',
            label='Reference: 10-class agent latents (all runs)')
    ax.set_xscale('log')
    ax.set_xticks(sorted(df['k'].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('k (neighbours)')
    ax.set_ylabel('10-class k-NN accuracy (test, 2-fold)')
    n_seeds = int(seeds.max())
    dims = df['dims'].iloc[0]
    ax.set_title(f'Digit identity in the even/odd agent\'s latent space (d = {dims.replace("x", " / ")})'
                 + (f' — mean ± std over {n_seeds} seeds' if n_seeds > 1 else ' — single seed'),
                 fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'knn_vs_k_d{dims}.{ext}', dpi=200)
    plt.close(fig)
    print(f'plot -> {out / f"knn_vs_k_d{dims}.png"}')


def _style(ax) -> None:
    ax.grid(alpha=0.25, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _plot_vs_lmb(df: pd.DataFrame, ax, k: int = 10) -> None:
    """k-NN probe (one k) and comm accuracy 1 -> 0 against DiSheafFRL's λ."""
    k = k if k in set(df['k']) else int(sorted(df['k'])[len(set(df['k'])) // 2])
    d = df[(df.method == 'disheaf') & (df.k == k)]
    nc = df[(df.method == 'noncoop') & (df.k == k)]
    for col, name, color in (('knn_parity_agent', f'{k}-NN 10-class, even/odd agent', COLORS['disheaf']),
                             ('comm_acc_1_to_0', 'Comm accuracy 1 → 0', '#1baf7a')):
        s = d.groupby('lmb')[col].agg(['mean', 'std'])
        ax.errorbar(s.index, s['mean'], yerr=s['std'].fillna(0.0), marker='o', ms=5, lw=2,
                    capsize=3, color=color, label=f'DiSheafFRL: {name}')
        if not nc.empty:
            ax.axhline(nc[col].mean(), color=color, ls=':', lw=1.5,
                       label=f'Non-coop: {name}')
    ax.set_xscale('log')
    ax.set_xlabel('DiSheafFRL λ')
    ax.set_ylabel('Accuracy (test)')
    ax.set_title(f'Sweep over λ (k={k}; dotted = non-coop)', fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8)


def summary(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index=['method', 'lmb'], columns='k', values='knn_parity_agent',
                         aggfunc='mean')
    other = df.groupby(['method', 'lmb'])[['parity_acc_agent1', 'fine_acc_agent0',
                                           'comm_acc_1_to_0', 'gate_mean', 'seed']].agg(
        {'parity_acc_agent1': 'mean', 'fine_acc_agent0': 'mean', 'comm_acc_1_to_0': 'mean',
         'gate_mean': 'mean', 'seed': 'nunique'}).rename(columns={'seed': 'n_seeds'})
    pd.set_option('display.width', 200)
    print('\n10-class k-NN accuracy on the even/odd agent\'s test latents (mean over seeds):')
    print(piv.round(4).to_string())
    print('\nTask accuracies (mean over seeds):')
    print(other.round(4).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS))
    p.add_argument('--lmb', nargs='+', type=float, default=[4e-3], help='DiSheafFRL λ values')
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--k', nargs='+', type=int, default=[1, 3, 5, 10, 20, 50, 100])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--pilots', type=int, default=512)
    p.add_argument('--train-samples', type=int, default=20000, help='training rows per agent')
    p.add_argument('--accelerator', default='auto', help="'auto' picks MPS on Apple silicon; 'cpu' also works")
    p.add_argument('--wandb', action='store_true', help='log runs to the mnist_2agents_parity project')
    p.add_argument('--progress', action='store_true', help='show Lightning progress bars')
    p.add_argument('--out', type=Path, default=ROOT / 'results/mnist_2agents_parity')
    p.add_argument('--plot-only', action='store_true')
    p.add_argument('--overrides', nargs='*', default=[], help='extra Hydra overrides, e.g. model.agents.1.encoder_hidden_dims=[16,64]')
    args = p.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning)
    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / 'runs.csv'
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    if not df.empty and 'dims' not in df:
        df['dims'] = '320x64'   # runs from before the dims column
    args.dims = dims_tag(args)

    if not args.plot_only:
        cells = [(m, lm, s) for s in args.seeds for m in args.methods
                 for lm in ([None] if m == 'noncoop' else args.lmb)]
        for method, lmb, seed in cells:
            cell = (not df.empty) and ((df.method == method) & (df.seed == seed)
                    & np.isclose(df.lmb, 0.0 if lmb is None else lmb)
                    & (df.epochs == args.epochs) & (df.pilots == args.pilots)
                    & (df.dims == args.dims))
            if cell is not False and cell.any() and latents_path(args, method, lmb, seed).exists():
                print(f'skip {method} λ={lmb} seed={seed}: already in {csv}')
                continue
            rows = pd.DataFrame(run_one(method, seed, lmb, args))
            if cell is not False:   # re-run of a cell without saved latents: replace its rows
                df = df[~cell]
            df = pd.concat([df, rows], ignore_index=True)
            df.to_csv(csv, index=False)

    if df.empty:
        print(f'no runs in {csv}')
        return
    df = df[(df.epochs == args.epochs) & (df.pilots == args.pilots) & (df.dims == args.dims)]
    summary(df)
    plot(df, args.out)


if __name__ == '__main__':
    main()
