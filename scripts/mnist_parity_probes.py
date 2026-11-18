"""Sharper probes of digit identity in the even/odd agent's latent space.

Reads the test latents cached by scripts/mnist_parity_knn.py
(<out>/latents/*.npz) and runs two probes on the even/odd agent (with the
10-class agent as reference). No training happens here and nothing is logged
to wandb; tables are printed and the figure is saved to <out>. Each run's
probe scores are cached in <out>/probes/ (keyed by setup and probe
parameters), so re-running only redraws unless k, n or repeats change.

1. Within-parity k-NN: tell the 5 even digits apart among even test images, and
   the 5 odd digits among odd ones (chance 0.20). Parity itself is what the agent
   was trained on, so this isolates the digit information it was never asked to
   keep. Same stratified 2-fold split of the test set as the 10-class probe; the
   score pools both parity groups and both folds.
2. Few-label k-NN: the 10-class probe with only n labelled references per class,
   drawn from one half of the test set and scored on the other (then swapped),
   averaged over --repeats random draws. k > n is skipped.

Cosine distance throughout (CenterNorm latents lie on the unit sphere).

Usage:
    uv run python scripts/mnist_parity_probes.py
    uv run python scripts/mnist_parity_probes.py --k 1 5 10 20 --n 1 5 10 50 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from mnist_parity_knn import (  # noqa: E402
    CONFIG_NAME,
    LABELS,
    apply_style,
    cached_probe,
    legend_below,
    load_runs,
    run_color,
    save_fig,
)

from src.utils import few_label, within_parity  # noqa: E402

AGENTS = {'parity': 'z_parity', 'fine': 'z_fine'}


def _tag(method: str, lmb: float) -> str:
    return LABELS[method] + (f' (λ={lmb:g})' if method == 'disheaf' else '')


def _axis_k(ax, ticks) -> None:
    ax.set_xscale('log')
    ax.set_xticks(sorted(ticks))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(alpha=0.25, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def _curves(d: pd.DataFrame):
    """List (method, λ, group), non-coop first and then λ ascending."""
    return sorted(
        ((m, lmb, g) for (m, lmb), g in d.groupby(['method', 'lmb'])),
        key=lambda t: (t[0] != 'noncoop', t[1]),
    )


def plot(wp: pd.DataFrame, fl: pd.DataFrame, out: Path, k_few: int) -> None:
    """Within-parity k-NN against k, and few-label k-NN against n."""
    apply_style()
    lmbs = sorted(wp.loc[wp.method == 'disheaf', 'lmb'].unique())
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.8), constrained_layout=True
    )
    for method, lmb, g in _curves(wp[wp.agent == 'parity']):
        s = g.groupby('k')['acc'].agg(['mean', 'std'])
        ax1.errorbar(
            s.index,
            s['mean'],
            yerr=s['std'].fillna(0.0),
            marker='o',
            capsize=2,
            elinewidth=0.8,
            color=run_color(method, lmb, lmbs),
            label=_tag(method, lmb),
        )
    ref = wp[wp.agent == 'fine'].groupby('k')['acc'].mean()
    ax1.plot(
        ref.index,
        ref.values,
        ls='--',
        lw=1.0,
        color='#6b6a63',
        label='Reference: 10-class agent',
    )
    ax1.axhline(0.2, color='#9a998f', lw=0.8, ls=':')
    ax1.text(wp['k'].min(), 0.21, 'chance', color='#6b6a63', fontsize=7)
    _axis_k(ax1, wp['k'].unique())
    ax1.set_xlabel('k (neighbours)')
    ax1.set_ylabel('Within-parity k-NN accuracy')
    ax1.set_title('(a) Digits within the same parity')

    f = fl[fl.k == k_few]
    for method, lmb, g in _curves(f[f.agent == 'parity']):
        s = g.groupby('n')['acc'].mean()
        sd = g.groupby('n')['std'].mean()
        ax2.errorbar(
            s.index,
            s.values,
            yerr=sd.values,
            marker='o',
            capsize=2,
            elinewidth=0.8,
            color=run_color(method, lmb, lmbs),
        )
    ref = f[f.agent == 'fine'].groupby('n')['acc'].mean()
    ax2.plot(ref.index, ref.values, ls='--', lw=1.0, color='#6b6a63')
    ax2.axhline(0.1, color='#9a998f', lw=0.8, ls=':')
    ax2.text(f['n'].min(), 0.11, 'chance', color='#6b6a63', fontsize=7)
    _axis_k(ax2, f['n'].unique())
    ax2.set_xlabel('Labelled references per class')
    ax2.set_ylabel(f'10-class {k_few}-NN accuracy')
    ax2.set_title('(b) Few-label probe')
    handles, labels = ax1.get_legend_handles_labels()
    legend_below(fig, handles, labels, ncol=4)
    save_fig(fig, out, 'probes')


def main() -> None:
    """Probe every cached run, print the tables and save the figure."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--config', default=CONFIG_NAME)
    p.add_argument(
        '--out', type=Path, default=None, help='default: results/<config>'
    )
    p.add_argument(
        '--k', nargs='+', type=int, default=[1, 3, 5, 10, 20, 50, 100]
    )
    p.add_argument('--n', nargs='+', type=int, default=[1, 5, 10, 50, 100])
    p.add_argument(
        '--few-k',
        nargs='+',
        type=int,
        default=[1, 5, 10],
        help='k values for the few-label probe',
    )
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    out = args.out or ROOT / 'results' / args.config

    runs = load_runs(out)
    print(f'probing {len(runs)} cached runs from {out / "latents"}')
    wp_parts, fl_parts = [], []
    for r in runs:
        meta = {'method': r['method'], 'lmb': r['lmb'], 'seed': r['seed']}

        def wp_rows(r=r):
            return pd.DataFrame(
                [
                    {'agent': agent, 'k': k, 'acc': acc}
                    for agent, key in AGENTS.items()
                    for k, acc in within_parity(r[key], r['y'], args.k).items()
                ]
            )

        def fl_rows(r=r):
            return pd.DataFrame(
                [
                    {'agent': agent, 'n': n, 'k': k, 'acc': acc, 'std': sd}
                    for agent, key in AGENTS.items()
                    for (n, k), (acc, sd) in few_label(
                        r[key], r['y'], args.n, args.few_k, args.repeats
                    ).items()
                ]
            )

        wp_parts.append(
            cached_probe(out, r, 'wp', {'ks': args.k}, wp_rows).assign(**meta)
        )
        fl_parts.append(
            cached_probe(
                out,
                r,
                'fl',
                {'ns': args.n, 'ks': args.few_k, 'repeats': args.repeats},
                fl_rows,
            ).assign(**meta)
        )
        print(
            f'done {r["method"]} λ={r["lmb"]:g} seed={r["seed"]}', flush=True
        )

    wp, fl = pd.concat(wp_parts), pd.concat(fl_parts)
    pd.set_option('display.width', 200)
    idx = ['method', 'lmb']
    print(
        '\nWithin-parity k-NN (5 digits per parity group, chance 0.20), '
        'even/odd agent:'
    )
    print(
        wp[wp.agent == 'parity']
        .pivot_table(index=idx, columns='k', values='acc')
        .round(4)
        .to_string()
    )
    print('Same probe, 10-class agent (reference):')
    print(
        wp[wp.agent == 'fine']
        .pivot_table(index=idx, columns='k', values='acc')
        .round(4)
        .to_string()
    )
    print(
        '\nFew-label 10-class k-NN, even/odd agent (rows: n labelled per class):'
    )
    print(
        fl[fl.agent == 'parity']
        .pivot_table(index=['n', 'k'], columns=idx, values='acc')
        .round(4)
        .to_string()
    )
    print('Same probe, 10-class agent (reference):')
    print(
        fl[fl.agent == 'fine']
        .pivot_table(index=['n', 'k'], columns=idx, values='acc')
        .round(4)
        .to_string()
    )
    plot(wp, fl, out, min(args.few_k))


if __name__ == '__main__':
    main()
