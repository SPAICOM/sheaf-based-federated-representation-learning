"""Sharper probes of digit identity in the even/odd agent's latent space.

Reads the test latents saved by scripts/mnist_parity_knn.py (<out>/latents/*.npz)
and runs two probes on the even/odd agent (with the 10-class agent as reference).
No training happens here.

1. Within-parity k-NN: tell the 5 even digits apart among even test images, and
   the 5 odd digits among odd ones (chance 0.20). Parity itself is what the agent
   was trained on, so this isolates the digit information it was never asked to
   keep. Same stratified 2-fold split of the test set as the 10-class probe; the
   score pools both parity groups and both folds.
2. Few-label k-NN: the 10-class probe with only n labelled references per class,
   drawn from one half of the test set and scored on the other half (then
   swapped), averaged over --repeats random draws. k > n is skipped.

Cosine distance throughout (CenterNorm latents lie on the unit sphere).

Usage:
    uv run python scripts/mnist_parity_probes.py
    uv run python scripts/mnist_parity_probes.py --k 1 5 10 20 --n 1 5 10 50 100
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from mnist_parity_knn import COLORS, LABELS, ROOT, _halves

NAME = re.compile(r'(?P<method>noncoop|disheaf)(_lmb(?P<lmb>[^_]+))?_s(?P<seed>\d+)_e(?P<e>\d+)_p(?P<p>\d+)(_d(?P<d>\d+x\d+))?')
AGENTS = {'parity': 'z_parity', 'fine': 'z_fine'}


def _knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k, metric='cosine', n_jobs=-1)


def within_parity(Z: np.ndarray, y: np.ndarray, ks: list[int]) -> dict[int, float]:
    hits = dict.fromkeys(ks, 0)
    total = 0
    for p in (0, 1):
        m = y % 2 == p
        Zp, yp = Z[m], y[m]
        a, b = _halves(yp)
        for f, s in ((a, b), (b, a)):
            total += len(s)
            for k in ks:
                hits[k] += int((_knn(k).fit(Zp[f], yp[f]).predict(Zp[s]) == yp[s]).sum())
    return {k: hits[k] / total for k in ks}


def few_label(Z: np.ndarray, y: np.ndarray, ns: list[int], ks: list[int],
              repeats: int) -> dict[tuple[int, int], tuple[float, float]]:
    a, b = _halves(y)
    rng = np.random.default_rng(0)   # same reference draws for every run
    scores: dict[tuple[int, int], list[float]] = {}
    for f, s in ((a, b), (b, a)):
        by_class = {c: f[y[f] == c] for c in range(10)}
        for n in ns:
            for _ in range(repeats):
                ref = np.concatenate([rng.choice(by_class[c], n, replace=False) for c in range(10)])
                for k in ks:
                    if k > n:   # more neighbours than references per class: the vote degenerates
                        continue
                    acc = float((_knn(k).fit(Z[ref], y[ref]).predict(Z[s]) == y[s]).mean())
                    scores.setdefault((n, k), []).append(acc)
    return {key: (float(np.mean(v)), float(np.std(v))) for key, v in scores.items()}


def plot(wp: pd.DataFrame, fl: pd.DataFrame, out: Path, k_few: int, dims: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for (method, lmb), g in wp[wp.agent == 'parity'].groupby(['method', 'lmb']):
        s = g.groupby('k')['acc'].agg(['mean', 'std'])
        ax1.errorbar(s.index, s['mean'], yerr=s['std'].fillna(0.0), marker='o', ms=5, lw=2, capsize=3,
                     color=COLORS[method], label=LABELS[method] + (f' (λ={lmb:g})' if method == 'disheaf' else ''))
    ref = wp[wp.agent == 'fine'].groupby('k')['acc'].mean()
    ax1.plot(ref.index, ref.values, ls='--', lw=1.5, color='#6b6a63', label='Reference: 10-class agent')
    ax1.axhline(0.2, color='#9a998f', lw=1, ls=':', label='Chance (5 digits)')
    ax1.set_xscale('log')
    ax1.set_xticks(sorted(wp['k'].unique()))
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel('k (neighbours)')
    ax1.set_ylabel('Within-parity k-NN accuracy (test, 2-fold)')
    ax1.set_title('Telling digits apart within the same parity', fontsize=11)

    f = fl[fl.k == k_few]
    for (method, lmb), g in f[f.agent == 'parity'].groupby(['method', 'lmb']):
        s = g.groupby('n')['acc'].mean()
        sd = g.groupby('n')['std'].mean()
        ax2.errorbar(s.index, s.values, yerr=sd.values, marker='o', ms=5, lw=2, capsize=3,
                     color=COLORS[method], label=LABELS[method] + (f' (λ={lmb:g})' if method == 'disheaf' else ''))
    ref = f[f.agent == 'fine'].groupby('n')['acc'].mean()
    ax2.plot(ref.index, ref.values, ls='--', lw=1.5, color='#6b6a63', label='Reference: 10-class agent')
    ax2.axhline(0.1, color='#9a998f', lw=1, ls=':', label='Chance (10 digits)')
    ax2.set_xscale('log')
    ax2.set_xticks(sorted(f['n'].unique()))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_xlabel('Labelled references per class')
    ax2.set_ylabel(f'10-class {k_few}-NN accuracy (test)')
    ax2.set_title('Few-label probe (error bars: spread over reference draws)', fontsize=11)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'probes_d{dims}.{ext}', dpi=200)
    plt.close(fig)
    print(f'plot -> {out / f"probes_d{dims}.png"}')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=ROOT / 'results/mnist_2agents_parity')
    p.add_argument('--k', nargs='+', type=int, default=[1, 3, 5, 10, 20, 50, 100])
    p.add_argument('--n', nargs='+', type=int, default=[1, 5, 10, 50, 100])
    p.add_argument('--few-k', nargs='+', type=int, default=[1, 5, 10], help='k values for the few-label probe')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--dims', default=None,
                   help="latent sizes to probe, e.g. 40x8 or 320x64 (default: the most recent run's)")
    args = p.parse_args()

    files = [f for f in (args.out / 'latents').glob('*.npz') if NAME.fullmatch(f.stem)]
    if not files:
        sys.exit(f'no latents in {args.out / "latents"}: re-run scripts/mnist_parity_knn.py first')
    dims_of = {f: NAME.fullmatch(f.stem)['d'] or '320x64' for f in files}   # unsuffixed = old 320/64 runs
    dims = args.dims or dims_of[max(files, key=lambda f: f.stat().st_mtime)]
    files = sorted(f for f in files if dims_of[f] == dims)
    if not files:
        sys.exit(f'no latents with dims {dims}; available: {sorted(set(dims_of.values()))}')
    print(f'probing latent dims {dims} (10-class / even-odd), {len(files)} runs')

    wp_rows, fl_rows = [], []
    for path in files:
        m = NAME.fullmatch(path.stem)
        meta = {'method': m['method'], 'lmb': float(m['lmb'] or 0.0), 'seed': int(m['seed']),
                'epochs': int(m['e']), 'pilots': int(m['p'])}
        d = np.load(path)
        y = d['y']
        for agent, key in AGENTS.items():
            for k, acc in within_parity(d[key], y, args.k).items():
                wp_rows.append({**meta, 'agent': agent, 'k': k, 'acc': acc})
            for (n, k), (acc, sd) in few_label(d[key], y, args.n, args.few_k, args.repeats).items():
                fl_rows.append({**meta, 'agent': agent, 'n': n, 'k': k, 'acc': acc, 'std': sd})
        print(f'done {path.name}', flush=True)

    wp, fl = pd.DataFrame(wp_rows), pd.DataFrame(fl_rows)
    wp.to_csv(args.out / f'probe_within_parity_d{dims}.csv', index=False)
    fl.to_csv(args.out / f'probe_few_label_d{dims}.csv', index=False)

    pd.set_option('display.width', 200)
    idx = ['method', 'lmb']
    print('\nWithin-parity k-NN (5 digits per parity group, chance 0.20), even/odd agent:')
    print(wp[wp.agent == 'parity'].pivot_table(index=idx, columns='k', values='acc').round(4).to_string())
    print('Same probe, 10-class agent (reference):')
    print(wp[wp.agent == 'fine'].pivot_table(index=idx, columns='k', values='acc').round(4).to_string())
    print('\nFew-label 10-class k-NN, even/odd agent (rows: n labelled per class):')
    print(fl[fl.agent == 'parity'].pivot_table(index=['n', 'k'], columns=idx, values='acc').round(4).to_string())
    print('Same probe, 10-class agent (reference):')
    print(fl[fl.agent == 'fine'].pivot_table(index=['n', 'k'], columns=idx, values='acc').round(4).to_string())
    plot(wp, fl, args.out, min(args.few_k), dims)


if __name__ == '__main__':
    main()
