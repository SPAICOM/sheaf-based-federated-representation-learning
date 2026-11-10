"""7v7 two ResNets, CenterNorm, no whitening, shared unlabeled pilots: non-coop vs DiSheafFRL
(λ=3e-2) as the number of pilots grows (tmp/2026-11-03_7v7_cnorm_pilotsweep.sh, 328/8128 from
tmp/2026-11-03_7v7_pilots_shift.sh, 32-256 from tmp/2026-11-05_7v7_smallpilots.sh), fetched from
the remote wandb project. The 2048-pilot points are the sharedpilots_cnorm_* runs.

    uv run python scripts/plot_pilot_count_study.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_6v6_cnorm_study import METRICS

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 'engrima18/cifar10_2agents_grayblur_7v7'
OUT = ROOT / 'results/cifar10_2agents_grayblur_7v7/pilot_count_study'
TICKS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8128]   # labelled subset of the pilot counts
METHODS = {  # label -> (groups, extra run ids at 2048 pilots, colour)
    'non-coop': (['pilotsweep_cnorm_noncoop'], ['qt7wgve3'], 'tab:gray'),
    'DiSheafFRL λ=3e-2': (['pilotsweep_cnorm_disheaf'], ['utjhk1eb'], 'tab:green'),
}


def row(r) -> dict:
    s, c = r.summary, r.config
    n = c.get('dataset', {}).get('pilot_num_samples') if isinstance(c.get('dataset'), dict) else None
    n = n or c.get('dataset/pilot_num_samples') or c.get('pilot_num_samples')
    if n is None:  # later runs log the override only on the command line
        args = dict(a.split('=', 1) for a in (r.metadata or {}).get('args', []) if '=' in a)
        n = args['dataset.pilot_num_samples']
    out = {'run': r.id, 'pilots': int(n)}
    for col, key in METRICS.items():
        if isinstance(key, tuple):
            v = [s.get(k) for k in key]
            out[col] = f'{v[0]:.0f} / {v[1]:.0f}' if None not in v else None
        else:
            out[col] = s.get(key)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    data = {}
    for label, (groups, extra, _) in METHODS.items():
        runs = list(api.runs(PROJECT, filters={'group': {'$in': groups}, 'state': 'finished'}))
        runs += [api.run(f'{PROJECT}/{rid}') for rid in extra]
        data[label] = pd.DataFrame([row(r) for r in runs]).sort_values('pilots')

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    panels = [('comm test', 'post-comm test accuracy (avg of both directions)'),
              ('private test', 'private test accuracy (avg)'),
              ('misalign test', 'test misalignment (energy-normalised, lower = better)')]
    for ax, (col, title) in zip(axes, panels):
        for label, d in data.items():
            ax.plot(d.pilots, d[col], 'o-', color=METHODS[label][2], label=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks(TICKS)
        ax.set_xticklabels(TICKS)
        ax.set_xlabel('number of shared pilots')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('accuracy (10 test classes, max 0.7)')
    axes[1].set_ylabel('accuracy (10 test classes, max 0.7)')
    axes[0].legend(fontsize=9)
    fig.suptitle('CIFAR-10 7v7 grayscale/blur, two ResNets (d=512 / 256), CenterNorm, no whitening, '
                 'shared unlabeled pilots, single seed', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'metrics_vs_pilots.png', dpi=140)
    plt.close(fig)

    # standalone curve: post-comm test accuracy vs number of pilots
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for label, d in data.items():
        ax.plot(d.pilots, d['comm test'], 'o-', color=METHODS[label][2], label=label, lw=2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(TICKS)
    ax.set_xticklabels(TICKS)
    ax.set_xlabel('number of shared pilots')
    ax.set_ylabel('post-comm test accuracy (avg of both directions)\n10 test classes, max 0.7')
    ax.set_title('CIFAR-10 7v7, two ResNets, CenterNorm, shared unlabeled pilots, single seed', fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'comm_vs_pilots.png', dpi=150)
    plt.close(fig)

    t = pd.concat([d.assign(method=label) for label, d in data.items()])
    t = t[['method', 'pilots', *[c for c in t.columns if c not in ('method', 'pilots')]]]
    t.to_csv(OUT / 'pilot_count_table.csv', index=False)
    fmt = t.copy()
    for col in fmt.columns:
        if fmt[col].dtype.kind == 'f':
            fmt[col] = fmt[col].map(lambda v: f'{v:.3f}')
    md = '\n'.join(['| ' + ' | '.join(fmt.columns) + ' |', '|' + '---|' * len(fmt.columns)]
                   + ['| ' + ' | '.join(map(str, r)) + ' |' for r in fmt.itertuples(index=False)])
    (OUT / 'pilot_count_table.md').write_text(md + '\n')
    print(md)
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
