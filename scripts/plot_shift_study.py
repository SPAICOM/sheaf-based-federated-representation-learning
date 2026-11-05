"""7v7 two ResNets, CenterNorm, no whitening, 4096 shared unlabeled pilots: non-coop vs
DiSheafFRL (λ=3e-2) vs label-shift strength, for grayscale/blur inputs and for clean inputs
(tmp/2026-11-03_7v7_pilots_shift.sh), fetched from the remote wandb projects. The grayblur
shift-1.0 points are the 4096-pilot runs of the pilot-count sweep.

    uv run python scripts/plot_shift_study.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.plotting import use_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SETTINGS = {  # label -> (wandb project, extra shift-1.0 run ids {method: id})
    'grayscale / blur': ('cifar10_2agents_grayblur_7v7', {'non-coop': 'iqx3e07o', r'DiSheafFRL $\lambda=3\times10^{-2}$': 'k8u9mndv'}),
    'clean inputs': ('cifar10_2agents_7v7', {}),
}
METHODS = {'non-coop': ('shiftsweep_cnorm_noncoop', 'tab:gray'),
           r'DiSheafFRL $\lambda=3\times10^{-2}$': ('shiftsweep_cnorm_disheaf', 'tab:green')}


def row(r) -> dict:
    # shift/pilot overrides are not in the logged config, only in the command line
    args = dict(a.split('=', 1) for a in (r.metadata or {}).get('args', []) if '=' in a)
    out = {'run': r.id, 'shift': float(args.get('dataset.shift_strength', 1.0))}
    for col, key in METRICS.items():
        if isinstance(key, tuple):
            v = [r.summary.get(k) for k in key]
            out[col] = f'{v[0]:.0f} / {v[1]:.0f}' if None not in v else None
        else:
            out[col] = r.summary.get(key)
    return out


def main():
    use_style()
    plt.rcParams.update({'lines.markersize': 6, 'lines.linewidth': 1.8})
    api = wandb.Api()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, (setting, (proj, extra)) in zip(axes, SETTINGS.items()):
        out = ROOT / f'results/{proj}/shift_study'
        out.mkdir(parents=True, exist_ok=True)
        tables = []
        for label, (group, colour) in METHODS.items():
            runs = list(api.runs(f'engrima18/{proj}', filters={'group': group, 'state': 'finished'}))
            if label in extra:
                runs.append(api.run(f'engrima18/{proj}/{extra[label]}'))
            d = pd.DataFrame([row(r) for r in runs]).sort_values('shift')
            ax.plot(d['shift'], d['comm test'], 'o-', color=colour, label=f'{label}, post-comm')
            ax.plot(d['shift'], d['private test'], 's:', color=colour, alpha=0.6, label=f'{label}, private')
            tables.append(d.assign(method=label))
        ax.set_xlabel('label-shift strength')
        ax.set_ylabel('test accuracy (10 classes)')
        ax.set_title(f'7v7 {setting}, 4096 shared pilots', fontsize=18)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 1.0])
        t = pd.concat(tables)
        # run list for plot_pilot_tsne.py (setups 7v7_{grayblur,clean}_s<shift>)
        pd.DataFrame({'variant': 'cnorm_s' + t['shift'].map('{:g}'.format), 'run': t['run'],
                      'lmb': [0.0 if m == 'non-coop' else 3e-2 for m in t['method']],
                      'method': ['non-coop' if m == 'non-coop' else 'DiSheafFRL' for m in t['method']],
                      'comm': t['comm test'], 'private': t['private test']}).to_csv(out / 'runs.csv', index=False)
        t = t[['method', 'shift', *[c for c in t.columns if c not in ('method', 'shift')]]]
        t.to_csv(out / 'shift_table.csv', index=False)
        fmt = t.copy()
        for col in fmt.columns:
            if fmt[col].dtype.kind == 'f' and col != 'shift':
                fmt[col] = fmt[col].map(lambda v: f'{v:.3f}')
        md = '\n'.join(['| ' + ' | '.join(fmt.columns) + ' |', '|' + '---|' * len(fmt.columns)]
                       + ['| ' + ' | '.join(map(str, r)) + ' |' for r in fmt.itertuples(index=False)])
        (out / 'shift_table.md').write_text(md + '\n')
        print(f'## {setting}\n{md}\n-> {out}\n')
    axes[0].legend(fontsize=12, loc='lower left')
    fig.suptitle('CIFAR-10 7v7, two ResNets, CenterNorm, no whitening, single seed')
    fig.tight_layout()
    for proj, _ in SETTINGS.values():
        for ext in ('png', 'pdf'):
            fig.savefig(ROOT / f'results/{proj}/shift_study/comm_vs_shift.{ext}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
