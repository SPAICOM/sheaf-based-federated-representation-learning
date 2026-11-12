"""Remote three-seed lambda report: all four receivers and private agents."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Repository imports require the project root on sys.path.
from src.utils.plotting import use_style  # noqa: E402

matplotlib.use('Agg')
use_style()
# Compact panels and portable rendering without an external LaTeX install.
plt.rcParams.update(
    {
        'text.usetex': False,
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.markersize': 6,
        'lines.linewidth': 1.5,
    }
)

api = wandb.Api()
project = 'mnist_4agents_shift'
prefix = 'mnist4_di_clean_bs256'
selected = [
    r
    for r in api.runs(f'{api.default_entity}/{project}')
    if r.group.startswith(prefix + '_lambda4096_')
    or r.group
    in (
        prefix + '_check_non_cooperative_p4096',
        prefix + '_check_di_sheaf_frl_p4096',
    )
]
assert len(selected) == 21 and all(r.state == 'finished' for r in selected)
records = []
for r in selected:
    r = api.run(f'{api.default_entity}/{project}/{r.id}')
    method = 'Non-coop' if 'non_cooperative' in r.group else 'DiSheafFRL'
    weight = r.config['orchestrator']['max_lmb']
    for kind in ('private', 'comm'):
        for agent in ('avg', 0, 1, 2, 3):
            key = (
                f'test/avg_{kind}_task_perf'
                if agent == 'avg'
                else f'test/{kind}_task_perf_agent_{agent}'
            )
            records.append(
                {
                    'run': r.id,
                    'seed': r.config['seed'],
                    'method': method,
                    'weight': weight,
                    'metric': f'{kind} {agent}',
                    'value': 100 * r.summary[key],
                }
            )
f = pd.DataFrame(records)
assert all(
    g.seed.nunique() == 3 and len(g) == 3
    for _, g in f.groupby(['method', 'weight', 'metric'])
)
s = (
    f.groupby(['method', 'weight', 'metric'])
    .value.agg(['mean', 'std', 'count'])
    .reset_index()
)
out = ROOT / 'results/mnist_4agents_shift/lambda'
out.mkdir(parents=True, exist_ok=True)
s.to_csv(out / 'summary.csv', index=False)
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
for ax, metric in zip(
    axes.flat,
    [f'{k} {a}' for k in ('private', 'comm') for a in ('avg', 0, 1, 2, 3)],
):
    d = s[(s.metric == metric) & (s.method == 'DiSheafFRL')].sort_values(
        'weight'
    )
    b = s[(s.metric == metric) & (s.method == 'Non-coop')].iloc[0]
    ax.errorbar(
        d.weight, d['mean'], yerr=d['std'], marker='o', label='DiSheafFRL'
    )
    ax.axhline(b['mean'], label='Non-coop', color='orange')
    ax.axhspan(
        b['mean'] - b['std'], b['mean'] + b['std'], alpha=0.15, color='orange'
    )
    ax.set(xscale='log', xlabel='Lambda', ylabel='Accuracy (%)', title=metric)
    ax.grid(alpha=0.2)
axes.flat[0].legend()
fig.tight_layout()
fig.savefig(out / 'metrics.png')
fig.savefig(out / 'metrics.pdf')
(out / 'README.md').write_text(
    'Mean and sample SD across seeds 42/43/44. Communication agent i averages all three senders into receiver i; avg covers all twelve directed edges. Private panels evaluate each agent locally. Noncoop and lambda=.01 reuse the valid baseline runs.\n\nRun IDs: '
    + ', '.join(r.id for r in selected)
    + '\n'
)
