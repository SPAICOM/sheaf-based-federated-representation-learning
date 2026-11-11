"""Compare overall map-study accuracy from remote map metrics and the promoted baseline summaries."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Repository imports require the project root on sys.path.
from src.utils.plotting import use_style  # noqa: E402

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

ROOT = ROOT / 'results/mnist_4agents_shift'
api = wandb.Api()
OUT = ROOT / 'maps'
OUT.mkdir(parents=True, exist_ok=True)
frames = {}
for method in ('zca', 'ols'):
    project = f'mnist_4agents_mapstudy_{method}'
    groups = {f'{method}_p{p}_l.01' for p in (384, 768, 1536, 2048, 4096)}
    groups |= {
        f'{method}_p4096_l{w}' for w in ('.003', '.03', '.1', '.3', '1')
    }
    runs = [
        api.run(f'{api.default_entity}/{project}/{r.id}')
        for r in api.runs(f'{api.default_entity}/{project}')
        if r.group in groups
    ]
    assert len(runs) == 30 and all(r.state == 'finished' for r in runs)
    df = pd.DataFrame(
        [
            dict(
                run=r.id,
                seed=r.config['seed'],
                weight=r.config['orchestrator']['max_lmb'],
                pilots=r.config['dataset']['pilot_num_samples'],
                **{
                    'comm test': r.summary['test/avg_comm_task_perf'],
                    'private test': r.summary['test/avg_private_task_perf'],
                },
            )
            for r in runs
        ]
    )
    assert all(
        sorted(g.seed) == [42, 43, 44]
        for _, g in df.groupby(['pilots', 'weight'])
    )
    (OUT / f'{method}_provenance.md').write_text(
        f'Project: {project}. Three seeds per point; posthoc Procrustes evaluation.\n\n'
        + '\n'.join(
            f'- {r.group}: {r.id} (seed {r.config["seed"]})' for r in runs
        )
        + '\n'
    )
    frames[method] = df
labels = [
    'Non cooperative',
    'DiSheafFRL + CenterNorm',
    'DiSheafFRL + ZCA',
    'DiSheafFRL + CenterNorm + OLS',
]
for study, directory, x in [
    ('lambda', 'lambda', 'weight'),
    ('pilot', 'pilots', 'pilots'),
]:
    old = pd.read_csv(ROOT / directory / 'summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    table = {}
    for ax, kind in zip(axes, ['comm', 'private']):
        metric = kind + ' avg' if study == 'lambda' else kind.capitalize()
        series = []
        for method in ['Non-coop', 'DiSheafFRL']:
            d = old[(old.method == method) & (old.metric == metric)].copy()
            if study == 'pilot':
                d = d.rename(columns={'sd': 'std'})
            d = d.set_index(x)[['mean', 'std']].sort_index()
            series.append(d)
        for method in ['zca', 'ols']:
            d = frames[method]
            d = (
                d[d.pilots == 4096]
                if study == 'lambda'
                else d[d.weight == 0.01]
            )
            series.append(
                d.groupby(x)[kind + ' test'].agg(['mean', 'std']) * 100
            )
        values = sorted(set().union(*(set(d.index) for d in series[1:])))
        if study == 'lambda':
            series[0] = pd.DataFrame(
                [series[0].iloc[0].to_dict()] * len(values), index=values
            )
        for label, color, d in zip(
            labels,
            ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple'],
            series,
        ):
            ax.errorbar(
                d.index,
                d['mean'],
                yerr=d['std'],
                marker='o',
                color=color,
                label=label,
            )
            if kind == 'comm':
                table[label] = d.apply(
                    lambda r: f'{r["mean"]:.2f} ± {r["std"]:.2f}', axis=1
                )
        ax.set(
            xlabel='Lambda' if study == 'lambda' else 'Pilot count',
            ylabel='Accuracy (%)',
            title=kind.capitalize() + ' test (overall)',
        )
        if study == 'lambda':
            ax.set_xscale('log')
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{study}_study.{ext}', dpi=180)
    plt.close(fig)
    result = pd.DataFrame(table).sort_index().fillna('—')
    result.index.name = 'Lambda' if study == 'lambda' else 'Pilots'
    result.to_csv(OUT / f'{study}_summary.csv')
    headers = [result.index.name, *result.columns]
    markdown_rows = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    markdown_rows.extend(
        '| ' + ' | '.join([str(index), *map(str, values)]) + ' |'
        for index, values in result.iterrows()
    )
    (OUT / f'{study}_summary.md').write_text(
        'Communication test accuracy (%), mean ± sample SD across runs. Lambda study: 4096 pilots; pilot study: lambda=0.01. Non-cooperative baseline is repeated across lambda.\n\n'
        + '\n'.join(markdown_rows)
        + '\n'
    )
print(OUT)
