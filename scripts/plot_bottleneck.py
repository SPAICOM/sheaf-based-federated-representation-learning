"""Accuracy vs bottleneck dimension, two CIFAR-10 agents, every method.

Reads the finished runs of ``just bottleneck`` (wandb project
cifar10_2agents_bottleneck, one group per method) from the wandb remote and
plots communication and private test accuracy against the latent dimension d
of agent 0 (agent 1 has d/2), mean and standard deviation over seeds.

Writes results/cifar10_2agents_bottleneck/bottleneck.{pdf,png} and the table
bottleneck.md next to them.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.plotting import use_style  # noqa: E402

PROJECT = 'cifar10_2agents_bottleneck'
OUT = ROOT / 'results' / PROJECT
# config name (= wandb group) -> label, in legend order
METHODS = {
    'non_cooperative': 'Non-coop',
    'sheaf_fmtl': 'Sheaf-FMTL',
    'comfed': 'ComFed',
    'fedproto': 'FedProto',
    'fedmuscle': 'FedMuscle',
    'fedre': 'FedRE',
    'sheaf_frl': 'SheafFRL',
    'di_sheaf_frl': 'DiSheafFRL',
}
METRICS = {'comm': 'Communication accuracy', 'private': 'Private accuracy'}


def latent_dim(config: dict, agent: int) -> int:
    """Width of the last encoder stage of one agent (wandb stores int keys as str)."""
    agents = config['model']['agents']
    return int(agents.get(str(agent), agents.get(agent))['encoder_hidden_dims'][-1])


def load_runs() -> pd.DataFrame:
    """One row per finished run: method, d, seed, accuracies in %."""
    api = wandb.Api()
    if PROJECT not in {p.name for p in api.projects(api.default_entity)}:
        raise SystemExit(f'{api.default_entity}/{PROJECT} does not exist yet: run `just bottleneck` first')
    rows = []
    for run in api.runs(f'{api.default_entity}/{PROJECT}', filters={'state': 'finished'}):
        if run.group not in METHODS:
            continue
        rows.append(
            {
                'method': run.group,
                'd': latent_dim(run.config, agent=0),
                'seed': run.config['seed'],
                'created': run.created_at,
                **{k: 100 * run.summary[f'test/avg_{k}_task_perf'] for k in METRICS},
            }
        )
    if not rows:
        raise SystemExit(f'no finished runs with a method group in {PROJECT}')
    runs = pd.DataFrame(rows)
    # a re-run of the same cell replaces the older one
    return runs.sort_values('created').drop_duplicates(['method', 'd', 'seed'], keep='last')


def main() -> None:
    """Plot both accuracies against d and write the table."""
    runs = load_runs()
    stats = runs.groupby(['method', 'd'])[list(METRICS)].agg(['mean', 'std', 'count'])

    matplotlib.use('Agg')
    use_style()
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10, 3.8), sharex=True)
    for ax, (key, title) in zip(axes, METRICS.items(), strict=True):
        for method, label in METHODS.items():
            if method not in stats.index.get_level_values('method'):
                continue
            s = stats.loc[method, key]
            ax.plot(s.index, s['mean'], marker='o', label=label)
            ax.fill_between(s.index, s['mean'] - s['std'].fillna(0), s['mean'] + s['std'].fillna(0), alpha=0.2)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sorted(runs['d'].unique()))
        ax.set_xticklabels([f'{d}/{d // 2}' for d in sorted(runs['d'].unique())])
        ax.set_xlabel('Latent dimension (agent 0 / agent 1)')
        ax.set_ylabel(f'{title} (%)')
    axes[0].legend(ncol=2)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'bottleneck.{ext}')
    lines = ['| method | d | ' + ' | '.join(METRICS.values()) + ' | seeds |', '|---|---|---|---|---|']
    for (method, d), row in stats.iterrows():
        cells = [f"{row[(k, 'mean')]:.1f} ± {row[(k, 'std')]:.1f}" for k in METRICS]
        lines.append(f"| {METHODS[method]} | {d}/{d // 2} | {' | '.join(cells)} | {row[('comm', 'count')]} |")
    (OUT / 'bottleneck.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {OUT}/bottleneck.{{pdf,png,md}}')


if __name__ == '__main__':
    main()
