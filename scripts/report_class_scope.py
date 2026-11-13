"""Seen/unseen-class breakdown of one experiment, at network, agent and edge level.

Reads the ``test/scope/*`` metrics (src/utils/class_scope.py) of the finished
runs of one wandb project, one group per method, and reports mean and standard
deviation over seeds of:

* private accuracy on the classes the agent trained on (seen) and the rest;
* communication accuracy in every class scope: all, both_seen,
  tx_unseen_rx_seen (only the receiver knows the class), tx_seen_rx_unseen,
  both_unseen.

Levels: network (pooled over all agents / directed edges), agent (as
receiver ``rx`` and as sender ``tx``), edge. With ``--x <config key>`` (for
example ``dataset.pilot_num_samples``, or ``latent_d`` for agent 0's latent
width) the network level is also plotted against that axis.

Writes results/<project>/class_scope/{network,agents,edges}.md, network.tex
and the figures network.{pdf,png} (agents.{pdf,png} per agent).

    uv run python scripts/report_class_scope.py cifar10_4agents_full
    uv run python scripts/report_class_scope.py mnist_4agents_shift --x dataset.pilot_num_samples --groups di_sheaf_frl,non_cooperative
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from sweep_report import config_value  # noqa: E402

from src.utils.plotting import use_style  # noqa: E402

# the house style's fonts are sized for single-panel figures
RC = {
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.markersize': 4,
    'lines.linewidth': 1.4,
}

# config name (= wandb group) -> label, in table order
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
# network-level columns: metric key suffix -> label
NETWORK = {
    'private': 'Private',
    'private_seen': 'Private seen',
    'private_unseen': 'Private unseen',
    'comm_all': 'Comm',
    'comm_both_seen': 'Comm both seen',
    'comm_tx_unseen_rx_seen': 'Comm tx unseen / rx seen',
    'comm_tx_seen_rx_unseen': 'Comm tx seen / rx unseen',
    'comm_both_unseen': 'Comm both unseen',
}
# agent-level columns
AGENT = {
    'private_seen': 'Private seen',
    'private_unseen': 'Private unseen',
    'rx_all': 'Comm as receiver',
    'rx_both_seen': 'rx: both seen',
    'rx_tx_unseen_rx_seen': 'rx: tx unseen / rx seen',
    'tx_all': 'Comm as sender',
    'tx_both_seen': 'tx: both seen',
    'tx_tx_unseen_rx_seen': 'tx: tx unseen / rx seen',
}
EDGE = {
    'all': 'Comm',
    'both_seen': 'both seen',
    'tx_unseen_rx_seen': 'tx unseen / rx seen',
}
PLOTTED = [
    'private_seen',
    'comm_all',
    'comm_both_seen',
    'comm_tx_unseen_rx_seen',
]


def latent_d(config: dict) -> int:
    """Width of agent 0's last encoder stage (wandb stores int keys as str)."""
    agents = config['model']['agents']
    return int(agents.get('0', agents.get(0))['encoder_hidden_dims'][-1])


def x_value(config: dict, key: str):
    """Value of the swept axis in one run's wandb config."""
    return latent_d(config) if key == 'latent_d' else config_value(config, key)


def load_runs(
    project: str, groups: list[str], x_key: str | None
) -> pd.DataFrame:
    """Long table: one row per (run, metric) with method, x, seed, value in %."""
    api = wandb.Api()
    entity = api.default_entity
    if project not in {p.name for p in api.projects(entity)}:
        raise SystemExit(f'{entity}/{project} does not exist')
    rows = []
    for run in api.runs(f'{entity}/{project}', filters={'state': 'finished'}):
        if run.group not in groups:
            continue
        summary = dict(run.summary)
        scope = {
            k.removeprefix('test/scope/'): v
            for k, v in summary.items()
            if k.startswith('test/scope/')
        }
        if not scope:
            print(
                f'{run.id} ({run.group}): no test/scope metrics (run predates the class-scope logging), skipped'
            )
            continue
        scope['private'] = summary.get('test/avg_private_task_perf')
        x = x_value(run.config, x_key) if x_key else None
        for key, value in scope.items():
            if key.startswith('n_') or value is None:
                continue
            rows.append(
                {
                    'run': run.id,
                    'created': run.created_at,
                    'method': run.group,
                    'x': x,
                    'seed': run.config['seed'],
                    'key': key,
                    'value': 100 * float(value),
                }
            )
    if not rows:
        raise SystemExit(
            f'no finished runs with test/scope metrics in {project} (groups {groups})'
        )
    runs = pd.DataFrame(rows)
    # a re-run of the same cell replaces the older one
    latest = runs.sort_values('created').drop_duplicates(
        ['method', 'x', 'seed'], keep='last'
    )['run']
    return runs[runs['run'].isin(set(latest))]


def cell(values: pd.Series) -> str:
    """'mean ± std' over seeds, '(n=k)' when the seed count is not 3."""
    values = values.dropna()
    if values.empty:
        return '–'
    text = (
        f'{values.mean():.1f} ± {values.std():.1f}'
        if len(values) > 1
        else f'{values.mean():.1f}'
    )
    return text if len(values) == 3 else f'{text} (n={len(values)})'


def to_markdown(table: pd.DataFrame, index_names: list[str]) -> str:
    """Pipe table with the index levels as leading columns."""
    header = index_names + [str(c) for c in table.columns]
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for idx, row in zip(table.index, table.values, strict=True):
        idx = list(idx) if isinstance(idx, tuple) else [idx]
        lines.append(
            '| ' + ' | '.join([str(i) for i in idx] + list(row)) + ' |'
        )
    return '\n'.join(lines) + '\n'


def label(method: str, x) -> str:
    """Row label: the method's display name, with the swept value if any."""
    return (
        METHODS.get(method, method)
        if x is None
        else f'{METHODS.get(method, method)} @ {x}'
    )


def network_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Build the method x network-level metric table."""
    table = {}
    for (method, x), group in runs.groupby(
        ['method', 'x'], dropna=False, sort=False
    ):
        table[label(method, None if pd.isna(x) else x)] = {
            col: cell(group.loc[group['key'] == key, 'value'])
            for key, col in NETWORK.items()
        }
    return pd.DataFrame(table).T


def agent_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Build the (method, agent) x metric table; rx = as receiver, tx = as sender."""
    pattern = re.compile(
        r'^(private_(?:seen|unseen))_agent_(\d+)$|^comm_(\w+?)_(rx|tx)_(\d+)$'
    )
    parsed = []
    for key in runs['key'].unique():
        m = pattern.match(key)
        if not m:
            continue
        if m.group(1):
            parsed.append((key, m.group(1), int(m.group(2))))
        else:
            parsed.append((key, f'{m.group(4)}_{m.group(3)}', int(m.group(5))))
    rows = {}
    for (method, x), group in runs.groupby(
        ['method', 'x'], dropna=False, sort=False
    ):
        for key, col, agent in parsed:
            if col in AGENT:
                values = group.loc[group['key'] == key, 'value']
                rows.setdefault(
                    (label(method, None if pd.isna(x) else x), agent), {}
                )[AGENT[col]] = cell(values)
    table = pd.DataFrame(rows).T.sort_index(level=1, sort_remaining=False)
    return table.reindex(
        columns=[c for c in AGENT.values() if c in table.columns]
    ).fillna('–')


def edge_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Build the (method, directed edge) x comm-scope table."""
    pattern = re.compile(r'^comm_(\w+?)_edge_(\d+)_(\d+)$')
    rows = {}
    for (method, x), group in runs.groupby(
        ['method', 'x'], dropna=False, sort=False
    ):
        for key in group['key'].unique():
            m = pattern.match(key)
            if m and m.group(1) in EDGE:
                edge = f'{m.group(2)} → {m.group(3)}'
                rows.setdefault(
                    (label(method, None if pd.isna(x) else x), edge), {}
                )[EDGE[m.group(1)]] = cell(
                    group.loc[group['key'] == key, 'value']
                )
    return (
        pd.DataFrame(rows).T.reindex(columns=list(EDGE.values())).fillna('–')
    )


def plot_network(runs: pd.DataFrame, x_key: str | None, out: Path) -> None:
    """Network-level accuracies: bars per method, or lines against --x."""
    sub = runs[runs['key'].isin(PLOTTED)].copy()
    if not x_key:
        sub['x'] = 0
    stats = sub.groupby(['method', 'key', 'x'])['value'].agg(['mean', 'std'])
    present = set(stats.index.droplevel('x'))
    methods = [m for m in METHODS if m in runs['method'].unique()] + sorted(
        set(runs['method']) - set(METHODS)
    )
    fig, axes = plt.subplots(
        1, len(PLOTTED), figsize=(3.4 * len(PLOTTED), 3.4)
    )
    for ax, key in zip(axes, PLOTTED, strict=True):
        for i, method in enumerate(methods):
            if (method, key) not in present:
                continue
            s = stats.loc[(method, key)].sort_index()
            if x_key:
                ax.errorbar(
                    s.index,
                    s['mean'],
                    yerr=s['std'].fillna(0),
                    marker='o',
                    capsize=2,
                    label=METHODS.get(method, method),
                    color=f'C{i}',
                )
            else:
                ax.bar(
                    i,
                    s['mean'].iloc[0],
                    yerr=np.nan_to_num(s['std'].iloc[0]),
                    capsize=3,
                    label=METHODS.get(method, method),
                    color=f'C{i}',
                )
        if x_key:
            ax.set_xlabel(x_key.rsplit('.', 1)[-1])
            if x_key in ('latent_d', 'dataset.pilot_num_samples'):
                ax.set_xscale('log', base=2)
        else:
            ax.set_xticks([])
        ax.set_title(NETWORK[key])
    axes[0].set_ylabel(r'Accuracy (\%)')
    axes[0].legend()
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'network.{ext}')
    plt.close(fig)


def plot_agents(runs: pd.DataFrame, out: Path) -> None:
    """Per agent (as receiver): both-seen and tx-unseen/rx-seen comm accuracy, bars per method."""
    keys = {
        'both_seen': 'both seen',
        'tx_unseen_rx_seen': 'tx unseen / rx seen',
    }
    agents = sorted(
        {
            int(k.rsplit('_', 1)[1])
            for k in runs['key']
            if re.match(r'^comm_all_rx_\d+$', k)
        }
    )
    methods = [m for m in METHODS if m in runs['method'].unique()] + sorted(
        set(runs['method']) - set(METHODS)
    )
    if not agents:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 3.4))
    width = 0.8 / len(methods)
    for ax, (scope, title) in zip(
        np.atleast_1d(axes), keys.items(), strict=True
    ):
        for i, method in enumerate(methods):
            sub = runs[runs['method'] == method]
            means, stds = [], []
            for a in agents:
                v = sub.loc[sub['key'] == f'comm_{scope}_rx_{a}', 'value']
                means.append(v.mean() if len(v) else np.nan)
                stds.append(v.std() if len(v) > 1 else 0.0)
            ax.bar(
                np.arange(len(agents)) + i * width,
                means,
                width,
                yerr=stds,
                capsize=2,
                label=METHODS.get(method, method),
                color=f'C{i}',
            )
        ax.set_xticks(np.arange(len(agents)) + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f'agent {a}' for a in agents])
        ax.set_title(f'Comm as receiver, {title}')
    np.atleast_1d(axes)[0].set_ylabel(r'Accuracy (\%)')
    np.atleast_1d(axes)[0].legend()
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'agents.{ext}')
    plt.close(fig)


def main() -> None:
    """Load the runs and write every table and figure."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        'project', help='wandb project (= experiment config name)'
    )
    parser.add_argument(
        '--groups',
        default=','.join(METHODS),
        help='comma-separated wandb groups (default: the 8 methods)',
    )
    parser.add_argument(
        '--x',
        default=None,
        help="config key of a swept axis, e.g. dataset.pilot_num_samples or 'latent_d'",
    )
    args = parser.parse_args()

    runs = load_runs(args.project, args.groups.split(','), args.x)
    out = ROOT / 'results' / args.project / 'class_scope'
    out.mkdir(parents=True, exist_ok=True)

    network = network_table(runs)
    agents = agent_table(runs)
    edges = edge_table(runs)
    (out / 'network.md').write_text(to_markdown(network, ['method']))
    (out / 'network.tex').write_text(network.to_latex().replace('±', r'$\pm$'))
    (out / 'agents.md').write_text(to_markdown(agents, ['method', 'agent']))
    (out / 'edges.md').write_text(to_markdown(edges, ['method', 'edge']))

    matplotlib.use('Agg')
    use_style()
    plt.rcParams.update(RC)
    plot_network(runs, args.x, out)
    plot_agents(runs, out)
    print(network.to_string())
    print(
        f'wrote {out}/{{network,agents,edges}}.md, network.tex, network.{{pdf,png}}, agents.{{pdf,png}}'
    )


if __name__ == '__main__':
    main()
