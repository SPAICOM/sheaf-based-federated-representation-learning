"""Accuracy vs bottleneck dimension, two agents (latent d / d/2), every method.

Reads the finished runs of ``just bottleneck-pipeline <dataset>`` (wandb project
<dataset>_2agents_bottleneck, one group per method) from the wandb remote and
plots test accuracies against the latent dimension d of agent 0 (agent 1 has
d/2), mean and standard deviation over seeds:

  bottleneck.{pdf,png}           communication, private, comm on receiver-only classes
  bottleneck_rx_scope.{pdf,png}  communication on every class the receiver trained on,
                                 and on the classes only the receiver trained on

Class scopes are the ones of src/utils/class_scope.py (micro-averaged over the
two directed edges). "Seen by Rx" pools the scopes both_seen and
tx_unseen_rx_seen with their logged sample counts. ComFed communicates through
its own projections and logs no scopes in training: its scoped accuracies come
from the re-scoring cache results/<project>/rescore/<run_id>.json, which
``just plot-bottleneck`` fills first (``just rescore-hetero <project> --groups comfed``).

Writes everything plus the table bottleneck.md to results/<project>/.

    uv run python scripts/plot_bottleneck.py femnist
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.plotting import use_style  # noqa: E402

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
TITLES = {
    'comm': 'Communication accuracy',
    'private': 'Private accuracy',
    'comm_rx_seen': 'Comm., classes seen by receiver',
    'comm_rx_only': 'Comm., classes seen by receiver only',
}
FIGURES = {
    'bottleneck': ('comm', 'private', 'comm_rx_only'),
    'bottleneck_rx_scope': ('comm_rx_seen', 'comm_rx_only'),
}
S = 'test/scope/'


def latent_dim(config: dict, agent: int) -> int:
    """Width of the last encoder stage of one agent (wandb stores int keys as str)."""
    agents = config['model']['agents']
    return int(
        agents.get(str(agent), agents.get(agent))['encoder_hidden_dims'][-1]
    )


def scoped(summary, cache: Path) -> dict[str, float]:
    """Comm accuracy on receiver-seen / receiver-only classes, NaN if unavailable."""
    nb, nr = (
        summary.get(S + 'n_comm_both_seen'),
        summary.get(S + 'n_comm_tx_unseen_rx_seen'),
    )
    if nb is not None and nr is not None:
        both, rx_only = (
            summary[S + 'comm_both_seen'],
            summary[S + 'comm_tx_unseen_rx_seen'],
        )
        return {
            'comm_rx_seen': (both * nb + rx_only * nr) / (nb + nr),
            'comm_rx_only': rx_only,
        }
    if cache.exists():  # ComFed: re-scored with its own projections
        res = json.loads(cache.read_text())
        return {k: res[k] for k in ('comm_rx_seen', 'comm_rx_only')}
    return {'comm_rx_seen': float('nan'), 'comm_rx_only': float('nan')}


def load_runs(project: str) -> pd.DataFrame:
    """One row per finished run: method, d, seed, accuracies in %."""
    api = wandb.Api()
    if project not in {p.name for p in api.projects(api.default_entity)}:
        raise SystemExit(
            f'{api.default_entity}/{project} does not exist yet: run `just bottleneck-pipeline` first'
        )
    rows = []
    path = f'{api.default_entity}/{project}'
    for listed in api.runs(path, filters={'state': 'finished'}):
        if listed.group not in METHODS:
            continue
        run = api.run(f'{path}/{listed.id}')  # listings carry no config
        acc = {
            'comm': run.summary.get('test/avg_comm_task_perf', float('nan')),
            'private': run.summary.get(
                'test/avg_private_task_perf', float('nan')
            ),
            **scoped(
                run.summary,
                ROOT / 'results' / project / 'rescore' / f'{run.id}.json',
            ),
        }
        if pd.isna(acc['comm_rx_only']):
            print(
                f'{run.group} {run.id}: no class-scope accuracies (not re-scored)'
            )
        rows.append(
            {
                'method': run.group,
                'd': latent_dim(run.config, agent=0),
                'seed': run.config['seed'],
                'created': run.created_at,
                **{k: 100 * v for k, v in acc.items()},
            }
        )
    if not rows:
        raise SystemExit(f'no finished runs with a method group in {project}')
    runs = pd.DataFrame(rows)
    # a re-run of the same cell replaces the older one
    return runs.sort_values('created').drop_duplicates(
        ['method', 'd', 'seed'], keep='last'
    )


def plot(stats: pd.DataFrame, ds: list[int], keys: tuple, path: Path) -> None:
    """One panel per metric against d, mean ± std over seeds."""
    fig, axes = plt.subplots(
        1, len(keys), figsize=(8 * len(keys), 6), sharex=True
    )
    for ax, key in zip(axes, keys, strict=True):
        for method, label in METHODS.items():
            if method not in stats.index.get_level_values('method'):
                continue
            s = stats.loc[method, key].dropna(subset=['mean'])
            ls = '-' if 'sheaf_frl' in method else '--'
            ax.plot(s.index, s['mean'], marker='o', ls=ls, label=label)
            ax.fill_between(
                s.index,
                s['mean'] - s['std'].fillna(0),
                s['mean'] + s['std'].fillna(0),
                alpha=0.2,
            )
        ax.set_xscale('log', base=2)
        ax.set_xticks(ds)
        ax.set_xticklabels([f'{d}/{d // 2}' for d in ds])
        ax.set_xlabel('Latent dimension (agent 0 / agent 1)')
        ax.set_title(TITLES[key])
        ax.set_ylabel(r'Accuracy (\%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))  # room for the shared legend below
    for ext in ('pdf', 'png'):
        fig.savefig(path.with_suffix(f'.{ext}'))
    plt.close(fig)


def main() -> None:
    """Plot the accuracies against d and write the table."""
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'cifar10'
    project = f'{dataset}_2agents_bottleneck'
    out = ROOT / 'results' / project
    runs = load_runs(project)
    stats = runs.groupby(['method', 'd'])[list(TITLES)].agg(
        ['mean', 'std', 'count']
    )
    ds = sorted(runs['d'].unique())

    matplotlib.use('Agg')
    use_style()
    plt.rcParams['lines.markersize'] = (
        9  # house size hides the overlapping curves
    )
    out.mkdir(parents=True, exist_ok=True)
    for name, keys in FIGURES.items():
        plot(stats, ds, keys, out / name)

    lines = [
        '| method | d | ' + ' | '.join(TITLES.values()) + ' | seeds |',
        '|' + '---|' * (len(TITLES) + 3),
    ]
    order = {m: i for i, m in enumerate(METHODS)}
    rows = sorted(stats.iterrows(), key=lambda kv: (order[kv[0][0]], kv[0][1]))
    for (method, d), row in rows:
        cells = [
            f'{row[(k, "mean")]:.1f} ± {row[(k, "std")]:.1f}' for k in TITLES
        ]
        lines.append(
            f'| {METHODS[method]} | {d}/{d // 2} | {" | ".join(cells)} | {row[("comm", "count")]} |'
        )
    (out / 'bottleneck.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'wrote {out}/{{{",".join(FIGURES)}}}.{{pdf,png}} and bottleneck.md')


if __name__ == '__main__':
    main()
