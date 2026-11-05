"""Quick report of a finished sweep from the remote wandb project: one table + curves.

    just report cifar10_2agents_grayblur_7v7 pilotsweep_cnorm_noncoop pilotsweep_cnorm_disheaf \
        --x dataset.pilot_num_samples --logx \
        --extra pilotsweep_cnorm_noncoop=qt7wgve3 --extra pilotsweep_cnorm_disheaf=utjhk1eb

One series per wandb group (``--extra GROUP=RUN`` adds runs from other groups to a series).
``--x`` is a config key (dotted; nested, flat or '/'-joined wandb configs all work);
runs without it get ``--x-default`` (e.g. 0 for non-coop in a λ sweep).

Output goes to ``tmp/YYYY-MM-DD_<project>_<slug>/`` (table.md, table.csv, metrics.png):
a report is scratch until a figure is promoted to a ``scripts/plot_*.py`` for the paper.
Runs that are not ``finished`` are listed, never tabulated.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = {  # column -> summary key(s); pairs are shown as "agent0 / agent1"
    'comm test': 'test/avg_comm_task_perf',
    'comm 0←1': 'test/comm_task_perf_agent_0',
    'comm 1←0': 'test/comm_task_perf_agent_1',
    'private test': 'test/avg_private_task_perf',
    'eff. rank test (a0 / a1)': (
        'test/effective_rank_agent_0',
        'test/effective_rank_agent_1',
    ),
    'eff. rank pilots (a0 / a1)': (
        'pilots/effective_rank_agent_0',
        'pilots/effective_rank_agent_1',
    ),
    'misalign test': 'test/misalignment_energy',
    'misalign pilots': 'pilots/misalignment_energy',
    'MRR test': 'test/mean_reciprocal_rank',
    'MRR pilots': 'pilots/mean_reciprocal_rank',
}
PANELS = ['comm test', 'private test', 'misalign test']


def config_value(config: dict, key: str):
    """Look a dotted key up in a wandb config, whatever way Hydra's dict was flattened."""
    for k in (key, key.replace('.', '/')):
        if k in config:
            return config[k]
    node = config
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            break
        node = node[part]
    else:
        return node
    return config.get(key.rsplit('.', 1)[-1])


def row(run, series: str, x_key: str | None, x_default) -> dict:
    s = run.summary
    out = {'series': series, 'run': run.id}
    if x_key:
        x = config_value(run.config, x_key)
        out[x_key] = x_default if x is None else x
    for col, key in METRICS.items():
        if isinstance(key, tuple):
            v = [s.get(k) for k in key]
            out[col] = f'{v[0]:.0f} / {v[1]:.0f}' if None not in v else None
        else:
            out[col] = s.get(key)
    return out


def to_markdown(t: pd.DataFrame, raw: tuple = ()) -> str:
    """Float columns to 3 decimals, except the ``raw`` ones (e.g. the x axis)."""
    fmt = t.copy()
    for col in fmt.columns:
        if fmt[col].dtype.kind == 'f' and col not in raw:
            fmt[col] = fmt[col].map(lambda v: '' if pd.isna(v) else f'{v:.3f}')
    head = [
        '| ' + ' | '.join(map(str, fmt.columns)) + ' |',
        '|' + '---|' * len(fmt.columns),
    ]
    return '\n'.join(
        head
        + [
            '| ' + ' | '.join(map(str, r)) + ' |'
            for r in fmt.itertuples(index=False)
        ]
    )


def plot(
    t: pd.DataFrame, x_key: str, logx: bool, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6 * len(PANELS), 4.5))
    for ax, col in zip(axes, PANELS):
        for series, d in t.groupby('series', sort=False):
            d = d.dropna(subset=[col])
            ax.plot(d[x_key], d[col], 'o-', label=series)
        if logx and (t[x_key] <= 0).any():  # e.g. λ = 0 for non-coop
            ax.set_xscale('symlog', linthresh=t[x_key][t[x_key] > 0].min())
        elif logx:
            ax.set_xscale('log')
        ax.set_xlabel(x_key)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('project')
    ap.add_argument('groups', nargs='+')
    ap.add_argument(
        '--x',
        help='config key for the x axis, e.g. max_lmb or dataset.pilot_num_samples',
    )
    ap.add_argument(
        '--x-default',
        type=float,
        default=None,
        help='x for runs without the key',
    )
    ap.add_argument('--logx', action='store_true')
    ap.add_argument(
        '--extra',
        action='append',
        default=[],
        metavar='GROUP=RUN',
        help='add a run from elsewhere to a series (repeatable)',
    )
    ap.add_argument('--slug', help='output dir suffix (default: first group)')
    a = ap.parse_args()

    import wandb

    api = wandb.Api()
    path = f'{api.default_entity}/{a.project}'
    rows, skipped = [], []
    for g in a.groups:
        for r in api.runs(path, filters={'group': g}):
            if r.state == 'finished':
                rows.append(row(r, g, a.x, a.x_default))
            else:
                skipped.append(f'{r.id} ({g}): {r.state}')
    for spec in a.extra:
        g, rid = spec.split('=', 1)
        r = api.run(f'{path}/{rid}')
        if r.state == 'finished':
            rows.append(row(r, g, a.x, a.x_default))
        else:
            skipped.append(f'{rid} (extra): {r.state}')
    if not rows:
        raise SystemExit(f'no finished runs in {path} for groups {a.groups}')

    t = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(a.groups)}  # series in CLI order
    if a.x:
        t[a.x] = pd.to_numeric(t[a.x], errors='coerce')
        t = t[['series', 'run', a.x, *METRICS]].sort_values(
            ['series', a.x],
            key=lambda c: c.map(order) if c.name == 'series' else c,
        )
    out = ROOT / 'tmp' / f'{date.today()}_{a.project}_{a.slug or a.groups[0]}'
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / 'table.csv', index=False)
    md = to_markdown(t, raw=(a.x,))
    (out / 'table.md').write_text(md + '\n')
    if a.x:
        plot(
            t,
            a.x,
            a.logx,
            f'{a.project}: {", ".join(a.groups)} (single runs, wandb remote)',
            out / 'metrics.png',
        )
    print(md)
    if skipped:
        print('\nnot finished (excluded):\n  ' + '\n  '.join(skipped))
    print(f'\n-> {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
