"""Topology study: DiSheafFRL - Non-coop comm gap on star / ring / fully connected graphs.

Comm accuracy overall / sender-unseen+receiver-seen / receiver-seen, rescored offline from the
final checkpoints (no W&B writes), pooled over the directed edges of each topology within a seed;
totals are asserted against the remote test metrics. Fully connected = the existing 4096-pilot,
lambda .01 runs of mnist_4agents_shift. The gap is paired by seed.
Promoted from tmp/2026-11-12_mnist4_topology_report.py; writes results/mnist_4agents_shift/topology/.
--plot-only redraws the gap figure + gap_table.md from the retained per_run.csv (no checkpoints needed);
without it every checkpoint is rescored (CUDA, logs/mnist_4agents_shift{,_ring,_star}/) and
per_run.csv / summary.csv / table.md are rewritten.
Usage: uv run python scripts/plot_mnist4_topology.py [--plot-only] [--topologies star ring fully_connected]
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['WANDB_MODE'] = 'disabled'
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import plot_pilot_tsne as enc  # noqa: E402
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

from src.utils.plotting import use_style  # noqa: E402

FULLY_CONNECTED = {  # mnist_4agents_shift, 4096 pilots, lambda .01, bs256 (LEDGER valid)
    'Non-coop': ['c8ix7vn9', 'x9u62pd7', 'dsnopa19'],
    'DiSheafFRL': ['cndm7lcm', '7x9j9uyl', '7zvxszuc'],
}
PRIVATE = 'Private'  # private test accuracy on all 10 classes (W&B test/avg_private_task_perf)
PRIV_SEEN = 'Private target classes'  # private accuracy on the agent's own target classes (figure panel a)
TABLE_SCOPES = {
    PRIVATE: 'Private accuracy (all classes)',
    PRIV_SEEN: 'Private accuracy (target classes)',
    'Comm': 'Communication, all classes',
    'Comm sender unseen / receiver seen': 'Sender unseen / receiver seen',
    'Comm receiver seen': 'Receiver seen (its target classes)',
}
SCOPES = {
    'Comm': 'Communication, all classes',
    'Comm sender unseen / receiver seen': 'Sender unseen / receiver seen',
    'Comm receiver seen': 'Receiver seen (its target classes)',
}
TOPOLOGIES = {
    'star': 'Star',
    'ring': 'Ring',
    'fully_connected': 'Fully connected',
}
METHOD_GROUP = {
    'Non-coop': 'non_cooperative_p4096',
    'DiSheafFRL': 'di_sheaf_frl_p4096',
}

parser = argparse.ArgumentParser()
parser.add_argument(
    '--topologies',
    nargs='+',
    default=list(TOPOLOGIES),
    choices=list(TOPOLOGIES),
)
parser.add_argument('--out', default='results/mnist_4agents_shift/topology')
parser.add_argument(
    '--plot-only',
    action='store_true',
    help='redraw the gap figure from <out>/per_run.csv',
)
args = parser.parse_args()
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)
enc.DEV = 'cuda'
torch.set_num_threads(4)
api = wandb.Api()
entity = api.default_entity


def collect(topology):
    """Return [(method, project, run)] for one topology; fail loudly if incomplete."""
    if topology == 'fully_connected':
        project = 'mnist_4agents_shift'
        return [
            (m, project, api.run(f'{entity}/{project}/{rid}'))
            for m, ids in FULLY_CONNECTED.items()
            for rid in ids
        ]
    project = f'mnist_4agents_shift_{topology}'
    out = []
    for method, group in METHOD_GROUP.items():
        runs = list(api.runs(f'{entity}/{project}', filters={'group': group}))
        runs = [
            api.run(f'{entity}/{project}/{r.id}')
            for r in runs
            if r.state == 'finished'
        ]
        assert sorted(r.config['seed'] for r in runs) == [42, 43, 44], (
            project,
            group,
            [(r.id, r.state) for r in runs],
        )
        out += [(method, project, r) for r in runs]
    return out


def load(project, r):
    """Rebuild the run's orchestrator and load its final checkpoint (eval mode, CUDA)."""
    cfg = OmegaConf.create(r.config)
    dm, neighbors = _setup(cfg)
    agents, dims = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    orch = _build_orchestrator(
        cfg, agents, neighbors, dims, _parse_per_agent_cfg(cfg)
    )
    ckpt = list(
        (ROOT / 'logs' / project / r.id / 'checkpoints').glob('epoch=99*.ckpt')
    )
    assert len(ckpt) == 1, (r.id, ckpt)
    orch.load_state_dict(
        torch.load(ckpt[0], map_location='cpu', weights_only=False)[
            'state_dict'
        ],
        strict=True,
    )
    orch.to('cuda').eval()
    orch._trainer = SimpleNamespace(datamodule=dm)
    return orch, dm, cfg, neighbors


def score(project, r, method):
    """Score one run from its final checkpoint; pool over the topology's directed edges."""
    orch, dm, cfg, neighbors = load(project, r)
    if method == 'Non-coop':
        assert orch._fit_alignment_maps(dm)
    seen = {
        i: torch.tensor(list(cfg.dataset.groups[str(i)].target_classes))
        for i in range(4)
    }
    counts = {(k, rc): [0, 0] for k in SCOPES for rc in range(4)}
    priv, priv_seen = [0, 0], [0, 0]
    with torch.no_grad():
        for sender in range(4):
            z, y = enc.encode(
                orch.agents[str(sender)],
                dm.test_datasets[sender],
                list(range(len(dm.test_datasets[sender]))),
                bs=256,
            )
            z = z.float().cuda()
            p = orch.agents[str(sender)].decoder(z).argmax(1).cpu()
            priv[0] += int((p == y).sum())
            priv[1] += len(y)
            ss = torch.isin(y, seen[sender])
            priv_seen[0] += int(((p == y) & ss).sum())
            priv_seen[1] += int(ss.sum())
            for receiver in sorted(neighbors[sender]):
                pred = (
                    orch.agents[str(receiver)]
                    .decoder(orch.send_message(sender, receiver, z).float())
                    .argmax(1)
                    .cpu()
                )
                rs = torch.isin(y, seen[receiver])
                for key, mask in [
                    ('Comm', torch.ones_like(ss)),
                    ('Comm sender unseen / receiver seen', ~ss & rs),
                    ('Comm receiver seen', rs),
                ]:
                    counts[key, receiver][0] += int(((pred == y) & mask).sum())
                    counts[key, receiver][1] += int(mask.sum())
    # per receiver over its in-edges, then mean over receivers (= W&B test/avg_comm_task_perf; matters when in-degrees differ, e.g. star)
    vals = {
        k: sum(counts[k, rc][0] / counts[k, rc][1] for rc in range(4)) / 4
        for k in SCOPES
    }
    assert (
        abs(priv[0] / priv[1] - r.summary['test/avg_private_task_perf']) < 2e-5
    ), (r.id, priv, r.summary['test/avg_private_task_perf'])
    assert abs(vals['Comm'] - r.summary['test/avg_comm_task_perf']) < 2e-5, (
        r.id,
        vals,
        r.summary['test/avg_comm_task_perf'],
    )
    vals[PRIVATE] = priv[0] / priv[1]
    vals[PRIV_SEEN] = priv_seen[0] / priv_seen[1]
    return vals


def fill_private(frame):
    """Add the private accuracy of runs scored before it was recorded (W&B summary, asserted equal at scoring)."""
    if PRIVATE in frame:
        return frame
    project = {t: f'mnist_4agents_shift_{t}' for t in TOPOLOGIES}
    project['fully_connected'] = 'mnist_4agents_shift'
    frame = frame.copy()
    frame[PRIVATE] = [
        api.run(f'{entity}/{project[r.topology]}/{r.run}').summary[
            'test/avg_private_task_perf'
        ]
        for r in frame.itertuples()
    ]
    return frame


def fill_private_seen(frame):
    """Add PRIV_SEEN to rows that lack it (one checkpoint load per run; Private re-asserted)."""
    if PRIV_SEEN in frame:
        return frame
    project = {t: f'mnist_4agents_shift_{t}' for t in TOPOLOGIES}
    project['fully_connected'] = 'mnist_4agents_shift'
    frame = frame.copy()
    values = []
    for r0 in frame.itertuples():
        r = api.run(f'{entity}/{project[r0.topology]}/{r0.run}')
        orch, dm, cfg, _ = load(project[r0.topology], r)
        allc, seen = [0, 0], [0, 0]
        with torch.no_grad():
            for i in range(4):
                ds = dm.test_datasets[i]
                z, y = enc.encode(
                    orch.agents[str(i)], ds, list(range(len(ds))), bs=256
                )
                ok = (
                    orch.agents[str(i)]
                    .decoder(z.float().cuda())
                    .argmax(1)
                    .cpu()
                    == y
                )
                mask = torch.isin(
                    y,
                    torch.tensor(
                        list(cfg.dataset.groups[str(i)].target_classes)
                    ),
                )
                allc[0] += int(ok.sum())
                allc[1] += len(y)
                seen[0] += int((ok & mask).sum())
                seen[1] += int(mask.sum())
        assert abs(allc[0] / allc[1] - r0.Private) < 2e-5, (r0.run, allc)
        values.append(seen[0] / seen[1])
        print(
            'PRIVATE SEEN',
            r0.run,
            f'{allc[0] / allc[1]:.4f} -> {values[-1]:.4f}',
            flush=True,
        )
        del orch
        torch.cuda.empty_cache()
    frame[PRIV_SEEN] = values
    return frame


def plot_gap(frame, order):
    """Paper figure: DiSheafFRL - Non-coop per seed (paired), mean ± sample SD, one panel per scope."""
    use_style()
    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.size': 9,
            'axes.titlesize': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'lines.linewidth': 1.2,
            'lines.markersize': 5,
        }
    )
    edges = {'star': 6, 'ring': 8, 'fully_connected': 12}
    short = {'star': 'Star', 'ring': 'Ring', 'fully_connected': 'Full'}
    # Same panels and titles as the lambda / pilot figures (the all-classes scope stays in the tables).
    panels = [
        (PRIV_SEEN, '(a) Private accuracy (target classes)'),
        (
            'Comm sender unseen / receiver seen',
            '(b) Sender unseen, receiver seen',
        ),
        ('Comm receiver seen', '(c) Receiver seen'),
    ]
    wide = frame.pivot_table(
        index=['topology', 'seed'],
        columns='method',
        values=[p for p, _ in panels],
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    xs = np.arange(len(order))
    lines = [
        '',
        'Gap DiSheafFRL - Non-coop (pts), paired by seed, mean ± sample SD:',
        '| Scope | ' + ' | '.join(short[t] for t in order) + ' |',
        '|---|' + '---|' * len(order),
    ]
    for ax, (scope, title) in zip(axes, panels):
        gap = (wide[scope, 'DiSheafFRL'] - wide[scope, 'Non-coop']) * 100
        stats = [gap.loc[t] for t in order]
        assert all(sorted(g.index) == [42, 43, 44] for g in stats)
        means, sds = [g.mean() for g in stats], [g.std(ddof=1) for g in stats]
        ax.errorbar(
            xs,
            means,
            yerr=sds,
            color='tab:blue',
            marker='o',
            capsize=2,
            label='DiSheafFRL + CenterNorm $-$ Non-coop',
        )
        ax.set_xticks(xs, [f'{short[t]}\n({edges[t]} edges)' for t in order])
        ax.set_xlim(-0.4, len(order) - 0.6)
        lo, hi = (
            min(m - s for m, s in zip(means, sds)),
            max(m + s for m, s in zip(means, sds)),
        )
        pad = max(0.5, 0.25 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Topology')
        lines.append(
            f'| {title} | '
            + ' | '.join(f'{m:+.2f} ± {sd:.2f}' for m, sd in zip(means, sds))
            + ' |'
        )
    axes[0].set_ylabel('Accuracy gap (pts)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'topology_gap.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'gap_table.md').write_text('\n'.join(lines[1:]) + '\n')
    print('\n'.join(lines))


def write_tables(frame, order):
    """summary.csv and table.md from per_run rows (private + communication scopes)."""
    summary = []
    for (topology, method), g in frame.groupby(['topology', 'method']):
        assert sorted(g.seed) == [42, 43, 44]
        for scope in TABLE_SCOPES:
            v = g[scope] * 100
            summary.append(
                {
                    'topology': topology,
                    'method': method,
                    'scope': scope,
                    'mean': v.mean(),
                    'sd': v.std(ddof=1),
                    'n': len(v),
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'summary.csv', index=False)
    lines = [
        'Test accuracy (%), mean ± sample SD over seeds 42/43/44. 4096 pilots, lambda .01, batch 256, 100 epochs, CenterNorm, no whitening.',
        'Communication pooled over the directed edges of each topology (star 6, ring 8, fully connected 12) within a seed; uncertainty across seeds only.',
        '',
        '| Scope | Method | '
        + ' | '.join(TOPOLOGIES[t] for t in order)
        + ' |',
        '|---|---|' + '---|' * len(order),
    ]
    for scope, label in TABLE_SCOPES.items():
        for method in METHOD_GROUP:
            cells = []
            for t in order:
                d = summary[
                    (summary.topology == t)
                    & (summary.method == method)
                    & (summary.scope == scope)
                ].iloc[0]
                cells.append(f'{d["mean"]:.2f} ± {d["sd"]:.2f}')
            lines.append(f'| {label} | {method} | ' + ' | '.join(cells) + ' |')
    lines += [
        '',
        'Run IDs: '
        + ', '.join(
            f'{r.topology}/{r.method}/{r.seed}={r.run}'
            for r in frame.itertuples()
        ),
    ]
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if args.plot_only:
    order = [t for t in TOPOLOGIES if t in args.topologies]
    frame = fill_private_seen(fill_private(pd.read_csv(OUT / 'per_run.csv')))
    frame.to_csv(OUT / 'per_run.csv', index=False)
    write_tables(frame, order)
    plot_gap(frame, order)
    sys.exit(0)

records = []
for topology in args.topologies:
    for method, project, r in collect(topology):
        vals = score(project, r, method)
        records.append(
            {
                'topology': topology,
                'method': method,
                'seed': r.config['seed'],
                'run': r.id,
                **vals,
            }
        )
        print('VERIFIED', topology, method, r.config['seed'], r.id, flush=True)
frame = pd.DataFrame(records)
frame.to_csv(OUT / 'per_run.csv', index=False)
order = [t for t in TOPOLOGIES if t in args.topologies]
write_tables(frame, order)
plot_gap(frame, order)
