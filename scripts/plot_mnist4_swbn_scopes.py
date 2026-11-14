"""MNIST 4 agents pilot study: comm accuracy by class scope, SWBN vs CenterNorm (+ZCA/OLS) vs non-coop.

Scopes (pooled over the 12 directed edges within a seed):
  Comm                               - all test samples
  Comm sender unseen / receiver seen - class outside the sender's target classes, inside the receiver's
  Comm receiver seen                 - class inside the receiver's target classes
Each method is rescored offline from its final checkpoint under the evaluation it reported remotely
(posthoc Procrustes for non-coop / CenterNorm / ZCA, learned SWBN + saved Stiefel maps for SWBN) and the
overall Comm is asserted against the remote test metric. SWBN is also scored posthoc as a like-for-like
reference to ZCA. CenterNorm + OLS (mapstudy_ols sources) is scored posthoc (asserted like the others) and
through its saved OLS maps; the saved-map refit is not bit-exact vs the *_saved_ols_eval runs (zp4ea8r3:
52.36 vs 52.40), so its Comm is taken from the remote eval run (= pilot_summary.csv) and only the two
seen scopes come from the checkpoint (local Comm kept as local_comm, must be within 2e-3).
'DiSheafFRL + CenterNorm (learned)' (saved Stiefel maps) is opt-in via --methods: it equals the posthoc
CenterNorm score (maps are refit as closed-form Procrustes during training; FINDINGS).
No W&B writes. Resumable: (method, run) rows already in per_run.csv are skipped.
Promoted from tmp/2026-11-12_mnist4_swbn_scopes.py; writes results/mnist_4agents_shift/swbn_scopes/.
per_run.csv / summary.csv / table.md keep every method; the figure shows the methods listed in FIGURES.
--plot-only redraws the figure from the retained summary.csv (no checkpoints / W&B needed):
results/mnist_4agents_shift/maps/pilot_study.* with the standard panels (a) Private accuracy (target classes),
(b) Sender unseen, receiver seen, (c) Receiver seen (non-coop, CenterNorm, ZCA, CenterNorm + OLS, SWBN).
--study lambda: same scoring on the 4096-pilot lambda study (lambda .003..1; non-coop, CenterNorm, ZCA,
CenterNorm + OLS saved maps, i.e. the lines of the former maps/lambda_study figure); tables in
results/mnist_4agents_shift/maps/lambda_scopes/, figure results/mnist_4agents_shift/maps/lambda_study.*
(Private / sender unseen-receiver seen / receiver seen; non-coop is lambda-free, drawn flat).
Usage: uv run python scripts/plot_mnist4_swbn_scopes.py [--study lambda] [--plot-only] [--methods ...] [--allow-partial]
"""

import argparse
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

os.environ['WANDB_MODE'] = 'disabled'
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

from src.communication.alignment_mixin import (  # noqa: E402
    PostTrainingAlignmentMixin,
)
from src.utils.plotting import use_style  # noqa: E402

PILOTS = (384, 768, 1536, 2048, 4096)
LAMBDAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SEEDS = [42, 43, 44]
SCOPES = ['Comm', 'Comm sender unseen / receiver seen', 'Comm receiver seen']
# method -> (project, group(pilots), eval mode used for the remote metric)
METHODS = {
    'Non-coop': ('mnist_4agents_shift', None, 'posthoc'),
    'DiSheafFRL + CenterNorm': ('mnist_4agents_shift', None, 'posthoc'),
    # same checkpoints, scored through the Stiefel maps learned in training (no whitening: identity)
    'DiSheafFRL + CenterNorm (learned)': (
        'mnist_4agents_shift',
        None,
        'stiefel',
    ),
    'DiSheafFRL + ZCA': (
        'mnist_4agents_mapstudy_zca',
        'zca_p{}_l.01',
        'posthoc',
    ),
    'DiSheafFRL + CenterNorm + OLS (posthoc)': (
        'mnist_4agents_mapstudy_ols',
        'ols_p{}_l.01',
        'posthoc',
    ),
    'DiSheafFRL + CenterNorm + OLS': (
        'mnist_4agents_mapstudy_ols',
        'ols_p{}_l.01',
        'saved',
    ),
    'DiSheafFRL + SWBN': (
        'mnist_4agents_mapstudy_swbn',
        'swbn_p{}_l.01',
        'learned',
    ),
}
# Lambda study (4096 pilots): method -> (project, groups, eval mode). Lambda .01 of non-coop / CenterNorm
# is the pilot-study check run, as in scripts/plot_mnist4_lambda.py; non-coop has no lambda.
_PRE = 'mnist4_di_clean_bs256'
LAMBDA_METHODS = {
    'Non-coop': (
        'mnist_4agents_shift',
        [f'{_PRE}_check_non_cooperative_p4096'],
        'posthoc',
    ),
    'DiSheafFRL + CenterNorm': (
        'mnist_4agents_shift',
        [f'{_PRE}_check_di_sheaf_frl_p4096']
        + [f'{_PRE}_lambda4096_{w:g}' for w in LAMBDAS if w != 0.01],
        'posthoc',
    ),
    'DiSheafFRL + ZCA': (
        'mnist_4agents_mapstudy_zca',
        [f'zca_p4096_l{w:g}'.replace('l0.', 'l.') for w in LAMBDAS],
        'posthoc',
    ),
    'DiSheafFRL + CenterNorm + OLS': (
        'mnist_4agents_mapstudy_ols',
        [f'ols_p4096_l{w:g}'.replace('l0.', 'l.') for w in LAMBDAS],
        'saved',
    ),
}
# Colours of scripts/plot_mnist4_maps.py; SWBN posthoc is the extra reference here.
COLORS = {
    'Non-coop': 'tab:orange',
    'DiSheafFRL + CenterNorm': 'tab:blue',
    'DiSheafFRL + CenterNorm (learned)': 'tab:cyan',
    'DiSheafFRL + ZCA': 'tab:green',
    'DiSheafFRL + CenterNorm + OLS (posthoc)': 'tab:purple',
    'DiSheafFRL + CenterNorm + OLS': 'tab:red',
    'DiSheafFRL + SWBN': 'tab:brown',
    'DiSheafFRL + SWBN (posthoc)': 'tab:gray',
}
# Legend text where it differs from the CSV method name (only one SWBN line is drawn).
LEGEND = {'DiSheafFRL + SWBN (posthoc)': 'DiSheafFRL + SWBN'}
DASHED = {
    'Non-coop',
    'DiSheafFRL + CenterNorm + OLS (posthoc)',
    'DiSheafFRL + SWBN (posthoc)',
}
# Private accuracy restricted to the sender's own target classes (filled per run by fill_private_seen).
PRIV_SEEN = 'Private target classes'
# Figures: file stem -> (methods, panel scopes, panel titles, line style per method).
_V2_SCOPES = [PRIV_SEEN, *SCOPES[1:]]
_V2_TITLES = [
    '(a) Private accuracy (target classes)',
    '(b) Sender unseen, receiver seen',
    '(c) Receiver seen',
]


def _v2_style(m):
    return ':' if m == 'Non-coop' else '-'


# Every MNIST-4 study figure uses the same three panels (_V2_SCOPES / _V2_TITLES); lambda and pilot
# studies differ only in the x axis and the methods drawn (SWBN is posthoc-scored, drawn as one line).
FIGURES = {
    'pilots': {
        '../maps/pilot_study': (
            [
                'Non-coop',
                'DiSheafFRL + CenterNorm',
                'DiSheafFRL + ZCA',
                'DiSheafFRL + CenterNorm + OLS',
                'DiSheafFRL + SWBN (posthoc)',
            ],
            _V2_SCOPES,
            _V2_TITLES,
            _v2_style,
        ),
    },
    'lambda': {
        '../lambda_study': (
            list(LAMBDA_METHODS),
            _V2_SCOPES,
            _V2_TITLES,
            _v2_style,
        ),
    },
}
# fit_procrustes adds 1e-6 jitter, so posthoc refits drift (see FINDINGS); learned maps are exact;
# saved OLS maps are re-solved (pseudoinverse) and drift up to ~4e-4 vs the remote eval runs.
TOL = {'posthoc': 3e-4, 'learned': 2e-5, 'saved': 2e-3}

parser = argparse.ArgumentParser()
parser.add_argument('--study', choices=['pilots', 'lambda'], default='pilots')
parser.add_argument(
    '--methods',
    nargs='+',
    help='default: every method of the study except CenterNorm (learned)',
    choices=list(METHODS),
)
parser.add_argument(
    '--allow-partial',
    action='store_true',
    help='score finished runs only; no table/plot',
)
parser.add_argument(
    '--out',
    help='default: swbn_scopes/ (pilots) or maps/lambda_scopes/ (lambda)',
)
parser.add_argument(
    '--plot-only',
    action='store_true',
    help='redraw the figures from <out>/summary.csv',
)
args = parser.parse_args()
X = args.study  # x column of per_run/summary: 'pilots' or 'lambda'
XS = PILOTS if X == 'pilots' else LAMBDAS
if X == 'lambda':
    METHODS = LAMBDA_METHODS
if args.methods is None:
    args.methods = [
        m for m in METHODS if m != 'DiSheafFRL + CenterNorm (learned)'
    ]
assert set(args.methods) <= set(METHODS), args.methods
OUT = ROOT / (
    args.out
    or {
        'pilots': 'results/mnist_4agents_shift/swbn_scopes',
        'lambda': 'results/mnist_4agents_shift/maps/lambda_scopes',
    }[X]
)
OUT.mkdir(parents=True, exist_ok=True)
enc.DEV = 'cuda'
torch.set_num_threads(4)
api = wandb.Api()
entity = api.default_entity


def collect_lambda(method):
    """[(lambda, run)] at 4096 pilots; non-coop runs get lambda None (lambda-free, repeated at every lambda in the summary)."""
    project, groups, _ = METHODS[method]
    full = [
        api.run(f'{entity}/{project}/{s.id}')
        for s in api.runs(
            f'{entity}/{project}', filters={'group': {'$in': groups}}
        )
        if s.state == 'finished'
    ]
    nocoop = method == 'Non-coop'
    out = []
    for r in full:
        assert r.config['dataset']['pilot_num_samples'] == 4096, r.id
        w = None if nocoop else float(r.config['orchestrator']['max_lmb'])
        assert nocoop or w in LAMBDAS, (r.id, w)
        out.append((w, r))
    for w in [None] if nocoop else LAMBDAS:
        seeds = sorted(r.config['seed'] for x, r in out if x == w)
        if not args.allow_partial:
            assert seeds == SEEDS, (method, w, seeds)
        assert len(seeds) == len(set(seeds)), (method, w, seeds)
    return project, out


def collect(method):
    """[(pilots, run)] for one method, every pilot x seed finished unless --allow-partial."""
    if X == 'lambda':
        return collect_lambda(method)
    project, group, _ = METHODS[method]
    if (
        group is None
    ):  # promoted pilot study groups (scripts/plot_mnist4_pilots.py)
        prefixes = (
            'mnist4_di_clean_bs256_check_',
            'mnist4_di_clean_bs256_pilots_',
        )
        stubs = [
            r
            for r in api.runs(f'{entity}/{project}')
            if r.group.startswith(prefixes)
            and r.group.rsplit('_p', 1)[-1] in {str(p) for p in PILOTS}
            and ('non_cooperative' in r.group) == (method == 'Non-coop')
        ]
    else:
        stubs = list(
            api.runs(
                f'{entity}/{project}',
                filters={'group': {'$in': [group.format(p) for p in PILOTS]}},
            )
        )
    # Listing stubs carry an empty config; fetch the full runs before filtering on it.
    full = [
        api.run(f'{entity}/{project}/{s.id}')
        for s in stubs
        if s.state == 'finished'
    ]
    out = []
    for p in PILOTS:
        # Finished runs without a dataset config are offline eval runs, not sources.
        runs = [
            r
            for r in full
            if r.config.get('dataset', {}).get('pilot_num_samples') == p
        ]
        seeds = sorted(r.config['seed'] for r in runs)
        if not args.allow_partial:
            assert seeds == SEEDS, (method, p, seeds)
        assert len(seeds) == len(set(seeds)), (method, p, seeds)
        out += [(p, r) for r in runs]
    return project, out


def saved_ols_evals():
    """Map source run id -> its saved-OLS-map eval run (remote Comm of 'DiSheafFRL + CenterNorm + OLS')."""
    project = 'mnist_4agents_mapstudy_ols'
    groups = ['lambda_saved_ols_eval', 'pilot_saved_ols_eval']
    evals = {}
    for s in api.runs(
        f'{entity}/{project}', filters={'group': {'$in': groups}}
    ):
        if s.state != 'finished':
            continue
        r = api.run(f'{entity}/{project}/{s.id}')
        if r.config.get('eval_solver') == 'saved_checkpoint_maps':
            assert r.config['source_run'] not in evals, (
                'duplicate eval',
                r.config['source_run'],
            )
            evals[r.config['source_run']] = r
    return evals


@contextmanager
def posthoc(orch, dm):
    """Swap in the non-cooperative post-training maps, as SheafFRL's posthoc eval does."""
    if not hasattr(
        orch, '_POSTHOC_METHODS'
    ):  # NonCooperativeLearning is the mixin itself
        assert orch._fit_alignment_maps(dm)
        yield
        return
    for name in orch._POSTHOC_METHODS:
        object.__setattr__(
            orch,
            name,
            types.MethodType(getattr(PostTrainingAlignmentMixin, name), orch),
        )
    try:
        assert orch._fit_alignment_maps(dm)
        yield
    finally:
        orch._cleanup_alignment()
        for name in orch._POSTHOC_METHODS:
            orch.__dict__.pop(name, None)


def scope_counts(orch, dm, cfg, neighbors):
    """Pooled correct/total per scope plus private accuracy, through orch.send_message."""
    seen = {
        i: torch.tensor(list(cfg.dataset.groups[str(i)].target_classes))
        for i in range(4)
    }
    counts = {k: [0, 0] for k in SCOPES}
    priv = [0, 0]
    with torch.no_grad():
        for sender in range(4):
            ds = dm.test_datasets[sender]
            z, y = enc.encode(
                orch.agents[str(sender)], ds, list(range(len(ds))), bs=256
            )
            z = z.float().cuda()
            priv[0] += int(
                (
                    orch.agents[str(sender)].decoder(z).argmax(1).cpu() == y
                ).sum()
            )
            priv[1] += len(y)
            ss = torch.isin(y, seen[sender])
            for receiver in sorted(neighbors[sender]):
                msg = orch.send_message(sender, receiver, z).float()
                pred = orch.agents[str(receiver)].decoder(msg).argmax(1).cpu()
                rs = torch.isin(y, seen[receiver])
                for key, mask in [
                    ('Comm', torch.ones_like(ss)),
                    (SCOPES[1], ~ss & rs),
                    (SCOPES[2], rs),
                ]:
                    counts[key][0] += int(((pred == y) & mask).sum())
                    counts[key][1] += int(mask.sum())
    return {k: a / b for k, (a, b) in counts.items()}, priv[0] / priv[1]


def load(project, r):
    """Orchestrator of run r with its final checkpoint, on CUDA in eval mode."""
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


def private_seen(orch, dm, cfg):
    """(private acc on all test samples, on the sender's target classes), pooled over the 4 agents."""
    allc, seen = [0, 0], [0, 0]
    with torch.no_grad():
        for i in range(4):
            ds = dm.test_datasets[i]
            z, y = enc.encode(
                orch.agents[str(i)], ds, list(range(len(ds))), bs=256
            )
            ok = (
                orch.agents[str(i)].decoder(z.float().cuda()).argmax(1).cpu()
                == y
            )
            mask = torch.isin(
                y,
                torch.tensor(list(cfg.dataset.groups[str(i)].target_classes)),
            )
            allc[0] += int(ok.sum())
            allc[1] += len(y)
            seen[0] += int((ok & mask).sum())
            seen[1] += int(mask.sum())
    return allc[0] / allc[1], seen[0] / seen[1]


def fill_private_seen(frame):
    """Add PRIV_SEEN to rows that lack it (one checkpoint load per run; Private re-asserted)."""
    if PRIV_SEEN not in frame:
        frame[PRIV_SEEN] = float('nan')
    for rid in frame.loc[frame[PRIV_SEEN].isna(), 'run'].unique():
        rows = frame.run == rid
        project = frame.loc[rows, 'project'].iloc[0]
        r = api.run(f'{entity}/{project}/{rid}')
        orch, dm, cfg, _ = load(project, r)
        priv, seen = private_seen(orch, dm, cfg)
        assert (abs(frame.loc[rows, 'Private'] - priv) < 2e-5).all(), (
            rid,
            priv,
        )
        frame.loc[rows, PRIV_SEEN] = seen
        print('PRIVATE SEEN', rid, f'{priv:.4f} -> {seen:.4f}', flush=True)
        del orch
        torch.cuda.empty_cache()
    return frame


def score(method, project, r):
    """Rows for one run: its reported eval mode (asserted vs remote) and, for SWBN, posthoc too."""
    orch, dm, cfg, neighbors = load(project, r)
    mode = METHODS[method][2]
    extra = {}
    if mode == 'learned':
        assert (
            cfg.orchestrator.eval_alignment == 'learned'
            and cfg.orchestrator.learn_whitening
        )
        vals, priv = scope_counts(orch, dm, cfg, neighbors)
        remote = r.summary['test/avg_comm_task_perf']
    elif (
        mode == 'stiefel'
    ):  # no remote learned-map score exists: nothing to assert Comm against
        assert (
            not cfg.orchestrator.whitening
            and not cfg.orchestrator.learn_whitening
        )
        vals, priv = scope_counts(orch, dm, cfg, neighbors)
        remote = None
        extra = {'remote_posthoc_comm': r.summary['test/avg_comm_task_perf']}
    elif (
        mode == 'saved'
    ):  # send_message without the posthoc swap = the OLS maps saved in the checkpoint
        vals, priv = scope_counts(orch, dm, cfg, neighbors)
        ev = OLS_EVALS[r.id]
        remote = ev.summary['test/avg_comm_task_perf']
        extra = {'eval_run': ev.id, 'local_comm': vals['Comm']}
    else:
        with posthoc(orch, dm):
            vals, priv = scope_counts(orch, dm, cfg, neighbors)
        remote = r.summary['test/avg_comm_task_perf']
    if remote is not None:
        assert abs(vals['Comm'] - remote) < TOL[mode], (
            r.id,
            mode,
            vals['Comm'],
            remote,
        )
    assert abs(priv - r.summary['test/avg_private_task_perf']) < 2e-5, (
        r.id,
        priv,
    )
    if mode == 'saved':
        vals['Comm'] = (
            remote  # overall from the existing remote eval; seen scopes from the checkpoint
        )
    rows = [{'method': method, 'eval': mode, 'Private': priv, **vals, **extra}]
    if method == 'DiSheafFRL + SWBN':
        with posthoc(orch, dm):
            vals_ph, _ = scope_counts(orch, dm, cfg, neighbors)
        rows.append(
            {
                'method': 'DiSheafFRL + SWBN (posthoc)',
                'eval': 'posthoc',
                'Private': priv,
                **vals_ph,
            }
        )
    return rows


def plot(summary):
    """Paper figures of the study (FIGURES): one panel per scope vs pilots/lambda; house style, compact overrides."""
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
            'lines.markersize': 3,
        }
    )
    for stem, (methods, scopes, titles, style) in FIGURES[X].items():
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
        for ax, scope, title in zip(axes, scopes, titles):
            for m in [m for m in methods if m in set(summary.method)]:
                d = summary[
                    (summary.method == m) & (summary.scope == scope)
                ].sort_values(X)
                ax.errorbar(
                    d[X],
                    d['mean'],
                    yerr=d.sd,
                    marker='o',
                    capsize=2,
                    color=COLORS[m],
                    label=LEGEND.get(m, m),
                    linestyle=style(m),
                )
            ax.set_xscale('log')
            # 1536/2048 sit close on a log axis: rotate and anchor the label end at the tick.
            ax.set_xticks(
                XS,
                [f'{x:g}' for x in XS],
                rotation=45,
                ha='right',
                rotation_mode='anchor',
            )
            ax.minorticks_off()
            ax.set(
                xlabel='Pilot count' if X == 'pilots' else 'Lambda',
                title=title,
            )
            ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel(
            'Comm. accuracy (%)' if scopes[0] == 'Comm' else 'Accuracy (%)'
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.8))
        for ext in ['png', 'pdf']:
            path = (OUT / f'{stem}.{ext}').resolve()
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print('wrote', path.relative_to(ROOT))
        plt.close(fig)


if args.plot_only:
    plot(pd.read_csv(OUT / 'summary.csv'))
    sys.exit(0)

per_run = OUT / 'per_run.csv'
frame = (
    pd.read_csv(per_run)
    if per_run.exists()
    else pd.DataFrame(columns=['method', 'run'])
)
OLS_EVALS = (
    saved_ols_evals()
    if 'DiSheafFRL + CenterNorm + OLS' in args.methods
    else {}
)
for method in args.methods:
    project, runs = collect(method)
    for p, r in runs:
        if (method, r.id) in set(zip(frame.method, frame.run)):
            continue
        rows = [
            {
                X: p,
                'seed': r.config['seed'],
                'run': r.id,
                'project': project,
                **x,
            }
            for x in score(method, project, r)
        ]
        frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
        frame.to_csv(per_run, index=False)
        print('VERIFIED', method, p, r.config['seed'], r.id, flush=True)
        torch.cuda.empty_cache()
frame = fill_private_seen(frame)
frame.to_csv(per_run, index=False)
if args.allow_partial:
    print(frame.groupby(['method', X], dropna=False).size())
    sys.exit(0)

frame = frame[
    frame.method.isin(set(args.methods) | {'DiSheafFRL + SWBN (posthoc)'})
]
summary = []
for (method, p), g in frame.groupby(['method', X], dropna=False):
    assert sorted(g.seed) == SEEDS, (method, p, sorted(g.seed))
    for scope in ['Private', PRIV_SEEN, *SCOPES]:
        v = g[scope] * 100
        summary.append(
            {
                'method': method,
                X: p,
                'scope': scope,
                'mean': v.mean(),
                'sd': v.std(ddof=1),
                'n': len(v),
            }
        )
summary = pd.DataFrame(summary)
if (
    X == 'lambda'
):  # non-coop has no lambda: repeat its 4096-pilot score at every lambda
    nc = summary[summary[X].isna()]
    summary = pd.concat(
        [summary[summary[X].notna()]] + [nc.assign(**{X: w}) for w in XS],
        ignore_index=True,
    )
summary.to_csv(OUT / 'summary.csv', index=False)

order = [m for m in COLORS if m in set(summary.method)]
lines = [
    'Test accuracy (%), mean ± sample SD over seeds 42/43/44; '
    + (
        'lambda .01'
        if X == 'pilots'
        else '4096 pilots (non-coop lambda-free, repeated)'
    )
    + ', batch 256, 100 epochs, shift 1.',
    'Scopes pool the 12 directed edges within a seed. Eval: posthoc Procrustes for non-coop / CenterNorm / ZCA '
    '(as reported remotely); learned SWBN + saved Stiefel maps for SWBN, plus SWBN posthoc as reference.',
    'CenterNorm (learned): same checkpoints as CenterNorm through the Stiefel maps saved in training (offline, '
    'no remote reference). CenterNorm + OLS: saved OLS maps; Comm from the remote saved-map eval runs (results/mnist_4agents_shift/maps/'
    f'{"pilot" if X == "pilots" else "lambda"}_summary.csv), seen scopes rescored from the checkpoints.',
    '',
]
if X == 'lambda':  # no SWBN / learned CenterNorm in the lambda study
    lines[1] = (
        'Scopes pool the 12 directed edges within a seed. Eval: posthoc Procrustes for non-coop / CenterNorm / '
        'ZCA (as reported remotely).'
    )
    lines[2] = lines[2][lines[2].index('CenterNorm + OLS') :]
for scope in ['Private', PRIV_SEEN, *SCOPES]:
    lines += [
        f'### {scope}',
        '',
        f'| {X.capitalize()} | ' + ' | '.join(order) + ' |',
        '|---|' + '---|' * len(order),
    ]
    for p in XS:
        cells = []
        for m in order:
            d = summary[
                (summary.method == m)
                & (summary[X] == p)
                & (summary.scope == scope)
            ].iloc[0]
            cells.append(f'{d["mean"]:.2f} ± {d["sd"]:.2f}')
        lines.append(f'| {p} | ' + ' | '.join(cells) + ' |')
    lines.append('')
lines.append(
    'Run IDs: '
    + ', '.join(
        f'{r["method"]}/{r[X]}/{r["seed"]}={r["run"]}'
        for r in frame.drop_duplicates(['method', 'run']).to_dict('records')
    )
)
(OUT / 'table.md').write_text('\n'.join(lines) + '\n')

plot(summary)
print('\n'.join(lines))
