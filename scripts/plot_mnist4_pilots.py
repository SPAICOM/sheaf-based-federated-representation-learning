"""Remote-verified pilot summary and checkpoint class-scope rescoring."""

import os

os.environ['WANDB_MODE'] = 'disabled'
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
# Repository imports require the project root on sys.path.
import plot_pilot_tsne as enc  # noqa: E402
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402
from sweep_report import config_value, row  # noqa: E402

from src.utils.plotting import use_style  # noqa: E402

enc.DEV = 'cuda'
torch.set_num_threads(4)
OUT = ROOT / 'results/mnist_4agents_shift/pilots'
OUT.mkdir(parents=True, exist_ok=True)
api = wandb.Api()
project = 'mnist_4agents_shift'
prefixes = ('mnist4_di_clean_bs256_check_', 'mnist4_di_clean_bs256_pilots_')
runs = [
    api.run(f'{api.default_entity}/{project}/{r.id}')
    for r in api.runs(f'{api.default_entity}/{project}')
    if r.group.startswith(prefixes)
    and r.group.rsplit('_p', 1)[-1] in ('384', '768', '1536', '2048', '4096')
]
assert len(runs) == 30 and all(r.state == 'finished' for r in runs)
records = []
for r in runs:
    cfg = OmegaConf.create(r.config)
    p = config_value(r.config, 'dataset.pilot_num_samples')
    method = 'Non-coop' if 'non_cooperative' in r.group else 'DiSheafFRL'
    remote = row(r, method, None, None)
    dm, neighbors = _setup(cfg)
    agents, dims = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    orch = _build_orchestrator(
        cfg, agents, neighbors, dims, _parse_per_agent_cfg(cfg)
    )
    checkpoints = list(
        (ROOT / 'logs' / project / r.id / 'checkpoints').glob('epoch=99*.ckpt')
    )
    assert len(checkpoints) == 1
    orch.load_state_dict(
        torch.load(checkpoints[0], map_location='cpu', weights_only=False)[
            'state_dict'
        ],
        strict=True,
    )
    orch.to('cuda').eval()
    orch._trainer = SimpleNamespace(datamodule=dm)
    if method == 'Non-coop':
        assert orch._fit_alignment_maps(dm)
    seen = {
        i: torch.tensor(list(cfg.dataset.groups[str(i)].target_classes))
        for i in range(4)
    }
    counts = {
        k: [0, 0]
        for k in [
            'Private',
            'Private seen',
            'Private unseen',
            'Comm',
            'Comm sender unseen / receiver seen',
            'Comm sender seen / receiver unseen',
        ]
    }

    def add(key, pred, y, mask, counts=counts):
        """Accumulate correct predictions and sample counts for a scope."""
        counts[key][0] += int(((pred == y) & mask).sum())
        counts[key][1] += int(mask.sum())

    with torch.no_grad():
        for sender in range(4):
            z, y = enc.encode(
                orch.agents[str(sender)],
                dm.test_datasets[sender],
                list(range(len(dm.test_datasets[sender]))),
                bs=256,
            )
            z = z.float().cuda()
            pred = orch.agents[str(sender)].decoder(z).argmax(1).cpu()
            ss = torch.isin(y, seen[sender])
            add('Private', pred, y, torch.ones_like(ss))
            add('Private seen', pred, y, ss)
            add('Private unseen', pred, y, ~ss)
            for receiver in range(4):
                if sender == receiver:
                    continue
                pred = (
                    orch.agents[str(receiver)]
                    .decoder(orch.send_message(sender, receiver, z).float())
                    .argmax(1)
                    .cpu()
                )
                rs = torch.isin(y, seen[receiver])
                add('Comm', pred, y, torch.ones_like(ss))
                add('Comm sender unseen / receiver seen', pred, y, ~ss & rs)
                add('Comm sender seen / receiver unseen', pred, y, ss & ~rs)
    vals = {k: a / b for k, (a, b) in counts.items()}
    assert abs(vals['Private'] - remote['private test']) < 2e-5, (
        r.id,
        vals,
        remote,
    )
    assert abs(vals['Comm'] - remote['comm test']) < 2e-5, (r.id, vals, remote)
    records.append(
        dict(pilots=p, method=method, seed=cfg.seed, run=r.id, **vals)
    )
    print('VERIFIED', p, method, cfg.seed, r.id, flush=True)
    (OUT / 'progress.txt').write_text(
        f'{len(records)}/30 checkpoints verified\n'
    )
    del orch, agents, dm
frame = pd.DataFrame(records)
assert all(
    sorted(g.seed) == [42, 43, 44]
    for _, g in frame.groupby(['pilots', 'method'])
)
metrics = list(counts)
cols = [
    (p, m)
    for p in sorted(frame.pilots.unique())
    for m in ['Non-coop', 'DiSheafFRL']
]
lines = [
    'Mean ± sample SD (%), seeds 42/43/44; shift=1, batch=256, 100 epochs.',
    '',
    '| Metric | ' + ' | '.join(f'{p} {m}' for p, m in cols) + ' |',
    '|---|' + '---|' * len(cols),
]
summary = []
for metric in metrics:
    cells = []
    for p, m in cols:
        v = frame.loc[(frame.pilots == p) & (frame.method == m), metric] * 100
        cells.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
        summary.append(
            {
                'pilots': p,
                'method': m,
                'metric': metric,
                'mean': v.mean(),
                'sd': v.std(ddof=1),
                'sem': v.std(ddof=1) / np.sqrt(3),
            }
        )
    lines.append('| ' + metric + ' | ' + ' | '.join(cells) + ' |')
lines += [
    '',
    'Scopes pool correct predictions and sample counts across four agents or twelve directed edges within each seed. Uncertainty is across seeds, not edges.',
    '',
    'All 30 checkpoint totals match remote W&B test metrics within 0.002 percentage points.',
    '',
    'Run IDs: ' + ', '.join(frame.run),
]
(OUT / 'table.md').write_text('\n'.join(lines) + '\n')
pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
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


fig, axs = plt.subplots(2, 3, figsize=(15, 8))
s = pd.DataFrame(summary)
for ax, metric in zip(axs.flat, metrics):
    for m in ['Non-coop', 'DiSheafFRL']:
        d = s[(s.metric == metric) & (s.method == m)]
        ax.errorbar(d.pilots, d['mean'], yerr=d.sd, marker='o', label=m)
    ax.set_title(metric)
    ax.set_xlabel('Pilots')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(alpha=0.2)
axs.flat[0].legend()
fig.tight_layout()
fig.savefig(OUT / 'metrics.png')
fig.savefig(OUT / 'metrics.pdf')
print('\n'.join(lines), flush=True)
