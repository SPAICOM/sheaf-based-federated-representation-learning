"""Fetch completed MNIST results and summarize the preliminary two-seed comparison."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
# Script imports require the scripts directory on sys.path.
from sweep_report import plot, row, to_markdown  # noqa: E402

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

out = ROOT / 'results/mnist_2agents_7v7/main'
out.mkdir(parents=True, exist_ok=True)
api = wandb.Api()
runs = list(
    api.runs(
        f'{api.default_entity}/mnist_2agents_7v7',
        filters={'group': {'$in': ['mnist_7v7_noncoop', 'mnist_7v7_disheaf']}},
    )
)
runs = [r for r in runs if r.state == 'finished']
if len(runs) != 4:
    raise RuntimeError(f'Expected four finished runs, found {len(runs)}')
table = pd.DataFrame([row(r, r.group, 'seed', None) for r in runs])
(out / 'table.md').write_text(
    'Preliminary two-seed comparison (seeds 42/43; ledger status: single-seed).\n\n'
    + to_markdown(table)
)
plot(
    table,
    'seed',
    False,
    'MNIST 7-vs-7, shift=1: two seeds',
    out / 'metrics.png',
)
plot(
    table,
    'seed',
    False,
    'MNIST 7-vs-7, shift=1: two seeds (preliminary)',
    out / 'metrics.pdf',
)
