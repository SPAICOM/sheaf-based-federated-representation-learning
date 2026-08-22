"""Plot SheafFMTL P_ij memory usage vs. gamma, for the 15-agent setup in
config/hydra/multiagent_mnist_shift_distr.yaml, up to where it explodes.

Combines two sources:

  1. A theoretical curve: exact P_ij storage (GB, for a given matrix_dtype)
     as a function of gamma, computed directly from d_ij = max(1, int(gamma
     * min(d_i, d_j))) (src/orchestrators/sheaf_fmtl.py) using the real
     15-agent architecture and class_overlap communication graph. No
     training needed for this part - it's closed-form and instant.
  2. Empirical points: peak process RSS actually observed by
     scripts/probe_gamma_memory.sh, read from its CSV trace(s). Gammas at
     which that probe self-killed or the run failed (per its summary file)
     are marked as the empirical "explosion" point.

Usage (run the probe first to get empirical points, then):
    uv run scripts/plot_gamma_memory_curve.py
"""

import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.agents import CNNClassifier
from src.utils import generate_neighbors

REPO_ROOT = Path(__file__).resolve().parent.parent
MPLSTYLE = REPO_ROOT / 'config' / 'plotting' / 'plt.mplstyle'

# Kept in sync by hand with the `model.agents` / `dataset.groups` blocks in
# config/hydra/multiagent_mnist_shift_distr.yaml (same values also used by
# the sheaf_fmtl-specific sibling config) - there's no cheap programmatic way
# to pull just this slice out of the full hydra config without instantiating
# the whole datamodule.
AGENTS_CFG = {
    0: dict(encoder_hidden_dims=[32, 64, 128], decoder_hidden_dims=[256, 128, 64], dropout=0.3, use_batchnorm=True),
    1: dict(encoder_hidden_dims=[32, 64, 128, 224], decoder_hidden_dims=[120], dropout=0.1, use_batchnorm=True),
    2: dict(encoder_hidden_dims=[16, 32, 64], decoder_hidden_dims=[128], dropout=0.2, use_batchnorm=True),
    3: dict(encoder_hidden_dims=[64, 128, 128], decoder_hidden_dims=[256, 128], dropout=0.25, use_batchnorm=True),
    4: dict(encoder_hidden_dims=[32, 48, 96, 192], decoder_hidden_dims=[192, 96], dropout=0.3, use_batchnorm=True),
    5: dict(encoder_hidden_dims=[24, 48, 96], decoder_hidden_dims=[96], dropout=0.15, use_batchnorm=True),
    6: dict(encoder_hidden_dims=[32, 64, 128, 256], decoder_hidden_dims=[256], dropout=0.4, use_batchnorm=True),
    7: dict(encoder_hidden_dims=[24, 48, 96, 144], decoder_hidden_dims=[64, 32], dropout=0.1, use_batchnorm=True),
    8: dict(encoder_hidden_dims=[48, 96, 160], decoder_hidden_dims=[160, 80], dropout=0.2, use_batchnorm=True),
    9: dict(encoder_hidden_dims=[32, 80], decoder_hidden_dims=[100], dropout=0.05, use_batchnorm=True),
    10: dict(encoder_hidden_dims=[40, 80, 120], decoder_hidden_dims=[120, 60], dropout=0.2, use_batchnorm=True),
    11: dict(encoder_hidden_dims=[48, 96, 192, 240], decoder_hidden_dims=[256, 128], dropout=0.2, use_batchnorm=True),
    12: dict(encoder_hidden_dims=[20, 40, 60, 80], decoder_hidden_dims=[80], dropout=0.2, use_batchnorm=True),
    13: dict(encoder_hidden_dims=[32, 64, 96], decoder_hidden_dims=[192, 96], dropout=0.25, use_batchnorm=True),
    14: dict(encoder_hidden_dims=[56, 112, 160], decoder_hidden_dims=[128], dropout=0.1, use_batchnorm=True),
}
TARGET_CLASSES = {
    0: {4, 5, 6, 7, 8}, 1: {0, 1, 2, 3, 4}, 2: {0, 1, 2, 3, 4, 5},
    3: {4, 5, 6, 7, 8, 9}, 4: {0, 1, 2, 7, 8, 9}, 5: {2, 3, 4, 5, 6},
    6: {1, 2, 3, 4, 5, 6}, 7: {3, 4, 5, 6, 7}, 8: {0, 2, 4, 6, 8},
    9: {5, 6, 7, 8, 9}, 10: {1, 2, 3, 4, 5}, 11: {0, 1, 2, 8, 9},
    12: {3, 4, 5, 6, 7, 8}, 13: {1, 4, 6, 8, 9}, 14: {0, 1, 2, 3, 7},
}


def _param_counts() -> dict[int, int]:
    return {
        idx: sum(
            p.numel()
            for p in CNNClassifier(in_features=1, num_classes=10, **cfg).parameters()
            if p.requires_grad
        )
        for idx, cfg in AGENTS_CFG.items()
    }


def _directed_edges(max_edge_frac: float = 0.4) -> list[tuple[int, int]]:
    neighbors = generate_neighbors(
        mode='class_overlap',
        n_agents=len(TARGET_CLASSES),
        target_classes=TARGET_CLASSES,
        max_edge_frac=max_edge_frac,
    )
    return [(i, j) for i, js in neighbors.items() for j in js]


def theoretical_gb(gammas: np.ndarray, dtype_bytes: int) -> np.ndarray:
    """Total P_ij storage (GB) across all directed edges, for each gamma."""
    d = _param_counts()
    edges = _directed_edges()
    totals = []
    for gamma in gammas:
        total_elems = sum(
            max(1, int(gamma * min(d[i], d[j]))) * d[i] for i, j in edges
        )
        totals.append(total_elems * dtype_bytes / 1e9)
    return np.array(totals)


def _load_probe_traces(trace_glob: str) -> pd.DataFrame | None:
    files = sorted(glob.glob(trace_glob))
    if not files:
        return None
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df['proc_rss_mb'] = pd.to_numeric(df['proc_rss_mb'], errors='coerce')
    return df.dropna(subset=['proc_rss_mb'])


def _load_crashed_gammas(summary_glob: str) -> set[float]:
    crashed = set()
    for f in sorted(glob.glob(summary_glob)):
        text = Path(f).read_text()
        for m in re.finditer(r'gamma=([0-9.eE+-]+): (SELF-KILLED|FAILED)', text):
            crashed.add(float(m.group(1)))
    return crashed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--trace-glob',
        default=str(REPO_ROOT / 'logs/gamma_memory_probe/trace_*.csv'),
    )
    parser.add_argument(
        '--summary-glob',
        default=str(REPO_ROOT / 'logs/gamma_memory_probe/summary_*.txt'),
    )
    parser.add_argument(
        '--dtype-bytes',
        type=int,
        default=2,
        help='Bytes per P_ij element: 2 for float16/bfloat16 (matrix_dtype '
        'default), 4 for float32.',
    )
    parser.add_argument('--sys-ram-gb', type=float, default=31.0)
    parser.add_argument(
        '--out',
        type=Path,
        default=REPO_ROOT / 'results/multi_agent/plots/gamma_memory_curve.png',
    )
    args = parser.parse_args()

    gammas = np.logspace(-5, -0.3, 300)
    theo_gb = theoretical_gb(gammas, args.dtype_bytes)

    empirical = _load_probe_traces(args.trace_glob)
    crashed_gammas = _load_crashed_gammas(args.summary_glob)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with plt.style.context(str(MPLSTYLE)):
        fig, ax = plt.subplots()
        ax.plot(
            gammas, theo_gb, color='C0',
            label=r'Theoretical $P_{ij}$ storage (all edges)',
        )
        ax.axhline(
            args.sys_ram_gb, color='red', linestyle='--',
            label=f'System RAM ({args.sys_ram_gb:g} GB)',
        )

        if empirical is not None and len(empirical):
            peak_gb = empirical.groupby('gamma')['proc_rss_mb'].max() / 1024
            ok = [g for g in peak_gb.index if g not in crashed_gammas]
            bad = [g for g in peak_gb.index if g in crashed_gammas]
            if ok:
                ax.scatter(
                    ok, peak_gb.loc[ok], color='C2', zorder=5, s=120,
                    label='Measured peak RSS (completed)',
                )
            if bad:
                ax.scatter(
                    bad, peak_gb.loc[bad], color='red', marker='x', s=250,
                    zorder=6, label='Measured at crash / self-kill',
                )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel('Memory (GB)')
        ax.legend()
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        fig.savefig(args.out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'Saved -> {args.out} (+ .pdf)')


if __name__ == '__main__':
    main()
