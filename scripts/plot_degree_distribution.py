"""Plot the per-agent degree distribution of a multi-agent communication graph.

Reconstructs the same communication graph ``multi_agent_experiment.py``
builds at run time (``generate_neighbors`` over ``cfg.graph``/``cfg.dataset``)
directly from a **local** Hydra config — no wandb run needed. The graph is
fully determined by ``graph.neighbors_mode``/``seed``/``max_edge_frac``/
``similarity`` and ``dataset.n_agents``/``dataset.groups`` (target classes);
none of those are part of the ``multiagent_mnist_shift_distr`` sweep (only
``orchestrator``/``dataset.shift_strength`` are swept, and neither affects
the graph), so the distribution shown here is the same one every run in that
sweep uses — including the per-agent weights (``degree_i / sum(degrees)``)
that :func:`plot_multiagent_metrics.agent_degree_weights` feeds into the
``'weighted_mean'`` comm-accuracy estimator in ``just plot-hetero``.

Produces one figure: a bar per agent (x = agent index, y = degree), sorted
by agent index so it lines up with the ``weight`` column in the saved CSV,
with a dashed line at the mean degree. Also writes
``degree_distribution.csv`` (agent, degree, weight) and prints min/max/mean
degree.

Usage:
    python scripts/plot_degree_distribution.py
    python scripts/plot_degree_distribution.py --config-name hetero_rate_2agents_mnist_hetero_bottleneck
    python scripts/plot_degree_distribution.py --override dataset.n_agents=20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Reuse the shared matplotlib style, figure-saving helper, and the
# multi_agent_experiment.py loader (for _parse_agent_target_classes) from the
# main plotting script, so this figure looks like every other one here.
from plot_multiagent_metrics import _MPLSTYLE, _load_mae, _savefig  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / 'config' / 'hydra'

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils.graph_generator import generate_neighbors  # noqa: E402


def load_cfg(config_name: str, overrides: list[str]) -> Any:
    """Compose the Hydra config the same way ``multi_agent_experiment.py`` does."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base='1.3'):
        return compose(config_name=config_name, overrides=overrides)


def compute_degrees(cfg: Any) -> dict[int, int]:
    """Return ``{agent_idx: degree}`` for ``cfg``'s communication graph."""
    mae = _load_mae()
    n = int(cfg.dataset.n_agents)
    atc = mae._parse_agent_target_classes(cfg)
    if atc is not None:
        for i in range(n):
            atc.setdefault(i, set())
    neighbors = generate_neighbors(
        mode=cfg.graph.neighbors_mode,
        n_agents=n,
        seed=cfg.graph.get('seed', 42),
        p=cfg.graph.get('p', 0.3),
        m=cfg.graph.get('m', 3),
        manual=cfg.graph.get('neighbors', {}),
        target_classes=atc,
        max_edge_frac=cfg.graph.get('max_edge_frac', 0.4),
        similarity=cfg.graph.get('similarity', 'intersection'),
    )
    return {i: len(neighbors[i]) for i in range(n)}


def plot_degree_distribution(
    degrees: dict[int, int], out_dir: Path, title: str
) -> pd.DataFrame:
    agents = sorted(degrees)
    values = np.array([degrees[i] for i in agents], dtype=float)
    total = values.sum()
    weights = values / total if total > 0 else np.zeros_like(values)
    mean_degree = values.mean()

    df = pd.DataFrame({'agent': agents, 'degree': values.astype(int), 'weight': weights})

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'degree_distribution.csv'
    df.to_csv(csv_path, index=False)
    print(f'  saved → {csv_path}')

    with plt.style.context(str(_MPLSTYLE)):
        fig, ax = plt.subplots()
        ax.bar(agents, values, color=sns.color_palette('tab10')[0])
        ax.axhline(
            mean_degree,
            color='black',
            linestyle='--',
            linewidth=1,
            label=f'mean = {mean_degree:.2f}',
        )
        ax.set_xlabel('Agent')
        ax.set_ylabel('Degree')
        ax.set_xticks(agents)
        ax.set_title(title)
        ax.legend(frameon=True)
        sns.despine()
        out = out_dir / 'degree_distribution.png'
        _savefig(fig, out)
        plt.close(fig)
    print(f'  saved → {out}')

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default='multiagent_mnist_shift_distr',
        help='Hydra config under config/hydra/ to load (default: the one '
        "`just plot-hetero` visualizes).",
    )
    parser.add_argument(
        '--override',
        action='append',
        default=[],
        dest='overrides',
        help='Extra Hydra override, e.g. --override dataset.n_agents=20 '
        '(repeatable).',
    )
    parser.add_argument(
        '--out_dir', type=Path, default=Path('results/multi_agent/plots')
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config_name, args.overrides)
    degrees = compute_degrees(cfg)

    n = len(degrees)
    vals = list(degrees.values())
    print(f'Communication graph ({cfg.graph.neighbors_mode}), {n} agents:')
    print(
        f'  degree: min={min(vals)}, max={max(vals)}, '
        f'mean={sum(vals) / n:.2f}, std={np.std(vals):.2f}'
    )

    plot_degree_distribution(
        degrees, args.out_dir, title=f'Degree distribution — {args.config_name}'
    )


if __name__ == '__main__':
    main()
