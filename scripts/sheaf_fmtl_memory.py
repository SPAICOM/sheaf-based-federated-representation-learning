"""RAM SheafFMTL needs for its projection matrices, per gamma, for one experiment config.

SheafFMTL keeps one matrix P_ij (d_ij x d_i, d_i = agent i's parameter count,
d_ij = max(1, int(gamma * min(d_i, d_j)))) per directed edge, in fp16 on the
CPU, and upcasts one edge at a time to fp32 for the epoch-end update (matrix
plus outer-product update: 2 fp32 copies of the largest edge). Peak RAM of a
run ≈ resting + transient + ~2 GB for the process itself (measured: CIFAR-10
4 agents, gamma 1e-3 → 6.9 GB). Pick the largest gamma that fits and set
``sheaf_fmtl_gamma`` in the experiment config.

    uv run python scripts/sheaf_fmtl_memory.py cifar10_4agents_full fashionmnist_4agents_shift
"""

import os

os.environ['WANDB_MODE'] = 'disabled'
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from hydra import compose, initialize_config_dir  # noqa: E402
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

GAMMAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
PROCESS_GB = 2.0


def main() -> None:
    """Print parameter counts and the memory table of every config given."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='+', help='experiment config names')
    args = parser.parse_args()
    for name in args.configs:
        with initialize_config_dir(str(ROOT / 'config' / 'hydra'), version_base=None):
            cfg = compose(name, overrides=[
                'orchestrator=sheaf_fmtl', 'orchestrator.max_lmb=0', 'study_name=x',
            ])
        dm, neighbors = _setup(cfg)
        agents, _ = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
        d = {
            int(i): sum(p.numel() for p in a.parameters() if p.requires_grad)
            for i, a in agents.items()
        }
        edges = [(i, j) for i in neighbors for j in neighbors[i]]
        print(f'{name}: parameters per agent {d}, {len(edges)} directed edges')
        print('  gamma    min d_ij  P fp16 GB  update fp32 GB  peak run GB')
        for g in GAMMAS:
            size = {(i, j): max(1, int(g * min(d[i], d[j]))) * d[i] for i, j in edges}
            rest, step = sum(size.values()) * 2 / 1e9, max(size.values()) * 8 / 1e9
            d_ij = min(max(1, int(g * min(d[i], d[j]))) for i, j in edges)
            print(f'  {g:<8g} {d_ij:>8d} {rest:>10.2f} {step:>15.2f} {rest + step + PROCESS_GB:>12.1f}')


if __name__ == '__main__':
    main()
