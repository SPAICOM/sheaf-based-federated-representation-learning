"""Post-hoc alignment quality vs number of pilots, with the encoders held FIXED.

The shift x pilots table mixes two effects in every cell: the pilot budget
used by the post-hoc fit AND a different training run (different encoders,
one seed each). This script removes the second: it loads one checkpoint,
rebuilds that run's datamodule, and refits the post-hoc pipeline (pilot
whitening + one Procrustes map per edge) on random subsets of n of the run's
pilots, then scores comm accuracy / misalignment on the shared test set.

Use a checkpoint trained with the LARGEST pilot budget, so none of the
subsampled pilots was ever a training image of any agent. The same subset of
sample ids is used on every agent (pilots are shared).

Usage (run spec = run_id:shift:pilots:orchestrator; empty shift/pilots keep the
config's values, and checkpoints are read from logs/<logger.project> by default):
    uv run python scripts/pilot_count_sweep.py --config-name cifar10_2agents_crops --run 7b0ggpe4:::non_cooperative
    uv run python scripts/pilot_count_sweep.py --run ecmzv1oi:0.8:2048:non_cooperative
    uv run python scripts/pilot_count_sweep.py --run ecmzv1oi:0.8:2048:non_cooperative \
        --overrides orchestrator.posthoc_whitening_shrinkage=auto --csv results/pilot_count_sweep/pilot_count_sweep_lw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).parent))
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _PilotSubset(torch.utils.data.Dataset):
    def __init__(self, ds, positions):
        self.ds, self.positions = ds, list(positions)
        self.sample_ids = [ds.sample_ids[p] for p in self.positions]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        return self.ds[self.positions[i]]


class _DMView:
    def __init__(self, dm, pilot_datasets):
        self._dm, self.pilot_datasets = dm, pilot_datasets

    def __getattr__(self, name):
        return getattr(self._dm, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', nargs='+', required=True, help='run_id:shift:pilots:orchestrator_config')
    parser.add_argument('--config-name', default='cifar10_4agents_shift')
    parser.add_argument(
        '--project_dir', type=Path, default=None,
        help='checkpoint root (default: logs/<logger.project> of the composed config)',
    )
    parser.add_argument('--n', nargs='+', type=int, default=[64, 128, 192, 256, 320, 384, 512, 768, 1024, 2048])
    parser.add_argument('--repeats', type=int, default=2, help='random subsets per n')
    parser.add_argument('--csv', type=Path, default=ROOT / 'results/pilot_count_sweep/pilot_count_sweep.csv')
    parser.add_argument(
        '--overrides', nargs='*', default=[],
        help='extra hydra overrides, e.g. orchestrator.posthoc_whitening_shrinkage=auto',
    )
    args = parser.parse_args()

    rows = []
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        for spec in args.run:
            run_id, shift, pilots, orch_name = spec.split(':')
            cfg = compose(args.config_name, overrides=[
                f'orchestrator={orch_name}',
                *([f'dataset.shift_strength={shift}'] if shift else []),
                *([f'dataset.pilot_num_samples={pilots}'] if pilots else []),
                *args.overrides,
            ])
            project_dir = args.project_dir or ROOT / 'logs' / cfg.logger.project
            dm, neighbors = _setup(cfg)
            per_agents_cfg = _parse_per_agent_cfg(cfg)
            agents, latent_dims = _build_agents(cfg, dm, per_agents_cfg)
            orch = _build_orchestrator(cfg, agents, neighbors, latent_dims, per_agents_cfg)
            ckpt = sorted((project_dir / run_id / 'checkpoints').glob('*.ckpt'))[-1]
            orch.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'])
            orch = orch.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
            n_pool = len(next(iter(dm.pilot_datasets.values())))
            # budgets above the pool size mean "all pilots"
            for n in sorted({min(n, n_pool) for n in args.n}):
                for rep in range(args.repeats if n < n_pool else 1):
                    g = torch.Generator().manual_seed(1000 * rep + n)
                    pos = sorted(torch.randperm(n_pool, generator=g)[:n].tolist())
                    view = _DMView(dm, {i: _PilotSubset(d, pos) for i, d in dm.pilot_datasets.items()})
                    # prefix 'sweep': comm accuracy + misalignment on the shared
                    # test set, skipping the (costly) 10k x 10k MRR ranking.
                    logs = orch.evaluate_communication_accuracy(view, prefix='sweep')
                    row = {
                        'run': run_id, 'orch': orch_name, 'shift': float(cfg.dataset.shift_strength),
                        'n_pilots': n, 'rep': rep,
                        'comm': logs.get('sweep/avg_comm_task_perf'),
                        'misalign': logs.get('sweep/misalignment_loss'),
                    }
                    rows.append(row)
                    print(row, flush=True)
            del dm, orch
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(df.groupby(['orch', 'shift', 'n_pilots'])[['comm', 'misalign']].mean().round(4).to_string())
    print(f'\nsaved → {args.csv}')


if __name__ == '__main__':
    main()
