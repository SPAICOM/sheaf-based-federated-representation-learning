"""Add the seen/unseen-class metrics to finished runs that predate them.

Runs logged before src/utils/class_scope.py have no ``test/scope/*`` metrics.
This rebuilds each run from its wandb config, loads its final checkpoint
(logs/<project>/<run_id>/checkpoints/), re-runs the test evaluation (same
post-hoc Procrustes evaluation as training) and compares the recomputed
``test/avg_comm_task_perf`` / ``test/avg_private_task_perf`` with the logged
ones. With ``--write`` the ``test/scope/*`` keys are added to the run's wandb
summary, only when both differences are below ``--tol`` (accuracy, default
0.002). Without ``--write`` nothing is sent to wandb.

    uv run python scripts/rescore_class_scope.py cifar10_4agents_full 3mgkbida x2b2r0f4
    uv run python scripts/rescore_class_scope.py cifar10_4agents_full --groups non_cooperative,di_sheaf_frl --write
"""

import os

os.environ['WANDB_MODE'] = (
    'disabled'  # rebuilding the orchestrator must not start a run
)
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

CHECKED = ('test/avg_comm_task_perf', 'test/avg_private_task_perf')


def rescore(run, project: str, device: str) -> dict[str, float]:
    """Recomputed test logs of one finished run, from its final checkpoint."""
    checkpoints = sorted(
        (ROOT / 'logs' / project / run.id / 'checkpoints').glob('*.ckpt')
    )
    if not checkpoints:
        raise FileNotFoundError(
            f'no checkpoint under logs/{project}/{run.id}/checkpoints'
        )
    cfg = OmegaConf.create(run.config)
    dm, neighbors = _setup(cfg)
    per_agent = _parse_per_agent_cfg(cfg)
    agents, dims = _build_agents(cfg, dm, per_agent)
    orch = _build_orchestrator(cfg, agents, neighbors, dims, per_agent)
    state = torch.load(
        checkpoints[-1], map_location='cpu', weights_only=False
    )['state_dict']
    orch.load_state_dict(state, strict=True)
    orch.to(device).eval()
    orch._trainer = SimpleNamespace(datamodule=dm)
    with torch.no_grad():
        return dict(orch.evaluate_communication_accuracy(dm, prefix='test'))


def main() -> None:
    """Rescore the selected runs and optionally write the scope metrics."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('project')
    parser.add_argument(
        'run_ids',
        nargs='*',
        help='run ids (default: every finished run of --groups)',
    )
    parser.add_argument(
        '--groups',
        default=None,
        help='comma-separated wandb groups to rescore',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='add test/scope/* to the wandb summaries',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='also rescore runs that already have test/scope/*',
    )
    parser.add_argument('--tol', type=float, default=0.002)
    args = parser.parse_args()

    api = wandb.Api()
    path = f'{api.default_entity}/{args.project}'
    if args.run_ids:
        runs = [api.run(f'{path}/{rid}') for rid in args.run_ids]
    elif args.groups:
        groups = set(args.groups.split(','))
        runs = [
            r
            for r in api.runs(path, filters={'state': 'finished'})
            if r.group in groups
        ]
    else:
        raise SystemExit('give run ids or --groups')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for run in runs:
        if not args.overwrite and any(
            k.startswith('test/scope/') for k in run.summary
        ):
            print(
                f'{run.id} ({run.group}): already has test/scope metrics, skipped'
            )
            continue
        logs = rescore(run, args.project, device)
        diffs = {
            k: abs(logs[k] - run.summary[k])
            for k in CHECKED
            if k in logs and k in run.summary
        }
        scope = {k: v for k, v in logs.items() if k.startswith('test/scope/')}
        ok = (
            len(diffs) == len(CHECKED)
            and max(diffs.values()) <= args.tol
            and scope
        )
        report = ', '.join(
            f'{k.split("/")[-1]} |Δ|={v:.4f}' for k, v in diffs.items()
        )
        print(
            f'{run.id} ({run.group}, seed {run.config.get("seed")}): {report}; '
            f'comm tx-unseen/rx-seen {scope.get("test/scope/comm_tx_unseen_rx_seen", float("nan")):.4f}, '
            f'{len(scope)} scope keys -> {"OK" if ok else "MISMATCH, not written"}'
        )
        if args.write and ok:
            run.summary.update(scope)
            print(f'  written to {path}/{run.id}')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
