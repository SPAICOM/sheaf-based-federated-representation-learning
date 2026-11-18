r"""Score the anchor-selection ablation by class scope, offline from final checkpoints (no W&B writes).

Scopes pooled over the 12 directed edges (i = receiver, j = sender): Comm (all test samples), Comm on C_i \ C_j (sender
unseen / receiver seen), Comm on C_i (receiver seen), Comm on C_i ∩ C_j (both seen), plus Private and Private on C_i.
Rows scored (appended to results/mnist_4agents_shift/anchors/per_run_scored.csv, resumable):
  * DiSheafFRL runs of mnist_4agents_shift_anchors_{kmeans,strat} (posthoc under their own anchor selection);
  * non-coop 4096-pilot checkpoints rescored under each (arm, K);
  * the original p128 runs (DiSheafFRL cnorm_p128_l.01, non-coop ..._p128) of mnist_4agents_shift, arm 'original'.
Original K=384/768/1536 come from results/mnist_4agents_shift/swbn_scopes/per_run.csv. Overall Comm is asserted vs the
remote test metric (2e-3) wherever a remote value for the same eval exists. Aggregation + figures: plot_mnist4_anchors.py.
Usage: uv run python scripts/score_mnist4_anchors.py [--plan] [--limit N] [--noncoop-seeds 42 43 44]
"""

import argparse
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

os.environ['WANDB_MODE'] = 'disabled'
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

OUT = (
    ROOT / 'results' / 'mnist_4agents_shift' / 'anchors' / 'per_run_scored.csv'
)
SCOPES = [
    'Comm',
    'Comm sender unseen / receiver seen',
    'Comm receiver seen',
    'Comm sender seen / receiver seen',
]
PRIV_SEEN = 'Private target classes'
# original (anchor_selection all, K pilots) K=128 runs: (project, group) -> method
ORIGINAL = {
    ('mnist_4agents_shift', 'cnorm_p128_l.01'): 'DiSheafFRL',
    (
        'mnist_4agents_shift',
        'mnist4_di_clean_bs256_pilots_non_cooperative_p128',
    ): 'Non-coop',
}
NONCOOP_SRC = (
    'mnist_4agents_shift',
    'mnist4_di_clean_bs256_check_non_cooperative_p4096',
)
ARMS = {'kmeans': 'proto_kmeans', 'strat': 'stratified'}
TOL = 2e-3
# DiSheafFRL source groups per arm: group -> K (the *_warm groups are k-means with warm start).
GROUPS = {
    'kmeans': {
        'K128': 128,
        'K128_warm': 128,
        'K384': 384,
        'K768_warm': 768,
        'K1536_warm': 1536,
    },
    'strat': {'K128': 128, 'K384': 384, 'K768': 768, 'K1536': 1536},
}


def posthoc(orch, dm):
    """Context-free swap-in of the post-hoc maps; returns a cleanup callable."""
    if hasattr(orch, '_POSTHOC_METHODS'):
        for name in orch._POSTHOC_METHODS:
            object.__setattr__(
                orch,
                name,
                types.MethodType(
                    getattr(PostTrainingAlignmentMixin, name), orch
                ),
            )
    assert orch._fit_alignment_maps(dm)

    def cleanup():
        orch._cleanup_alignment()
        for name in getattr(orch, '_POSTHOC_METHODS', ()):
            orch.__dict__.pop(name, None)

    return cleanup


def scope_counts(orch, dm, cfg, neighbors):
    """Pooled correct/total per scope plus private accuracy."""
    seen = {
        i: torch.tensor(list(cfg.dataset.groups[str(i)].target_classes))
        for i in range(4)
    }
    counts = {k: [0, 0] for k in SCOPES}
    priv = [0, 0]
    priv_seen = [0, 0]
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
            ok = orch.agents[str(sender)].decoder(z).argmax(1).cpu() == y
            priv_seen[0] += int((ok & ss).sum())
            priv_seen[1] += int(ss.sum())
            for receiver in sorted(neighbors[sender]):
                msg = orch.send_message(sender, receiver, z).float()
                pred = orch.agents[str(receiver)].decoder(msg).argmax(1).cpu()
                rs = torch.isin(y, seen[receiver])
                for key, mask in [
                    (SCOPES[0], torch.ones_like(ss)),
                    (SCOPES[1], ~ss & rs),
                    (SCOPES[2], rs),
                    (SCOPES[3], ss & rs),
                ]:
                    counts[key][0] += int(((pred == y) & mask).sum())
                    counts[key][1] += int(mask.sum())
    vals = {k: a / b for k, (a, b) in counts.items()}
    vals[PRIV_SEEN] = priv_seen[0] / priv_seen[1]
    return vals, priv[0] / priv[1]


def load(project, run, cfg):
    """Orchestrator for ``cfg`` with the final checkpoint of ``run``; the dm/seed come from cfg (seed_everything)."""
    dm, neighbors = _setup(cfg)
    agents, dims = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    orch = _build_orchestrator(
        cfg, agents, neighbors, dims, _parse_per_agent_cfg(cfg)
    )
    ckpt = list(
        (ROOT / 'logs' / project / run.id / 'checkpoints').glob(
            'epoch=99*.ckpt'
        )
    )
    assert len(ckpt) == 1, (run.id, ckpt)
    orch.load_state_dict(
        torch.load(ckpt[0], map_location='cpu', weights_only=False)[
            'state_dict'
        ],
        strict=True,
    )
    orch.to('cuda').eval()
    orch._trainer = SimpleNamespace(datamodule=dm)
    return orch, dm, neighbors


def jobs(api, entity):
    """All scoring jobs: (method, arm, group, K, seed, project, run, anchor overrides or None)."""
    out = []
    for arm, groups in GROUPS.items():
        project = f'mnist_4agents_shift_anchors_{arm}'
        try:
            runs = list(api.runs(f'{entity}/{project}'))
        except ValueError as exc:  # project not created yet
            if 'Could not find project' not in str(exc):
                raise
            continue
        for r in runs:
            if r.group in groups and r.state == 'finished':
                r = api.run(f'{entity}/{project}/{r.id}')
                if (
                    r.summary.get('epoch', 0) >= 99
                    or r.summary.get('test/avg_comm_task_perf') is not None
                ):
                    out.append(
                        (
                            'DiSheafFRL',
                            arm,
                            r.group,
                            groups[r.group],
                            r.config['seed'],
                            project,
                            r,
                            None,
                        )
                    )
    src_project, src_group = NONCOOP_SRC
    src = [
        api.run(f'{entity}/{src_project}/{r.id}')
        for r in api.runs(f'{entity}/{src_project}')
        if r.group == src_group
    ]
    out += [
        (
            'Non-coop',
            arm,
            f'noncoop_K{k}',
            k,
            r.config['seed'],
            src_project,
            r,
            (ARMS[arm], k),
        )
        for r in src
        for arm in ARMS
        for k in (128, 384, 768, 1536)
    ]
    for (project, group), method in ORIGINAL.items():
        for r in api.runs(f'{entity}/{project}'):
            if r.group == group and r.state == 'finished':
                r = api.run(f'{entity}/{project}/{r.id}')
                out.append(
                    (
                        method,
                        'original',
                        group,
                        128,
                        r.config['seed'],
                        project,
                        r,
                        None,
                    )
                )
    return out


def main():
    """Plan or run the scoring."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--limit', type=int)
    ap.add_argument(
        '--noncoop-seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44],
        help='non-coop rescoring seeds',
    )
    ap.add_argument('--methods', nargs='+', default=['DiSheafFRL', 'Non-coop'])
    args = ap.parse_args()
    api = wandb.Api()
    entity = api.default_entity
    done = (
        pd.read_csv(OUT)
        if OUT.exists()
        else pd.DataFrame(columns=['method', 'arm', 'group', 'seed'])
    )
    have = {(r.method, r.arm, r.group, int(r.seed)) for r in done.itertuples()}
    todo = [
        j
        for j in jobs(api, entity)
        if j[0] in args.methods
        and (j[7] is None or int(j[4]) in args.noncoop_seeds)
        and (j[0], j[1], j[2], int(j[4])) not in have
    ]
    todo.sort(key=lambda j: (j[0], j[1], j[3], j[4]))
    print(f'{len(todo)} to score, {len(have)} done', flush=True)
    if args.plan:
        for j in todo:
            print(j[0], j[1], j[2], 'K', j[3], 's', j[4], j[6].id)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for method, arm, group, k, seed, project, run, override in todo[
        : args.limit
    ]:
        cfg = OmegaConf.create(run.config)
        if override:
            cfg.orchestrator.anchor_selection, cfg.orchestrator.num_anchors = (
                override
            )
            cfg.orchestrator.stratified_bias = 0.7
        orch, dm, neighbors = load(project, run, cfg)
        cleanup = posthoc(orch, dm)
        try:
            vals, priv = scope_counts(orch, dm, cfg, neighbors)
        finally:
            cleanup()
        remote = (
            run.summary.get('test/avg_comm_task_perf')
            if override is None
            else None
        )
        if arm == 'original':
            assert cfg.dataset.pilot_num_samples == k
        if remote is not None:
            assert abs(vals['Comm'] - remote) < TOL, (
                run.id,
                vals['Comm'],
                remote,
            )
        assert abs(priv - run.summary['test/avg_private_task_perf']) < 2e-5, (
            run.id,
            priv,
        )
        row = {
            'method': method,
            'arm': arm,
            'group': group,
            'K': k,
            'seed': seed,
            'source_run': run.id,
            'private': priv,
            **vals,
            'remote_comm': remote,
        }
        pd.DataFrame([row]).to_csv(
            OUT, mode='a', header=not OUT.exists(), index=False
        )
        print(
            method,
            arm,
            group,
            'K',
            k,
            's',
            seed,
            {s: round(v, 4) for s, v in vals.items()},
            flush=True,
        )
        del orch, dm
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
