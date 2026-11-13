"""Re-score finished heterogeneity runs from their checkpoints, one protocol for every method.

For each finished run of the selected wandb groups (one group per method),
the final checkpoint (logs/<project>/<run_id>/checkpoints/) is reloaded into
the run's orchestrator and evaluated on the shared test split with the
post-hoc pipeline of ``PostTrainingAlignmentMixin`` (one map per edge fitted
on the pilots, ``alignment_method`` of the run), the SheafFRL family included.
ComFed is the exception: it communicates through its own learned projections
(``ComFed.send_message``), so no post-hoc map is fitted and no sheaf Laplacian
is built for it (no tv / lmin).

    private          agent's own decoder on its test latents (mean over agents)
    private_seen     same, only the classes the agent trained on
    comm             receiver's decoder on the sender's mapped test latents
                     (pooled over the directed edges)
    comm_rx_seen     comm on classes the receiver trained on, whether the
                     sender did or not (sender-only classes excluded)
    comm_rx_only     comm on classes only the receiver trained on (the
                     sender never saw them)
    mrr              instance-retrieval MRR on the test set through the maps
    misalign         coboundary residual per edge / (d_s + d_r), mean over
                     edges (both from ``evaluate_misalignment_loss``; for
                     ComFed the same formulas with its own transport of the
                     lower-dim receiver into the higher-dim sender, see
                     ``transport_misalignment``)
    tv, lmin         sheaf total variation (mean over test samples of
                     x^T L x) and smallest eigenvalue of the sheaf Laplacian
                     built from the post-hoc maps (src/utils/sheaf_laplacian.py)
    tv_learned,      same from the learned restriction maps (SheafFRL /
    lmin_learned     DiSheafFRL only)

Recomputed ``private`` / ``comm`` are checked against the logged
``test/avg_*_task_perf`` (|Δ| reported).
Per-run results are cached in results/<project>/rescore/<run_id>.json
(``--force`` recomputes); the table (mean ± std over seeds, in %) goes to
results/<project>/rescore/table.{md,csv}.

    uv run python scripts/rescore_hetero.py cifar10_4agents_full
    uv run python scripts/rescore_hetero.py cifar10_4agents_full --groups comfed,sheaf_frl --force
"""

import os

os.environ['WANDB_MODE'] = 'disabled'  # rebuilding the orchestrator must not start a run
import argparse
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _build_orchestrator,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

from src.communication.alignment_mixin import (  # noqa: E402
    PostTrainingAlignmentMixin,
)
from src.orchestrators.sheaf_frl import SheafFRL  # noqa: E402
from src.utils.class_scope import ClassScopeAccuracy  # noqa: E402
from src.utils.sheaf_laplacian import (  # noqa: E402
    min_eigenvalue,
    sheaf_laplacian,
    total_variation,
)

# config name (= wandb group) -> label, in table order
METHODS = {
    'non_cooperative': 'Non-coop',
    'fedre': 'FedRE',
    'fedproto': 'FedProto',
    'fedmuscle': 'FedMuscle',
    'comfed': 'ComFed',
    'sheaf_fmtl': 'Sheaf-FMTL',
    'sheaf_frl': 'SheafFRL',
    'di_sheaf_frl': 'DiSheafFRL',
}
# column -> header; accuracies and MRR in %, the rest raw
COLUMNS = {
    'private': 'private',
    'private_seen': 'private seen',
    'comm': 'comm',
    'comm_rx_seen': 'comm RX seen',
    'comm_rx_only': 'comm only RX seen',
    'mrr': 'MRR',
    'misalign': 'misalign',
    'tv': 'TV',
    'lmin': 'λ_min',
    'tv_learned': 'TV learned',
    'lmin_learned': 'λ_min learned',
}
PERCENT = {
    'private', 'private_seen', 'comm', 'comm_rx_seen', 'comm_rx_only', 'mrr',
}
POSTHOC = SheafFRL._POSTHOC_METHODS  # the mixin methods that make up the pipeline


def load(run, project: str, device: str):
    """Orchestrator and datamodule of one finished run, final checkpoint loaded."""
    ckpts = sorted((ROOT / 'logs' / project / run.id / 'checkpoints').glob('*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f'no checkpoint under logs/{project}/{run.id}/checkpoints')
    cfg = OmegaConf.create(run.config)
    dm, neighbors = _setup(cfg)
    per_agent = _parse_per_agent_cfg(cfg)
    agents, dims = _build_agents(cfg, dm, per_agent)
    orch = _build_orchestrator(cfg, agents, neighbors, dims, per_agent)
    state = torch.load(ckpts[-1], map_location='cpu', weights_only=False)['state_dict']
    orch.load_state_dict(state, strict=True)
    orch.to(device).eval()
    orch._trainer = SimpleNamespace(datamodule=dm)
    return orch, dm, cfg


def bind_posthoc(orch) -> None:
    """Give SheafFRL the post-hoc pipeline (``SheafFRL._evaluate_with_posthoc_alignment``)."""
    if isinstance(orch, PostTrainingAlignmentMixin):
        return
    for name in POSTHOC:
        object.__setattr__(
            orch, name, types.MethodType(getattr(PostTrainingAlignmentMixin, name), orch)
        )


def _collate(batch):
    xs = [to_tensor(b[0]) if isinstance(b[0], Image.Image) else b[0] for b in batch]
    ys = [torch.as_tensor(b[1]) for b in batch]
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def encode(agent, dataset, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval-mode latents and labels of one dataset."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=_collate
    )
    Z, Y = [], []
    for x, y in loader:
        Z.append(agent.encode(x.to(device)).float())
        Y.append(y)
    return torch.cat(Z), torch.cat(Y)


def canonical_edges(neighbors: dict[int, set[int]], dims: dict[int, int]):
    """One (sender, receiver) per undirected edge: sender = larger dim, ties larger index."""
    edges = set()
    for i, nbrs in neighbors.items():
        for j in nbrs - {i}:
            edges.add((i, j) if (dims[i], i) > (dims[j], j) else (j, i))
    return sorted(edges)


@torch.no_grad()
def transport_misalignment(orch, Z, neighbors) -> dict[str, float]:
    """``evaluate_misalignment_loss`` with the orchestrator's own transport.

    Per canonical edge (s, r): X = Z_s, AY = send_message(r -> s, Z_r);
    misalign = mean ||X - AY||² / (d_s + d_r), MRR = reciprocal rank of the
    matching row of AY among all rows by cosine similarity to X. Used for
    ComFed, which has no post-hoc embedding map.
    """
    dims = {i: z.shape[1] for i, z in Z.items()}
    mis, mrr = [], []
    for s, r in canonical_edges(neighbors, dims):
        X, AY = Z[s], orch.send_message(r, s, Z[r]).float()
        mis.append(float(((X - AY) ** 2).sum(1).mean()) / (dims[s] + dims[r]))
        a, b = torch.nn.functional.normalize(X, dim=1), torch.nn.functional.normalize(AY, dim=1)
        ranks = []
        for start in range(0, len(a), 1024):
            sim = a[start:start + 1024] @ b.T
            rows = torch.arange(sim.shape[0], device=sim.device)
            ranks.append((sim >= sim[rows, rows + start].unsqueeze(1)).sum(1).clamp(min=1))
        mrr.append(float((1.0 / torch.cat(ranks).float()).mean()))
    return {'misalign': sum(mis) / len(mis), 'mrr': sum(mrr) / len(mrr)}


@torch.no_grad()
def score(run, project: str, device: str) -> dict[str, float]:
    """All table metrics of one run (see the module docstring)."""
    orch, dm, cfg = load(run, project, device)
    o = cfg.orchestrator
    if o.get('posthoc_whitening', True) or o.get('whitening', False) or o.get(
        'learn_whitening', False
    ):
        raise ValueError(
            f'{run.id}: TV/λ_min assume raw latents (no whitening); got '
            f'posthoc_whitening={o.get("posthoc_whitening")}, whitening={o.get("whitening")}'
        )
    out: dict[str, float] = {}
    agents = {int(k): a for k, a in orch.agents.items()}
    neighbors = {int(k): {int(n) for n in v} for k, v in orch.hparams.neighbors.items()}
    native = hasattr(orch, 'projection_matrices')  # ComFed: its own projections
    if not native:
        bind_posthoc(orch)
        if not orch._fit_alignment_maps(dm):
            raise RuntimeError(f'{run.id}: no post-hoc map fitted')
    try:
        Z, y = {}, None
        for i, agent in agents.items():
            agent.eval()
            Z[i], yi = encode(agent, dm.test_datasets[i], device)
            if y is not None and not torch.equal(y, yi):
                raise ValueError(f'{run.id}: test sets differ across agents')
            y = yi
        dims = {i: Z[i].shape[1] for i in Z}
        scope = ClassScopeAccuracy(orch._resolve_agent_target_classes())
        for i, agent in agents.items():
            scope.add_private(i, agent.decoder(Z[i]), y)
        for i, receivers in neighbors.items():
            for j in receivers - {i}:
                scope.add_comm(i, j, agents[j].decoder(orch.send_message(i, j, Z[i])), y)
        acc = scope.logs('test')
        seen, unseen = scope.counts['private_seen'], scope.counts['private_unseen']
        rx_seen = ('comm_both_seen', 'comm_tx_unseen_rx_seen')  # pooled over edges
        out.update(
            private=(seen[0] + unseen[0]) / (seen[1] + unseen[1]),
            private_seen=acc['test/scope/private_seen'],
            comm=acc['test/scope/comm_all'],
            comm_rx_seen=sum(scope.counts[k][0] for k in rx_seen)
            / sum(scope.counts[k][1] for k in rx_seen),
            comm_rx_only=acc['test/scope/comm_tx_unseen_rx_seen'],
        )
        if native:
            out.update(transport_misalignment(orch, Z, neighbors))
        else:
            mis = orch.evaluate_misalignment_loss(dm, prefix='test')
            out['mrr'] = mis['test/mean_reciprocal_rank']
            out['misalign'] = mis['test/misalignment_loss']
            L = sheaf_laplacian(dict(orch._embedding_maps), dims)
            out['tv'] = total_variation(
                L, {i: orch._whiten_own_latents(i, Z[i].cpu()) for i in Z}
            )
            out['lmin'] = min_eigenvalue(L)
    finally:
        if not native:
            orch._cleanup_alignment()
            for name in POSTHOC:
                orch.__dict__.pop(name, None)

    if hasattr(orch, 'stiefel_matrices'):  # SheafFRL family: learned restriction maps
        learned = {
            tuple(int(t) for t in key.split('_')): V.detach().float().T
            for key, V in orch.stiefel_matrices.items()
        }
        L = sheaf_laplacian(learned, dims)
        out['tv_learned'] = total_variation(L, {i: Z[i].cpu() for i in Z})
        out['lmin_learned'] = min_eigenvalue(L)

    for key, logged in (('private', 'test/avg_private_task_perf'), ('comm', 'test/avg_comm_task_perf')):
        if logged in run.summary:
            out[f'delta_{key}'] = out[key] - float(run.summary[logged])
    return out


def table(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds per method, accuracies and MRR in %."""
    lines = []
    for group, label in METHODS.items():
        sub = rows[rows.group == group]
        if sub.empty:
            continue
        line = {'method': label, 'seeds': len(sub)}
        for col, head in COLUMNS.items():
            if col not in sub or sub[col].isna().all():
                line[head] = ''
                continue
            v = sub[col].dropna() * (100 if col in PERCENT else 1)
            fmt = '{:.1f} ± {:.1f}' if col in PERCENT else '{:.3g} ± {:.2g}'
            line[head] = fmt.format(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
        lines.append(line)
    return pd.DataFrame(lines)


def markdown(df: pd.DataFrame) -> str:
    """Plain GitHub-markdown table."""
    lines = ['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)]
    lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Score every finished run of the groups (cached) and write the table."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('project')
    parser.add_argument('--groups', default=','.join(METHODS), help='comma-separated wandb groups')
    parser.add_argument('--force', action='store_true', help='recompute cached runs')
    args = parser.parse_args()

    out_dir = ROOT / 'results' / args.project / 'rescore'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    api = wandb.Api()
    path = f'{api.default_entity}/{args.project}'

    rows = []
    for group in args.groups.split(','):
        for listed in api.runs(path, filters={'group': group, 'state': 'finished'}):
            run = api.run(f'{path}/{listed.id}')
            cache = out_dir / f'{run.id}.json'
            if cache.exists() and not args.force:
                res = json.loads(cache.read_text())
            else:
                res = score(run, args.project, device)
                cache.write_text(json.dumps(res, indent=1))
                torch.cuda.empty_cache()
            seed = run.config.get('seed')
            deltas = ', '.join(f'{k} {v:+.4f}' for k, v in res.items() if k.startswith('delta_'))
            print(f'{group:16s} {run.id} seed {seed}: comm {res["comm"]:.4f} ({deltas})', flush=True)
            rows.append({'group': group, 'run_id': run.id, 'seed': seed, **res})

    df = table(pd.DataFrame(rows))
    df.to_csv(out_dir / 'table.csv', index=False)
    (out_dir / 'table.md').write_text(markdown(df))
    print(markdown(df))
    print(f'-> {out_dir}/table.md')


if __name__ == '__main__':
    main()
