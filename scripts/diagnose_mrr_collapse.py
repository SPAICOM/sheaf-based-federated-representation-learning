"""Diagnose *why* pilot MRR and misalignment loss disagree across the comm ablation.

The ``comm_ablation`` table shows two things that look contradictory:

  * ``sheaf_frl`` drives ``test/misalignment_loss`` to ~0.002 (28x lower than
    the post-hoc ``non_cooperative`` Procrustes baseline) yet scores a *lower*
    ``pilots/mean_reciprocal_rank`` (0.30 vs 0.57);
  * within the ZCA CESheafFRL families, MRR *falls* monotonically as
    misalignment *falls* with more communication.

Both metrics are computed by ``BaseOrchestrator.evaluate_misalignment_loss`` on
the same rows, in the same whitened space, with the same single stored map per
edge -- so the disagreement is not a scoping artefact.  It is a difference in
*what the two numbers are sensitive to*:

  ``misalignment_loss``  absolute residual, ``E||Z_i M - Z_j||^2 / (2 d_e)``.
  ``MRR``                a *ranking* of the transported row against every other
                         receiver pilot row by cosine similarity -- i.e. the
                         residual measured **relative to how far apart distinct
                         pilot rows are** in the receiver's whitened space.

So a penalty that pulls the two agents' whitened clouds onto each other while
also squeezing the rows of a *class* together drives the residual down and the
rank up at the same time.  This script measures exactly that, per edge:

  ``mrr``          the logged metric, reproduced (all rows are distractors).
  ``mrr_cross``    MRR with every *same-class* distractor removed, so only
                   rows the model is not being asked to merge can outrank the
                   match.  If a method's MRR deficit is same-class collapse,
                   ``mrr_cross`` closes the gap; if its map is genuinely worse,
                   the gap survives.
  ``same_cls_err`` share of top-1 retrieval errors that land on a *same-class*
                   row -- the direct fingerprint of class-consensus collapse.
  ``resid/nn``     RMS alignment residual divided by the mean nearest-other-row
                   distance in the receiver's whitened cloud: the scale-free
                   quantity retrieval actually depends on. > 1 means the
                   residual is larger than the gap between distinct samples,
                   so retrieval must fail however small the residual is.
  ``cos_same``/``cos_diff``  mean cosine similarity between distinct receiver
                   rows of the same / different class -- how much room the
                   ranking has left.
  ``erank_*``      effective rank (entropy of the normalised spectrum) of each
                   side's whitened pilot cloud.

It also draws the t-SNE of one edge's whitened pilots -- the transported sender
rows and the receiver rows in one joint embedding, coloured by class -- for
every model, which is the qualitative version of the same story.

Checkpoints are read from ``logs/comm_ablation/<run_id>/checkpoints/*.ckpt``;
the run ids are the wandb ids of the rows in ``results/comm_ablation/table.md``.
SWBN (``learn_whitening=true``) runs carry their whitening in the checkpoint;
``learn_whitening=false`` runs fit their closed-form ZCA from training latents
during training, which is not checkpointed, so it is refit here from the train
split exactly as the post-hoc baselines do (flagged in the output).

Usage:
    uv run scripts/diagnose_mrr_collapse.py
    uv run scripts/diagnose_mrr_collapse.py --runs sheaf_frl=e0q8m6u7 noncoop=gplav88p
    uv run scripts/diagnose_mrr_collapse.py --edge 2 11 --no-tsne
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(sys.path[0]).parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate
from lightning import seed_everything
from omegaconf import OmegaConf
from PIL import Image as _PILImage
from sklearn.manifold import TSNE
from torchvision.transforms.functional import to_tensor as _pil_to_tensor

from src.communication.whitening import common_pilot_indices

REPO = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO / 'logs' / 'comm_ablation'
CONFIG_DIR = REPO / 'config' / 'hydra'
CONFIG_NAME = 'multiagent_mnist_comm_ablation'

# label -> wandb run id.  These are the runs behind the rows of
# results/comm_ablation/table.md (see `uv run scripts/comm_ablation_table.py`).
# NOTE: these five all carry their whitening in the checkpoint (SWBN) or refit
# it exactly as they did at eval time (the post-hoc baseline), so every row
# reproduces its logged ``pilots/mean_reciprocal_rank``.  The learn_whitening=
# false (ZCA) cells of the table do NOT -- their training-time ZCA operator is
# fit from the task-latent buffer and never checkpointed -- so they are omitted
# here; pass them explicitly if you want them, and read them as indicative only.
DEFAULT_RUNS: dict[str, str] = {
    'non_coop (post-hoc)': 'gplav88p',
    'sheaf_frl every-step': 'e0q8m6u7',
    'ce p=2 reg=T': '7s4qu163',
    'ce p=90 reg=T': 'vunctvsb',
    'ce p=90 reg=F': 'yvjuje61',
}


def _collate_xy(batch):
    xs, ys = [], []
    for item in batch:
        x = item[0]
        if isinstance(x, _PILImage.Image):
            x = _pil_to_tensor(x)
        xs.append(x)
        y = item[1]
        ys.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    return torch.stack(xs), torch.stack(ys)


def build_datamodule(cfg) -> Any:
    """The ablation's datamodule, built with the config's own seed and splits."""
    seed_everything(cfg.seed, workers=True)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dm = instantiate(dataset_cfg)
    dm.prepare_data()
    dm.setup()
    return dm


def run_target(run_id: str) -> str:
    """The orchestrator ``_target_`` of a run, from its local wandb config.

    Hydra strips ``_target_`` before instantiation, so the checkpoint's saved
    hyper-parameters do not carry it; the wandb run directory does.
    """
    matches = sorted((REPO / 'logs' / 'wandb').glob(f'run-*-{run_id}/files/config.yaml'))
    if not matches:
        raise FileNotFoundError(
            f'{run_id}: no logs/wandb/run-*-{run_id}/files/config.yaml to read '
            '_target_ from (pass the run whose wandb dir is still on disk)'
        )
    import yaml

    cfg = yaml.safe_load(matches[-1].read_text())
    orch = cfg.get('orchestrator', {})
    orch = orch.get('value', orch) if isinstance(orch, dict) else {}
    target = orch.get('_target_')
    if not target:
        raise RuntimeError(f'{run_id}: wandb config has no orchestrator._target_')
    return str(target)


def load_orchestrator(run_id: str, cfg, dm, device: torch.device):
    """Restore an orchestrator from its comm_ablation checkpoint.

    The checkpoint's ``hyper_parameters`` carry the fully built agent modules,
    the neighbour graph and the latent dims, so ``load_from_checkpoint`` needs
    no reconstruction of its own.
    """
    # Runs live under logs/<wandb project>/<run id>/, and a single analysis may
    # span several projects (comm_ablation, comm_ablation_seeds, ...), so search
    # every project directory rather than assuming one.
    ckpts = sorted((CKPT_ROOT / run_id / 'checkpoints').glob('*.ckpt'))
    if not ckpts:
        ckpts = sorted(CKPT_ROOT.parent.glob(f'*/{run_id}/checkpoints/*.ckpt'))
    if not ckpts:
        raise FileNotFoundError(
            f'no checkpoint for run {run_id} under {CKPT_ROOT.parent}/*/{run_id}/checkpoints/'
        )
    cls = get_class(run_target(run_id))
    ck = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
    extra: dict[str, Any] = {}
    if 'agents' not in ck['hyper_parameters']:
        # Orchestrators that don't pickle their agent modules into the
        # hyper-parameters (e.g. NonCooperativeLearning) need them rebuilt from
        # the config; the checkpoint then supplies the trained weights.
        from multi_agent_experiment import _build_agents, _parse_per_agent_cfg

        agents, latent_dims = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
        extra = {'agents': agents, 'latent_dims': latent_dims}
    del ck
    orch = cls.load_from_checkpoint(
        ckpts[-1], map_location='cpu', weights_only=False, **extra
    )
    orch.eval()
    orch.to(device)
    # _build_agent_target_classes() reads self.trainer.datamodule; there is no
    # trainer here, so seed the cache the same way SheafFRL does at train start.
    groups = getattr(dm, 'groups', None)
    group_tc = getattr(dm, 'group_target_classes', None)
    if groups and group_tc:
        orch._agent_target_classes = {
            int(a): set(group_tc[g])
            for g, agent_ids in groups.items()
            if g in group_tc
            for a in agent_ids
        }
    return orch


def encode_pilots(orch, dm, device) -> tuple[dict, dict]:
    """Per-agent raw pilot latents and labels (same loop as the eval)."""
    pilot_Z, pilot_y = {}, {}
    for idx_str, agent in orch.agents.items():
        idx = int(idx_str)
        ds = dm.pilot_datasets.get(idx)
        if ds is None or len(ds) == 0:
            continue
        loader = torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=0,
            collate_fn=_collate_xy,
        )
        agent.eval()
        Zs, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                Zs.append(agent.encode(xb.to(device)).detach().cpu().float())
                ys.append(yb.cpu())
        pilot_Z[idx] = torch.cat(Zs)
        pilot_y[idx] = torch.cat(ys)
    return pilot_Z, pilot_y


def prepare_maps(orch, dm) -> str:
    """Make ``_directed_alignment_map`` / ``_whiten_own_latents`` answerable.

    Returns a short description of where the whitening came from.
    """
    # Post-hoc baselines (non_cooperative & friends) fit both on demand.
    if hasattr(orch, '_fit_alignment_maps'):
        orch._fit_alignment_maps(dm)
        return 'post-hoc ZCA (train split) + Procrustes'
    if getattr(orch, '_use_learnable_whitening', lambda: False)():
        return 'SWBN (from checkpoint)'
    # learn_whitening=false: the closed-form ZCA op is fit during training from
    # the task-latent buffer and is not checkpointed. Refit it from the train
    # split, exactly as the post-hoc baselines do.
    from src.communication.whitening import fit_whitening

    device = orch.device
    for idx_str, agent in orch.agents.items():
        idx = int(idx_str)
        ds = dm.train_datasets.get(idx)
        if ds is None or len(ds) == 0:
            continue
        loader = torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=0,
            collate_fn=_collate_xy,
        )
        agent.eval()
        Zs = []
        with torch.no_grad():
            for xb, _ in loader:
                Zs.append(agent.encode(xb.to(device)).detach().cpu().float())
        orch._whitening_ops[idx] = fit_whitening(torch.cat(Zs))
    return 'ZCA REFIT from train split (not checkpointed)'


def effective_rank(Z: torch.Tensor) -> float:
    """exp(entropy of the normalised singular-value spectrum) of centred Z."""
    Zc = Z - Z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Zc)
    p = s / s.sum().clamp(min=1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def edge_rows(orch, dm, pilot_Z, pilot_y, i, j):
    """Whitened matched pilot rows for the canonical direction of edge (i, j).

    Mirrors ``BaseOrchestrator.evaluate_misalignment_loss`` exactly: one stored
    direction per edge, matched by sample id, optionally scoped to the edge's
    class intersection.
    """
    M = orch._directed_alignment_map(i, j)
    if M is None:
        return None
    pi, pj = dm.pilot_datasets.get(i), dm.pilot_datasets.get(j)
    if pi is None or pj is None:
        return None
    idx_i, idx_j = common_pilot_indices(pi, pj)
    if len(idx_i) < 2:
        return None

    if getattr(orch, '_eval_on_class_intersection', False):
        labels_i = pilot_y[i][torch.as_tensor(idx_i, dtype=torch.long)]
        mask = orch._class_intersection_mask(
            orch._resolve_agent_target_classes(), i, j, labels_i
        )
        if mask is not None:
            keep = mask.nonzero(as_tuple=True)[0].tolist()
            if len(keep) < 2:
                return None
            idx_i = [idx_i[k] for k in keep]
            idx_j = [idx_j[k] for k in keep]

    dev = orch.device
    with torch.no_grad():
        Zi = orch._whiten_own_latents(i, pilot_Z[i][idx_i].to(dev)).cpu().float()
        Zj = orch._whiten_own_latents(j, pilot_Z[j][idx_j].to(dev)).cpu().float()
    y = pilot_y[i][torch.as_tensor(idx_i, dtype=torch.long)]
    return Zi @ M.detach().cpu().float(), Zj, Zi, y


def edge_metrics(Zi2j, Zj, Zi, y) -> dict[str, float]:
    """The logged metrics plus the geometry that explains their disagreement."""
    n, d = Zj.shape
    diff = Zi2j - Zj
    out = {'n': n, 'd_e': d}
    out['misalign'] = float(((diff**2).sum(1).mean() / (2.0 * d)).item())

    a = torch.nn.functional.normalize(Zi2j, dim=1)
    b = torch.nn.functional.normalize(Zj, dim=1)
    sim = a @ b.T
    correct = sim.diagonal().unsqueeze(1)

    # MRR exactly as logged: every receiver row is a distractor.
    ranks = (sim >= correct).sum(1).clamp(min=1)
    out['mrr'] = float((1.0 / ranks.float()).mean().item())
    out['top1'] = float((ranks == 1).float().mean().item())

    # MRR with same-class distractors removed: only rows the model is NOT being
    # asked to merge may outrank the match.
    same_cls = y.unsqueeze(0) == y.unsqueeze(1)          # (n, n)
    eye = torch.eye(n, dtype=torch.bool)
    distract_ok = (~same_cls) | eye                       # keep self + off-class
    sim_cross = sim.masked_fill(~distract_ok, -float('inf'))
    ranks_cross = (sim_cross >= correct).sum(1).clamp(min=1)
    out['mrr_cross'] = float((1.0 / ranks_cross.float()).mean().item())

    # Of the rows whose top-1 is wrong, how many retrieve a same-class row?
    top1_idx = sim.argmax(1)
    wrong = top1_idx != torch.arange(n)
    if wrong.any():
        out['same_cls_err'] = float(
            same_cls[torch.arange(n)[wrong], top1_idx[wrong]].float().mean().item()
        )
    else:
        out['same_cls_err'] = float('nan')

    # Scale-free geometry: residual vs the gap between distinct receiver rows.
    D = torch.cdist(Zj, Zj)
    D.fill_diagonal_(float('inf'))
    out['resid_rms'] = float(diff.pow(2).sum(1).mean().sqrt().item())
    out['nn_dist'] = float(D.min(1).values.mean().item())
    out['resid/nn'] = out['resid_rms'] / max(out['nn_dist'], 1e-12)

    off = ~eye
    out['cos_same'] = float(sim_off_mean(b @ b.T, same_cls & off))
    out['cos_diff'] = float(sim_off_mean(b @ b.T, (~same_cls) & off))
    out['erank_send'] = effective_rank(Zi)
    out['erank_recv'] = effective_rank(Zj)
    return out


def sim_off_mean(S: torch.Tensor, mask: torch.Tensor) -> float:
    return S[mask].mean().item() if mask.any() else float('nan')


def pick_edge(orch, dm, pilot_Z, pilot_y, forced=None):
    """The canonical edge with the most scored rows (or the forced one)."""
    if forced:
        return tuple(forced)
    best, best_n = None, -1
    for i, nbrs in orch.hparams.neighbors.items():
        for j in nbrs:
            r = edge_rows(orch, dm, pilot_Z, pilot_y, int(i), int(j))
            if r is not None and r[0].shape[0] > best_n:
                best, best_n = (int(i), int(j)), r[0].shape[0]
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs', nargs='*', default=None,
                    help='label=run_id pairs (default: the table.md rows)')
    ap.add_argument('--edge', nargs=2, type=int, default=None,
                    help='sender receiver for the t-SNE panel (canonical direction)')
    ap.add_argument('--max-edges', type=int, default=None,
                    help='only score the first N edges (quick pass)')
    ap.add_argument('--scope', choices=('run', 'on', 'off'), default='run',
                    help="class-intersection scoping of the scored rows: "
                         "'run' honours each run's own eval_on_class_intersection, "
                         "'on'/'off' force it (use to compare runs whose configs differ)")
    ap.add_argument('--no-tsne', action='store_true')
    ap.add_argument('--tsne-max-rows', type=int, default=600,
                    help='subsample this many matched pairs for the t-SNE panels')
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'results' / 'comm_ablation' / 'diagnostics')
    args = ap.parse_args()

    runs = DEFAULT_RUNS
    if args.runs:
        runs = dict(kv.split('=', 1) for kv in args.runs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CONFIG_NAME)
    # logger.name interpolates ${hydra:...}, which only resolves inside a hydra
    # run; nothing here logs, so drop the node rather than fake a HydraConfig.
    OmegaConf.set_struct(cfg, False)
    cfg.pop('logger', None)
    dm = build_datamodule(cfg)

    per_edge_rows: list[dict] = []
    tsne_payload: dict[str, tuple] = {}
    edge_for_tsne = tuple(args.edge) if args.edge else None

    for label, run_id in runs.items():
        print(f'\n=== {label}  (run {run_id}) ===', flush=True)
        orch = load_orchestrator(run_id, cfg, dm, device)
        if args.scope != 'run':
            orch._eval_on_class_intersection = args.scope == 'on'
        src = prepare_maps(orch, dm)
        print(f'    whitening: {src}')
        pilot_Z, pilot_y = encode_pilots(orch, dm, device)

        if edge_for_tsne is None:
            edge_for_tsne = pick_edge(orch, dm, pilot_Z, pilot_y)
            print(f'    t-SNE edge (most scored rows): {edge_for_tsne}')

        seen: set[frozenset] = set()
        for i, nbrs in orch.hparams.neighbors.items():
            for j in nbrs:
                i, j = int(i), int(j)
                if frozenset((i, j)) in seen:
                    continue
                r = edge_rows(orch, dm, pilot_Z, pilot_y, i, j)
                if r is None:
                    continue
                seen.add(frozenset((i, j)))
                m = edge_metrics(*r)
                m.update(model=label, run=run_id, edge=f'{i}->{j}',
                         whitening=src)
                per_edge_rows.append(m)
                if args.max_edges and len(seen) >= args.max_edges:
                    break
            if args.max_edges and len(seen) >= args.max_edges:
                break

        if not args.no_tsne:
            r = edge_rows(orch, dm, pilot_Z, pilot_y, *edge_for_tsne)
            if r is not None:
                tsne_payload[label] = r

        del orch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    df = pd.DataFrame(per_edge_rows)
    csv_path = args.out_dir / 'mrr_diagnostics_per_edge.csv'
    df.to_csv(csv_path, index=False)

    num = df.select_dtypes('number').columns
    summary = df.groupby('model', sort=False)[list(num)].mean()
    order = ['n', 'd_e', 'misalign', 'mrr', 'mrr_cross', 'top1',
             'same_cls_err', 'resid_rms', 'nn_dist', 'resid/nn',
             'cos_same', 'cos_diff', 'erank_send', 'erank_recv']
    summary = summary[[c for c in order if c in summary.columns]]
    summary.to_csv(args.out_dir / 'mrr_diagnostics_summary.csv')

    pd.set_option('display.width', 220)
    print('\n' + '=' * 110)
    print('Edge-averaged diagnostics (same edges/direction the logged metrics use)')
    print('=' * 110)
    print(summary.to_string(float_format=lambda v: f'{v:.4f}'))
    print(f'\nper-edge CSV : {csv_path}')

    if not args.no_tsne and tsne_payload:
        plot_tsne(tsne_payload, edge_for_tsne, args.out_dir, args.tsne_max_rows)


def plot_tsne(payload: dict[str, tuple], edge, out_dir: Path,
              max_rows: int = 600) -> None:
    """Joint t-SNE of transported sender rows and receiver rows, per model."""
    n_models = len(payload)
    fig, axes = plt.subplots(1, n_models, figsize=(4.4 * n_models, 4.8),
                             squeeze=False)
    cmap = plt.get_cmap('tab10')

    # Same matched rows for every panel, so the models are visually comparable.
    n_full = min(v[0].shape[0] for v in payload.values())
    sel = torch.arange(n_full)
    if n_full > max_rows:
        sel = torch.randperm(n_full, generator=torch.Generator().manual_seed(0))
        sel = sel[:max_rows].sort().values

    for ax, (label, (Zi2j, Zj, _Zi, y)) in zip(axes[0], payload.items()):
        Zi2j, Zj, y = Zi2j[sel], Zj[sel], y[sel]
        X = torch.cat([Zi2j, Zj]).numpy()
        n = Zi2j.shape[0]
        perp = float(min(30, max(5, (2 * n - 1) / 3)))
        emb = TSNE(n_components=2, init='pca', perplexity=perp,
                   random_state=0).fit_transform(X)
        e_send, e_recv = emb[:n], emb[n:]
        classes = sorted(set(y.tolist()))
        cidx = {c: k for k, c in enumerate(classes)}
        col = np.array([cmap(cidx[int(v)] % 10) for v in y])

        # matched pairs: short segments = instance identity survived transport
        step = max(1, n // 120)
        for k in range(0, n, step):
            ax.plot([e_send[k, 0], e_recv[k, 0]], [e_send[k, 1], e_recv[k, 1]],
                    color='0.75', lw=0.4, zorder=1)
        ax.scatter(e_send[:, 0], e_send[:, 1], c=col, s=13, marker='o',
                   alpha=0.85, linewidths=0, zorder=2, label='sender → receiver')
        ax.scatter(e_recv[:, 0], e_recv[:, 1], c=col, s=22, marker='x',
                   alpha=0.85, linewidths=0.9, zorder=2, label='receiver')
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    handles = [
        plt.Line2D([], [], marker='o', ls='', color='0.35', label='transported sender'),
        plt.Line2D([], [], marker='x', ls='', color='0.35', label='receiver'),
        plt.Line2D([], [], color='0.75', lw=0.8, label='matched pair'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'Whitened pilot latents, edge {edge[0]}→{edge[1]} '
        '(colour = class; short grey links = instance identity preserved)',
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    for ext in ('png', 'pdf'):
        p = out_dir / f'tsne_whitened_pilots_edge_{edge[0]}_{edge[1]}.{ext}'
        fig.savefig(p, dpi=180, bbox_inches='tight')
    print(f'figure      : {out_dir}/tsne_whitened_pilots_edge_{edge[0]}_{edge[1]}.png')


if __name__ == '__main__':
    main()
