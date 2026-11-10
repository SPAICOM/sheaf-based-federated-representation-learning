"""t-SNE of the two agents' pilot latents, before and after the post-hoc alignment.

For every run of a λ study (read from its lambda_study.csv, written by
plot_lambda_study.py) the final checkpoint is loaded and both encoders embed the
pilots, each through its own corruption. One PNG per λ, four panels:
    top:    agent 0 alone | agent 1 alone | both, whitened, no map
    bottom: one t-SNE of both after the map, shown as agent 0 only | agent 1 only | both
coloured by class. The map is the evaluation's: per-agent Ledoit-Wolf ZCA fitted on
the pilots, then orthogonal Procrustes 0 -> 1 in whitened space, then agent 1's
colouring back into agent 1's latent space, which is what its decoder reads (the
bottom row). Plotting the whitened space instead is misleading off the pilots: pilot
statistics at n ~ d blow up weak directions on new rows and hide the class structure. "No map" puts the
two whitened clouds in one space by zero-padding the smaller latent, so it shows
what the clouds look like before any rotation. t-SNE is non-linear: it shows which
clusters overlap, not distances.

    uv run python scripts/plot_pilot_tsne.py --setup grayblur_harsh
    uv run python scripts/plot_pilot_tsne.py --setup halves_scarce_5000
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _parse_per_agent_cfg,
)
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

from src.communication.whitening import ledoit_wolf_shrinkage  # noqa: E402

# agent 1 architectures of the 6v6 runs (tmp/2026-10-30_6v6_noncoop.sh)
GRAYBLUR_6V6_ARCHS = {
    'resnet': [],
    'vgg': ['+model.agents.1._target_=src.agents.CNNClassifier', '+model.agents.1.encoder_hidden_dims=[64,128,256,384]'],
    'vit': ['+model.agents.1._target_=src.agents.TransformerClassifier', '+model.agents.1.pool=mean',
            '+model.agents.1.use_batchnorm=false', '+model.agents.1.embed_dim=192', '+model.agents.1.depth=6',
            '+model.agents.1.num_heads=6'],
}
SETUPS = {  # config, overrides, wandb project dir, study csv, variant
    'grayblur_harsh': ('cifar10_2agents_grayblur_harsh', ['dataset.pilot_num_samples=2048'],
                       'cifar10_2agents_grayblur_harsh_lmb',
                       'results/cifar10_2agents_grayblur_harsh_lmb/lambda_study.csv', 'swbn'),
    'halves_scarce_5000': ('cifar10_2agents_halves_scarce', ['dataset.agent_train_samples=5000'],
                           'cifar10_2agents_halves_scarce',
                           'results/cifar10_2agents_halves_scarce/lmb_study_5000/lambda_study.csv',
                           'swbn_5000'),
    'grayblur_7v7': ('cifar10_2agents_grayblur_7v7', ['dataset.pilot_num_samples=2048'],
                     'cifar10_2agents_grayblur_7v7',
                     'results/cifar10_2agents_grayblur_7v7/lmb_study_swbn/lambda_study.csv', 'swbn'),
    'mnist_labelnoise': ('mnist_2agents_labelnoise', [], 'mnist_2agents_labelnoise',
                         'results/mnist_2agents_labelnoise/lmb_study/lambda_study.csv', 'swbn'),
    'mnist_labelnoise_p06_17': (
        'mnist_2agents_labelnoise',
        ['dataset.agent_label_noise={0:{classes:[1,7],fraction:0.6},1:{classes:[3,5],fraction:0.6}}',
         'dataset.groups={0:{agents:[0],target_classes:[1,7]},1:{agents:[1],target_classes:[3,5]}}'],
        'mnist_2agents_labelnoise',
        'results/mnist_2agents_labelnoise/lmb_study/lambda_study.csv', 'swbn_p06_17'),
    **{f'mnist_labelnoise_oneway_p{p}': (
        'mnist_2agents_labelnoise',
        [f'dataset.agent_label_noise={{0:{{classes:[1,7],fraction:{p},direction:one_way}},'
         f'1:{{classes:[3,5],fraction:{p},direction:one_way}}}}',
         'dataset.groups={0:{agents:[0],target_classes:[1,7]},1:{agents:[1],target_classes:[3,5]}}'],
        'mnist_2agents_labelnoise',
        'results/mnist_2agents_labelnoise/oneway/runs.csv', f'noncoop_p{p}') for p in (0.2, 0.4)},
    **{f'grayblur_6v6_{arch}': ('cifar10_2agents_grayblur_6v6', ov, 'cifar10_2agents_grayblur_6v6',
                                'results/cifar10_2agents_grayblur_6v6/runs.csv', f'noncoop_{arch}')
       for arch, ov in GRAYBLUR_6V6_ARCHS.items()},
    # CenterNorm bottleneck, no whitening anywhere (tmp/2026-11-02_6v6_cnorm.sh)
    **{f'grayblur_6v6_cnorm_{arch}': ('cifar10_2agents_grayblur_6v6', [*ov, 'model.latent_norm=true'],
                                      'cifar10_2agents_grayblur_6v6',
                                      'results/cifar10_2agents_grayblur_6v6/runs.csv', f'noncoop_cnorm_{arch}')
       for arch, ov in GRAYBLUR_6V6_ARCHS.items()},
    # non-coop (λ=0) vs SheafFRL under the SAME pilot strategy, CenterNorm, no whitening
    # (tmp/2026-11-02_6v6_cnorm_lmb.sh: deer/dog pilots; 6v6_cnorm_sharedpilots.sh: shared unlabeled pilots)
    'grayblur_6v6_cnorm_ddpilots': ('cifar10_2agents_grayblur_6v6', ['model.latent_norm=true'],
                                    'cifar10_2agents_grayblur_6v6',
                                    'results/cifar10_2agents_grayblur_6v6/runs.csv', 'cnorm_ddpilots'),
    'grayblur_6v6_cnorm_sharedpilots': (
        'cifar10_2agents_grayblur_6v6',
        ['model.latent_norm=true', 'dataset.pilot_source=pool', 'dataset.pilot_exclusive=true',
         '+dataset.pilot_class_filter=false'],
        'cifar10_2agents_grayblur_6v6', 'results/cifar10_2agents_grayblur_6v6/runs.csv', 'cnorm_sharedpilots'),
    # 7v7 shift study, 4096 shared pilots, CenterNorm, no whitening: non-coop vs DiSheafFRL per shift
    # (tmp/2026-11-03_7v7_pilots_shift.sh; runs.csv written by plot_shift_study.py)
    **{f'7v7_{setting}_s{s:g}': (proj, ['model.latent_norm=true', 'dataset.pilot_source=pool',
                                        'dataset.pilot_exclusive=true', '+dataset.pilot_class_filter=false',
                                        'dataset.pilot_num_samples=4096', f'dataset.shift_strength={s}'],
                                 proj, f'results/{proj}/shift_study/runs.csv', f'cnorm_s{s:g}')
       for setting, proj in (('grayblur', 'cifar10_2agents_grayblur_7v7'), ('clean', 'cifar10_2agents_7v7'))
       for s in (0.5, 0.6, 0.7, 0.8, 1.0)},
    # ResNet-10 on 10 classes vs plain CNN on animate/inanimate (per_class_comm.py has the coarse scores)
    'coarse': ('cifar10_2agents_coarse', [], 'cifar10_2agents_coarse',
               'results/cifar10_2agents_coarse/runs.csv', 'noncoop_coarse'),
    'coarse_control': ('cifar10_2agents_coarse', ['dataset.agent_label_map=null'], 'cifar10_2agents_coarse',
                       'results/cifar10_2agents_coarse/runs.csv', 'noncoop_fine_control'),
}
CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASSES = CIFAR_CLASSES
METHOD = 'SheafFRL'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TT = ToTensor()


@torch.no_grad()
def encode(agent, ds, idx, bs=500):
    Z, Y = [], []
    for s in range(0, len(idx), bs):
        items = [ds[k] for k in idx[s:s + bs]]
        x = torch.stack([it[0] if torch.is_tensor(it[0]) else TT(it[0]) for it in items]).to(DEV)
        Z.append(agent.encode(x).float().cpu())
        Y.append(torch.as_tensor([int(it[1]) for it in items]))
    return torch.cat(Z).double(), torch.cat(Y)


WHITEN = True   # False: identity "whitener" (runs with posthoc_whitening=false)


def whitener(Z):
    """Ledoit-Wolf ZCA fitted on Z, returned as a function."""
    if not WHITEN:
        return (lambda z: z), (lambda w: w)
    mu = Z.mean(0)
    X = Z - mu
    C = X.T @ X / len(X)
    g = min(ledoit_wolf_shrinkage(Z), 0.999)
    C = (1 - g) * C + g * C.trace() / C.shape[0] * torch.eye(C.shape[0], dtype=C.dtype)
    e, V = torch.linalg.eigh(C)
    W = V @ torch.diag(e.clamp_min(1e-12) ** -0.5) @ V.T
    Wi = V @ torch.diag(e.clamp_min(1e-12) ** 0.5) @ V.T
    return (lambda z: (z - mu) @ W), (lambda w: w @ Wi + mu)


def tsne(X):
    X = np.asarray(X, dtype=np.float32)
    X = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    return TSNE(n_components=2, perplexity=30, init='pca', random_state=0).fit_transform(X)


def scatter(ax, P, y, title, marker='o', hollow=False):
    c = plt.cm.tab10(np.asarray(y) % 10)
    if hollow:
        ax.scatter(P[:, 0], P[:, 1], s=7, marker=marker, facecolors='none', edgecolors=c, linewidths=0.5)
    else:
        ax.scatter(P[:, 0], P[:, 1], s=4, marker=marker, c=c, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('t-SNE 1', fontsize=7)
    ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.grid(alpha=0.2)


def load_agent(agent, sd, a, run):
    pre = next(k for k in (f'agents.{a}.', f'agents.{a}.model.') if any(s.startswith(k) for s in sd))
    missing = agent.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                    strict=False).missing_keys
    assert not missing, f'{run} agent {a}: missing {missing[:3]}'
    return agent.to(DEV).eval()


def plot_train(args, r, agents, dm, sd, out):
    """Each agent's own training set through its own encoder (no augmentation)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4))
    for a, ax in zip((0, 1), axes):
        ds = dm.train_datasets[a]
        aug, ds.transform = ds.transform, None
        Z, y = encode(load_agent(agents[a], sd, a, r.run), ds, list(range(len(ds))))
        ds.transform = aug
        present = sorted(set(y.tolist()))
        title = f'agent {a}: all {len(ds)} training images, classes {present} (d={Z.shape[1]})'
        clean_key = f'{ds.label_key}_clean'
        flipped = None
        if clean_key in ds.dataset.column_names:          # label noise: colour by the TRUE label
            y_true = torch.as_tensor([int(v) for v in ds.dataset[clean_key]])
            flipped = (y_true != y).numpy()
            y = y_true
            title += f'\ncoloured by true label; x = {int(flipped.sum())} images trained with a swapped label'
        E = tsne(Z)
        scatter(ax, E, y.numpy(), title, marker='o' if a == 0 else '^')
        for t in ax.collections:
            t.set_sizes([1.5])
        if flipped is not None and flipped.any():
            ax.scatter(E[flipped, 0], E[flipped, 1], s=6, marker='x', c='k', linewidths=0.4)
    handles = [plt.Line2D([], [], ls='', marker='o', color=plt.cm.tab10(c), label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=8, frameon=False)
    fig.suptitle(f'{args.setup}: {METHOD} λ={r.lmb:g} (run {r.run}), training latents; '
                 f'test comm {r.comm:.3f}, private {r.private:.3f}', fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / f'lmb_{r.lmb:g}_train.png', dpi=130)
    plt.close(fig)
    print(f'λ={r.lmb:g} {r.run} -> {out}/lmb_{r.lmb:g}_train.png', flush=True)


@torch.no_grad()
def plot_pred(args, r, agents, Z, Zf, y, out, suffix):
    """Per agent: its latents coloured by true label | own decoder | other decoder after the map."""
    wh = {a: whitener(Zf[a]) for a in (0, 1)}
    dec = lambda a, z: agents[a].decoder(z.float().to(DEV)).argmax(1).cpu()
    yn = y.numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.8))
    for i, j in ((0, 1), (1, 0)):                       # sender i -> receiver j
        (wi, _), (wj, cj) = wh[i], wh[j]
        U, _, Vt = torch.linalg.svd(wi(Zf[i]).T @ wj(Zf[j]), full_matrices=False)
        own, comm = dec(i, Z[i]), dec(j, cj(wi(Z[i]) @ (U @ Vt)))
        E = tsne(Z[i])
        mk = 'o' if i == 0 else '^'
        acc = lambda pr: float((pr == y).float().mean())
        scatter(axes[i][0], E, yn, f'agent {i} latents, coloured by TRUE label', marker=mk)
        scatter(axes[i][1], E, own.numpy(), f'same points, coloured by agent {i}\'s own decoder\n'
                f'(private accuracy on these rows {acc(own):.3f})', marker=mk)
        scatter(axes[i][2], E, comm.numpy(), f'same points mapped {i} -> {j}, coloured by agent {j}\'s '
                f'decoder\n(comm accuracy {i}->{j} on these rows {acc(comm):.3f})', marker=mk)
    handles = [plt.Line2D([], [], ls='', marker='o', color=plt.cm.tab10(c), label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=8, frameon=False)
    fig.suptitle(f'{args.setup}: {METHOD} λ={r.lmb:g} (run {r.run}), {len(y)} {args.split} rows; '
                 f'colour = true label / predicted label', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = out / f'lmb_{r.lmb:g}{suffix}_pred.png'
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f'λ={r.lmb:g} {r.run} -> {path}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--setup', choices=SETUPS, required=True)
    p.add_argument('--max-pilots', type=int, default=2048)
    p.add_argument('--split', choices=['pilots', 'test', 'train'], default='pilots',
                   help='rows to plot; the map is always fitted on the pilots. train = each '
                        'agent\'s own whole training set (no augmentation), one t-SNE per agent, no map')
    p.add_argument('--max-test', type=int, default=2500)
    p.add_argument('--test-classes', type=int, nargs='+', default=None,
                   help='with --split test: plot only test rows of these classes')
    p.add_argument('--color-by', choices=['label', 'pred'], default='label',
                   help='pred: one t-SNE per agent, coloured by true label, by its own decoder '
                        '(private) and by the other agent\'s decoder after the map (comm)')
    p.add_argument('--map-fit', choices=['pilots', 'test'], default='pilots',
                   help='rows the map is fitted on; test = oracle fitted on the plotted test rows')
    p.add_argument('--out', default='results/pilot_tsne')
    args = p.parse_args()
    cfg_name, overrides, proj, csv, variant = SETUPS[args.setup]
    global CLASSES
    CLASSES = [str(c) for c in range(10)] if 'mnist' in cfg_name else CIFAR_CLASSES
    out = ROOT / args.out / args.setup
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(ROOT / csv)
    runs = runs[runs.variant == variant].sort_values('lmb')
    global METHOD, WHITEN
    WHITEN = 'cnorm' not in variant   # CenterNorm runs: no whitening, map on raw latents

    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        cfg = compose(cfg_name, overrides=['orchestrator=non_cooperative', *overrides])
    dm, _ = _setup(cfg)
    agents, _ = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    n = len(dm.pilot_datasets[0])
    idx = sorted(np.random.default_rng(0).choice(n, min(n, args.max_pilots), replace=False).tolist())
    tds = dm.test_datasets[0]
    pool = list(range(len(tds)))
    if args.test_classes:
        keep = set(args.test_classes)
        pool = [k for k in pool if int(tds[k][1]) in keep]
    tidx = sorted(np.random.default_rng(1).choice(pool, min(len(pool), args.max_test), replace=False).tolist())
    suffix = '' if args.split == 'pilots' else '_test'
    if args.split == 'test' and args.test_classes:
        suffix += '_classes' + ''.join(map(str, sorted(args.test_classes)))
    if args.map_fit == 'test':
        suffix += '_oraclemap'

    rows = []
    for _, r in runs.iterrows():
        METHOD = 'non-coop' if variant.startswith('noncoop') or r.lmb == 0 else 'SheafFRL'
        if isinstance(r.get('method'), str) and r.get('method'):
            METHOD = r.method
        if not WHITEN:
            METHOD += ' + CenterNorm, no whitening'
        ck = sorted((ROOT / 'logs' / proj / r.run / 'checkpoints').glob('*.ckpt'))[-1]
        sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
        Z, Y, Zf = {}, {}, {}
        if args.split == 'train':
            plot_train(args, r, agents, dm, sd, out)
            continue
        for a in (0, 1):
            pre = next(k for k in (f'agents.{a}.', f'agents.{a}.model.') if any(s.startswith(k) for s in sd))
            missing = agents[a].load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                                strict=False).missing_keys
            assert not missing, f'{r.run} agent {a}: missing {missing[:3]}'
            agents[a].to(DEV).eval()
            Zf[a], Y[a] = encode(agents[a], dm.pilot_datasets[a], idx)
            if args.split == 'test':
                Z[a], Y[a] = encode(agents[a], dm.test_datasets[a], tidx)
            else:
                Z[a] = Zf[a]
        assert torch.equal(Y[0], Y[1]), 'rows are not paired'
        if args.map_fit == 'test':      # cheating oracle: the map is fitted on the plotted test rows
            assert args.split == 'test', '--map-fit test needs --split test'
            Zf = Z
        if args.color_by == 'pred':
            plot_pred(args, r, agents, Z, Zf, Y[0], out, suffix)
            continue
        (w0, _), (w1, c1) = whitener(Zf[0]), whitener(Zf[1])   # fitted on the pilots
        U, _, Vt = torch.linalg.svd(w0(Zf[0]).T @ w1(Zf[1]), full_matrices=False)
        W0, W1 = w0(Z[0]), w1(Z[1])
        M0 = W0 @ (U @ Vt)                                         # agent 0 in agent 1's whitened space
        d0, d1 = W0.shape[1], W1.shape[1]
        pad = torch.cat([W1, torch.zeros(len(W1), d0 - d1, dtype=W1.dtype)], 1) if d0 > d1 else W1
        W0p = torch.cat([W0, torch.zeros(len(W0), d1 - d0, dtype=W0.dtype)], 1) if d1 > d0 else W0
        k = len(W0)
        # evaluation's convention on the plotted rows, direction 0 -> 1: 1 = unaligned
        mis = float(((W0 @ (U @ Vt) - W1) ** 2).sum(1).mean()) / (2 * W1.shape[1])
        J_raw = tsne(torch.cat([W0p, pad]))
        # after the map, show agent 1's own latent space: what its decoder reads
        J_map = tsne(torch.cat([c1(M0), Z[1]]))
        y = Y[0].numpy()

        P0, P1 = tsne(Z[0]), tsne(Z[1])
        order = np.random.default_rng(0).permutation(2 * k)
        both, who = J_map[order], order >= k
        lim = (J_map.min(0) - 2, J_map.max(0) + 2)
        before = f'before the map: both {"whitened" if WHITEN else "CenterNorm"}, one space (• agent 0, △ agent 1)'
        panels = {   # one figure per panel: name -> (title, [(points, labels, marker, hollow)], share map limits)
            'agent0': (f'agent 0 latents alone (d={d0})', [(P0, y, 'o', False)], False),
            'agent1': (f'agent 1 latents alone (d={d1})', [(P1, y, '^', False)], False),
            'before_map': (before, [(J_raw[:k], y, 'o', False), (J_raw[k:], y, '^', True)], False),
            'mapped_agent0': ("after the map, in agent 1's latent space: agent 0 points only",
                              [(J_map[:k], y, 'o', False)], True),
            'mapped_agent1': ("after the map, in agent 1's latent space: agent 1 points only",
                              [(J_map[k:], y, '^', False)], True),
            'mapped_both': (f'after the map: both (• agent 0, △ agent 1), {args.split} misalignment {mis:.2f}',
                            [(both[~who], y[order[~who]], 'o', False),
                             (both[who], y[order[who] - k], '^', True)], True),
        }
        rows.append((f'{METHOD.split(" + ")[0]}, λ={r.lmb:g} ({r.run})\ntest comm {r.comm:.3f}, private {r.private:.3f}', panels, lim))

    if not rows:
        return
    # one figure per setup (= pilot strategy): a row per run (non-coop, SheafFRL), all panels side by side
    fig, axes = plt.subplots(len(rows), 6, figsize=(30, 5.4 * len(rows)), squeeze=False)
    for axrow, (label, panels, lim) in zip(axes, rows):
        for ax, (title, layers, shared) in zip(axrow, panels.values()):
            for P, lab, mk, hol in layers:
                scatter(ax, P, lab, title, marker=mk, hollow=hol)
            ax.title.set_fontsize(8)
            if shared:
                ax.set_xlim(lim[0][0], lim[1][0])
                ax.set_ylim(lim[0][1], lim[1][1])
        axrow[0].set_ylabel(label + '\n\nt-SNE 2', fontsize=11, fontweight='bold')
    handles = [plt.Line2D([], [], ls='', marker='o', color=plt.cm.tab10(c), label=CLASSES[c]) for c in range(10)]
    fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=10, frameon=False)
    fig.suptitle(f'{args.setup} ({"CenterNorm, no whitening" if not WHITEN else "whitened"}): {k} {args.split} rows, map fitted on '
                 + (f'the {k} plotted test rows (ORACLE)' if args.map_fit == 'test' else f'{len(idx)} pilots'),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = out / f'compare{suffix}.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f'-> {path}', flush=True)


if __name__ == '__main__':
    main()
