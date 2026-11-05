"""Linear probes on the 7v7 shift study (tmp/2026-11-03_7v7_pilots_shift.sh): non-coop vs
DiSheafFRL (λ=3e-2), 4096 shared pilots, CenterNorm, no whitening, two settings
(grayscale/blur, clean inputs) x shift {0.5, 0.6, 0.7, 0.8, 1.0}.

For every run the final checkpoint is loaded and both encoders embed the official test
split (each through its own corruption) and the pilots. A 10-class multinomial logistic
regression (L2, sklearn C=1 equivalent, standardised features) is cross-fitted on two
halves of the test rows:
  private probe  agent a: fit on half A of agent a's test latents, score half B (and swap)
  post-comm probe i -> j: fit on half A of receiver j's test latents, score half B of
                 sender i's test latents mapped into j's space by the evaluation's map
                 (orthogonal Procrustes on the raw pilot latents, no whitening)
so the post-comm probe replaces the receiver's trained decoder by the best linear head on
its own latents. Scores are cached per run in results/<project>/shift_study/probe_cache.csv.

    uv run python scripts/plot_shift_probe.py [--refresh]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from multi_agent_experiment import (  # noqa: E402
    _build_agents,
    _parse_per_agent_cfg,
)
from plot_pilot_tsne import DEV, encode, load_agent  # noqa: E402
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

from src.utils.plotting import use_style  # noqa: E402

SHIFTS = (0.5, 0.6, 0.7, 0.8, 1.0)
COMMON = ['model.latent_norm=true', 'dataset.pilot_source=pool', 'dataset.pilot_exclusive=true',
          '+dataset.pilot_class_filter=false', 'dataset.pilot_num_samples=4096']
NONCOOP, DISHEAF = 'non-coop', r'DiSheafFRL $\lambda=3\times10^{-2}$'
COLOURS = {NONCOOP: 'tab:gray', DISHEAF: 'tab:green'}
SETTINGS = {  # label -> (config = wandb project, {method: run ids in SHIFTS order})
    'grayscale / blur': ('cifar10_2agents_grayblur_7v7', {
        NONCOOP: ['kb1kwjm3', 'une3bw5o', 'gkkzgz9m', 'u5laxld2', 'iqx3e07o'],
        DISHEAF: ['55udmvti', 'h3urkz17', 'h5e1kvvc', '1lgg1zzz', 'k8u9mndv']}),
    'clean inputs': ('cifar10_2agents_7v7', {
        NONCOOP: ['kcczl13t', '66h3lcky', '2bwdcj60', 'graon9m8', 'x5yik4aa'],
        DISHEAF: ['hmns3cxg', 'iyy0ic0w', 'wfit6w5e', 'fuf9ctip', '38c47yen']}),
}


@torch.enable_grad()
def probe(Xf: torch.Tensor, yf: torch.Tensor, Xs: torch.Tensor) -> torch.Tensor:
    """Multinomial logistic regression fitted on (Xf, yf), predictions on Xs."""
    mu, sd = Xf.mean(0), Xf.std(0).clamp_min(1e-6)
    X, Y = ((Xf - mu) / sd).float().to(DEV), yf.to(DEV)
    W = torch.zeros(X.shape[1], 10, device=DEV, requires_grad=True)
    b = torch.zeros(10, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=1000, tolerance_grad=1e-7, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, Y) + 0.5 / len(X) * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return ((((Xs - mu) / sd).float().to(DEV) @ W + b).argmax(1)).cpu()


@torch.no_grad()
def score(cfg_name: str, run: str, shift: float) -> dict:
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        cfg = compose(cfg_name, overrides=['orchestrator=non_cooperative', *COMMON,
                                           f'dataset.shift_strength={shift}'])
    dm, _ = _setup(cfg)
    agents, _ = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    ck = sorted((ROOT / 'logs' / cfg_name / run / 'checkpoints').glob('*.ckpt'))[-1]
    sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
    Zt, Zp, Y = {}, {}, {}
    for a in (0, 1):
        ag = load_agent(agents[a], sd, a, run)
        Zt[a], Y[a] = encode(ag, dm.test_datasets[a], list(range(len(dm.test_datasets[a]))))
        Zp[a], _ = encode(ag, dm.pilot_datasets[a], list(range(len(dm.pilot_datasets[a]))))
    assert torch.equal(Y[0], Y[1]), 'test rows are not paired'
    y, n = Y[0], len(Y[0])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    folds = [(perm[: n // 2], perm[n // 2:]), (perm[n // 2:], perm[: n // 2])]   # (fit, score)
    out = {'run': run, 'shift': shift}
    for a in (0, 1):
        hits = sum(int((probe(Zt[a][f], y[f], Zt[a][s]) == y[s]).sum()) for f, s in folds)
        out[f'probe private {a}'] = hits / n
    for i, j in ((1, 0), (0, 1)):                       # sender i -> receiver j
        U, _, Vt = torch.linalg.svd(Zp[i].T @ Zp[j], full_matrices=False)
        Zm = Zt[i] @ (U @ Vt)
        hits = sum(int((probe(Zt[j][f], y[f], Zm[s]) == y[s]).sum()) for f, s in folds)
        out[f'probe comm {i}->{j}'] = hits / n
    out['probe private'] = (out['probe private 0'] + out['probe private 1']) / 2
    out['probe comm'] = (out['probe comm 1->0'] + out['probe comm 0->1']) / 2
    print(f'{cfg_name} {run} shift {shift}: ' + ', '.join(f'{k} {v:.3f}' for k, v in out.items()
                                                           if k.startswith('probe')), flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--refresh', action='store_true', help='recompute cached runs')
    refresh = p.parse_args().refresh
    use_style()
    plt.rcParams.update({'lines.markersize': 6, 'lines.linewidth': 1.8})
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (setting, (proj, methods)) in zip(axes, SETTINGS.items()):
        out = ROOT / f'results/{proj}/shift_study'
        out.mkdir(parents=True, exist_ok=True)
        cache_path = out / 'probe_cache.csv'
        cache = pd.read_csv(cache_path) if cache_path.exists() and not refresh else pd.DataFrame(columns=['run'])
        rows = []
        for method, runs in methods.items():
            for shift, run in zip(SHIFTS, runs):
                hit = cache[cache.run == run]
                if len(hit):
                    r = hit.iloc[0].to_dict()
                else:   # score and cache right away, so an interrupted pass resumes
                    r = score(proj, run, shift)
                    cache = pd.concat([cache, pd.DataFrame([r])], ignore_index=True)
                    cache.to_csv(cache_path, index=False)
                rows.append({**r, 'method': method})
        t = pd.DataFrame(rows)
        for method, d in t.groupby('method', sort=False):
            d = d.sort_values('shift')
            ax.plot(d['shift'], d['probe comm'], 'o-', color=COLOURS[method], label=f'{method}, post-comm probe')
            ax.plot(d['shift'], d['probe private'], 's:', color=COLOURS[method], alpha=0.6,
                    label=f'{method}, private probe')
        ax.set_xlabel('label-shift strength')
        ax.set_ylabel('linear-probe test accuracy (10 classes)')
        ax.set_title(f'7v7 {setting}, 4096 shared pilots', fontsize=18)
        ax.set_xticks(list(SHIFTS))
        cols = ['method', 'shift', 'run', 'probe comm', 'probe comm 1->0', 'probe comm 0->1',
                'probe private', 'probe private 0', 'probe private 1']
        fmt = t[cols].copy()
        fmt['method'] = fmt['method'].replace({DISHEAF: 'DiSheafFRL λ=3e-2'})
        for col in cols[3:]:
            fmt[col] = fmt[col].map(lambda v: f'{v:.3f}')
        md = '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
                       + ['| ' + ' | '.join(map(str, r)) + ' |' for r in fmt.itertuples(index=False)])
        (out / 'probe_table.md').write_text(md + '\n')
        print(f'## {setting}\n{md}\n-> {out}\n', flush=True)
    axes[0].legend(fontsize=12, loc='lower left')
    fig.suptitle('CIFAR-10 7v7, two ResNets, CenterNorm, linear probes on the test latents, single seed')
    fig.tight_layout()
    for proj, _ in SETTINGS.values():
        for ext in ('png', 'pdf'):
            fig.savefig(ROOT / f'results/{proj}/shift_study/probe_vs_shift.{ext}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
