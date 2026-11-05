"""Per-class private and post-communication test accuracy of a 2-agent run.

Loads the final checkpoint, fits the evaluation's map (per-agent Ledoit-Wolf ZCA
on the pilots + orthogonal Procrustes in whitened space, then the receiver's
colouring) and decodes with the receiver's decoder. Overall numbers should match
the run's test/comm_task_perf_agent_* and test/private_task_perf_agent_*.

    uv run python scripts/per_class_comm.py --config cifar10_2agents_grayblur_7v7 \
        --project cifar10_2agents_grayblur_7v7 --run eal9ef4q \
        --overrides dataset.pilot_num_samples=2048 --groups 3,4,5,6

--fit-on test fits the map on the test latents it is scored on (a cheating
oracle: the best map of its kind); test_cv cross-fits it on two halves of the
test set, so each row is scored by a map fitted on the other half.
--map lstsq replaces whitened Procrustes by an unconstrained linear map
(least squares with bias, raw latent spaces).
--label-map 1:0,0,1,1,1,1,1,1,0,0 scores agent 1 (its private predictions and
whatever its decoder reads) against those coarse targets instead of the classes.
--probe adds a linear probe (logistic regression, 2-fold on the test latents)
of the 10 classes on each agent's latents: how much class information is
linearly readable at all, whatever the other agent's decoder can use.
"""
import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))
from multi_agent_experiment import _build_agents, _parse_per_agent_cfg  # noqa: E402
import plot_pilot_tsne  # noqa: E402
from plot_pilot_tsne import DEV, encode, load_agent, whitener  # noqa: E402
from posthoc_on_sheaf_encoders import _setup  # noqa: E402

CIFAR = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--overrides', nargs='*', default=[])
    p.add_argument('--groups', nargs='*', default=[],
                   help='extra class subsets to summarise, e.g. 3,4,5,6')
    p.add_argument('--fit-on', choices=['pilots', 'test', 'test_cv'], default='pilots')
    p.add_argument('--map', choices=['procrustes', 'lstsq'], default='procrustes')
    p.add_argument('--label-map', nargs='*', default=[], help='agent:t0,t1,...,t9 (coarse targets)')
    p.add_argument('--probe', action='store_true')
    p.add_argument('--no-whiten', action='store_true', help='Procrustes on raw latents (posthoc_whitening=false)')
    args = p.parse_args()
    plot_pilot_tsne.WHITEN = not args.no_whiten
    with initialize_config_dir(config_dir=str(ROOT / 'config/hydra'), version_base='1.3'):
        cfg = compose(args.config, overrides=['orchestrator=non_cooperative', *args.overrides])
    names = [str(c) for c in range(10)] if 'mnist' in args.config else CIFAR
    dm, _ = _setup(cfg)
    agents, _ = _build_agents(cfg, dm, _parse_per_agent_cfg(cfg))
    ck = sorted((ROOT / 'logs' / args.project / args.run / 'checkpoints').glob('*.ckpt'))[-1]
    sd = torch.load(ck, map_location='cpu', weights_only=False)['state_dict']
    Zt, Zp, P = {}, {}, {}
    for a in (0, 1):
        ag = load_agent(agents[a], sd, a, args.run)
        Zt[a], y = encode(ag, dm.test_datasets[a], list(range(len(dm.test_datasets[a]))))
        Zp[a], _ = encode(ag, dm.pilot_datasets[a], list(range(len(dm.pilot_datasets[a]))))
        P[a] = ag.decoder(Zt[a].float().to(DEV)).argmax(1).cpu()
    def fit_map(Fi, Fj):
        """Linear map from sender rows Fi to receiver rows Fj (paired)."""
        if args.map == 'lstsq':
            X = torch.cat([Fi, torch.ones(len(Fi), 1)], 1).double()
            B = torch.linalg.lstsq(X, Fj.double()).solution
            return lambda Z: (torch.cat([Z, torch.ones(len(Z), 1)], 1).double() @ B).float()
        (wi, _), (wj, cj) = whitener(Fi), whitener(Fj)
        U, _, Vt = torch.linalg.svd(wi(Fi).T @ wj(Fj), full_matrices=False)
        return lambda Z: cj(wi(Z) @ (U @ Vt))

    n = len(y)
    if args.fit_on == 'test_cv':
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        folds = [(perm[: n // 2], perm[n // 2:]), (perm[n // 2:], perm[: n // 2])]   # (fit, score)
    else:
        folds = [(None, torch.arange(n))]
    C, mis = {}, {}
    for i, j in ((1, 0), (0, 1)):                       # sender i -> receiver j
        C[j], err = torch.empty(n, dtype=torch.long), torch.empty(n)
        for fit, score in folds:
            Fi, Fj = (Zp[i], Zp[j]) if fit is None and args.fit_on == 'pilots' else \
                     (Zt[i], Zt[j]) if fit is None else (Zt[i][fit], Zt[j][fit])
            M = fit_map(Fi, Fj)
            Zm = M(Zt[i][score])
            C[j][score] = agents[j].decoder(Zm.float().to(DEV)).argmax(1).cpu()
            # residual in the receiver's latent space, relative to its total variance
            err[score] = (((Zm - Zt[j][score]) ** 2).sum(1) / Zt[j].var(0).sum()).float()
        mis[j] = err
    print(f'map: {args.map}, fitted on {args.fit_on} ({len(Zp[0])} pilots, {n} test rows)')
    T = {a: y for a in (0, 1)}                  # targets of agent a's decoder
    for spec in args.label_map:
        a, ts = spec.split(':')
        T[int(a)] = torch.tensor([int(t) for t in ts.split(',')])[y]
    # P[a] / C[a] are decoded by agent a, so they are scored against T[a]
    acc_a = lambda a, pr, m: float((pr[m] == T[a][m]).float().mean())
    if args.label_map:
        print('coarse targets:', {a: sorted(set(T[a].tolist())) for a in T if T[a] is not y})
        for a in (0, 1):
            if T[a] is y:     # a fine decoder: also score its predictions after coarsening
                for spec in args.label_map:
                    b, ts = spec.split(':')
                    cm = torch.tensor([int(t) for t in ts.split(',')])
                    print(f'agent {a} decoder coarsened with agent {b}\'s map: private {float((cm[P[a]] == cm[y]).float().mean()):.3f}, '
                          f'comm {1 - a}->{a} {float((cm[C[a]] == cm[y]).float().mean()):.3f}')
    if args.probe:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0)).numpy()
        halves = (perm[: len(y) // 2], perm[len(y) // 2:])
        for a in (0, 1):
            X, Yn = Zt[a].numpy(), y.numpy()
            hits = 0
            for fit, sc in (halves, halves[::-1]):
                sca = StandardScaler().fit(X[fit])
                clf = LogisticRegression(max_iter=3000, C=1.0).fit(sca.transform(X[fit]), Yn[fit])
                hits += (clf.predict(sca.transform(X[sc])) == Yn[sc]).sum()
            print(f'linear probe, 10 classes, agent {a} test latents (2-fold): {hits / len(y):.3f}')

    print(f'{"class":8s} {"priv 0":>7s} {"priv 1":>7s} {"1->0":>7s} {"0->1":>7s}')
    for k in range(10):
        m = y == k
        print(f'{k} {names[k]:6s} {acc_a(0, P[0], m):7.3f} {acc_a(1, P[1], m):7.3f} {acc_a(0, C[0], m):7.3f} {acc_a(1, C[1], m):7.3f}')
    for g in [*args.groups, ','.join(map(str, range(10)))]:
        ks = [int(c) for c in g.split(',')]
        m = torch.isin(y, torch.tensor(ks))
        print(f'{"classes " + g:20s} {acc_a(0, P[0], m):7.3f} {acc_a(1, P[1], m):7.3f} {acc_a(0, C[0], m):7.3f} {acc_a(1, C[1], m):7.3f}'
              f'   rel. residual 1->0 {float(mis[0][m].mean()):.3f}, 0->1 {float(mis[1][m].mean()):.3f}')
    # where do the noisy pairs go? confusion among a class subset
    for g in args.groups:
        ks = [int(c) for c in g.split(',')]
        for lab, pr in (('private 0', P[0]), ('private 1', P[1]), ('comm 1->0', C[0]), ('comm 0->1', C[1])):
            rows = []
            for k in ks:
                m = y == k
                rows.append(' '.join(f'{float((pr[m] == c).float().mean()):.2f}' for c in ks))
            print(f'confusion {lab:10s} classes {ks} (rows = true, cols = predicted): ' + ' | '.join(rows))


if __name__ == '__main__':
    main()
