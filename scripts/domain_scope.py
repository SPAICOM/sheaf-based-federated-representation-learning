"""Per-domain accuracy of the DomainNet runs (one visual domain per agent), report and CSV.

The shared test set pools every agent's held-out rows, so the logged ``private`` / ``comm``
accuracies average over all domains. This script reloads the final checkpoint of every finished
run of the selected wandb groups (``rescore_hetero.load``), fits the run's post-hoc maps on the
pilots and splits the accuracies by the domain of the test row (``dm.row_writers``):

    private[i][d]   agent i's decoder on its own latents of the domain-d rows
    comm[s->r][d]   receiver r's decoder on the sender's latents, mapped s -> r

Both are counted on the agent's (for comm, the receiver's) target classes: under label shift, the
rows of a class an agent never trained on are left out, since it cannot predict them. Per-run
scores are cached in results/<project>/domain_scope/<run_id>.json.

Runs of the same method (and lambda) form one column, mean ± std over their seeds. The report,
results/<project>/domain_scope_report.md, has:

    per test domain   comm averaged over the directed pairs, private averaged over the agents
    own vs unseen     per method, mean over agents of private and comm (agent as receiver) on its
                      own domain vs the other domains, and the logged test comm
    best method       highest logged test/avg_comm_task_perf over seeds (the tune_and_run.py rule):
                      one table per agent, rows = test domain, columns = private and comm from
                      each neighbour, with the change vs --baseline paired by seed (the seed also
                      draws the split and the pilots); bold when it has the same sign in every seed

results/<project>/domain_scope.csv holds one row per run, cell and test domain; it feeds
scripts/plot_domain_scope.py.

    uv run python scripts/domain_scope.py domainnet_6agents
    uv run python scripts/domain_scope.py domainnet_6agents_shift
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import wandb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from rescore_hetero import POSTHOC, bind_posthoc, encode, load  # noqa: E402

from src.communication.alignment_mixin import (  # noqa: E402
    PostTrainingAlignmentMixin,
)

BASELINE = 'NonCooperativeLearning'
OWN_UNSEEN = {
    'priv_mine': 'private, own domain',
    'priv_other': 'private, unseen domains',
    'comm_mine': 'comm, own domain',
    'comm_other': 'comm, unseen domains',
}


# ── Scoring ──────────────────────────────────────────────────────────────────


def _acc(
    logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> list[int]:
    return [int((logits.argmax(1)[mask] == y[mask]).sum()), int(mask.sum())]


@torch.no_grad()
def score(run, project: str, device: str) -> dict:
    """[correct, total] of every agent (private) and directed pair (comm), per test domain."""
    orch, dm, cfg = load(run, project, device)
    agents = {int(k): a for k, a in orch.agents.items()}
    neighbors = {
        int(k): {int(n) for n in v} for k, v in orch.hparams.neighbors.items()
    }
    feature = dm.test_datasets[0].dataset.features[dm.writer_key]
    name = feature.int2str if hasattr(feature, 'int2str') else str
    own = {
        int(i): {feature.str2int(w) if isinstance(w, str) else w for w in ws}
        for i, ws in cfg.dataset.writer_groups.items()
    }
    target = orch._resolve_agent_target_classes() or {}

    Z, y = {}, None
    for i, agent in agents.items():
        agent.eval()
        Z[i], yi = encode(agent, dm.test_datasets[i], device)
        y = yi if y is None else y
        if not torch.equal(y, yi):
            raise ValueError(f'{run.id}: test sets differ across agents')
    dom = torch.tensor(
        [dm.row_writers[k] for k in dm.test_datasets[0].sample_ids]
    )
    domains = sorted(set(dom.tolist()))
    y = y.cpu()

    def in_classes(i: int) -> torch.Tensor:
        if not target.get(i):
            return torch.ones_like(y, dtype=torch.bool)
        return torch.tensor([int(c) in target[i] for c in y.tolist()])

    out = {
        'run_id': run.id,
        'seed': cfg.seed,
        'group': run.group,
        'domains': {d: name(d) for d in domains},
        'agent_domains': {
            i: [name(d) for d in sorted(own[i])] for i in agents
        },
        'private': {},
        'comm': {},
    }
    for i, agent in agents.items():
        pred, cls = agent.decoder(Z[i]).cpu(), in_classes(i)
        out['private'][i] = {
            name(d): _acc(pred, y, cls & (dom == d)) for d in domains
        }

    # ComFed communicates through its own projections; every other method through post-hoc maps
    native = hasattr(orch, 'projection_matrices') and not isinstance(
        orch, PostTrainingAlignmentMixin
    )
    if not native:
        bind_posthoc(orch)
        if not orch._fit_alignment_maps(dm):
            raise RuntimeError(f'{run.id}: no post-hoc map fitted')
    try:
        for s, receivers in neighbors.items():
            for r in sorted(receivers - {s}):
                pred, cls = (
                    agents[r].decoder(orch.send_message(s, r, Z[s])).cpu(),
                    in_classes(r),
                )
                out['comm'][f'{s}->{r}'] = {
                    name(d): _acc(pred, y, cls & (dom == d)) for d in domains
                }
    finally:
        if not native:
            orch._cleanup_alignment()
            for n in POSTHOC:
                orch.__dict__.pop(n, None)
    return out


# ── Aggregation ──────────────────────────────────────────────────────────────


def _rate(c: list[int]) -> float:
    return 100 * c[0] / c[1] if c[1] else float('nan')


def _stats(xs) -> tuple[float, float, int]:
    """Mean, sample std (nan for a single value) and count of the non-nan values."""
    xs = [x for x in xs if x == x]
    if not xs:
        return float('nan'), float('nan'), 0
    m = sum(xs) / len(xs)
    s = (
        (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        if len(xs) > 1
        else float('nan')
    )
    return m, s, len(xs)


def _ms(xs) -> str:
    m, s, n = _stats(xs)
    return '–' if not n else (f'{m:.1f} ± {s:.1f}' if n > 1 else f'{m:.1f}')


def domain_order(res: dict) -> list[str]:
    """Test domains in agent order (agent 0's domain first)."""
    seen = [d for ds in res['agent_domains'].values() for d in ds]
    return list(dict.fromkeys(seen)) + [
        d for d in res['domains'].values() if d not in seen
    ]


def own_unseen(res: dict) -> dict[str, float]:
    """Mean over agents of private / comm (agent as receiver) on its own domain vs the others (%)."""
    per_agent = []
    for i, own in res['agent_domains'].items():
        m = {k: [0, 0] for k in OWN_UNSEEN}
        for d in res['domains'].values():
            side = 'mine' if d in own else 'other'
            cells = [('priv', res['private'][i][d])]
            cells += [
                ('comm', per[d])
                for e, per in res['comm'].items()
                if e.split('->')[1] == i
            ]
            for kind, (c, n) in cells:
                m[f'{kind}_{side}'][0] += c
                m[f'{kind}_{side}'][1] += n
        per_agent.append({k: _rate(v) for k, v in m.items()})
    return {k: _stats(p[k] for p in per_agent)[0] for k in OWN_UNSEEN}


def domain_means(res: dict) -> dict[str, dict[str, float]]:
    """Per test domain: private averaged over the agents, comm averaged over the directed pairs (%)."""
    doms = domain_order(res)
    return {
        'private': {
            d: _stats(_rate(a[d]) for a in res['private'].values())[0]
            for d in doms
        },
        'comm': {
            d: _stats(_rate(a[d]) for a in res['comm'].values())[0]
            for d in doms
        },
    }


def short_labels(labels: list[str]) -> list[str]:
    """'NC' for the baseline; 'λ0.03' when a single penalised method is compared, else 'Method λ0.03'."""
    methods = {lab.split(' λ=')[0] for lab in labels if 'λ=' in lab}
    out = []
    for lab in labels:
        if lab == BASELINE:
            out.append('NC')
        elif 'λ=' in lab:
            m, lam = lab.split(' λ=')
            out.append(f'λ{lam}' if len(methods) == 1 else f'{m} λ{lam}')
        else:
            out.append(lab)
    return out


# ── Report ───────────────────────────────────────────────────────────────────
# A method = (label, {seed: run}), a run = {'id', 'label', 'seed', 'res', 'logged'}.


def summary_tables(methods: list[tuple[str, dict]]) -> list[str]:
    """One row per test domain, one column per method (mean ± std over seeds), comm then private."""
    labels = short_labels([lab for lab, _ in methods])
    first = next(iter(methods[0][1].values()))['res']
    per = [
        [domain_means(r['res']) for r in seeds.values()]
        for _, seeds in methods
    ]
    head = [
        '| test domain | ' + ' | '.join(labels) + ' |',
        '|---|' + '---:|' * len(labels),
    ]
    out = []
    for kind, title in [
        (
            'comm',
            f'Communication accuracy per test domain, mean over the {len(first["comm"])} directed pairs',
        ),
        (
            'private',
            f'Private accuracy per test domain, mean over the {len(first["private"])} agents',
        ),
    ]:
        out += [f'**{title}**', '', *head]
        out += [
            f'| {d} | '
            + ' | '.join(_ms(m[kind][d] for m in p) for p in per)
            + ' |'
            for d in domain_order(first)
        ]
        out.append('')
    return out


def agent_tables(seeds: dict, base: dict | None) -> list[str]:
    """One table per agent: rows = test domain (own first), private + comm from each neighbour.

    Cell: mean over ``seeds``, then the change vs ``base`` (the baseline's runs by seed) paired by
    seed, mean ± std; bold when it has the same sign in every seed (two seeds or more).
    """
    res0 = next(iter(seeds.values()))['res']
    own = res0['agent_domains']

    def cell(get) -> str:
        m, _, n = _stats([_rate(get(r['res'])) for r in seeds.values()])
        if not n:
            return '–'
        txt = f'{m:.1f}'
        if base:
            ds = [
                _rate(get(seeds[s]['res'])) - _rate(get(base[s]['res']))
                for s in seeds
                if s in base
            ]
            dm, dsd, dn = _stats(ds)
            if dn:
                t = f'{dm:+.1f}' + (f' ± {dsd:.1f}' if dn > 1 else '')
                if dn > 1 and (
                    all(x > 0 for x in ds) or all(x < 0 for x in ds)
                ):
                    t = f'**{t}**'
                txt += f' ({t})'
        return txt

    def name(i: str) -> str:
        return f'{i} {",".join(own[i])}'

    out = []
    for i in own:
        senders = sorted(
            (e.split('->')[0] for e in res0['comm'] if e.split('->')[1] == i),
            key=int,
        )
        out += [
            f'**Agent {name(i)}**',
            '',
            '| test domain | private | '
            + ' | '.join(f'comm from {name(s)}' for s in senders)
            + ' |',
            '|---|' + '---:|' * (1 + len(senders)),
        ]
        doms = domain_order(res0)
        for d in [d for d in doms if d in own[i]] + [
            d for d in doms if d not in own[i]
        ]:
            row = [cell(lambda r, d=d: r['private'][i][d])]
            row += [
                cell(lambda r, d=d, e=f'{s}->{i}': r['comm'][e][d])
                for s in senders
            ]
            out.append(
                f'| {f"**{d}** (own)" if d in own[i] else d} | '
                + ' | '.join(row)
                + ' |'
            )
        out.append('')
    return out


def report(
    project: str, methods: list[tuple[str, dict]], base_label: str | None
) -> str:
    """Markdown report of every method (see the module docstring)."""
    short = dict(
        zip([m for m, _ in methods], short_labels([m for m, _ in methods]))
    )
    n_seeds = sorted({len(s) for _, s in methods})
    seeds_all = sorted({s for _, seeds in methods for s in seeds})
    base = dict(methods).get(base_label)
    best_lab, best = max(
        methods, key=lambda m: _stats(r['logged'] for r in m[1].values())[0]
    )
    use_base = base if base is not None and best_lab != base_label else None
    lines = [
        f'# {project}: accuracy per test domain',
        '',
        'Accuracies in %, on the classes the agent (the receiver, for comm) trained on; mean ± std over '
        f'{"/".join(map(str, n_seeds))} seeds.',
        '',
        *summary_tables(methods),
        '**Mean over agents, own domain vs the others** (comm: the agent as receiver; '
        'logged comm: test/avg_comm_task_perf)',
        '',
        '| method | ' + ' | '.join(OWN_UNSEEN.values()) + ' | logged comm |',
        '|---|' + '---:|' * (len(OWN_UNSEEN) + 1),
    ]
    lines += [
        f'| {short[lab]} | '
        + ' | '.join(
            _ms(r['mean'][k] for r in seeds.values()) for k in OWN_UNSEEN
        )
        + f' | {_ms(r["logged"] for r in seeds.values())} |'
        for lab, seeds in methods
    ]
    lines += [
        '',
        f'## Best: {best_lab} (highest logged test comm, {_ms(r["logged"] for r in best.values())})',
        '',
        'Per agent: rows are test domains (own first); private is its own encoder + decoder, '
        '"comm from s" is s encoding the row, the post-hoc map sending it, the agent decoding. '
        'Cell: mean over seeds'
        + (
            f'; (Δ ± std) vs {short[base_label]}, paired by seed; **bold**: same sign in every seed.'
            if use_base
            else '.'
        ),
        '',
        *agent_tables(best, use_base),
        '**Runs**',
        '',
        '| method | ' + ' | '.join(f'seed {s}' for s in seeds_all) + ' |',
        '|---|' + '---|' * len(seeds_all),
    ]
    lines += [
        f'| {short[lab]} | '
        + ' | '.join(
            f'`{seeds[s]["id"]}`' if s in seeds else '–' for s in seeds_all
        )
        + ' |'
        for lab, seeds in methods
    ]
    return '\n'.join(lines) + '\n'


def long_rows(run: dict) -> list[dict]:
    """One CSV row per (kind, sender, receiver, test domain) of one run."""
    res, own = run['res'], run['res']['agent_domains']
    cells = [('private', i, i, accs) for i, accs in res['private'].items()]
    cells += [
        ('comm', *e.split('->'), accs) for e, accs in res['comm'].items()
    ]
    return [
        {
            'run_id': res['run_id'],
            'group': res['group'],
            'method': run['label'],
            'seed': res['seed'],
            'sender_domain': ','.join(own[s]),
            'receiver_domain': ','.join(own[r]),
            'kind': kind,
            'sender': s,
            'receiver': r,
            'test_domain': d,
            'own_domain': d in own[r],
            'acc': _rate(c),
            'n': c[1],
            'logged_comm': run['logged'],
        }
        for kind, s, r, accs in cells
        for d, c in accs.items()
        if not d.startswith('@')
    ]


def run_label(run) -> str:
    """Return the method class name, plus lambda for methods with a penalty."""
    orch = run.config.get('orchestrator', {})
    method = str(orch.get('_target_', run.group)).split('.')[-1]
    lmb = orch.get('max_lmb')
    return (
        method
        if not lmb or method == BASELINE
        else f'{method} λ={float(lmb):g}'
    )


def main() -> None:
    """Score every finished run of the groups (cached), then write the report and the CSV."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('project')
    p.add_argument(
        '--groups',
        default='non_cooperative,di_sheaf_frl_lambda',
        help='wandb groups to read',
    )
    p.add_argument(
        '--runs',
        default=None,
        help='comma-separated run ids (default: every finished run)',
    )
    p.add_argument(
        '--baseline',
        default=BASELINE,
        help='method label the per-agent changes refer to',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='re-score runs that are already cached',
    )
    a = p.parse_args()
    out = ROOT / 'results' / a.project
    cache_dir = out / 'domain_scope'
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = (
        'cuda'
        if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    api = wandb.Api()
    path = f'{api.default_entity}/{a.project}'

    wanted = set(a.runs.split(',')) if a.runs else None
    runs = []
    for group in a.groups.split(','):
        for listed in api.runs(
            path, filters={'group': group, 'state': 'finished'}
        ):
            if wanted and listed.id not in wanted:
                continue
            cache = cache_dir / f'{listed.id}.json'
            run = api.run(
                f'{path}/{listed.id}'
            )  # a listed run's config is lazy (empty)
            if a.force or not cache.exists():
                cache.write_text(
                    json.dumps(score(run, a.project, device), indent=1)
                )
            res = json.loads(
                cache.read_text()
            )  # str keys, as every cached result
            logged = run.summary.get('test/avg_comm_task_perf')
            runs.append(
                {
                    'id': listed.id,
                    'label': run_label(run),
                    'seed': int(res['seed']),
                    'res': res,
                    'mean': own_unseen(res),
                    'logged': 100 * logged
                    if logged is not None
                    else float('nan'),
                }
            )
            print(
                f'scored {listed.id}  {runs[-1]["label"]}  seed {res["seed"]}',
                flush=True,
            )
    if not runs:
        raise SystemExit(f'no finished run in groups {a.groups} of {path}')

    lam = lambda lab: float(lab.split('λ=')[1]) if 'λ=' in lab else -1.0  # noqa: E731
    methods = []
    for lab in sorted({r['label'] for r in runs}, key=lam):
        seeds = {}
        for r in sorted(
            (r for r in runs if r['label'] == lab), key=lambda r: r['seed']
        ):
            if r['seed'] in seeds:
                print(
                    f'warning: {lab} seed {r["seed"]}: {r["id"]} ignored, {seeds[r["seed"]]["id"]} kept'
                )
                continue
            seeds[r['seed']] = r
        methods.append((lab, seeds))

    text = report(a.project, methods, a.baseline)
    (out / 'domain_scope_report.md').write_text(text)
    rows = [
        row
        for _, seeds in methods
        for r in seeds.values()
        for row in long_rows(r)
    ]
    with open(out / 'domain_scope.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(text)
    print(
        f'wrote {out}/domain_scope_report.md and domain_scope.csv ({len(rows)} rows)'
    )


if __name__ == '__main__':
    main()
