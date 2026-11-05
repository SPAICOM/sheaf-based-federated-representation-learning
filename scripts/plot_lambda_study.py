"""Curves vs λ for the SheafFRL λ study on grayblur_harsh (tmp/2026-10-28_lmb_study.sh).

Reads the finished runs of the wandb project (variant = run group, i.e.
study_name 'lmb_<variant>') and saves, in --out:
    comm_vs_lambda.png            post-communication accuracy (test, avg of the 2 directions)
    private_vs_lambda.png         private accuracy (test, avg of the 2 agents)
    effective_rank_vs_lambda.png  effective rank of each agent's test latents
    pilot_misalignment_vs_lambda.png  post-hoc residual on the pilots the map is fitted on
    test_misalignment_vs_lambda.png   the same residual on the test set
    lambda_study.csv              the table behind the plots

λ=0 trains like non-coop whatever the SWBN setting, so the SWBN variants all
reuse the 'swbn' λ=0 run. The penalty_lr variants are plotted at their nominal
λ; under SGD their effective weight is λ * penalty_lr / lr.

    uv run python scripts/plot_lambda_study.py
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

METRICS = {
    'comm': 'test/avg_comm_task_perf',
    'private': 'test/avg_private_task_perf',
    'rank_0': 'test/effective_rank_agent_0',
    'rank_1': 'test/effective_rank_agent_1',
    'pilot_misalign': 'pilots/misalignment_loss',
    'test_misalign': 'test/misalignment_loss',
}
LABELS = {
    'swbn': 'SWBN (momentum 0.95)',
    'swbn_mom0.5': 'SWBN momentum 0.5',
    'swbn_mom0.2': 'SWBN momentum 0.2',
    'zca': 'closed-form ZCA',
    'swbn_plr10x': 'SWBN, penalty lr x10',
    'swbn_plr0.1x': 'SWBN, penalty lr x0.1',
    'swbn_warm40': 'SWBN, 40-epoch warmup',
    'swbn_vicreg': 'SWBN + VICReg var/cov terms',
    'swbn_p06_17': 'p=0.6, agent 0: 1<->7, agent 1: 3<->5',
}


def load(project: str, entity: str | None) -> pd.DataFrame:
    api = wandb.Api()
    entity = entity or api.default_entity
    rows = []
    for run in api.runs(f'{entity}/{project}', filters={'state': 'finished'}):
        group = str(run.group or '')
        if not group.startswith('lmb_'):
            continue
        # The listing returns runs with an empty config: fetch each one.
        # Orchestrator hparams are logged at the top level of the run config.
        run = api.run(f'{entity}/{project}/{run.id}')
        row = {'variant': group[4:], 'lmb': float(run.config['max_lmb']),
               'run': run.id, 'created': str(run.created_at)}
        row.update({k: run.summary.get(v) for k, v in METRICS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f'no finished lmb_* runs in {entity}/{project}')
    # newest run per (variant, λ)
    df = df.sort_values('created').groupby(['variant', 'lmb'], as_index=False).last()
    base0 = df[(df.variant == 'swbn') & (df.lmb == 0)]
    extra = [base0.assign(variant=v) for v in df.variant.unique()
             if v.startswith('swbn_') and not ((df.variant == v) & (df.lmb == 0)).any()]
    return pd.concat([df, *extra], ignore_index=True).sort_values(['variant', 'lmb'])


def plot(df: pd.DataFrame, cols: list[str], ylabel: str, path: Path, titles=None,
         suptitle='SheafFRL on grayblur_harsh (2048 pilots, 100 epochs, seed 42)'):
    fig, axes = plt.subplots(1, len(cols), figsize=(6.4 * len(cols), 4.4), squeeze=False)
    for ax, col, title in zip(axes[0], cols, titles or [None] * len(cols)):
        variants = [v for v in LABELS if v in set(df.variant)]
        variants += sorted(set(df.variant) - set(variants))
        for v in variants:
            d = df[df.variant == v]
            ax.plot(d.lmb, d[col], marker='o', ms=4, label=LABELS.get(v, v))
        ax.set_xscale('symlog', linthresh=1e-4)
        ax.set_xlim(-2e-5, 1.3)
        ax.set_xlabel('λ')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='cifar10_2agents_grayblur_harsh_lmb')
    p.add_argument('--entity', default=None)
    p.add_argument('--out', default='results/cifar10_2agents_grayblur_harsh_lmb')
    p.add_argument('--title', default='SheafFRL on grayblur_harsh (2048 pilots, 100 epochs, seed 42)')
    p.add_argument('--rank-titles', nargs=2, default=['agent 0 (grayscale, d=512)', 'agent 1 (blur, d=256)'])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.project, args.entity)
    df.to_csv(out / 'lambda_study.csv', index=False)
    kw = {'suptitle': args.title}
    plot(df, ['comm'], 'post-communication accuracy', out / 'comm_vs_lambda.png', **kw)
    plot(df, ['private'], 'private accuracy', out / 'private_vs_lambda.png', **kw)
    plot(df, ['rank_0', 'rank_1'], 'effective rank (test latents)',
         out / 'effective_rank_vs_lambda.png', titles=args.rank_titles, **kw)
    plot(df, ['pilot_misalign'], 'misalignment loss on the pilots',
         out / 'pilot_misalignment_vs_lambda.png', **kw)
    plot(df, ['test_misalign'], 'misalignment loss on the test set',
         out / 'test_misalignment_vs_lambda.png', **kw)
    print(df.drop(columns=['created']).round(4).to_string(index=False))
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
