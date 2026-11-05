"""Per-agent accuracies vs λ for the SheafFRL λ study on the coarse-task setup
(tmp/2026-10-31_coarse_lmb.sh): ResNet-10 on 10 classes vs a plain CNN on
animate/inanimate, plus the all-fine control.

Agent 1's logged numbers are wrong in the coarse setup (scored on 10 classes),
so every run is re-scored with scripts/per_class_comm.py (map fitted on the
pilots, as the evaluation; and cross-fitted on the test set), with a 10-class
linear probe of each agent's latents. Outputs are cached in --out/raw/ and
re-used. λ=0 is the non-coop run (SheafFRL at λ=0 trains exactly like it).

    uv run python scripts/plot_coarse_lmb_study.py
"""
import argparse
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NONCOOP = {'coarse': 'apddia1c', 'control': 'b0k50l6w'}
# (setup, λ) -> wandb run id of the finished sweep (project cifar10_2agents_coarse)
SHEAF = {
    ('coarse', 1e-3): 'p2zpejxt', ('coarse', 1e-2): 'eo2mm6sx', ('coarse', 3e-2): 'gsked4kc',
    ('coarse', 1e-1): '8i7wgnp7', ('coarse', 3e-1): '3aw1il2x',
    ('control', 1e-3): '8b103qzs', ('control', 1e-2): 'je375hqh', ('control', 3e-2): 'ukwr4sh7',
    ('control', 1e-1): '83dlo9ug', ('control', 3e-1): 'xoy2qagk',
}
SETUP_ARGS = {'coarse': ['--label-map', '1:0,0,1,1,1,1,1,1,0,0'],
              'control': ['--overrides', 'dataset.agent_label_map=null']}
ALL = 'classes 0,1,2,3,4,5,6,7,8,9'


def runs() -> list[tuple[str, float, str]]:
    return [(s, 0.0, r) for s, r in NONCOOP.items()] + [(s, lmb, r) for (s, lmb), r in SHEAF.items()]


def score(setup: str, run: str, fit: str, raw: Path) -> str:
    path = raw / f'{run}_{fit}.txt'
    if not path.exists():
        cmd = ['uv', 'run', 'python', 'scripts/per_class_comm.py', '--config', 'cifar10_2agents_coarse',
               '--project', 'cifar10_2agents_coarse', '--run', run, '--fit-on', fit,
               '--groups', '3,5', *(['--probe'] if fit == 'pilots' else []), *SETUP_ARGS[setup]]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if ALL not in res.stdout:
            raise RuntimeError(f'{run} {fit}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}')
        path.write_text(res.stdout)
    return path.read_text()


def parse(text: str, prefix: str) -> list[float]:
    line = next(l for l in text.splitlines() if l.startswith(prefix))
    return [float(x) for x in line[len(prefix):].split()[:4]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/cifar10_2agents_coarse/lmb_study')
    args = p.parse_args()
    out = ROOT / args.out
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for setup, lmb, run in runs():
        pil, cv = score(setup, run, 'pilots', raw), score(setup, run, 'test_cv', raw)
        p0, p1, c0, c1 = parse(pil, ALL)
        _, _, cv0, cv1 = parse(cv, ALL)
        probe = [float(x) for x in re.findall(r'linear probe.*: ([0-9.]+)', pil)]
        rows.append({'setup': setup, 'lmb': lmb, 'run': run,
                     'private_0': p0, 'private_1': p1, 'comm_into_0': c0, 'comm_into_1': c1,
                     'comm_into_0_cv': cv0, 'comm_into_1_cv': cv1,
                     'probe_0': probe[0], 'probe_1': probe[1],
                     'cat_dog_comm_into_0': parse(pil, 'classes 3,5')[2]})
    df = pd.DataFrame(rows).sort_values(['setup', 'lmb'])
    df.to_csv(out / 'lambda_study.csv', index=False)

    panels = [
        ('private_0', 'private, agent 0 (ResNet-10, 10 classes)'),
        ('private_1', 'private, agent 1 (CNN; coarse: animate/inanimate)'),
        ('comm_into_0', 'comm into agent 0: agent 1 latents -> 10-class decoder'),
        ('comm_into_1', 'comm into agent 1: agent 0 latents -> agent 1 decoder'),
        ('probe_1', '10-class linear probe on agent 1 latents'),
        ('cat_dog_comm_into_0', 'comm into agent 0 on cat/dog only'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (col, title) in zip(axes.flat, panels):
        for setup, style in (('coarse', '-'), ('control', '--')):
            d = df[df.setup == setup]
            ax.plot(d.lmb, d[col], style, marker='o', ms=4, label=f'{setup}, map on pilots'
                    if col.startswith('comm') else setup)
            if col in ('comm_into_0', 'comm_into_1'):
                ax.plot(d.lmb, d[col + '_cv'], style, marker='x', ms=5, alpha=0.6,
                        label=f'{setup}, map on test (2-fold oracle)')
        ax.set_xscale('symlog', linthresh=1e-3)
        ax.set_xlabel('λ (0 = non-coop)')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('SheafFRL (SWBN) λ study: ResNet-10 on 10 classes vs plain CNN on animate/inanimate '
                 '(solid) and all-fine control (dashed); full CIFAR-10 test set', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / 'per_agent_vs_lambda.png', dpi=140)
    print(df.round(3).to_string(index=False))
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
