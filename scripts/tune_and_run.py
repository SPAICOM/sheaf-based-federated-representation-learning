"""Tune lambda, then train the final seeds: the heterogeneity-table pipeline of one dataset.

For every orchestrator given (default: every method the experiment config sweeps over,
``hydra.sweeper.params.orchestrator``):
  * methods in ``lmb_tuning.no_lmb`` (non-coop, FedRE) train every seed of the experiment
    config (``hydra.sweeper.params.seed``) -> wandb group ``<method>``;
  * methods in ``lmb_tuning.fixed`` (lambda already final) do the same at ``tuned_lmb.<method>``;
  * the others train the first seed at every ``lmb_tuning.grid`` value -> group
    ``<method>_lambda``. The lambda with the best ``test/avg_comm_task_perf`` is written into
    ``tuned_lmb`` of the experiment config and its run moves to group ``<method>``, where it
    is the first seed; only the remaining seeds then train, at that lambda.

Each job is one run of ``scripts/multi_agent_experiment.py``; one at a time unless ``--jobs N``, and
a method's final seeds start as soon as its grid is done. A job whose run already finished on
wandb (same group, seed, lambda) is skipped and one still running there is waited for, so a
stopped pipeline resumes where it was. Console output: ``logs/console/<sweep>_<job>.log``.

    just hetero-pipeline cifar10                              # every method of the config
    just hetero-pipeline cifar10 fedproto,fedre               # only these
    just hetero-pipeline mnist fedproto,comfed --dry-run    # list the jobs, train nothing
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import wandb
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / 'config' / 'hydra'
METRIC = 'test/avg_comm_task_perf'
POLL_S = 300
STALE_S = 900  # a 'running' run without a heartbeat for this long is dead (killed job)
_print_lock = threading.Lock()


def log(msg: str) -> None:
    """Timestamped, thread-safe progress line."""
    with _print_lock:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def experiment_config(dataset: str) -> str:
    """Experiment config of a dataset (same rule as `just hetero`)."""
    return (
        'cifar10_4agents_full'
        if dataset == 'cifar10'
        else f'{dataset}_4agents_shift'
    )


def lmb_key(lmb: float | None) -> float | None:
    """Lambda rounded to 6 significant digits, so config and wandb values compare equal."""
    return None if lmb is None else float(f'{lmb:.6g}')


@dataclass(frozen=True)
class Job:
    """One training run: method, seed, lambda and pipeline stage."""

    method: str
    seed: int
    lmb: float | None  # None: the method has no lambda
    stage: str  # 'grid' | 'final'

    @property
    def group(self) -> str:
        """Wandb group the run is logged to."""
        return f'{self.method}_lambda' if self.stage == 'grid' else self.method

    @property
    def lookup(self) -> tuple[str, ...]:
        """Groups searched for an existing run (a selected grid run lives in the official one)."""
        return (
            (self.group, self.method)
            if self.stage == 'grid'
            else (self.group,)
        )

    @property
    def name(self) -> str:
        """Short job id used in logs."""
        return f'{self.method}_s{self.seed}' + (
            '' if self.lmb is None else f'_l{self.lmb:g}'
        )


class Runs:
    """wandb runs of the experiment project, keyed by (seed, lambda)."""

    def __init__(self, project: str):
        self.api = wandb.Api()
        self.entity, self.project = self.api.default_entity, project
        self.path = f'{self.entity}/{project}'
        self.exists = False

    def project_exists(self) -> bool:
        """Whether wandb has the project yet (a new experiment creates it with its first run)."""
        if not self.exists:
            names = {p.name for p in self.api.projects(self.entity)}
            self.exists = self.project in names
        return self.exists

    def fetch(self, groups: tuple[str, ...], with_lmb: bool) -> dict:
        """Return the runs of the given groups as {(seed, lambda): [runs]}."""
        out: dict = {}
        if not self.project_exists():
            return out
        self.api.flush()  # the Api caches run listings: without this, states never update
        for group in groups:
            for listed in self.api.runs(self.path, filters={'group': group}):
                r = self.api.run(
                    '/'.join(listed.path)
                )  # listings carry no config
                cfg = r.config
                lmb = (
                    lmb_key(float(cfg['orchestrator']['max_lmb']))
                    if with_lmb
                    else None
                )
                out.setdefault((int(cfg['seed']), lmb), []).append(r)
        return out

    def matching(self, job: Job) -> list:
        """Existing runs with the job's seed and lambda."""
        return self.fetch(job.lookup, job.lmb is not None).get(
            (job.seed, lmb_key(job.lmb)), []
        )


def alive(run) -> bool:
    """Run is running and still sending heartbeats (a killed job stays 'running' for a while)."""
    if run.state != 'running':
        return False
    beat = datetime.fromisoformat(run.heartbeatAt.replace('Z', '+00:00'))
    return (datetime.now(UTC) - beat).total_seconds() < STALE_S


def finished(run) -> bool:
    """Run finished and logged the selection metric."""
    return run.state == 'finished' and run.summary.get(METRIC) is not None


def write_tuned_lmb(path: Path, method: str, lmb: float) -> None:
    """Set ``tuned_lmb.<method>`` in an experiment config, keeping its comments."""
    lines = path.read_text().splitlines(keepends=True)
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith('tuned_lmb:')
    )
    end = start + 1
    while end < len(lines) and lines[end].startswith((' ', '\t')):
        end += 1
    entry = f'  {method}: {lmb:g}\n'
    for i in range(start + 1, end):
        if lines[i].split(':')[0].strip() == method:
            comment = lines[i].partition('#')[2]
            lines[i] = entry if not comment else f'{entry[:-1]}   #{comment}'
            break
    else:
        lines.insert(end, entry)
    tmp = path.with_suffix('.yaml.tmp')
    tmp.write_text(''.join(lines))
    os.replace(tmp, path)


class Pipeline:
    """Job scheduling, lambda selection and bookkeeping for one dataset."""

    def __init__(self, args):
        self.config = experiment_config(args.dataset)
        with initialize_config_dir(
            config_dir=str(CONFIG_DIR), version_base='1.3'
        ):
            cfg = compose(config_name=self.config, return_hydra_config=True)
        self.config_path = CONFIG_DIR / f'{self.config}.yaml'
        self.seeds = [
            int(s) for s in str(cfg.hydra.sweeper.params.seed).split(',')
        ]
        grid = args.grid.split(',') if args.grid else cfg.lmb_tuning.grid
        self.grid = [float(x) for x in grid]
        self.no_lmb = set(cfg.lmb_tuning.no_lmb)
        self.fixed = {m: float(cfg.tuned_lmb[m]) for m in cfg.lmb_tuning.fixed}
        methods = args.orchestrators or str(
            cfg.hydra.sweeper.params.orchestrator
        )
        self.methods = methods.split(',')
        for m in self.methods:
            if not (CONFIG_DIR / 'orchestrator' / f'{m}.yaml').exists():
                sys.exit(f'unknown orchestrator {m!r}')
        self.runs = Runs(cfg.logger.project)
        self.dry = args.dry_run
        self.sweep = os.environ.get('SWEEP', f'hetero_pipeline_{args.dataset}')
        self.jobs = args.jobs
        self.write_lock = threading.Lock()
        self.launched: list[str] = []
        log(
            f'{self.config} (wandb {cfg.logger.project}): seeds {self.seeds}, '
            f'grid {[f"{x:g}" for x in self.grid]}, no lambda: {sorted(self.no_lmb)}, fixed lambda: {self.fixed}'
        )

    def cmd(self, job: Job) -> list[str]:
        """Command line of a job."""
        cmd = [
            sys.executable, 'scripts/multi_agent_experiment.py', '-m',
            '--config-name', self.config,
            f'orchestrator={job.method}', f'seed={job.seed}', f'study_name={job.group}',
        ]  # fmt: skip
        if job.lmb is not None:
            cmd.append(f'orchestrator.max_lmb={job.lmb!r}')
        return cmd

    def run(self, job: Job) -> bool:
        """Train one job unless wandb already has it; True if its run is finished."""
        while True:
            runs = self.runs.matching(job)
            if any(finished(r) for r in runs):
                log(f'{job.name}: finished on wandb ({runs[0].id}), skipped')
                return True
            if not any(alive(r) for r in runs):
                break
            if self.dry:
                log(
                    f'{job.name}: running on wandb, the pipeline would wait for it'
                )
                return True
            log(f'{job.name}: running on wandb, waiting')
            time.sleep(POLL_S)
        self.launched.append(job.name)
        if self.dry:
            log(f'{job.name}: would run  {" ".join(self.cmd(job)[1:])}')
            return True
        console = ROOT / 'logs' / 'console' / f'{self.sweep}_{job.name}.log'
        console.parent.mkdir(
            parents=True, exist_ok=True
        )  # git-ignored: absent in a fresh clone
        log(f'{job.name}: start  (log {console.relative_to(ROOT)})')
        with console.open('a') as f:
            rc = subprocess.run(
                self.cmd(job), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
            ).returncode
        log(f'{job.name}: ' + ('done' if rc == 0 else f'FAILED (exit {rc})'))
        return rc == 0

    def select(self, method: str) -> float | None:
        """Best grid lambda of a method -> tuned_lmb + official group; None if the grid is incomplete."""
        runs = self.runs.fetch((f'{method}_lambda', method), with_lmb=True)
        rows = []
        for lmb in self.grid:
            done = [
                r
                for r in runs.get((self.seeds[0], lmb_key(lmb)), [])
                if finished(r)
            ]
            if not done:
                log(
                    f'{method}: no finished grid run at lambda={lmb:g}, lambda not selected'
                )
                return None
            rows.append((done[0].summary[METRIC], lmb, done[0]))
        score, best, run = max(rows, key=lambda t: t[0])
        table = '  '.join(f'{lmb:g}:{100 * s:.1f}' for s, lmb, _ in rows)
        log(
            f'{method}: comm per lambda  {table}  -> best {best:g} ({100 * score:.1f}, run {run.id})'
        )
        if not self.dry:
            with self.write_lock:
                write_tuned_lmb(self.config_path, method, best)
            if run.group != method:
                run.group = method
                run.update()
            log(
                f'{method}: tuned_lmb.{method}={best:g} in {self.config_path.name}, run {run.id} -> group {method}'
            )
        return best

    def main(self) -> int:
        """Run every job; exit code 1 if any failed."""
        pool = ThreadPoolExecutor(self.jobs)
        pending: dict = {}
        grid_left: dict[str, int] = {}
        grid_ok: dict[str, bool] = {}
        failed: list[str] = []

        def submit(job: Job) -> None:
            pending[pool.submit(self.run, job)] = job

        for m in self.methods:
            if m in self.no_lmb:
                for s in self.seeds:
                    submit(Job(m, s, None, 'final'))
            elif m in self.fixed:
                for s in self.seeds:
                    submit(Job(m, s, self.fixed[m], 'final'))
            else:
                grid_left[m], grid_ok[m] = len(self.grid), True
                for lmb in self.grid:
                    submit(Job(m, self.seeds[0], lmb, 'grid'))

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                job = pending.pop(fut)
                ok = fut.result()
                if not ok:
                    failed.append(job.name)
                if job.stage != 'grid':
                    continue
                grid_ok[job.method] &= ok
                grid_left[job.method] -= 1
                if grid_left[job.method]:
                    continue
                best = self.select(job.method) if grid_ok[job.method] else None
                if best is not None:
                    for s in self.seeds[1:]:
                        submit(Job(job.method, s, best, 'final'))
                elif self.dry:
                    self.launched += [
                        f'{job.method}_s{s}_l<best>' for s in self.seeds[1:]
                    ]
                    log(
                        f'{job.method}: after the grid, seeds {self.seeds[1:]} at the best lambda'
                    )
                else:
                    log(
                        f'{job.method}: grid incomplete, final seeds not trained'
                    )
        pool.shutdown()

        verb = 'would train' if self.dry else 'trained'
        log(
            f'{verb} {len(self.launched)} runs'
            + (f'; FAILED: {", ".join(failed)}' if failed else '')
        )
        return 1 if failed else 0


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('dataset', help='mnist | fashionmnist | femnist | cifar10')
    p.add_argument(
        'orchestrators',
        nargs='?',
        help='comma-separated orchestrator configs, e.g. non_cooperative,fedproto '
        '(default: hydra.sweeper.params.orchestrator of the config)',
    )
    p.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='runs trained at once (default 1; raise it only if GPU memory and RAM allow)',
    )
    p.add_argument(
        '--grid',
        help='comma-separated lambda values (default: lmb_tuning.grid of the config)',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='list the jobs and the selection, train nothing',
    )
    return Pipeline(p.parse_args()).main()


if __name__ == '__main__':
    sys.exit(main())
