"""Stop Hydra from writing ``outputs/`` / ``multirun/`` run directories.

Only ``logs/hydra/multirun.yaml`` (one file, overwritten) is left behind.

wandb already stores the resolved config and metrics, so Hydra's per-run
directory (``.hydra/`` config copies + a ``<job>.log`` file) is redundant.
Call :func:`disable_hydra_outputs` right before a ``@hydra.main`` entry point::

    if __name__ == '__main__':
        from src.utils.hydra_outputs import disable_hydra_outputs

        disable_hydra_outputs()
        main()

Each override is appended to ``sys.argv`` unless the user passed that key
explicitly, so e.g. ``hydra.run.dir=...`` on the command line still wins.
"""

from __future__ import annotations

import sys

NO_OUTPUT_OVERRIDES = {
    'hydra.output_subdir': 'null',  # no .hydra/ config copies
    'hydra.run.dir': '.',  # single run: no outputs/<date>/<time>/
    # multirun: no multirun/<date>/<time>/. The basic sweeper always writes
    # <sweep.dir>/multirun.yaml, so park that single file in git-ignored logs/.
    'hydra.sweep.dir': 'logs/hydra',
    'hydra.sweep.subdir': '.',  # ... and no per-job 0/, 1/, ... dirs
    'hydra/job_logging': 'stdout',  # console only, no <job>.log file
}


def disable_hydra_outputs() -> None:
    given = {
        arg.split('=', 1)[0].lstrip('+~') for arg in sys.argv[1:] if '=' in arg
    }
    sys.argv += [
        f'{key}={value}'
        for key, value in NO_OUTPUT_OVERRIDES.items()
        if key not in given
    ]
