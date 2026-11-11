#!/usr/bin/env bash
# Pilot-count x method x seed sweep for the 4-agent FashionMNIST shift=1.0
# study (config/hydra/fashionmnist_4agents_shift.yaml), non_cooperative vs
# DiSheafFRL at a FIXED lambda (orchestrator.max_lmb, set in the config;
# lmb_study stays off for every cell).
#
#   PILOTS x METHODS x SEEDS = 4 x 2 x 3 = 24 runs
#
# Pilots change each agent's training set (pilot_exclusive: true), so every
# cell — non-coop included — is its own training run. Do NOT reuse one
# non-coop run across pilot budgets via scripts/pilot_count_sweep.py here.
#
# All 24 runs share one wandb group (the config's study_name -> logger.group)
# and log to logger.project: fashionmnist_4agents_shift. Each run's wandb
# NAME also carries the method, pilot budget and seed (multi_agent_experiment
# .py's run-name builder: "<orchestrator>_..._shift_1.0_p<pilots>_seed<seed>_
# <job_num>"), so a cell can be found by filtering runs on name or on config
# (dataset.pilot_num_samples / seed / orchestrator_name).
#
# Resumability: every job writes a marker under
# logs/fashionmnist_4agents_shift_sweep/done/ on a clean exit, and a job with
# a marker is skipped — interrupt and restart freely. This is a LOCAL marker
# only (no wandb API call before each job); if you deleted a run from wandb
# after it finished, remove its marker too, or `grep -rl` the pilots/method/
# seed with scripts/list_wandb_runs.py to find it.
#
# Usage:
#   scripts/run_fashionmnist_4agents_shift_sweep.sh --dry-run   # print the 24 commands, run nothing
#   scripts/run_fashionmnist_4agents_shift_sweep.sh             # launch (seeds 1 2 3)
#   SEEDS="1 2" scripts/run_fashionmnist_4agents_shift_sweep.sh # subset of seeds
#   NUM_WORKERS=0 scripts/run_fashionmnist_4agents_shift_sweep.sh  # force dataset.num_workers=0
#                                                                 # (e.g. Apple MPS DataLoader-worker crash;
#                                                                 # default: leave the config's own value, 6)
#   WAIT_PID=12345 scripts/run_fashionmnist_4agents_shift_sweep.sh  # start once that pid exits

set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1

PILOTS="${PILOTS:-384 768 2048 4096}"
METHODS="${METHODS:-non_cooperative di_sheaf_frl}"
SEEDS="${SEEDS:-1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-}"
WAIT_PID="${WAIT_PID:-}"
DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

OUT_DIR="logs/fashionmnist_4agents_shift_sweep"
DONE_DIR="$OUT_DIR/done"
mkdir -p "$DONE_DIR"

log() { echo "[$(date '+%F %T')] $*"; }

if [[ -n "$WAIT_PID" && "$DRY_RUN" -eq 0 ]]; then
  log "waiting for pid $WAIT_PID to exit before starting …"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid $WAIT_PID has exited; starting."
  sleep 30  # let the GPU settle
fi

# run <pilots> <method> <seed>
run() {
  local pilots="$1" method="$2" seed="$3"
  local tag="p${pilots}_${method}_seed${seed}"
  local marker="$DONE_DIR/${tag}.done"

  local overrides=(
    --config-name=fashionmnist_4agents_shift
    "orchestrator=$method"
    "dataset.pilot_num_samples=$pilots"
    "seed=$seed"
  )
  [[ -n "$NUM_WORKERS" ]] && overrides+=("dataset.num_workers=$NUM_WORKERS")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "uv run python scripts/multi_agent_experiment.py ${overrides[*]}"
    return 0
  fi

  if [[ -f "$marker" ]]; then
    log "SKIP $tag (already done)"
    return 0
  fi
  log "RUN  $tag  (${overrides[*]})"
  if uv run python scripts/multi_agent_experiment.py "${overrides[@]}" \
      >>"$OUT_DIR/${tag}.log" 2>&1; then
    touch "$marker"
    log "OK   $tag"
  else
    log "FAIL $tag — see $OUT_DIR/${tag}.log"
  fi
}

for pilots in $PILOTS; do
  for method in $METHODS; do
    for seed in $SEEDS; do
      run "$pilots" "$method" "$seed"
    done
  done
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  log "all done."
fi
