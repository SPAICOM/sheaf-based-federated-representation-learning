#!/usr/bin/env bash
# Re-run the shift_distr study (5 shifts) for several seeds, one method at a
# time, logging to the study's own wandb project (config/hydra/
# multiagent_mnist_shift_distr.yaml -> logger.project: shift_distr).
#
# SheafFRL is NOT run here: it is launched separately (see README note below).
# The other three methods are driven as follows:
#
#   non_cooperative  one multirun over (5 shifts x SEEDS)
#   sheaf_fmtl       one multirun over SEEDS per shift, because each shift has
#   comfed           its own tuned max_lmb (the Optuna values of the reference
#                    seed-42 runs; lmb_study stays off, so the seeds differ
#                    only by the seed itself)
#
# Every job writes a marker under logs/shift_distr_seeds/done/ on success, and
# jobs with a marker are skipped, so the script can be interrupted and
# restarted without repeating finished work.
#
# Usage:
#   scripts/run_shift_distr_seeds.sh                 # seeds 1 2 3
#   SEEDS="4 5" scripts/run_shift_distr_seeds.sh     # other seeds
#   WAIT_PID=12345 scripts/run_shift_distr_seeds.sh  # start once that pid exits

set -u -o pipefail

cd "$(dirname "$0")/.." || exit 1

SEEDS="${SEEDS:-1 2 3}"
SHIFTS="${SHIFTS:-0.5 0.6 0.7 0.8 0.9}"
WAIT_PID="${WAIT_PID:-}"
OUT_DIR="logs/shift_distr_seeds"
DONE_DIR="$OUT_DIR/done"
mkdir -p "$DONE_DIR"

# Tuned max_lmb of the reference (seed 42) runs, per shift — reused verbatim so
# that the seed is the only thing that changes across repetitions.
declare -A FMTL_LMB=(
  [0.5]=0.02895654762869968
  [0.6]=0.0004826518755478656
  [0.7]=0.03512344297785497
  [0.8]=0.02550179380009556
  [0.9]=0.0008452505040955813
)
declare -A COMFED_LMB=(
  [0.5]=0.0006339058506331136
  [0.6]=0.0008642071376349362
  [0.7]=0.013553036660758452
  [0.8]=0.0002976806287407863
  [0.9]=0.015124049623355238
)

SEED_CSV="$(echo "$SEEDS" | tr ' ' ',')"
SHIFT_CSV="$(echo "$SHIFTS" | tr ' ' ',')"

log() { echo "[$(date '+%F %T')] $*"; }

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for pid $WAIT_PID to exit before starting …"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  log "pid $WAIT_PID has exited; starting."
  sleep 30  # let the GPU settle
fi

# run <marker-name> <hydra overrides…>
run() {
  local marker="$DONE_DIR/$1.done"
  shift
  if [[ -f "$marker" ]]; then
    log "SKIP $(basename "$marker" .done) (already done)"
    return 0
  fi
  log "RUN  $* "
  if uv run python scripts/multi_agent_experiment.py \
      --config-name=multiagent_mnist_shift_distr "$@" \
      >>"$OUT_DIR/$(basename "$marker" .done).log" 2>&1; then
    touch "$marker"
    log "OK   $(basename "$marker" .done)"
  else
    log "FAIL $(basename "$marker" .done) — see $OUT_DIR/$(basename "$marker" .done).log"
  fi
}

# ── 1. Non-cooperative: identical config to SheafFRL, so one sweep covers it ──
run "non_cooperative_all" \
  hydra.sweeper.params.orchestrator=non_cooperative \
  "dataset.shift_strength=$SHIFT_CSV" "seed=$SEED_CSV"

# ── 2/3. Sheaf-FMTL and ComFed: per-shift lambda, so one sweep per shift ──────
for shift in $SHIFTS; do
  run "sheaf_fmtl_shift${shift}" \
    hydra.sweeper.params.orchestrator=sheaf_fmtl \
    "dataset.shift_strength=$shift" \
    "orchestrator.max_lmb=${FMTL_LMB[$shift]}" \
    dataset.num_workers=0 "seed=$SEED_CSV"
done

for shift in $SHIFTS; do
  run "comfed_shift${shift}" \
    hydra.sweeper.params.orchestrator=comfed \
    "dataset.shift_strength=$shift" \
    "orchestrator.max_lmb=${COMFED_LMB[$shift]}" \
    dataset.num_workers=0 "seed=$SEED_CSV"
done

log "all done."
