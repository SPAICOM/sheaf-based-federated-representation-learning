#!/usr/bin/env bash
# Escalating-gamma memory probe for sheaf_fmtl on the 15-agent
# multiagent_mnist_shift_distr config.
#
# Runs one single (non-multirun) sheaf_fmtl job per gamma value, ascending
# from the confirmed-safe ceiling, while an external sampling loop logs
# system/GPU/process memory every few seconds to a CSV. Logging is external
# to the training process on purpose: if the process gets OOM-killed it
# can't log its own death, but the last CSV row written just before the gap
# shows what memory looked like right up to the crash.
#
# This machine is shared with other users' jobs, so rather than waiting for
# the kernel OOM-killer (which may not pick *this* process as the victim),
# the loop self-kills once system-available RAM drops below MIN_AVAILABLE_MB.
#
# Usage: bash scripts/probe_gamma_memory.sh [gamma1 gamma2 ...]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/logs/gamma_memory_probe"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
CSV="$LOGDIR/trace_${STAMP}.csv"
SUMMARY="$LOGDIR/summary_${STAMP}.txt"
echo "timestamp,gamma,proc_rss_mb,gpu_used_mb,sys_available_mb" > "$CSV"

# Gammas to try, ascending from the confirmed-safe ceiling (~3e-3 on this
# 15-agent/84-edge graph). Override by passing your own list as args.
GAMMAS=("$@")
if [ ${#GAMMAS[@]} -eq 0 ]; then
  GAMMAS=(0.003 0.004 0.005 0.007 0.01 0.015 0.02 0.03)
fi

MIN_AVAILABLE_MB=3000  # self-kill if system available RAM drops below this
SAMPLE_INTERVAL=3       # seconds between memory samples

cd "$REPO"
for GAMMA in "${GAMMAS[@]}"; do
  echo "=== gamma=$GAMMA ===" | tee -a "$SUMMARY"
  RUNLOG="$LOGDIR/run_gamma_${GAMMA}_${STAMP}.log"

  WANDB_MODE=offline uv run python scripts/multi_agent_experiment.py \
    --config-name multiagent_mnist_shift_distr \
    orchestrator=sheaf_fmtl \
    "orchestrator.gamma=$GAMMA" \
    trainer.max_epochs=2 \
    lmb_study.enabled=false \
    hydra.mode=RUN \
    > "$RUNLOG" 2>&1 &
  WRAPPER_PID=$!

  KILLED_FOR_SAFETY=0
  while kill -0 "$WRAPPER_PID" 2>/dev/null; do
    # The actual worker is .venv/bin/python3 (uv run/sh wrapper PIDs have
    # near-zero RSS and would otherwise be matched too).
    PY_PID=$(pgrep -f "\.venv/bin/python3.*orchestrator\.gamma=$GAMMA " | head -1)
    RSS_MB="NA"
    if [ -n "${PY_PID:-}" ]; then
      RSS_KB=$(ps -o rss= -p "$PY_PID" 2>/dev/null | tr -d ' ')
      [ -n "$RSS_KB" ] && RSS_MB=$(( RSS_KB / 1024 ))
    fi
    GPU_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
    echo "$(date +%H:%M:%S),$GAMMA,$RSS_MB,${GPU_MB:-NA},$AVAIL_MB" >> "$CSV"

    if [ -n "$AVAIL_MB" ] && [ "$AVAIL_MB" -lt "$MIN_AVAILABLE_MB" ]; then
      echo "  !! system available RAM ${AVAIL_MB}MB < ${MIN_AVAILABLE_MB}MB - self-killing before the OOM-killer does (this machine is shared)" | tee -a "$SUMMARY"
      [ -n "${PY_PID:-}" ] && kill -9 "$PY_PID" 2>/dev/null
      kill -9 "$WRAPPER_PID" 2>/dev/null
      KILLED_FOR_SAFETY=1
      break
    fi
    sleep "$SAMPLE_INTERVAL"
  done

  wait "$WRAPPER_PID" 2>/dev/null
  STATUS=$?

  if [ "$KILLED_FOR_SAFETY" -eq 1 ]; then
    echo "gamma=$GAMMA: SELF-KILLED (system RAM nearly exhausted) -> this is your ceiling. Stopping." | tee -a "$SUMMARY"
    break
  elif [ $STATUS -ne 0 ]; then
    echo "gamma=$GAMMA: FAILED (exit=$STATUS) - see $RUNLOG. Stopping escalation." | tee -a "$SUMMARY"
    tail -20 "$RUNLOG" | tee -a "$SUMMARY"
    break
  else
    echo "gamma=$GAMMA: OK" | tee -a "$SUMMARY"
  fi
done

echo ""
echo "Memory trace: $CSV"
echo "Summary:      $SUMMARY"
