#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHEAF_GPU="${SHEAF_GPU:-1}"
NONCOOP_GPU="${NONCOOP_GPU:-0}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SHARED_VALUES="${SHARED_VALUES:-0.20 0.30 0.40 0.50 0.60 0.70 0.80}"
RANDOM_PRIVATE_VISIBLE_PROBABILITY="${RANDOM_PRIVATE_VISIBLE_PROBABILITY:-0.85}"
RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY="${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY:-0.05}"
RANDOM_BLOCK_SIZE="${RANDOM_BLOCK_SIZE:-4}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 127
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="logs/reconstruction_random_spatial_sweep_${RUN_ID}"
RESULTS_DIR="results/reconstruction"
DOC_PATH="${RESULTS_DIR}/random_spatial_sweep_${RUN_ID}.md"
mkdir -p "$SWEEP_DIR" "$RESULTS_DIR"

export WANDB_MODE
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$DOC_PATH" ]]; then
  cat > "$DOC_PATH" <<EOF
# Masked CIFAR-10 VAE Random-Spatial Sweep

- Started: ${RUN_ID}
- Epochs per run: ${EPOCHS}
- SheafFRL GPU: ${SHEAF_GPU}
- NonCooperative GPU: ${NONCOOP_GPU}
- Seed: ${SEED}
- W&B mode: ${WANDB_MODE}
- Shared visible probability values: ${SHARED_VALUES}
- Private visible probability: ${RANDOM_PRIVATE_VISIBLE_PROBABILITY}
- Off-focus visible probability: ${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY}
- Random block size: ${RANDOM_BLOCK_SIZE}

| setting | mask_mode | visible target | overlap/shared | focus | off-focus | sheaf local PSNR | sheaf comm PSNR | sheaf local MSE | sheaf comm MSE | noncoop local PSNR | noncoop comm PSNR | noncoop local MSE | noncoop comm MSE | sheaf parquet | noncoop parquet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
EOF
else
  echo "Resuming existing results doc: ${DOC_PATH}" | tee -a "$SWEEP_DIR/sweep.log"
fi

COMMON_OVERRIDES=(
  "trainer.max_epochs=${EPOCHS}"
  "trainer.devices=1"
  "trainer.accelerator=gpu"
  "+trainer.enable_checkpointing=false"
  "seed=${SEED}"
  "dataset.n_agents=2"
  "dataset.batch_size=256"
  "dataset.num_workers=0"
  "dataset.return_mask=true"
  "dataset.include_mask_in_input=true"
  "model.out_features=3"
  "orchestrator.alignment_method=procrustes"
  "orchestrator.anchor_selection=all"
  "logger.group=random_spatial_sweep_${EPOCHS}ep"
  "logger.log_model=false"
  "diagnostics.masked_reconstruction.every_n_epochs=10"
)

run_one() {
  local orchestrator="$1"
  local gpu="$2"
  local setting="$3"
  local logfile="$4"
  shift 4

  echo "[${setting}] start ${orchestrator} on GPU ${gpu}" | tee -a "$SWEEP_DIR/sweep.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/reconstruction_experiment.py \
    "orchestrator=${orchestrator}" \
    "study_name=random_spatial_sweep_${EPOCHS}ep_${setting}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@" \
    > "$logfile" 2>&1
  if ! grep -q 'Results saved ->' "$logfile"; then
    echo "[${setting}] missing results for ${orchestrator}; see ${logfile}" \
      | tee -a "$SWEEP_DIR/sweep.log"
    return 1
  fi
  echo "[${setting}] done ${orchestrator}" | tee -a "$SWEEP_DIR/sweep.log"
}

extract_result_path() {
  local logfile="$1"
  grep 'Results saved ->' "$logfile" | tail -n 1 | sed 's/^.*Results saved -> //'
}

expected_visible_fraction() {
  local shared="$1"
  "$PYTHON_BIN" - "$shared" "$RANDOM_PRIVATE_VISIBLE_PROBABILITY" \
    "$RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY" <<'PY'
import sys

shared, private, off_focus = map(float, sys.argv[1:])
private_visible = 0.5 * private + 0.5 * off_focus
visible = shared + (1.0 - shared) * private_visible
print(f"{visible:.3f}")
PY
}

append_doc_row() {
  local setting="$1"
  local visible="$2"
  local shared="$3"
  local sheaf_path="$4"
  local noncoop_path="$5"

  "$PYTHON_BIN" - "$DOC_PATH" "$setting" "$visible" "$shared" \
    "$RANDOM_PRIVATE_VISIBLE_PROBABILITY" \
    "$RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY" \
    "$sheaf_path" "$noncoop_path" <<'PY'
from pathlib import Path
import sys

import pandas as pd

(
    doc,
    setting,
    visible,
    shared,
    focus,
    off_focus,
    sheaf_path,
    noncoop_path,
) = sys.argv[1:]

def fmt(value):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"

def summarize(path):
    df = pd.read_parquet(path)
    return {
        "local_psnr": df["private_psnr_full"].mean(),
        "comm_psnr": df["comm_psnr"].mean(),
        "local_mse": df["private_mse_full"].mean(),
        "comm_mse": df["comm_mse"].mean(),
    }

sheaf = summarize(sheaf_path)
noncoop = summarize(noncoop_path)
row = (
    f"| {setting} | agent_random_spatial | {visible} | {shared} | "
    f"{focus} | {off_focus} | "
    f"{fmt(sheaf['local_psnr'])} | {fmt(sheaf['comm_psnr'])} | "
    f"{fmt(sheaf['local_mse'])} | {fmt(sheaf['comm_mse'])} | "
    f"{fmt(noncoop['local_psnr'])} | {fmt(noncoop['comm_psnr'])} | "
    f"{fmt(noncoop['local_mse'])} | {fmt(noncoop['comm_mse'])} | "
    f"{Path(sheaf_path).name} | {Path(noncoop_path).name} |\n"
)
with open(doc, "a", encoding="utf-8") as f:
    f.write(row)
PY
}

run_pair() {
  local shared="$1"
  local tag
  tag="$(printf '%03d' "$(awk -v x="$shared" 'BEGIN { printf "%.0f", x * 100 }')")"
  local setting="random_spatial_s${tag}"
  local visible
  visible="$(expected_visible_fraction "$shared")"
  local sheaf_log="${SWEEP_DIR}/${setting}__sheaf_frl.log"
  local noncoop_log="${SWEEP_DIR}/${setting}__non_cooperative.log"

  echo "[${setting}] queued" | tee -a "$SWEEP_DIR/sweep.log"

  local sheaf_status=0
  local noncoop_status=0
  if [[ "$SHEAF_GPU" == "$NONCOOP_GPU" ]]; then
    run_one "sheaf_frl" "$SHEAF_GPU" "$setting" "$sheaf_log" \
      "dataset.mask_mode=agent_random_spatial" \
      "dataset.random_focus_regions=[left,right]" \
      "dataset.random_private_visible_probability=${RANDOM_PRIVATE_VISIBLE_PROBABILITY}" \
      "dataset.random_off_focus_visible_probability=${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY}" \
      "dataset.random_shared_visible_probability=${shared}" \
      "dataset.random_block_size=${RANDOM_BLOCK_SIZE}" || sheaf_status=$?

    run_one "non_cooperative" "$NONCOOP_GPU" "$setting" "$noncoop_log" \
      "dataset.mask_mode=agent_random_spatial" \
      "dataset.random_focus_regions=[left,right]" \
      "dataset.random_private_visible_probability=${RANDOM_PRIVATE_VISIBLE_PROBABILITY}" \
      "dataset.random_off_focus_visible_probability=${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY}" \
      "dataset.random_shared_visible_probability=${shared}" \
      "dataset.random_block_size=${RANDOM_BLOCK_SIZE}" || noncoop_status=$?
  else
    run_one "sheaf_frl" "$SHEAF_GPU" "$setting" "$sheaf_log" \
      "dataset.mask_mode=agent_random_spatial" \
      "dataset.random_focus_regions=[left,right]" \
      "dataset.random_private_visible_probability=${RANDOM_PRIVATE_VISIBLE_PROBABILITY}" \
      "dataset.random_off_focus_visible_probability=${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY}" \
      "dataset.random_shared_visible_probability=${shared}" \
      "dataset.random_block_size=${RANDOM_BLOCK_SIZE}" &
    local sheaf_pid=$!

    run_one "non_cooperative" "$NONCOOP_GPU" "$setting" "$noncoop_log" \
      "dataset.mask_mode=agent_random_spatial" \
      "dataset.random_focus_regions=[left,right]" \
      "dataset.random_private_visible_probability=${RANDOM_PRIVATE_VISIBLE_PROBABILITY}" \
      "dataset.random_off_focus_visible_probability=${RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY}" \
      "dataset.random_shared_visible_probability=${shared}" \
      "dataset.random_block_size=${RANDOM_BLOCK_SIZE}" &
    local noncoop_pid=$!

    wait "$sheaf_pid" || sheaf_status=$?
    wait "$noncoop_pid" || noncoop_status=$?
  fi

  if [[ "$sheaf_status" -ne 0 || "$noncoop_status" -ne 0 ]]; then
    echo "[${setting}] failed sheaf=${sheaf_status} noncoop=${noncoop_status}" \
      | tee -a "$SWEEP_DIR/sweep.log"
    printf '| %s | agent_random_spatial | %s | %s | %s | %s | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | %s | %s |\n' \
      "$setting" "$visible" "$shared" \
      "$RANDOM_PRIVATE_VISIBLE_PROBABILITY" \
      "$RANDOM_OFF_FOCUS_VISIBLE_PROBABILITY" \
      "$(basename "$sheaf_log")" "$(basename "$noncoop_log")" \
      >> "$DOC_PATH"
    return 0
  fi

  append_doc_row "$setting" "$visible" "$shared" \
    "$(extract_result_path "$sheaf_log")" \
    "$(extract_result_path "$noncoop_log")"
  echo "[${setting}] documented in ${DOC_PATH}" | tee -a "$SWEEP_DIR/sweep.log"
}

for shared in $SHARED_VALUES; do
  run_pair "$shared"
done

"$PYTHON_BIN" scripts/plot_reconstruction_overlap_metrics.py \
  --results-dir "$RESULTS_DIR" \
  --out-dir "$RESULTS_DIR/plots" \
  --seed "$SEED" \
  --mask-mode agent_random_spatial \
  --min-value 0.1 \
  --max-value 0.8 \
  | tee -a "$SWEEP_DIR/sweep.log"

echo "Sweep complete. Results doc: ${DOC_PATH}" | tee -a "$SWEEP_DIR/sweep.log"
