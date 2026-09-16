#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SHEAF_GPU="${SHEAF_GPU:-1}"
NONCOOP_GPU="${NONCOOP_GPU:-0}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
WANDB_MODE="${WANDB_MODE:-offline}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
OVERLAP_VALUES="${OVERLAP_VALUES:-0.40 0.50 0.60 0.70 0.80 0.90}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 127
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/reconstruction_masking_sweep_${RUN_ID}"
RESULTS_DIR="results/reconstruction"
DOC_PATH="${RESULTS_DIR}/masking_sweep_${RUN_ID}.md"
mkdir -p "$SWEEP_DIR" "$RESULTS_DIR"

export WANDB_MODE
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

cat > "$DOC_PATH" <<EOF
# Masked CIFAR-10 VAE Masking Sweep

- Started: ${RUN_ID}
- Epochs per run: ${EPOCHS}
- SheafFRL GPU: ${SHEAF_GPU}
- NonCooperative GPU: ${NONCOOP_GPU}
- Seed: ${SEED}
- W&B mode: ${WANDB_MODE}
- Overlap values: ${OVERLAP_VALUES}

| setting | mask_mode | visible target | overlap/shared | focus | off-focus | sheaf local PSNR | sheaf comm PSNR | sheaf local MSE | sheaf comm MSE | noncoop local PSNR | noncoop comm PSNR | noncoop local MSE | noncoop comm MSE | sheaf parquet | noncoop parquet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
EOF

COMMON_OVERRIDES=(
  "trainer.max_epochs=${EPOCHS}"
  "trainer.devices=1"
  "trainer.accelerator=gpu"
  "seed=${SEED}"
  "dataset.n_agents=2"
  "dataset.batch_size=256"
  "dataset.num_workers=0"
  "dataset.return_mask=true"
  "dataset.include_mask_in_input=true"
  "model.out_features=3"
  "orchestrator.alignment_method=procrustes"
  "orchestrator.anchor_selection=all"
  "logger.group=masking_sweep_20ep"
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
    "study_name=masking_sweep_20ep_${setting}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@" \
    > "$logfile" 2>&1
  echo "[${setting}] done ${orchestrator}" | tee -a "$SWEEP_DIR/sweep.log"
}

extract_result_path() {
  local logfile="$1"
  grep 'Results saved ->' "$logfile" | tail -n 1 | sed 's/^.*Results saved -> //'
}

append_doc_row() {
  local setting="$1"
  local mask_mode="$2"
  local visible="$3"
  local overlap="$4"
  local focus="$5"
  local off_focus="$6"
  local sheaf_path="$7"
  local noncoop_path="$8"

  "$PYTHON_BIN" - "$DOC_PATH" "$setting" "$mask_mode" "$visible" "$overlap" \
    "$focus" "$off_focus" "$sheaf_path" "$noncoop_path" <<'PY'
from pathlib import Path
import sys

import pandas as pd

doc, setting, mask_mode, visible, overlap, focus, off_focus, sheaf_path, noncoop_path = sys.argv[1:]

def fmt(value):
    if pd.isna(value):
        return "nan"
    return f"{float(value):.4f}"

def summarize(path):
    df = pd.read_parquet(path)
    return {
        "local_psnr": df["private_psnr"].mean(),
        "comm_psnr": df["comm_psnr"].mean(),
        "local_mse": df["private_mse_visible"].mean(),
        "comm_mse": df["comm_mse_tx_missing_rx_visible"].mean(),
    }

sheaf = summarize(sheaf_path)
noncoop = summarize(noncoop_path)
row = (
    f"| {setting} | {mask_mode} | {visible} | {overlap} | {focus} | {off_focus} | "
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
  local setting="$1"
  local mask_mode="$2"
  local visible="$3"
  local overlap="$4"
  local focus="$5"
  local off_focus="$6"
  shift 6

  local sheaf_log="${SWEEP_DIR}/${setting}__sheaf_frl.log"
  local noncoop_log="${SWEEP_DIR}/${setting}__non_cooperative.log"

  echo "[${setting}] queued" | tee -a "$SWEEP_DIR/sweep.log"
  run_one "sheaf_frl" "$SHEAF_GPU" "$setting" "$sheaf_log" "$@" &
  local sheaf_pid=$!
  run_one "non_cooperative" "$NONCOOP_GPU" "$setting" "$noncoop_log" "$@" &
  local noncoop_pid=$!

  local sheaf_status=0
  local noncoop_status=0
  wait "$sheaf_pid" || sheaf_status=$?
  wait "$noncoop_pid" || noncoop_status=$?

  if [[ "$sheaf_status" -ne 0 || "$noncoop_status" -ne 0 ]]; then
    echo "[${setting}] failed sheaf=${sheaf_status} noncoop=${noncoop_status}" \
      | tee -a "$SWEEP_DIR/sweep.log"
    {
      printf '| %s | %s | %s | %s | %s | %s | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | %s | %s |\n' \
        "$setting" "$mask_mode" "$visible" "$overlap" "$focus" "$off_focus" \
        "$(basename "$sheaf_log")" "$(basename "$noncoop_log")"
    } >> "$DOC_PATH"
    return 0
  fi

  local sheaf_path
  local noncoop_path
  sheaf_path="$(extract_result_path "$sheaf_log")"
  noncoop_path="$(extract_result_path "$noncoop_log")"
  append_doc_row "$setting" "$mask_mode" "$visible" "$overlap" "$focus" "$off_focus" \
    "$sheaf_path" "$noncoop_path"
  echo "[${setting}] documented in ${DOC_PATH}" | tee -a "$SWEEP_DIR/sweep.log"
}

visible_from_overlap() {
  local overlap="$1"
  "$PYTHON_BIN" - "$overlap" <<'PY'
import sys

overlap = float(sys.argv[1])
print(f"{(1.0 + overlap) / 2.0:.2f}")
PY
}

for overlap in $OVERLAP_VALUES; do
  tag="$(printf '%03d' "$(awk -v x="$overlap" 'BEGIN { printf "%.0f", x * 100 }')")"
  visible="$(visible_from_overlap "$overlap")"
  run_pair "region_overlap_o${tag}" "region_overlap" "$visible" "$overlap" "-" "-" \
    "dataset.mask_mode=region_overlap" \
    "dataset.region_visible_fraction=${visible}" \
    "dataset.region_overlap_fraction=${overlap}" \
    "dataset.region_boundary_jitter_px=2"
done

"$PYTHON_BIN" scripts/plot_reconstruction_overlap_metrics.py \
  --results-dir "$RESULTS_DIR" \
  --out-dir "$RESULTS_DIR/plots" \
  --seed "$SEED" \
  --mask-mode region_overlap \
  --min-overlap 0.1 \
  --max-overlap 0.9 \
  | tee -a "$SWEEP_DIR/sweep.log"

echo "Sweep complete. Results doc: ${DOC_PATH}" | tee -a "$SWEEP_DIR/sweep.log"
