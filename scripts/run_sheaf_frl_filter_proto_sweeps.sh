#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="$ROOT_DIR/results/sheaf_frl_filter_proto_sweeps/$RUN_TAG"
SUMMARY_FILE="$SWEEP_DIR/best_results.jsonl"

mkdir -p "$SWEEP_DIR"

run_combo() {
  local filter_unseen_classes="$1"
  local use_prototypes="$2"
  local label="filter_${filter_unseen_classes}__proto_${use_prototypes}"
  local study_name="sheaf_frl_hetero_cifar10_${RUN_TAG}_${label}"
  local log_file="$SWEEP_DIR/${label}.log"

  echo "=== Running ${label} ===" | tee "$log_file"
  echo "study_name=${study_name}" | tee -a "$log_file"

  uv run scripts/experiment.py \
    --config-name hetero_cifar10_experiment \
    -m \
    +hpo=sweep_sheaf_frl \
    "study_name=${study_name}" \
    "orchestrator.filter_unseen_classes=${filter_unseen_classes}" \
    "orchestrator.use_prototypes=${use_prototypes}" \
    2>&1 | tee -a "$log_file"

  python3 - "$study_name" "$label" "$SUMMARY_FILE" <<'PY'
import json
import sys
from pathlib import Path

study_name = sys.argv[1]
label = sys.argv[2]
summary_file = Path(sys.argv[3])
results_dir = Path("results/experiment")

best = None
for result_path in sorted(results_dir.glob("*.json")):
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    config = payload.get("config", {})
    if config.get("study_name") != study_name:
        continue

    objective_value = payload.get("objective_value")
    if objective_value is None:
        continue

    if best is None or float(objective_value) > float(best["objective_value"]):
        best = {
            "label": label,
            "study_name": study_name,
            "objective_metric": payload.get("objective_metric"),
            "objective_value": float(objective_value),
            "results_file": str(result_path),
            "hydra_run_dir": payload.get("hydra", {}).get("run_dir"),
            "override_dirname": payload.get("hydra", {}).get("override_dirname"),
            "filter_unseen_classes": config.get("orchestrator", {}).get(
                "filter_unseen_classes"
            ),
            "use_prototypes": config.get("orchestrator", {}).get(
                "use_prototypes"
            ),
            "lambda_sheaf": config.get("orchestrator", {}).get(
                "lambda_sheaf"
            ),
            "optimizer": config.get("optimizer", {}),
        }

if best is None:
    raise SystemExit(
        f"No persisted result files found for study_name={study_name}"
    )

with summary_file.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(best, sort_keys=True) + "\n")

print(json.dumps(best, indent=2, sort_keys=True))
PY
}

run_combo true true
run_combo true false
run_combo false true
run_combo false false

echo
echo "Saved sweep logs and best-result summaries under:"
echo "  $SWEEP_DIR"
echo "Summary file:"
echo "  $SUMMARY_FILE"
