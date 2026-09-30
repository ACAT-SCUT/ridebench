#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
LOSS_STUDY_RUN_RESULTS_DIR="${LOSS_STUDY_RUN_RESULTS_DIR:-run_results/loss_study}"
LOSS_STUDY_SUMMARY_CSV="${LOSS_STUDY_SUMMARY_CSV:-run_results/summary_csv/loss_study.csv}"
LOSS_STUDY_LOSSES=("SmoothL1:smooth_l1" "MSE:mse" "MAE:mae" "FreDF:fredf" "DBLoss:dbloss")
run_loss_study_model() {
  local algorithm="$1" model_name="$2"; shift 2
  local spec label value
  for spec in "${LOSS_STUDY_LOSSES[@]}"; do
    label="${spec%%:*}"; value="${spec#*:}"
    echo "[LOSS-STUDY] ${model_name}_Loss${label}: train_loss=${value}, val_loss=${value}"
    run_benchmark "$algorithm" --model_name "${model_name}_Loss${label}" \
      --run_results_dir "$LOSS_STUDY_RUN_RESULTS_DIR" --benchmark_summary_csv_path "$LOSS_STUDY_SUMMARY_CSV" \
      --train_loss "$value" --val_loss "$value" "$@"
  done
}
