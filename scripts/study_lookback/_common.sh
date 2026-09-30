#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"

LOOKBACK_STUDY_RUN_RESULTS_DIR="${LOOKBACK_STUDY_RUN_RESULTS_DIR:-run_results/lookback_study}"
LOOKBACK_STUDY_SUMMARY_CSV="${LOOKBACK_STUDY_SUMMARY_CSV:-run_results/summary_csv/lookback_study.csv}"
LOOKBACK_STUDY_OUTPUT_LEN="${LOOKBACK_STUDY_OUTPUT_LEN:-336}"
LOOKBACK_STUDY_DATA_STRIDE="${LOOKBACK_STUDY_DATA_STRIDE:-48}"
LOOKBACK_LABELS=(1d 3d 1w 2w 4w 8w 14w)
LOOKBACK_INPUTS=(48 144 336 672 1344 2688 4704)

run_lookback_study_model() {
  local algorithm="$1" model_name="$2"; shift 2
  local idx
  for idx in "${!LOOKBACK_LABELS[@]}"; do
    local label="${LOOKBACK_LABELS[idx]}" input_len="${LOOKBACK_INPUTS[idx]}"
    echo "[LOOKBACK] ${model_name}_Lookback${label}: input_len=${input_len}, output_len=${LOOKBACK_STUDY_OUTPUT_LEN}"
    run_benchmark "$algorithm" --model_name "${model_name}_Lookback${label}" \
      --run_results_dir "$LOOKBACK_STUDY_RUN_RESULTS_DIR" --benchmark_summary_csv_path "$LOOKBACK_STUDY_SUMMARY_CSV" \
      --input_len "$input_len" --output_len "$LOOKBACK_STUDY_OUTPUT_LEN" --data_stride "$LOOKBACK_STUDY_DATA_STRIDE" "$@"
  done
}
