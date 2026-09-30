#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"

ULTRA_LONG_RUN_RESULTS_DIR="${ULTRA_LONG_RUN_RESULTS_DIR:-run_results/ultra_long_forecast}"
ULTRA_LONG_SUMMARY_CSV="${ULTRA_LONG_SUMMARY_CSV:-run_results/summary_csv/ultra_long_forecast.csv}"
ULTRA_LONG_INPUT_LEN="${ULTRA_LONG_INPUT_LEN:-4704}"
ULTRA_LONG_OUTPUT_LEN="${ULTRA_LONG_OUTPUT_LEN:-2688}"
ULTRA_LONG_DATA_STRIDE="${ULTRA_LONG_DATA_STRIDE:-336}"
ULTRA_LONG_BATCH_SIZE="${ULTRA_LONG_BATCH_SIZE:-32}"

run_ultra_long_model() {
  local algorithm="$1"
  shift
  run_benchmark "$algorithm" \
    --benchmark_variant ultra_long \
    --run_results_dir "$ULTRA_LONG_RUN_RESULTS_DIR" \
    --benchmark_summary_csv_path "$ULTRA_LONG_SUMMARY_CSV" \
    --input_len "$ULTRA_LONG_INPUT_LEN" \
    --output_len "$ULTRA_LONG_OUTPUT_LEN" \
    --data_stride "$ULTRA_LONG_DATA_STRIDE" \
    --batch_size "$ULTRA_LONG_BATCH_SIZE" \
    "$@"
}
