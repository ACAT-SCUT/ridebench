#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
AREA_STUDY_RUN_RESULTS_DIR="${AREA_STUDY_RUN_RESULTS_DIR:-run_results/area_study}"
AREA_STUDY_SUMMARY_CSV="${AREA_STUDY_SUMMARY_CSV:-run_results/summary_csv/area_study.csv}"
AREA_STUDY_TEST_AREA_IDS="${AREA_STUDY_TEST_AREA_IDS:-0,1,2,3,4,5,6,7,8,9}"
AREA_STUDY_TRAIN_SIZES=(10 30 60 100 150 200)

_join_range() {
  local start="$1" end="$2" result="" number
  for ((number=start; number<=end; number++)); do
    [[ -n "$result" ]] && result+=","
    result+="$number"
  done
  printf '%s\n' "$result"
}
csv_range() { _join_range "$@"; }
run_area_study_model() {
  local algorithm="$1" model_name="$2"; shift 2
  local size area_ids
  for size in "${AREA_STUDY_TRAIN_SIZES[@]}"; do
    area_ids="$(_join_range 0 $((size - 1)))"
    echo "[AREA-SCALE] ${model_name}_AreaScale_Tr${size}_Te10"
    run_benchmark "$algorithm" --model_name "${model_name}_AreaScale_Tr${size}_Te10" \
      --run_results_dir "$AREA_STUDY_RUN_RESULTS_DIR" --benchmark_summary_csv_path "$AREA_STUDY_SUMMARY_CSV" \
      --enable_custom_area_split --train_val_area_ids "$area_ids" --test_area_ids "$AREA_STUDY_TEST_AREA_IDS" "$@"
  done
}
