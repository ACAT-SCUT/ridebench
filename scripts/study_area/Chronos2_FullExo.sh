#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/study_area/_common.sh"
if ! has_arg_option "-h" "$@" && ! has_arg_option "--help" "$@"; then
  bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" Chronos2
fi

run_benchmark Chronos2 \
  --model_name Chronos2_FullExo_AreaZeroShot_Te10 \
  --run_results_dir "$AREA_STUDY_RUN_RESULTS_DIR" \
  --benchmark_summary_csv_path "$AREA_STUDY_SUMMARY_CSV" \
  --enable_custom_area_split \
  --train_val_area_ids "$AREA_STUDY_TEST_AREA_IDS" \
  --test_area_ids "$AREA_STUDY_TEST_AREA_IDS" \
  --use_exog 1 \
  "$@"
