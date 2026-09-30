#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
run_benchmark DAG \
  --lr_sweep \
  --model_name DAG_Ordinal \
  --use_exog 1 --exog_source both --discrete_exog_mode ordinal \
  "$@"
