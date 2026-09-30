#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
run_benchmark DAG \
  --lr_sweep \
  --model_name DAG_PureEndo \
  --use_exog 0 --exog_source none --discrete_exog_mode none \
  "$@"
