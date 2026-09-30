#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
run_benchmark SparseTSF --model_name SparseTSF_MLP --model_type mlp --lr_sweep \
  "$@"
