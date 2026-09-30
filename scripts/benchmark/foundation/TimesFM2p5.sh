#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/_common.sh"
if ! has_arg_option "-h" "$@" && ! has_arg_option "--help" "$@"; then
  bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" TimesFM2p5
fi
run_benchmark TimesFM2p5 \
  --model_name TimesFM2p5 \
  "$@"
