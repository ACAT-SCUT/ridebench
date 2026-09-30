#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/study_lookback/_common.sh"

run_lookback_study_model TiDE TiDE_OneHot \
  --lr_sweep \
  --use_exog 1 --exog_source both --discrete_exog_mode onehot \
  "$@"
