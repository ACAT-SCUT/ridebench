#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/study_area/_common.sh"

run_area_study_model PMDformer PMDformer --lr_sweep \
  "$@"
