#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

CHRONOS_DIR="${BENCH_CHRONOS2_MODEL:-./pretrained_models/chronos-2}"
MOIRAI_DIR="${BENCH_MOIRAI2_MODEL:-./pretrained_models/moirai-2.0-R-small}"
TIMESFM_DIR="${BENCH_TIMESFM2P5_MODEL:-./pretrained_models/timesfm-2.5-200m-pytorch}"
TIMERS1_DIR="${BENCH_TIMERS1_MODEL:-./pretrained_models/Timer-S1}"

mkdir -p pretrained_models

has_chronos() {
  [ -f "$CHRONOS_DIR/config.json" ] && [ -f "$CHRONOS_DIR/model.safetensors" ]
}

has_moirai() {
  [ -f "$MOIRAI_DIR/config.json" ] && [ -f "$MOIRAI_DIR/model.safetensors" ]
}

has_timesfm() {
  [ -f "$TIMESFM_DIR/config.json" ] && [ -f "$TIMESFM_DIR/model.safetensors" ]
}

has_timers1() {
  [ -f "$TIMERS1_DIR/config.json" ] && [ -f "$TIMERS1_DIR/modeling_TimerS1.py" ] && [ -f "$TIMERS1_DIR/model.safetensors.index.json" ]
}

download_chronos() {
  if has_chronos; then
    echo "[SKIP] Chronos-2 checkpoint already present: $CHRONOS_DIR"
    return
  fi
  echo "[DOWNLOAD] Chronos-2 checkpoint -> $CHRONOS_DIR"
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "amazon/chronos-2",
    local_dir="./pretrained_models/chronos-2",
)
PY
}

download_timesfm() {
  if has_timesfm; then
    echo "[SKIP] TimesFM 2.5 checkpoint already present: $TIMESFM_DIR"
    return
  fi
  echo "[DOWNLOAD] TimesFM 2.5 checkpoint -> $TIMESFM_DIR"
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "google/timesfm-2.5-200m-pytorch",
    local_dir="./pretrained_models/timesfm-2.5-200m-pytorch",
)
PY
}

download_moirai() {
  if has_moirai; then
    echo "[SKIP] Moirai 2.0 checkpoint already present: $MOIRAI_DIR"
    return
  fi
  echo "[DOWNLOAD] Moirai 2.0 checkpoint -> $MOIRAI_DIR"
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "Salesforce/moirai-2.0-R-small",
    local_dir="./pretrained_models/moirai-2.0-R-small",
)
PY
}

download_timers1() {
  if has_timers1; then
    echo "[SKIP] Timer-S1 checkpoint already present: $TIMERS1_DIR"
    return
  fi
  echo "[DOWNLOAD] Timer-S1 checkpoint -> $TIMERS1_DIR"
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "bytedance-research/Timer-S1",
    local_dir="./pretrained_models/Timer-S1",
)
PY
}

download_model() {
  case "$1" in
    Chronos2|chronos)
      download_chronos
      ;;
    Moirai2|moirai)
      download_moirai
      ;;
    TimesFM2p5|timesfm)
      download_timesfm
      ;;
    TimerS1|timer-s1|timers1)
      download_timers1
      ;;
    all)
      download_chronos
      download_moirai
      download_timesfm
      download_timers1
      ;;
    *)
      echo "Unknown foundation model target: $1" >&2
      echo "Usage: bash scripts/benchmark/foundation/download_pretrained_models.sh [all|Chronos2|Moirai2|TimesFM2p5|TimerS1]" >&2
      exit 1
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  download_model all
else
  for target in "$@"; do
    download_model "$target"
  done
fi

echo "Foundation model checkpoint check/download complete."
