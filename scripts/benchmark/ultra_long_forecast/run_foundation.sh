#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts/benchmark/ultra_long_forecast"

scripts=(
  "Chronos2_NoExo"
  "Chronos2_FullExo"
  "Moirai2"
  "TimesFM2p5"
  "TimerS1"
)

failed=()

for script in "${scripts[@]}"; do
  echo "[RUN] ultra_long_forecast/$script"
  if bash "$SCRIPTS_DIR/run_single.sh" "$script" "$@"; then
    echo "[DONE] ultra_long_forecast/$script"
  else
    echo "[FAIL] ultra_long_forecast/$script"
    failed+=("$script")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "Ultra-long foundation scripts completed with failures. Failed scripts:"
  printf ' - %s\n' "${failed[@]}"
  exit 1
fi

echo "Ultra-long foundation scripts completed."
