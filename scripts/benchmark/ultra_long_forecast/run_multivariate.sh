#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts/benchmark/ultra_long_forecast"

scripts=(
  "Crossformer"
  "CrossGNN"
  "DUET"
  "Leddam"
  "ModernTCN"
  "PMDformer"
  "SOFTS"
  "TQNet"
  "TimeFilter"
  "iTransformer"
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
  echo "Ultra-long multivariate scripts completed with failures. Failed scripts:"
  printf ' - %s\n' "${failed[@]}"
  exit 1
fi

echo "Ultra-long multivariate scripts completed."
