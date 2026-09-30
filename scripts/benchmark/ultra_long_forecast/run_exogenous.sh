#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts/benchmark/ultra_long_forecast"

scripts=(
  "CATS_PureEndo"
  "CATS_Ordinal"
  "CATS_OneHot"
  "CATS_CatEmb"
  "CrossLinear_PureEndo"
  "CrossLinear_Ordinal"
  "CrossLinear_OneHot"
  "CrossLinear_CatEmb"
  "DAG_PureEndo"
  "DAG_Ordinal"
  "DAG_OneHot"
  "DAG_CatEmb"
  "TiDE_PureEndo"
  "TiDE_Ordinal"
  "TiDE_OneHot"
  "TiDE_CatEmb"
  "TimeXer_PureEndo"
  "TimeXer_Ordinal"
  "TimeXer_OneHot"
  "TimeXer_CatEmb"
  "XLinear_PureEndo"
  "XLinear_Ordinal"
  "XLinear_OneHot"
  "XLinear_CatEmb"
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
  echo "Ultra-long exogenous scripts completed with failures. Failed scripts:"
  printf ' - %s\n' "${failed[@]}"
  exit 1
fi

echo "Ultra-long exogenous scripts completed."
