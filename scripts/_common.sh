#!/usr/bin/env bash
set -euo pipefail

# All benchmark entrypoints call this adapter. Keeping option assembly here means
# experiments differ only in the model recipe they declare.
REGULAR_LONG_RUN_RESULTS_DIR="${REGULAR_LONG_RUN_RESULTS_DIR:-run_results/regular_long_forecast}"
REGULAR_LONG_SUMMARY_CSV="${REGULAR_LONG_SUMMARY_CSV:-run_results/summary_csv/regular_long_forecast.csv}"

contains_option() {
  local needle="$1"; shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" || "$item" == "$needle="* ]] && return 0
  done
  return 1
}
has_arg_option() { contains_option "$@"; }

runner_command() {
  local runner="${BENCH_RUNNER:-auto}"
  case "$runner" in
    uv) command -v uv >/dev/null || { echo "uv not found" >&2; return 1; }; printf '%s\n' "uv run python -m ridebench" ;;
    pip|python)
      if command -v python >/dev/null; then printf '%s\n' "python -m ridebench";
      elif command -v python3 >/dev/null; then printf '%s\n' "python3 -m ridebench";
      else echo "python not found" >&2; return 1; fi ;;
    auto)
      if command -v uv >/dev/null; then printf '%s\n' "uv run python -m ridebench";
      elif command -v python >/dev/null; then printf '%s\n' "python -m ridebench";
      elif command -v python3 >/dev/null; then printf '%s\n' "python3 -m ridebench";
      else echo "Neither uv nor python found" >&2; return 1; fi ;;
    *) echo "Unsupported BENCH_RUNNER='$runner'. Use: auto | uv | pip" >&2; return 1 ;;
  esac
}

_run_once() {
  local argv=("$@")
  local extras=()
  if [[ -n "${BENCH_EXTRA_ARGS:-}" ]]; then read -r -a extras <<< "$BENCH_EXTRA_ARGS"; fi

  if ((${#argv[@]})); then
    local assembled=("${argv[0]}")
    ((${#extras[@]})) && assembled+=("${extras[@]}")
    ((${#argv[@]} > 1)) && assembled+=("${argv[@]:1}")
    argv=("${assembled[@]}")
  fi

  if ((${#argv[@]})) && ! contains_option -h "${argv[@]}" && ! contains_option --help "${argv[@]}"; then
    local output_defaults=()
    contains_option --run_results_dir "${argv[@]}" || output_defaults+=(--run_results_dir "$REGULAR_LONG_RUN_RESULTS_DIR")
    contains_option --benchmark_summary_csv_path "${argv[@]}" || output_defaults+=(--benchmark_summary_csv_path "$REGULAR_LONG_SUMMARY_CSV")
    ((${#output_defaults[@]})) && argv=("${argv[0]}" "${output_defaults[@]}" "${argv[@]:1}")
  fi

  local command_line
  command_line="$(runner_command)"
  # shellcheck disable=SC2086
  $command_line "${argv[@]}"
}

supports_lr_sweep() {
  case "$1" in
    Chronos2|Moirai2|SeasonalMean|SeasonalNaive|TimerS1|TimesFM2p5) return 1 ;;
    *) return 0 ;;
  esac
}

run_benchmark() {
  local raw=("$@")
  local model="${raw[0]:-}"
  [[ -n "$model" ]] || { echo "Model name is required." >&2; return 1; }
  local args=() lr_list="${BENCH_LR_LIST:-0.003 0.001 0.0003 0.0001 0.00003}"
  local sweep=0 explicit_lr=0 help=0 i=0
  while ((i < ${#raw[@]})); do
    case "${raw[i]}" in
      --lr_sweep) sweep=1 ;;
      --lr_sweep_list)
        i=$((i + 1)); ((i < ${#raw[@]})) || { echo "--lr_sweep_list requires a value." >&2; return 1; }
        lr_list="${raw[i]}" ;;
      --lr_sweep_list=*) lr_list="${raw[i]#*=}" ;;
      --learning_rate)
        explicit_lr=1; args+=("${raw[i]}"); i=$((i + 1)); ((i < ${#raw[@]})) || { echo "--learning_rate requires a value." >&2; return 1; }; args+=("${raw[i]}") ;;
      --learning_rate=*) explicit_lr=1; args+=("${raw[i]}") ;;
      -h|--help) help=1; args+=("${raw[i]}") ;;
      *) args+=("${raw[i]}") ;;
    esac
    i=$((i + 1))
  done

  if (( ! sweep || explicit_lr || help )) || ! supports_lr_sweep "$model"; then
    ((sweep && !explicit_lr && !help)) && ! supports_lr_sweep "$model" && echo "[LR-SWEEP] $model: unsupported for script-level LR sweep, running once."
    _run_once "${args[@]}"
    return
  fi
  local lr
  for lr in $lr_list; do
    echo "[LR-SWEEP] $model: learning_rate=$lr"
    _run_once "${args[@]}" --learning_rate "$lr"
  done
}
