#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/benchmark/ultra_long_forecast/_common.sh"

target="${1:-}"
if [ -z "$target" ]; then
  echo "Usage: $0 <target> [benchmark args...]" >&2
  exit 1
fi
shift
extra_args=("$@")

run_target() {
  run_ultra_long_model "$@" "${extra_args[@]}"
}

needs_preflight() {
  ! has_arg_option "-h" "${extra_args[@]}" && ! has_arg_option "--help" "${extra_args[@]}"
}

case "$target" in
  DLinear)
    run_target DLinear --lr_sweep
    ;;
  PatchTST)
    run_target PatchTST --lr_sweep
    ;;
  PETformer)
    run_target PETformer --lr_sweep
    ;;
  PhaseFormer)
    run_target PhaseFormer --lr_sweep
    ;;
  RMLP)
    run_target RMLP --lr_sweep
    ;;
  SegRNN)
    run_target SegRNN --lr_sweep
    ;;
  SparseTSF_MLP)
    run_target SparseTSF --model_name SparseTSF_MLP --model_type mlp --lr_sweep
    ;;
  STID)
    run_target STID --lr_sweep
    ;;
  TimeBase)
    run_target TimeBase --lr_sweep
    ;;
  TimeMixer)
    run_target TimeMixer --lr_sweep
    ;;
  Crossformer)
    run_target Crossformer --lr_sweep --batch_size 16
    ;;
  CrossGNN)
    run_target CrossGNN --lr_sweep
    ;;
  DUET)
    run_target DUET --lr_sweep
    ;;
  Leddam)
    run_target Leddam --lr_sweep
    ;;
  ModernTCN)
    run_target ModernTCN --lr_sweep
    ;;
  PMDformer)
    run_target PMDformer --lr_sweep
    ;;
  SOFTS)
    run_target SOFTS --lr_sweep
    ;;
  TQNet)
    run_target TQNet --lr_sweep
    ;;
  TimeFilter)
    run_target TimeFilter --lr_sweep
    ;;
  iTransformer)
    run_target iTransformer --lr_sweep
    ;;
  CATS_PureEndo)
    run_target CATS --lr_sweep --model_name CATS_PureEndo --use_exog 0
    ;;
  CATS_Ordinal)
    run_target CATS --lr_sweep --model_name CATS_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  CATS_OneHot)
    run_target CATS --lr_sweep --model_name CATS_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  CATS_CatEmb)
    run_target CATS --lr_sweep --model_name CATS_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  CrossLinear_PureEndo)
    run_target CrossLinear --lr_sweep --model_name CrossLinear_PureEndo --use_exog 0
    ;;
  CrossLinear_Ordinal)
    run_target CrossLinear --lr_sweep --model_name CrossLinear_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  CrossLinear_OneHot)
    run_target CrossLinear --lr_sweep --model_name CrossLinear_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  CrossLinear_CatEmb)
    run_target CrossLinear --lr_sweep --model_name CrossLinear_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  DAG_PureEndo)
    run_target DAG --lr_sweep --model_name DAG_PureEndo --use_exog 0
    ;;
  DAG_Ordinal)
    run_target DAG --lr_sweep --model_name DAG_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  DAG_OneHot)
    run_target DAG --lr_sweep --model_name DAG_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  DAG_CatEmb)
    run_target DAG --lr_sweep --model_name DAG_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  TiDE_PureEndo)
    run_target TiDE --lr_sweep --model_name TiDE_PureEndo --use_exog 0
    ;;
  TiDE_Ordinal)
    run_target TiDE --lr_sweep --model_name TiDE_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  TiDE_OneHot)
    run_target TiDE --lr_sweep --model_name TiDE_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  TiDE_CatEmb)
    run_target TiDE --lr_sweep --model_name TiDE_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  TimeXer_PureEndo)
    run_target TimeXer --lr_sweep --model_name TimeXer_PureEndo --use_exog 0
    ;;
  TimeXer_Ordinal)
    run_target TimeXer --lr_sweep --model_name TimeXer_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  TimeXer_OneHot)
    run_target TimeXer --lr_sweep --model_name TimeXer_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  TimeXer_CatEmb)
    run_target TimeXer --lr_sweep --model_name TimeXer_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  XLinear_PureEndo)
    run_target XLinear --lr_sweep --model_name XLinear_PureEndo --use_exog 0
    ;;
  XLinear_Ordinal)
    run_target XLinear --lr_sweep --model_name XLinear_Ordinal --use_exog 1 --exog_source both --discrete_exog_mode ordinal
    ;;
  XLinear_OneHot)
    run_target XLinear --lr_sweep --model_name XLinear_OneHot --use_exog 1 --exog_source both --discrete_exog_mode onehot
    ;;
  XLinear_CatEmb)
    run_target XLinear --lr_sweep --model_name XLinear_CatEmb --use_exog 1 --exog_source both --discrete_exog_mode catemb --catemb_dim 8
    ;;
  SeasonalNaive)
    run_target SeasonalNaive --model_name SeasonalNaive
    ;;
  SeasonalMean)
    run_target SeasonalMean --model_name SeasonalMean
    ;;
  Chronos2_NoExo)
    if needs_preflight; then
      bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" Chronos2
    fi
    run_target Chronos2 --model_name Chronos2_NoExo --use_exog 0 --exo_con_vars "" --exo_dis_dict '{}'
    ;;
  Chronos2_FullExo)
    if needs_preflight; then
      bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" Chronos2
    fi
    run_target Chronos2 --model_name Chronos2_FullExo --use_exog 1
    ;;
  Moirai2)
    if needs_preflight; then
      bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" Moirai2
      bash "$ROOT_DIR/scripts/benchmark/foundation/bootstrap_moirai2_backend.sh"
    fi
    run_target Moirai2 --model_name Moirai2
    ;;
  TimesFM2p5)
    if needs_preflight; then
      bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" TimesFM2p5
    fi
    run_target TimesFM2p5 --model_name TimesFM2p5
    ;;
  TimerS1)
    if needs_preflight; then
      bash "$ROOT_DIR/scripts/benchmark/foundation/download_pretrained_models.sh" TimerS1
    fi
    run_target TimerS1 --model_name TimerS1 --timers1_batch_size 4 --timers1_use_cache 1
    ;;
  *)
    echo "Unknown ultra-long target: $target" >&2
    exit 1
    ;;
esac
