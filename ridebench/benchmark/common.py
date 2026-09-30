"""Runtime helpers and compatibility exports for benchmark utilities."""

import logging
import random

import numpy as np
import torch

from .checkpoints import (
    BENCHMARK_CHECKPOINT_MATCH_KEYS, TRAINED_CHECKPOINT_ARGS_EXCLUDED,
    _collect_model_checkpoint_match_keys, _extract_checkpoint_match_args,
    _normalize_checkpoint_value, build_trained_checkpoint_args, load_nn_checkpoint,
    save_nn_checkpoint,
)
from .paths import (
    REGULAR_LONG_RUN_RESULTS_DIR, ULTRA_LONG_RUN_RESULTS_DIR,
    RunPaths, _learning_rate_tag, build_run_paths,
)


def set_seed(seed):
    if seed is not None:
        for initialize in (random.seed, np.random.seed, torch.manual_seed):
            initialize(seed)


def get_data_params(dataset_factory):
    methods = {"input_len": "input_len", "pred_len": "output_len", "x_dim": "x_dim",
               "exo_con_list": "exo_con_list", "exo_dis_dict": "exo_dis_dict"}
    return {key: getattr(dataset_factory, method)() for key, method in methods.items()}


def get_logger(log_level):
    logging.basicConfig(level=get_log_level(log_level),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("TS-Benchmark")


def _lookup(table, key, category):
    if key not in table:
        raise NotImplementedError(f"{category} {key} is defined but not supported yet.")
    return table[key]


def get_log_level(log_level):
    levels = {name: getattr(logging, name.upper()) for name in ("debug", "info", "warning", "error", "critical")}
    return _lookup(levels, log_level, "Log level")


def get_torch_dtype(dtype):
    types = {name: getattr(torch, name) for name in ("bfloat16", "float16", "float32", "float64")}
    return _lookup(types, dtype, "Data type")


def get_numpy_dtype(dtype):
    types = {"bfloat16": np.float16, "float16": np.float16, "float32": np.float32, "float64": np.float64}
    return _lookup(types, dtype, "Data type")
