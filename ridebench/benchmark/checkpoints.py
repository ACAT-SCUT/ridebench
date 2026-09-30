"""Checkpoint serialization and the model-compatibility contract."""

from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import torch


TRAINED_CHECKPOINT_ARGS_EXCLUDED = {'run_results_dir', 'batch_size', 'device', 'epochs', 'log_level', 'num_workers', 'optimizer', 'patience', 'persistent_workers', 'pin_memory', 'save_data', 'save_losses', 'seed', 'test_with_checkpoint', 'train_loss', 'val_loss', 'lr_scheduler'}
BENCHMARK_CHECKPOINT_MATCH_KEYS = {'benchmark_variant', 'enable_custom_area_split', 'input_len', 'output_len', 'endo_vars', 'exo_con_vars', 'exo_dis_dict', 'learning_rate', 'train_val_area_ids', 'test_area_ids'}


def normalize_value(value, *, sort_keys=False):
    if isinstance(value, dict):
        keys = sorted(value, key=str) if sort_keys else value
        return {str(key): normalize_value(value[key], sort_keys=sort_keys) for key in keys}
    if isinstance(value, (tuple, list)):
        return [normalize_value(item, sort_keys=sort_keys) for item in value]
    if isinstance(value, Path):
        return str(value)
    return getattr(value, "value", value)


def _normalize_checkpoint_value(value):
    return normalize_value(value, sort_keys=True)


def _collect_model_checkpoint_match_keys(alg_config):
    parser = ArgumentParser(add_help=False, argument_default=SUPPRESS)
    alg_config.add_arguments(parser)
    return {action.dest for action in parser._actions} - {"help", "algorithm"}


def _extract_checkpoint_match_args(checkpoint_args):
    if not checkpoint_args:
        return {}
    if not {"match_args", "all_args"}.issubset(checkpoint_args):
        raise ValueError("Checkpoint args payload must contain both all_args and match_args. "
                         "Please regenerate the checkpoint with the current benchmark code.")
    return checkpoint_args["match_args"] or {}


def build_trained_checkpoint_args(args, alg_config):
    saved = {key: _normalize_checkpoint_value(value) for key, value in vars(args).items()
             if key not in TRAINED_CHECKPOINT_ARGS_EXCLUDED}
    keys = _collect_model_checkpoint_match_keys(alg_config) | BENCHMARK_CHECKPOINT_MATCH_KEYS
    return {"all_args": saved, "match_args": {key: saved[key] for key in sorted(keys) if key in saved}}


def save_nn_checkpoint(model, *, run_paths, algorithm, alg_name, checkpoint_args):
    torch.save(dict(state_dict=model.state_dict(), algorithm=algorithm, model_name=alg_name,
                    run_name=run_paths.run_name, checkpoint_args=checkpoint_args), run_paths.checkpoint_path)


def _validate_payload(payload, path, algorithm, current_args):
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must be a payload dict with state_dict and checkpoint_args.")
    if not {"state_dict", "checkpoint_args"}.issubset(payload):
        raise ValueError(f"Checkpoint {path} is missing required keys: state_dict/checkpoint_args.")
    if payload.get("algorithm") != algorithm:
        raise ValueError(f"Checkpoint algorithm mismatch: expected {algorithm}, got {payload.get('algorithm')} from {path}")
    saved, current = map(_extract_checkpoint_match_args, (payload["checkpoint_args"], current_args))
    changes = [f"{key}: checkpoint={saved.get(key)!r}, current={current.get(key)!r}"
               for key in sorted(saved.keys() | current.keys()) if saved.get(key) != current.get(key)]
    if changes:
        raise ValueError(f"Checkpoint model-match args mismatch for {path}: {', '.join(changes)}")


def load_nn_checkpoint(model, *, run_paths, algorithm, current_checkpoint_args):
    path = run_paths.checkpoint_path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for --test_with_checkpoint: {path}")
    payload = torch.load(path, map_location="cpu")
    _validate_payload(payload, path, algorithm, current_checkpoint_args)
    model.load_state_dict(payload["state_dict"])
    return model
