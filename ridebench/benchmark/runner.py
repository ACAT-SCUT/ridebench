"""Coordinate data preparation, model execution, and result publication."""

import logging
import math
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from ridebench.data import BenchmarkDataProvider
from .batches import numpy_batch, torch_batch
from .common import (
    build_run_paths, build_trained_checkpoint_args, get_data_params, get_logger,
    get_numpy_dtype, get_torch_dtype, load_nn_checkpoint, save_nn_checkpoint, set_seed,
)
from .nn_loop import NNLoopContext, test_model_loop, train_model_loop, validate_model_loop
from .reporting import analyze_and_report, resolve_benchmark_results_csv, save_optional_run_artifacts


class Benchmark:
    def __init__(self, args, alg_config):
        self.args, self.alg_config = args, alg_config
        self.algorithm = args.algorithm
        self.alg_name = getattr(args, "model_name", None) or self.algorithm
        for name in ("run_results_dir", "dtype", "seed", "save_data", "log_level"):
            setattr(self, name, getattr(args, name))
        for name in ("epochs", "patience", "learning_rate", "device", "optimizer",
                     "lr_scheduler", "train_loss", "val_loss", "save_losses"):
            setattr(self, name, getattr(args, name, None))
        self.test_with_checkpoint = bool(getattr(args, "test_with_checkpoint", False))
        self.trained_checkpoint_args = build_trained_checkpoint_args(args, alg_config) if alg_config.need_train() else None
        self.run_paths = build_run_paths(self.run_results_dir, self.algorithm, self.alg_name,
                                         self.learning_rate, getattr(args, "benchmark_variant", "regular"))
        self.th_dtype, self.np_dtype = get_torch_dtype(self.dtype), get_numpy_dtype(self.dtype)
        self.log_batch_interval = max(0, int(getattr(args, "log_batch_interval", 200)))
        self.logger = get_logger(self.log_level)
        self._active_progress_bar = None
        self._configure_runtime_devices()

    def log(self, level, msg, *args, **kwargs):
        self.logger.log(level, msg, *args, **kwargs)

    def _configure_runtime_devices(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")

    def _progress(self, iterable, **kwargs):
        interactive = sys.stderr.isatty()
        bar = tqdm(iterable, dynamic_ncols=True, leave=False, disable=not interactive, **kwargs)
        self._active_progress_bar = bar if interactive else None
        return bar

    def _maybe_log_batch_progress(self, *, split_name, batch_idx, total_batches,
                                  epoch=None, avg_loss=None):
        if self.log_batch_interval <= 0 or batch_idx % self.log_batch_interval:
            return
        progress = f"batch {batch_idx}"
        if total_batches:
            progress += f"/{total_batches} ({100.0 * batch_idx / total_batches:.1f}%)"
        if self._active_progress_bar is not None:
            suffix = "" if avg_loss is None else f", avg_loss={avg_loss:.6f}"
            self._active_progress_bar.set_postfix_str(progress + suffix, refresh=False)
        else:
            prefix = "" if epoch is None else f"Epoch {epoch} | "
            message = f"{prefix}{split_name} progress | {progress}"
            if avg_loss is not None:
                message += f" | avg_loss={avg_loss:.6f}"
            self.log(logging.INFO, message)

    @staticmethod
    def _reduce_aux_loss(aux_loss):
        return aux_loss.mean() if torch.is_tensor(aux_loss) and aux_loss.dim() > 0 else aux_loss

    @staticmethod
    def _is_finite_loss(loss):
        return bool(torch.isfinite(loss).all().item()) if torch.is_tensor(loss) else math.isfinite(float(loss))

    def _nn_loop_context(self):
        fields = ("algorithm", "args", "device", "th_dtype", "optimizer", "lr_scheduler",
                  "learning_rate", "train_loss", "val_loss", "epochs", "patience", "log", "get_torch_X_y")
        callbacks = {"progress": self._progress, "maybe_log_batch_progress": self._maybe_log_batch_progress,
                     "is_finite_loss": self._is_finite_loss, "reduce_aux_loss": self._reduce_aux_loss}
        return NNLoopContext(**{name: getattr(self, name) for name in fields}, **callbacks)

    def _show_configuration(self, training):
        entries = [("Algorithm", self.alg_name), ("Base Algorithm", self.algorithm)]
        if training:
            entries += [("Test With Checkpoint", self.test_with_checkpoint), ("Epochs", self.epochs)]
        entries.append(("Batch Size", self.args.batch_size))
        if training:
            entries.append(("Early Stopping Patience", self.patience))
        variant = getattr(self.args, "benchmark_variant", "regular")
        entries += [("Run Name", self.run_paths.run_name), ("Run Results Dir", self.run_paths.root_dir),
                    ("Benchmark Variant", variant),
                    ("Benchmark Summary CSV", resolve_benchmark_results_csv(self.args.benchmark_summary_csv_path, variant)),
                    ("Data Type", self.dtype)]
        if training:
            entries.append(("Device", self.device))
        entries.append(("Random Seed", "-" if self.seed is None else self.seed))
        if training:
            entries += [("Optimizer", self.optimizer), ("Learning Rate", self.learning_rate),
                        ("Learning Rate Scheduler", self.lr_scheduler)]
        for title, attribute in (("Dataset Path", "dataset_path"), ("Input Length", "input_len"),
                                 ("Predict Length", "output_len"), ("Endogenous Variables", "endo_vars"),
                                 ("Exogenous Continuous Variables", "exo_con_vars"), ("Exogenous Discrete Variables", "exo_dis_dict")):
            entries.append((title, getattr(self.args, attribute)))
        for split, label in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
            entries.append((f"{label} Start", f"day_index={getattr(self.args, split + '_start_day_index')}, time={getattr(self.args, split + '_start_time')}"))
        self.log(logging.INFO, "*****************Benchmark Config*****************")
        for label, value in entries:
            self.log(logging.INFO, "%s: %s", label, value)
        self.log(logging.INFO, "**************************************************")

    def show_inference_config(self):
        self._show_configuration(False)

    def show_nn_config(self):
        self._show_configuration(True)

    def show_config(self):
        (self.show_nn_config if self.alg_config.need_train() else self.show_inference_config)()

    def show_dataset_summary(self, data_provider, datasets):
        factory = data_provider.dataset_factory
        summary = factory.data_summary()
        self.log(logging.INFO, "Dataset Timeline: [%s -> %s] | csv_engine=%s | time_points=%d | areas=%d",
                 *(summary[key] for key in ("dataset_start", "dataset_end", "csv_engine", "num_time_points", "num_areas")))
        if summary["requested_train_start"] != summary["actual_train_start"]:
            self.log(logging.INFO, "Effective Train Start adjusted from %s to %s.",
                     summary["requested_train_start"], summary["actual_train_start"])
        for warning in factory.load_warnings():
            self.log(logging.WARNING, warning)
        if factory.custom_area_split_enabled():
            preview = []
            for areas in (self.args.train_val_area_ids, self.args.test_area_ids):
                preview.extend((len(areas), list(areas[:8]), "..." if len(areas) > 8 else ""))
            self.log(logging.INFO, "Custom area split enabled | train/val areas=%d %s%s | test areas=%d %s%s | area_id discrete feature disabled", *preview)
        for split in ("train", "val", "test"):
            summary = datasets[split].split_summary()
            self.log(logging.INFO, "%s Split: areas=%d, windows=%d, samples=%d, target=[%s -> %s]", split.upper(),
                     *(summary[key] for key in ("num_areas", "num_windows", "num_samples", "first_target_start", "last_target_end")))
            details = [summary[f"{position}_{part}_{edge}"] for position in ("first", "last")
                       for part in ("input", "target") for edge in ("start", "end")]
            self.log(logging.DEBUG, "%s Split Detail: first input=[%s -> %s], first target=[%s -> %s], last input=[%s -> %s], last target=[%s -> %s]", split.upper(), *details)

    def run(self):
        self.show_config()
        set_seed(self.seed)
        self.log(logging.INFO, "Preprocessing data.")
        provider = BenchmarkDataProvider(self.args)
        loaders = {split: getattr(provider, f"{split}_loader")() for split in ("train", "val", "test")}
        self.show_dataset_summary(provider, {split: loader.dataset for split, loader in loaders.items()})
        self.log(logging.INFO, "Initializing model.")
        model = self.alg_config.model(args=self.args, **get_data_params(provider.dataset_factory))
        model, losses = self.prepare_model(model, loaders["train"], loaders["val"])
        selected = self.summarize_losses(losses)
        if not self.test_with_checkpoint:
            self.save_model(model)
        if selected is not None:
            selected["test_loss"] = float(self.validate_model(model, loaders["test"], split_name="Test"))
            self.log(logging.INFO, "Selected epoch %d losses | train: %.6f, val: %.6f, test: %.6f",
                     *(selected[key] for key in ("selected_epoch", "train_loss", "val_loss", "test_loss")))
        self._evaluate_and_report(model, loaders["test"], losses, selected)

    def _evaluate_and_report(self, model, loader, losses, loss_summary):
        self.log(logging.INFO, "Testing algorithm %s.", self.alg_name)
        started = time.perf_counter()
        predicted, observed, scenarios = self.test(model, loader)
        elapsed = time.perf_counter() - started
        samples = int(predicted.shape[0]) if hasattr(predicted, "shape") and len(predicted.shape) else 0
        per_sample = elapsed * 1000.0 / samples if samples > 0 else float("inf")
        self.log(logging.INFO, "Test runtime: %.3fs | samples: %d | ms/sample: %.3f", elapsed, samples, per_sample)
        predicted, observed = (loader.dataset.inverse_transform_endo(values) for values in (predicted, observed))
        self.save_losses_and_result(losses, predicted, observed)
        analyze_and_report(logger=self.logger, args=self.args, alg_name=self.alg_name, run_paths=self.run_paths,
                           learning_rate=self.learning_rate, pred=predicted, true=observed,
                           scenario_features=scenarios, test_dataset=loader.dataset, loss_summary=loss_summary,
                           test_elapsed_sec=elapsed, test_ms_per_sample=per_sample)

    def save_model(self, model):
        if self.alg_config.need_train() and not self.test_with_checkpoint:
            self.log(logging.INFO, "Saving the model to %s.", self.run_paths.checkpoint_path)
            save_nn_checkpoint(model, run_paths=self.run_paths, algorithm=self.algorithm,
                               alg_name=self.alg_name, checkpoint_args=self.trained_checkpoint_args)

    def save_losses_and_result(self, losses, pred, true):
        save_optional_run_artifacts(logger=self.logger, run_paths=self.run_paths, save_data=bool(self.save_data),
                                    save_losses=bool(self.save_losses), losses=losses, pred=pred, true=true)

    def summarize_losses(self, losses):
        if losses is None or len(losses[1]) == 0:
            return None
        index = int(np.argmin(np.asarray(losses[1])))
        return dict(selected_epoch=index + 1, train_loss=float(losses[0][index]), val_loss=float(losses[1][index]))

    def test_inference(self, model, test_dataloader):
        buckets = ([], [], [])
        for batch in self._progress(test_dataloader):
            inputs, target = self.get_numpy_X_y(batch)
            values = (model.predict_batch(*inputs), target, batch["y_scenario_features"].numpy(force=True))
            for bucket, value in zip(buckets, values):
                bucket.append(value)
        return tuple(np.vstack(bucket) for bucket in buckets)

    def test(self, model, test_dataloader):
        evaluate = self.test_model if self.alg_config.need_train() else self.test_inference
        return evaluate(model, test_dataloader)

    def prepare_model(self, model, train_dataloader, val_dataloader):
        if self.alg_config.need_train() and self.test_with_checkpoint:
            self.log(logging.INFO, "Skipping training and testing with the saved trained checkpoint for %s.", self.alg_name)
            self.log(logging.INFO, "Loading checkpoint from %s.", self.run_paths.checkpoint_path)
            return load_nn_checkpoint(model, run_paths=self.run_paths, algorithm=self.algorithm,
                                      current_checkpoint_args=self.trained_checkpoint_args), None
        return self.train(model, train_dataloader, val_dataloader)

    def train(self, model, train_dataloader, val_dataloader):
        if self.alg_config.need_train():
            self.log(logging.INFO, "Training algorithm %s.", self.alg_name)
            return self.train_model(model, train_dataloader, val_dataloader)
        if getattr(model, "is_global_model", False):
            self.log(logging.INFO, "Fitting algorithm %s on the training windows.", self.alg_name)
            options = {"val_dataloader": val_dataloader} if getattr(model, "fit_uses_validation", False) else {}
            model.fit_loader(train_dataloader, self.get_numpy_X_y, **options)
        return model, None

    def train_model(self, model, train_dataloader, val_dataloader):
        return train_model_loop(model, train_dataloader, val_dataloader, self._nn_loop_context())

    def validate_model(self, model, val_dataloader, *, split_name="Validation"):
        return validate_model_loop(model, val_dataloader, self._nn_loop_context(), split_name=split_name)

    def test_model(self, model, test_dataloader):
        return test_model_loop(model, test_dataloader, self._nn_loop_context())

    def get_torch_X_y(self, batch):
        return torch_batch(batch, self.device, self.th_dtype)

    def get_numpy_X_y(self, batch):
        return numpy_batch(batch, self.np_dtype)
