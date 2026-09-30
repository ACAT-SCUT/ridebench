"""Epoch execution with shared loss accounting and best-state restoration."""

from __future__ import annotations

import copy
import logging
import statistics
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from .ops import (
    forward_step, get_lr_scheduler, get_optimizer, loss_step,
    lr_scheduler_batch_step, lr_scheduler_epoch_step,
)


class EarlyStopper:
    def __init__(self, patience, params_getter, init_epoch=0, init_loss=float("inf")):
        self.patience, self.params_getter = patience, params_getter
        self.best_epoch, self.loss = init_epoch, init_loss
        self.params = params_getter()

    def update(self, epoch, loss):
        improved = loss < self.loss
        if improved:
            self.best_epoch, self.loss, self.params = epoch, loss, self.params_getter()
        return improved

    def stop(self, epoch):
        return epoch - self.best_epoch >= self.patience


@dataclass(frozen=True)
class NNLoopContext:
    algorithm: str
    args: Namespace
    device: str
    th_dtype: torch.dtype
    optimizer: Any
    lr_scheduler: Any
    learning_rate: float | None
    train_loss: Any
    val_loss: Any
    epochs: int
    patience: int
    log: Callable[..., None]
    progress: Callable[..., Any]
    maybe_log_batch_progress: Callable[..., None]
    get_torch_X_y: Callable[[dict[str, Tensor]], tuple[tuple, Tensor]]
    is_finite_loss: Callable[[Tensor | float], bool]
    reduce_aux_loss: Callable[[Tensor | float], Tensor | float]

@dataclass
class _LossMeter:
    values: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def record(self, loss, batch_size):
        self.values.append(loss)
        self.weights.append(batch_size)
        self.total += loss * batch_size
        self.count += batch_size

    @property
    def running(self):
        return self.total / self.count if self.count > 0 else None

    def mean(self):
        # fmean retains the original weighted summation and empty-loader errors.
        return statistics.fmean(self.values, self.weights)


def _move_model(model, ctx):
    return model.to(device=ctx.device, dtype=ctx.th_dtype)


def _build_early_stopper(model, ctx):
    return EarlyStopper(ctx.patience, lambda: copy.deepcopy(model.state_dict()))


def _restore_best_params(model, ctx, early_stopper):
    model.load_state_dict(early_stopper.params)


def _epoch_pass(model, loader, ctx, *, optimizer=None, scheduler=None,
                epoch=None, split_name="Validation"):
    training = optimizer is not None
    model.train(training)
    meter = _LossMeter()
    options = {"desc": f"Train {epoch}/{ctx.epochs}"} if training else {}
    # Training inherits the caller's autograd mode; evaluation disables it.
    from contextlib import nullcontext
    with nullcontext() if training else torch.no_grad():
        for index, batch in enumerate(ctx.progress(loader, **options), 1):
            inputs, targets = ctx.get_torch_X_y(batch)
            if training:
                model.zero_grad()
            prediction = forward_step(ctx.algorithm, ctx.args, model, inputs, targets,
                                      phase="train" if training else "eval")
            objective = loss_step(
                ctx.train_loss if training else ctx.val_loss, prediction.pred_y, targets,
                aux_loss=prediction.aux_loss, aux_weight=prediction.aux_weight,
                reduce_aux_loss=ctx.reduce_aux_loss,
            )
            if not ctx.is_finite_loss(objective):
                if training:
                    ctx.log(logging.WARNING,
                            "Non-finite training loss detected at epoch %d batch %d; stopping training early. learning_rate=%s",
                            epoch, index, ctx.learning_rate)
                else:
                    ctx.log(logging.WARNING, "Non-finite validation loss detected at batch %d; returning inf for this split.", index)
                return float("inf"), True
            meter.record(objective.item(), len(targets))
            progress = dict(split_name=split_name, batch_idx=index,
                            total_batches=len(loader), avg_loss=meter.running)
            if training:
                progress["epoch"] = epoch
            ctx.maybe_log_batch_progress(**progress)
            if training:
                objective.backward()
                optimizer.step()
                lr_scheduler_batch_step(ctx.lr_scheduler, scheduler)
    return meter.mean(), False


def train_model_loop(model, train_dataloader, val_dataloader, ctx):
    ctx.log(logging.INFO, "Setting optimizer.")
    model = _move_model(model, ctx)
    optimizer = get_optimizer(ctx.optimizer, model, ctx.learning_rate)
    scheduler = get_lr_scheduler(ctx.lr_scheduler, model, optimizer, ctx.epochs)
    best = _build_early_stopper(model, ctx)
    history = ([], [], [])
    ctx.log(logging.INFO, "Start training.")
    for epoch in range(1, ctx.epochs + 1):
        train_loss, invalid = _epoch_pass(
            model, train_dataloader, ctx, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, split_name="Train",
        )
        lr_scheduler_epoch_step(ctx.lr_scheduler, scheduler)
        validation = float("inf") if invalid else validate_model_loop(model, val_dataloader, ctx)
        for series, value in zip(history, (train_loss, validation, float("nan"))):
            series.append(value)
        improved = best.update(epoch, validation)
        suffix = ", model best parameters updated." if improved else "."
        ctx.log(logging.INFO, "Epoch %d, train loss: %f, validation loss: %f" + suffix,
                epoch, train_loss, validation)
        reason = "Early stopped." if best.stop(epoch) else (
            "Training stopped because the loss became non-finite." if invalid else None
        )
        if reason:
            ctx.log(logging.INFO, reason)
            break
    ctx.log(logging.INFO, "Best epoch %d, validation loss %f.", best.best_epoch, best.loss)
    _restore_best_params(model, ctx, best)
    return model, history


def validate_model_loop(model, dataloader, ctx, *, split_name="Validation"):
    average, _ = _epoch_pass(_move_model(model, ctx), dataloader, ctx, split_name=split_name)
    return average


def test_model_loop(model, dataloader, ctx):
    model = _move_model(model, ctx)
    model.eval()
    collected = ([], [], [])
    with torch.no_grad():
        for batch in ctx.progress(dataloader):
            inputs, targets = ctx.get_torch_X_y(batch)
            prediction = forward_step(ctx.algorithm, ctx.args, model, inputs, targets, phase="predict")
            for bucket, tensor in zip(collected, (prediction.pred_y, targets, batch["y_scenario_features"])):
                bucket.append(tensor.numpy(force=True))
    return tuple(np.vstack(bucket) for bucket in collected)
