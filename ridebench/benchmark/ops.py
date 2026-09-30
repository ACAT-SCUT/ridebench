"""Training policies: prediction adapters, objectives, and optimizer selection."""

from argparse import Namespace
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


@dataclass(frozen=True)
class ForwardStepResult:
    pred_y: Tensor
    aux_loss: Tensor | float = 0.0
    aux_weight: float = 0.0


def fredf_loss(pred_y, y):
    spectra = [torch.fft.rfft(value.float(), dim=1) for value in (pred_y, y)]
    return (spectra[0] - spectra[1]).abs().mean()


def _ema_trend(x, alpha=0.5):
    values = x.float().split(1, dim=1)
    trend = [values[0]]
    for current in values[1:]:
        trend.append(alpha * current + (1.0 - alpha) * trend[-1])
    return torch.cat(trend, dim=1)


def dbloss(pred_y, y, alpha=0.5, beta=0.5, eps=1e-8):
    trends = [_ema_trend(value, alpha) for value in (pred_y, y)]
    residuals = [value.float() - trend for value, trend in zip((pred_y, y), trends)]
    seasonal = F.mse_loss(*residuals)
    trend = F.l1_loss(*trends)
    scaled_trend = trend * (seasonal / (trend + eps)).detach()
    return beta * seasonal + (1.0 - beta) * scaled_trend


_LOSSES = {"smooth_l1": F.smooth_l1_loss, "mse": F.mse_loss, "mae": F.l1_loss,
           "fredf": fredf_loss, "dbloss": dbloss}
_OPTIMIZERS = {"adam": Adam, "adamw": AdamW}


def compute_loss(loss_type, pred_y, y):
    if loss_type not in _LOSSES:
        raise NotImplementedError(f"Method to compute loss {loss_type} is defined but not supported yet.")
    return _LOSSES[loss_type](pred_y, y)


def _check_scheduler(name):
    if name not in ("none", "cosine"):
        raise NotImplementedError(f"Learning rate scheduler {name} is defined but not supported yet.")


def lr_scheduler_batch_step(lr_scheduler_type, lr_scheduler):
    _check_scheduler(lr_scheduler_type)


def lr_scheduler_epoch_step(lr_scheduler_type, lr_scheduler):
    _check_scheduler(lr_scheduler_type)
    if lr_scheduler_type == "cosine":
        lr_scheduler.step()


def get_lr_scheduler(lr_scheduler, model, optimizer, epochs):
    _check_scheduler(lr_scheduler)
    if lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, 1 if epochs is None else epochs))
    return None


def get_optimizer(optimizer, model, learning_rate=1e-3):
    if optimizer not in _OPTIMIZERS:
        raise NotImplementedError(f"Optimizer {optimizer} is defined but not supported yet.")
    return _OPTIMIZERS[optimizer](model.parameters(), lr=learning_rate)


# Model-specific auxiliary outputs are interpreted here, outside the epoch loop.
_AUXILIARY = {
    "TimeFilter": (frozenset({"train"}), "moe_loss_weight", 0.05),
    "TimeBase": (frozenset({"train"}), "orthogonal_weight", 0.0),
    "CATS": (frozenset({"train"}), None, 1.0),
    "DAG": (frozenset({"train", "eval"}), None, 1.0),
    "DUET": (frozenset({"train", "eval"}), None, 1.0),
}
_TRAIN_KEYWORDS = {"TimeFilter": {"is_training": True}, "CATS": {"return_aux_loss": True}}


def forward_step(algorithm, args: Namespace, model, X, y, *, phase):
    keywords = _TRAIN_KEYWORDS.get(algorithm, {}) if phase == "train" else {}
    output = model(*X, **keywords)
    phases, weight_arg, default = _AUXILIARY.get(algorithm, ((), None, 0.0))
    uses_auxiliary = phase in phases and (algorithm != "TimeBase" or isinstance(output, tuple))
    if uses_auxiliary:
        prediction, auxiliary = output
        weight = default if weight_arg is None else float(getattr(args, weight_arg, default))
        return ForwardStepResult(prediction, auxiliary, weight)
    return ForwardStepResult(output[0] if isinstance(output, tuple) else output)


def loss_step(loss_type, pred_y, y, *, aux_loss=0.0, aux_weight=0.0, reduce_aux_loss=None):
    objective = compute_loss(loss_type, pred_y, y)
    if aux_weight != 0.0:
        auxiliary = aux_loss if reduce_aux_loss is None else reduce_aux_loss(aux_loss)
        objective = objective + aux_weight * auxiliary
    return objective
