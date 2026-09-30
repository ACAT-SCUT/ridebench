"""Forecast metrics with channel-wise WMAPE and explicit scenario selection."""

import numpy as np


def _to_metric_float64(array):
    return np.asarray(array, dtype=np.float64)


def _errors(pred, true):
    observed = _to_metric_float64(true)
    return _to_metric_float64(pred) - observed, observed


def finite_sample_mask(pred, true):
    masks = [np.isfinite(value).all(axis=tuple(range(1, value.ndim))) for value in (pred, true)]
    return masks[0] & masks[1]


def mae(pred, true):
    return np.abs(_errors(pred, true)[0]).mean().item()


def mse(pred, true):
    return np.square(_errors(pred, true)[0]).mean().item()


def mape(pred, true):
    residual, observed = _errors(pred, true)
    absolute = np.abs(residual)
    return np.where(observed != 0, absolute / np.abs(observed), absolute).mean().item()


def wmape(pred, true):
    residual, observed = _errors(pred, true)
    # Reduce samples and time, retaining one ratio per channel.
    axes = tuple(range(residual.ndim - 1)) if residual.ndim > 1 else None
    numerator = np.abs(residual).sum(axis=axes, dtype=np.float64)
    denominator = np.abs(observed).sum(axis=axes, dtype=np.float64)
    ratios = np.full(np.shape(denominator), np.nan, dtype=np.float64)
    usable = denominator != 0
    if not np.any(usable):
        return float("nan")
    np.divide(numerator, denominator, out=ratios, where=usable)
    return np.nanmean(ratios).item()


def _base_scores(pred, true):
    return {"WMAPE": wmape(pred, true), "MAE": mae(pred, true), "RMSE": np.sqrt(mse(pred, true)).item()}


def steps_mean(data, steps):
    if data.ndim != 3:
        raise RuntimeError("Need 3D matrix!")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    samples, length, channels = data.shape
    padded = np.pad(data, ((0, 0), (0, -length % steps), (0, 0)), constant_values=np.nan)
    return np.nanmean(padded.reshape(samples, -1, steps, channels), axis=2)


def _has_time_axis(pred, true, metric):
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError(f"{metric} expects predictions and targets shaped as [samples, horizon, channels].")
    return min(pred.shape[1], true.shape[1]) >= 2


def direction_accuracy(pred, true, *, eps=1e-6):
    if not _has_time_axis(pred, true, "direction_accuracy"):
        return float("nan")
    forecast_delta, actual_delta = (np.diff(value, axis=1) for value in (pred, true))
    moving = np.abs(actual_delta) > float(eps)
    if not moving.any():
        return float("nan")
    return (np.sign(forecast_delta[moving]) == np.sign(actual_delta[moving])).mean().item()


def pearson_corr(pred, true, *, eps=1e-12):
    if not _has_time_axis(pred, true, "pearson_corr"):
        return float("nan")
    forecast, actual = (value - value.mean(axis=1, keepdims=True) for value in (pred, true))
    cross_product = (forecast * actual).sum(axis=1)
    magnitude = np.sqrt((forecast * forecast).sum(axis=1) * (actual * actual).sum(axis=1))
    valid = magnitude > float(eps)
    return (cross_product[valid] / magnitude[valid]).mean().item() if valid.any() else float("nan")


def get_overall_scores(pred, true):
    return _base_scores(pred, true)


def get_ultra_long_overall_scores(pred, true, *, day_steps=48):
    daily = tuple(steps_mean(value, day_steps) for value in (pred, true))
    return {**_base_scores(pred, true), "DIR_ACC_DAY": direction_accuracy(*daily), "CORR_DAY": pearson_corr(*daily)}


def _slice_horizon(pred, true, steps):
    stop = min(int(steps), pred.shape[1], true.shape[1])
    return pred[:, :stop, :], true[:, :stop, :]


def get_first_week_scores(pred, true, *, week_steps=48 * 7):
    return _base_scores(*_slice_horizon(pred, true, week_steps))


def get_scenario_scores(pred, true, mask):
    active = mask.ravel() != 0
    if not active.any():
        return dict.fromkeys(("WMAPE", "MAE", "RMSE"), float("nan"))
    selected = (value.reshape(-1, value.shape[-1])[active] for value in (pred, true))
    return _base_scores(*selected)
