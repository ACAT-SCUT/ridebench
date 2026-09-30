"""Persist benchmark outputs and render the result figures."""

from __future__ import annotations

import csv
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from .common import RunPaths
from .scoring import (
    finite_sample_mask,
    get_first_week_scores,
    get_overall_scores,
    get_scenario_scores,
    get_ultra_long_overall_scores,
)

DEFAULT_BENCHMARK_VARIANT = "regular"
ULTRA_LONG_BENCHMARK_VARIANT = "ultra_long"
REGULAR_LONG_BENCHMARK_RESULTS_CSV = "run_results/summary_csv/regular_long_forecast.csv"
ULTRA_LONG_BENCHMARK_RESULTS_CSV = "run_results/summary_csv/ultra_long_forecast.csv"


def _format_benchmark_result_value(name: str, value: Any) -> str:
    if value in ("", None):
        return ""
    if name in {"model_name", "learning_rate"}:
        return str(value)
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def normalize_benchmark_summary_csv_path(path_like: str | Path,
                                         benchmark_variant: str = DEFAULT_BENCHMARK_VARIANT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if benchmark_variant == ULTRA_LONG_BENCHMARK_VARIANT and str(path) == REGULAR_LONG_BENCHMARK_RESULTS_CSV:
        return Path(ULTRA_LONG_BENCHMARK_RESULTS_CSV)
    return path


def resolve_benchmark_results_csv(path_like: str | Path,
                                  benchmark_variant: str = DEFAULT_BENCHMARK_VARIANT) -> Path:
    return normalize_benchmark_summary_csv_path(path_like, benchmark_variant)


def save_optional_run_artifacts(logger: logging.Logger, run_paths: RunPaths, save_data: bool,
                                save_losses: bool, losses, pred: np.ndarray, true: np.ndarray) -> None:
    if save_data:
        logger.info("Saving pred and true data to %s and %s.", run_paths.pred_path, run_paths.true_path)
        np.save(run_paths.pred_path, pred)
        np.save(run_paths.true_path, true)
    if save_losses and losses is not None:
        train_losses, val_losses, test_losses = losses
        logger.info("Saving train, validation, and placeholder test losses to %s.", run_paths.losses_path)
        np.savez(run_paths.losses_path, train=train_losses, val=val_losses, test=test_losses)


def save_scores_json(logger: logging.Logger, run_paths: RunPaths, alg_name: str,
                     scores: dict[str, object]) -> None:
    logger.info("Saving scores for %s to %s.", alg_name, run_paths.score_json_path)
    with run_paths.score_json_path.open("w", encoding="utf-8") as output:
        json.dump(scores, output, indent=2)


def serialize_args(args: Namespace) -> dict[str, object]:
    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "value"):
            return value.value
        return value

    return {key: normalize(value) for key, value in vars(args).items()}


def append_benchmark_result_csv(logger: logging.Logger, benchmark_summary_csv_path: str | Path,
                                benchmark_variant: str, row: dict[str, object]) -> None:
    formatted = {key: _format_benchmark_result_value(key, value) for key, value in row.items()}
    columns = list(formatted)
    destination = resolve_benchmark_results_csv(benchmark_summary_csv_path, benchmark_variant)
    destination.parent.mkdir(parents=True, exist_ok=True)

    exists = destination.exists()
    nonempty = exists and destination.stat().st_size > 0
    if nonempty:
        with destination.open("r", newline="", encoding="utf-8") as source:
            existing = next(csv.reader(source), [])
        if existing and existing != columns:
            raise ValueError(
                f"Benchmark summary CSV schema mismatch for {destination}. "
                f"Existing header={existing}, new header={columns}. "
                "Use a separate --benchmark_summary_csv_path for different benchmark variants."
            )
    with destination.open("a", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        if not nonempty:
            writer.writeheader()
        writer.writerow(formatted)
    logger.info("Appended benchmark summary row to %s.", destination)


def _axes_for(plt, variable_count: int):
    figure, axes = plt.subplots(variable_count, 1, figsize=(12, 3.2 * variable_count), sharex=True)
    return figure, [axes] if variable_count == 1 else axes


def _draw_series(axis, horizon: np.ndarray, truth: np.ndarray, prediction: np.ndarray,
                 variable_name: str, spans: list[tuple[float, float]], shade: str) -> None:
    axis.plot(horizon, truth, label="True", linewidth=2.0, color="#1f77b4")
    axis.plot(horizon, prediction, label="Pred", linewidth=2.0, color="#d62728", alpha=0.9)
    for left, right in spans:
        axis.axvspan(left, right, color=shade, alpha=0.16, linewidth=0)
    axis.set_ylabel(variable_name, fontsize=10)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=True)


def _save_overall_showcase_plot(logger: logging.Logger, alg_name: str, run_paths: RunPaths,
                                endo_vars: list[str], pred: np.ndarray, true: np.ndarray,
                                sample_idx: int = 0) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    variable_count = min(3, pred.shape[2])
    horizon = np.arange(1, pred.shape[1] + 1)
    figure, axes = _axes_for(plt, variable_count)
    for axis, variable in zip(axes, range(variable_count)):
        _draw_series(axis, horizon, true[sample_idx, :, variable].astype(np.float64),
                     pred[sample_idx, :, variable].astype(np.float64), endo_vars[variable], [], "#bbbbbb")
    axes[-1].set_xlabel("Forecast Horizon Step", fontsize=10)
    figure.suptitle(f"{alg_name} Overall Showcase (sample={sample_idx}, True vs Pred)",
                    fontsize=13, fontweight="bold")
    figure.tight_layout()
    figure.savefig(Path(f"{run_paths.plot_prefix}_overall_showcase.pdf"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    logger.info("Saved comparison plots to %s.", run_paths.plot_dir)


def save_showcase_plots(logger: logging.Logger, alg_name: str, run_paths: RunPaths,
                        endo_vars: list[str], pred: np.ndarray, true: np.ndarray,
                        weather_events: np.ndarray, holiday_events: np.ndarray,
                        large_scale_event_flags: np.ndarray, sample_idx: int = 0) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    variable_count = min(3, pred.shape[2])
    horizon = np.arange(1, pred.shape[1] + 1)
    scenarios = (
        ("weather", weather_events != 0, "#55a868"),
        ("holiday", holiday_events != 0, "#8172b2"),
        ("large_scale_event", large_scale_event_flags != 0, "#dd8452"),
    )

    def contiguous_spans(mask: np.ndarray) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        start: int | None = None
        for index, active in enumerate(mask.tolist()):
            if active and start is None:
                start = index
            elif not active and start is not None:
                spans.append((start + 0.5, index + 0.5))
                start = None
        if start is not None:
            spans.append((start + 0.5, len(mask) + 0.5))
        return spans

    def save_figure(path: Path, title: str, mask: np.ndarray | None = None,
                    shade: str = "#bbbbbb") -> None:
        figure, axes = _axes_for(plt, variable_count)
        spans = [] if mask is None else contiguous_spans(mask)
        for axis, variable in zip(axes, range(variable_count)):
            _draw_series(axis, horizon, true[sample_idx, :, variable].astype(np.float64),
                         pred[sample_idx, :, variable].astype(np.float64), endo_vars[variable], spans, shade)
        axes[-1].set_xlabel("Forecast Horizon Step", fontsize=10)
        figure.suptitle(title, fontsize=13, fontweight="bold")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)

    save_figure(Path(f"{run_paths.plot_prefix}_overall_showcase.pdf"),
                f"{alg_name} Overall Showcase (sample={sample_idx}, True vs Pred)")
    for name, mask, shade in scenarios:
        pretty_name = " ".join(part.title() for part in name.split("_"))
        save_figure(Path(f"{run_paths.plot_prefix}_{name}_showcase.pdf"),
                    f"{alg_name} {pretty_name} Showcase (sample={sample_idx}, hits={int(mask[sample_idx].sum())})",
                    mask[sample_idx], shade)
    logger.info("Saved comparison plots to %s.", run_paths.plot_dir)


def _show_scores(logger: logging.Logger, title: str, scores: dict[str, float], end: str) -> None:
    logger.info(title)
    for name, score in scores.items():
        logger.info("%s: %s", name, score)
    logger.info(end)


def _training_fields(loss_summary: dict[str, float | int] | None) -> dict[str, object]:
    keys = ("selected_epoch", "train_loss", "val_loss", "test_loss")
    return {key: ("" if loss_summary is None else loss_summary[key]) for key in keys}


def _default_result_row(*, alg_name: str, learning_rate, overall: dict[str, float],
                        weather: dict[str, float], holiday: dict[str, float],
                        large_scale_event: dict[str, float], loss_summary: dict[str, float | int] | None,
                        test_elapsed_sec: float, test_ms_per_sample: float) -> dict[str, object]:
    row: dict[str, object] = {"model_name": alg_name, "learning_rate": learning_rate}
    row.update(_training_fields(loss_summary))
    row.update({
        "test_elapsed_sec": test_elapsed_sec, "test_ms_per_sample": test_ms_per_sample,
        "overall_wmape": overall["WMAPE"], "overall_mae": overall["MAE"], "overall_rmse": overall["RMSE"],
        "weather_wmape": weather["WMAPE"], "weather_mae": weather["MAE"], "weather_rmse": weather["RMSE"],
        "holiday_wmape": holiday["WMAPE"], "holiday_mae": holiday["MAE"], "holiday_rmse": holiday["RMSE"],
        "large_scale_event_wmape": large_scale_event["WMAPE"],
        "large_scale_event_mae": large_scale_event["MAE"], "large_scale_event_rmse": large_scale_event["RMSE"],
    })
    return row


def _ultra_long_result_row(*, alg_name: str, learning_rate, overall: dict[str, float],
                           first_week: dict[str, float], loss_summary: dict[str, float | int] | None,
                           test_elapsed_sec: float, test_ms_per_sample: float) -> dict[str, object]:
    row: dict[str, object] = {"model_name": alg_name, "learning_rate": learning_rate}
    row.update(_training_fields(loss_summary))
    row.update({
        "test_elapsed_sec": test_elapsed_sec, "test_ms_per_sample": test_ms_per_sample,
        "overall_wmape": overall["WMAPE"], "overall_mae": overall["MAE"], "overall_rmse": overall["RMSE"],
        "overall_dir_acc_day": overall["DIR_ACC_DAY"], "overall_corr_day": overall["CORR_DAY"],
        "first_week_wmape": first_week["WMAPE"], "first_week_mae": first_week["MAE"],
        "first_week_rmse": first_week["RMSE"],
    })
    return row


def _store_report(logger: logging.Logger, args: Namespace, run_paths: RunPaths, alg_name: str,
                  row: dict[str, object], scores: dict[str, object]) -> None:
    append_benchmark_result_csv(
        logger=logger,
        benchmark_summary_csv_path=getattr(args, "benchmark_summary_csv_path", REGULAR_LONG_BENCHMARK_RESULTS_CSV),
        benchmark_variant=getattr(args, "benchmark_variant", DEFAULT_BENCHMARK_VARIANT),
        row=row,
    )
    save_scores_json(logger=logger, run_paths=run_paths, alg_name=alg_name, scores=scores)


def _analyze_default_profile(logger: logging.Logger, *, args: Namespace, alg_name: str, run_paths: RunPaths,
                             learning_rate, pred: np.ndarray, true: np.ndarray, scenario_features: np.ndarray,
                             test_dataset, loss_summary: dict[str, float | int] | None,
                             test_elapsed_sec: float, test_ms_per_sample: float) -> None:
    weather_events = test_dataset.weather_events(scenario_features)
    holiday_events = test_dataset.holiday_events(scenario_features)
    event_flags = test_dataset.large_scale_events(scenario_features)
    overall = get_overall_scores(pred, true)
    weather = get_scenario_scores(pred, true, weather_events)
    holiday = get_scenario_scores(pred, true, holiday_events)
    large_event = get_scenario_scores(pred, true, event_flags)
    _show_scores(logger, "============Overall=============", overall, "================================")
    _show_scores(logger, "===========Weather=============", weather, "================================")
    _show_scores(logger, "===========Holiday=============", holiday, "================================")
    _show_scores(logger, "======Large Scale Event======", large_event, "================================")
    try:
        save_showcase_plots(logger, alg_name, run_paths, list(args.endo_vars), pred, true,
                            weather_events, holiday_events, event_flags, 0)
    except Exception:
        logger.warning("Failed to generate comparison plots.", exc_info=True)

    row = _default_result_row(
        alg_name=alg_name, learning_rate=learning_rate, overall=overall, weather=weather,
        holiday=holiday, large_scale_event=large_event, loss_summary=loss_summary,
        test_elapsed_sec=test_elapsed_sec, test_ms_per_sample=test_ms_per_sample,
    )
    scores: dict[str, object] = {
        "params": serialize_args(args), "overall": overall, "weather": weather,
        "holiday": holiday, "large_scale_event": large_event,
        "runtime": {"test_elapsed_sec": float(test_elapsed_sec), "test_ms_per_sample": float(test_ms_per_sample)},
    }
    if loss_summary is not None:
        scores["training"] = loss_summary
    _store_report(logger, args, run_paths, alg_name, row, scores)


def _analyze_ultra_long_profile(logger: logging.Logger, *, args: Namespace, alg_name: str,
                                run_paths: RunPaths, learning_rate, pred: np.ndarray, true: np.ndarray,
                                loss_summary: dict[str, float | int] | None, test_elapsed_sec: float,
                                test_ms_per_sample: float) -> None:
    overall = get_ultra_long_overall_scores(pred, true)
    first_week = get_first_week_scores(pred, true)
    _show_scores(logger, "=========Ultra Overall=========", overall, "================================")
    _show_scores(logger, "==========First Week===========", first_week, "================================")
    try:
        _save_overall_showcase_plot(logger, alg_name, run_paths, list(args.endo_vars), pred, true, 0)
    except Exception:
        logger.warning("Failed to generate comparison plots.", exc_info=True)
    row = _ultra_long_result_row(
        alg_name=alg_name, learning_rate=learning_rate, overall=overall, first_week=first_week,
        loss_summary=loss_summary, test_elapsed_sec=test_elapsed_sec, test_ms_per_sample=test_ms_per_sample,
    )
    scores: dict[str, object] = {
        "params": serialize_args(args), "overall": overall, "first_week": first_week,
        "runtime": {"test_elapsed_sec": float(test_elapsed_sec), "test_ms_per_sample": float(test_ms_per_sample)},
    }
    if loss_summary is not None:
        scores["training"] = loss_summary
    _store_report(logger, args, run_paths, alg_name, row, scores)


def analyze_and_report(logger: logging.Logger, *, args: Namespace, alg_name: str, run_paths: RunPaths,
                       learning_rate, pred: np.ndarray, true: np.ndarray, scenario_features: np.ndarray,
                       test_dataset, loss_summary: dict[str, float | int] | None,
                       test_elapsed_sec: float, test_ms_per_sample: float) -> None:
    logger.info("Analyzing result.")
    valid_samples = finite_sample_mask(pred, true)
    invalid_count = int(valid_samples.size - valid_samples.sum())
    if invalid_count > 0:
        logger.warning(
            "Non-finite predictions or targets detected in test outputs: %d / %d samples.",
            invalid_count, valid_samples.size,
        )
    variant = getattr(args, "benchmark_variant", DEFAULT_BENCHMARK_VARIANT)
    if variant == ULTRA_LONG_BENCHMARK_VARIANT:
        _analyze_ultra_long_profile(
            logger=logger, args=args, alg_name=alg_name, run_paths=run_paths, learning_rate=learning_rate,
            pred=pred, true=true, loss_summary=loss_summary, test_elapsed_sec=test_elapsed_sec,
            test_ms_per_sample=test_ms_per_sample,
        )
        return
    _analyze_default_profile(
        logger=logger, args=args, alg_name=alg_name, run_paths=run_paths, learning_rate=learning_rate,
        pred=pred, true=true, scenario_features=scenario_features, test_dataset=test_dataset,
        loss_summary=loss_summary, test_elapsed_sec=test_elapsed_sec, test_ms_per_sample=test_ms_per_sample,
    )
