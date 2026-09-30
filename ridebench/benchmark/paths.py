"""Stable output layout for both benchmark variants."""

from dataclasses import dataclass
from pathlib import Path

REGULAR_LONG_RUN_RESULTS_DIR = "run_results/regular_long_forecast"
ULTRA_LONG_RUN_RESULTS_DIR = "run_results/ultra_long_forecast"


@dataclass(frozen=True)
class RunPaths:
    root_dir: Path
    run_name: str
    algorithm_dirname: str
    score_dir: Path
    checkpoint_dir: Path
    plot_dir: Path
    score_json_path: Path
    pred_path: Path
    true_path: Path
    losses_path: Path
    checkpoint_path: Path
    plot_prefix: Path


def _learning_rate_tag(learning_rate):
    if learning_rate is None:
        return ""
    coefficient, exponent = f"{learning_rate:.12e}".split("e")
    return f"_lr_{coefficient.rstrip('0').rstrip('.')}e{int(exponent)}"


def build_run_paths(run_results_dir, algorithm, run_name, learning_rate=None, benchmark_variant="regular"):
    destination = run_results_dir
    if benchmark_variant == "ultra_long" and destination == REGULAR_LONG_RUN_RESULTS_DIR:
        destination = ULTRA_LONG_RUN_RESULTS_DIR
    root = Path(destination)
    name = run_name + _learning_rate_tag(learning_rate)
    folders = {kind: root / kind / algorithm for kind in ("scores", "checkpoints", "plots")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "score_json_path": ("scores", "_scores.json"), "pred_path": ("scores", "_pred.npy"),
        "true_path": ("scores", "_true.npy"), "losses_path": ("scores", "_losses.npz"),
        "checkpoint_path": ("checkpoints", "_model.pth"), "plot_prefix": ("plots", ""),
    }
    return RunPaths(
        root_dir=root, run_name=name, algorithm_dirname=algorithm,
        score_dir=folders["scores"], checkpoint_dir=folders["checkpoints"], plot_dir=folders["plots"],
        **{key: folders[kind] / (name + suffix) for key, (kind, suffix) in artifacts.items()},
    )
