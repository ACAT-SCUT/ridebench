"""CLI entry point and compatibility exports for argument helpers."""

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)

from ridebench import algorithms
from ridebench.benchmark import Benchmark
from ridebench.cli import parse_args, validate_args


def algorithm_configs():
    registry = {config.name(): config for config in algorithms.configs}
    if len(registry) != len(algorithms.configs):
        raise RuntimeError("Duplicate names found.")
    return registry


def main():
    registry = algorithm_configs()
    options = parse_args(registry)
    validate_args(options)
    Benchmark(options, registry[options.algorithm]).run()


if __name__ == "__main__":
    main()
