"""Command-line schema for datasets and benchmark execution."""

import json
from argparse import ArgumentParser
from copy import deepcopy
from datetime import time as Time


def parse_int_list(arg):
    return [int(item) for item in (arg or "").split(",") if item.strip()]


def parse_time(arg):
    try:
        value = Time.fromisoformat(arg)
    except ValueError as exc:
        raise ValueError(f"Invalid time {arg!r}; expected HH:MM:SS.") from exc
    if value.tzinfo is not None:
        raise ValueError("Split times must not include a timezone.")
    if value.microsecond:
        raise ValueError("Split times must use whole-second precision.")
    return value.isoformat()


def _variable_names(value):
    return value.split(",") if value else []


SHARED_OPTIONS = (
    ('--model_name', dict(type=str, help='run/model name shown in outputs; defaults to algorithm name')),
    ('--run_results_dir', dict(metavar='DIR', type=str, default='run_results/regular_long_forecast', help='root directory for benchmark outputs')),
    ('--benchmark_summary_csv_path', dict(metavar='PATH', type=str, default='run_results/summary_csv/regular_long_forecast.csv', help='benchmark summary CSV path; relative paths are resolved from the current working directory')),
    ('--benchmark_variant', dict(type=str, choices=['regular', 'ultra_long'], default='regular', help='benchmark experiment variant; ultra_long switches to the extra 14-week input, 8-week forecast experiment schema')),
    ('--dtype', dict(choices=['bfloat16', 'float16', 'float32', 'float64'], default='float32', help='tensor dtype used by the benchmark; numpy bfloat16 falls back to float16')),
    ('--seed', dict(type=int, default=42, help='the random seed for reproducibility')),
    ('--log_level', dict(choices=['debug', 'info', 'warning', 'error', 'critical'], default='info', help='logging level')),
    ('--save_data', dict(action='store_true', help='whether to save pred and true data to disk')),
)

TRAINING_OPTIONS = (
    ('--epochs', dict(type=int, default=30, help='total training epochs')),
    ('--patience', dict(type=int, default=3, help='early stopping patience')),
    ('--test_with_checkpoint', dict(action='store_true', help='skip training and test by loading the saved trained checkpoint for the current run name')),
    ('--device', dict(choices=['cpu', 'cuda'], default='cuda', help='device placement for training and evaluation')),
    ('--log_batch_interval', dict(type=int, default=200, help='log intra-epoch train/validation progress every N batches; set 0 to disable')),
    ('--optimizer', dict(choices=['adam', 'adamw'], default='adamw', help='model optimizer')),
    ('--learning_rate', dict(type=float, default=0.001, help='optimizer learning rate')),
    ('--lr_scheduler', dict(choices=['none', 'cosine'], default='cosine', help='model learning rate scheduler')),
    ('--train_loss', dict(choices=['smooth_l1', 'mse', 'mae', 'fredf', 'dbloss'], default='mae', help='method to compute model training loss')),
    ('--val_loss', dict(choices=['smooth_l1', 'mse', 'mae', 'fredf', 'dbloss'], default='mae', help='method to compute model validation loss')),
    ('--save_losses', dict(action='store_true', help='whether to save train, validation and test losses to disk')),
)

DATA_OPTIONS = (
    ('--dataset_path', dict(metavar='PATH', type=str, default='datasets/ride_hailing.csv', help='dataset file path')),
    ('--enable_custom_area_split', dict(action='store_true', help='enable custom train/val vs test area split; disables area_id discrete feature for this run')),
    ('--train_val_area_ids', dict(metavar='IDS', type=parse_int_list, default=None, help='comma-separated area_id list used for both train and val when enable_custom_area_split is enabled')),
    ('--test_area_ids', dict(metavar='IDS', type=parse_int_list, default=None, help='comma-separated area_id list used for test when enable_custom_area_split is enabled')),
    ('--input_len', dict(metavar='LEN', type=int, default=48 * 28, help='length of input data')),
    ('--output_len', dict(metavar='LEN', type=int, default=48 * 7, help='length of output data')),
    ('--data_stride', dict(metavar='LEN', type=int, default=48, help='sliding stride when generating window samples')),
    ('--endo_vars', dict(metavar='VARS', type=_variable_names, default=[f"endo_{index}" for index in range(1, 10)], help='used endogenous variable names')),
    ('--exo_con_vars', dict(metavar='VARS', type=_variable_names, default=["weather_factor"], help='used exogenous continuous variable names')),
    ('--exo_dis_dict', dict(metavar='VARS', type=json.loads, default={"area_id": 200, "half_hour_of_day": 48, "day_of_week": 7, "public_holiday": 10, "traditional_festival": 10, "western_festival": 15, "large_scale_event_1": 2, "large_scale_event_2": 2}, help='used exogenous discrete variable dictionary, key is variable name and value is total number of distinct values')),
    ('--batch_size', dict(metavar='SIZE', type=int, default=128, help='sample batch size')),
    ('--num_workers', dict(metavar='N', type=int, default=4, help='DataLoader worker process count')),
    ('--no_persistent_workers', dict(dest='persistent_workers', action='store_false', default=True, help='disable persistent DataLoader workers')),
    ('--no_pin_memory', dict(dest='pin_memory', action='store_false', default=True, help='disable DataLoader pinned host memory')),
    ('--train_start_day_index', dict(metavar='DAY', type=int, default=0, help='day_index where training input may start')),
    ('--train_start_time', dict(metavar='TIME', type=parse_time, default='00:00:00', help='time where training input may start')),
    ('--val_start_day_index', dict(metavar='DAY', type=int, default=1095, help='day_index where validation targets start')),
    ('--val_start_time', dict(metavar='TIME', type=parse_time, default='00:00:00', help='time where validation targets start')),
    ('--test_start_day_index', dict(metavar='DAY', type=int, default=1277, help='day_index where test targets start')),
    ('--test_start_time', dict(metavar='TIME', type=parse_time, default='00:00:00', help='time where test targets start')),
)


def _register(parser, options):
    for flag, settings in options:
        parser.add_argument(flag, **deepcopy(settings))


def add_shared_benchmark_arguments(parser):
    _register(parser, SHARED_OPTIONS)


def add_training_benchmark_arguments(parser):
    _register(parser, (*SHARED_OPTIONS, *TRAINING_OPTIONS))


def add_inference_benchmark_arguments(parser):
    add_shared_benchmark_arguments(parser)


def add_dataset_arguments(parser):
    _register(parser, DATA_OPTIONS)


def add_model_arguments(parser, configs, train_parents, inference_parents):
    commands = parser.add_subparsers(dest="algorithm", required=True)
    for name, config in configs.items():
        parents = train_parents if config.need_train() else inference_parents
        config.add_arguments(commands.add_parser(name, parents=parents))


def build_parser(configs):
    shared = ArgumentParser(add_help=False)
    add_dataset_arguments(shared)
    parents = []
    for configure in (add_training_benchmark_arguments, add_inference_benchmark_arguments):
        parent = ArgumentParser(add_help=False)
        configure(parent)
        parents.append([parent, shared])
    parser = ArgumentParser(description="TS benchmark framework.")
    add_model_arguments(parser, configs, *parents)
    return parser


def parse_args(configs, argv=None):
    return build_parser(configs).parse_args(argv)


def validate_args(args):
    if not args.endo_vars:
        raise ValueError("endo_vars cannot be empty. At least one endogenous variable must be specified.")
    points = [
        (getattr(args, f"{split}_start_day_index"), parse_time(getattr(args, f"{split}_start_time")))
        for split in ("train", "val", "test")
    ]
    if any(left >= right for left, right in zip(points, points[1:])):
        raise ValueError("Train, validation, and test start points must satisfy train < validation < test.")
    if getattr(args, "enable_custom_area_split", False):
        for field in ("train_val_area_ids", "test_area_ids"):
            if not getattr(args, field):
                raise ValueError(f"--{field} must be provided when --enable_custom_area_split is enabled.")
