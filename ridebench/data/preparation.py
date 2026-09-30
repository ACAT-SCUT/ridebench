"""Load one shared panel and fit scalers on its training interval."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .timeline import TimePoint, make_time_point, parse_time, time_offset, validate_timeline


@dataclass(frozen=True)
class PreparedDataset:
    time_points: tuple[TimePoint, ...]
    areas: list[int]
    endo: np.ndarray
    exo_con: np.ndarray
    exo_dis: np.ndarray
    scenario_features: np.ndarray
    scaler_endo: StandardScaler
    scaler_exo_con: StandardScaler
    dataset_start: TimePoint
    dataset_end: TimePoint
    requested_train_start: TimePoint
    train_start: TimePoint
    val_start: TimePoint
    test_start: TimePoint
    csv_engine: str
    train_start_idx: int
    val_start_idx: int
    test_start_idx: int
    warnings: tuple[str, ...]
    endo_vars: tuple[str, ...]
    exo_con_vars: tuple[str, ...]
    exo_dis_vars: tuple[str, ...]
    scenario_feature_vars: tuple[str, ...]


def _read_panel(path, freq, scenarios):
    if not path:
        raise ValueError("dataset_path must be provided when prepared_data is not supplied.")
    table = pd.read_csv(path, engine="pyarrow")
    time_columns = ["day_index", "time"]
    missing = [key for key in time_columns if key not in table]
    if missing:
        raise ValueError(f"Missing required time columns: {missing}")
    if not pd.api.types.is_integer_dtype(table["day_index"]):
        raise ValueError("day_index must contain integer values.")
    if table[time_columns].isna().any().any():
        raise ValueError("day_index and time must not contain missing values.")
    table = table.sort_values([*time_columns, "area_id"]).reset_index(drop=True)
    timeline = tuple(
        TimePoint(int(day), parse_time(clock))
        for day, clock in table[time_columns].drop_duplicates().itertuples(index=False, name=None)
    )
    validate_timeline(table, timeline, freq)
    missing = [key for key in scenarios if key not in table]
    if missing:
        raise ValueError(f"Missing required scenario feature columns for benchmark evaluation: {missing}")
    return table, timeline


def _split_boundaries(points, requested, validation, test):
    start, end = points[0], points[-1]
    train = max(start, requested)
    notices = () if requested >= start else (
        f"Configured train start {requested} is earlier than dataset start {start}; using dataset start instead.",
    )
    for invalid, message in (
        (train > end, "The train start must not be later than the dataset end."),
        (not train < validation < test, "Split boundaries must satisfy train start < validation start < test start."),
        (validation > end, "The validation start must not be later than the dataset end."),
        (test >= end, "The test start must be earlier than the dataset end."),
    ):
        if invalid:
            raise ValueError(message)
    offsets = np.fromiter(map(time_offset, points), dtype=np.int64, count=len(points))
    indices = tuple(int(offsets.searchsorted(time_offset(p))) for p in (train, validation, test))
    for left, right, message in (
        (*indices[:2], "No training range remains before the validation start."),
        (*indices[1:], "No validation range remains before the test start."),
    ):
        if left >= right:
            raise ValueError(message)
    return train, indices, notices


def _standardize(panel, training_slice):
    scaler = StandardScaler()
    width = panel.shape[-1]
    if width:
        scaler.fit(panel[training_slice].reshape(-1, width))
        panel = scaler.transform(panel.reshape(-1, width)).reshape(panel.shape)
    return panel, scaler


def prepare_table(*, dataset_path, train_start_day_index, train_start_time,
                  val_start_day_index, val_start_time, test_start_day_index,
                  test_start_time, endo_vars, exo_con_vars, exo_dis_vars,
                  scenario_vars, freq="30min"):
    endogenous = tuple(endo_vars or ["endo_1", "endo_2", "endo_3"])
    continuous = tuple(["weather_factor"] if exo_con_vars is None else exo_con_vars)
    discrete = tuple(exo_dis_vars or ())
    table, points = _read_panel(dataset_path, freq, scenario_vars)
    requested, validation, test = (
        make_time_point(day, clock) for day, clock in (
            (train_start_day_index, train_start_time),
            (val_start_day_index, val_start_time),
            (test_start_day_index, test_start_time),
        )
    )
    train, indices, notices = _split_boundaries(points, requested, validation, test)
    areas = sorted(map(int, table["area_id"].unique()))
    shape = (len(points), len(areas))

    def panel(columns, dtype=None):
        if not columns:
            return np.empty((*shape, 0), dtype=dtype or np.float64)
        values = table[list(columns)].to_numpy()
        if dtype is not None:
            values = values.astype(dtype)
        return values.reshape(*shape, -1)

    training_slice = slice(indices[0], indices[1])
    endo, endo_scaler = _standardize(panel(endogenous), training_slice)
    exo_con, con_scaler = _standardize(panel(continuous), training_slice)
    return PreparedDataset(
        time_points=points, areas=areas, endo=endo, exo_con=exo_con,
        exo_dis=panel(discrete, np.int64), scenario_features=panel(scenario_vars, np.float64),
        scaler_endo=endo_scaler, scaler_exo_con=con_scaler,
        dataset_start=points[0], dataset_end=points[-1], requested_train_start=requested,
        train_start=train, val_start=validation, test_start=test, csv_engine="pyarrow",
        train_start_idx=indices[0], val_start_idx=indices[1], test_start_idx=indices[2],
        warnings=notices, endo_vars=endogenous, exo_con_vars=continuous,
        exo_dis_vars=discrete, scenario_feature_vars=tuple(scenario_vars),
    )
