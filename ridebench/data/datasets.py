"""Area-wise views of a single preprocessed time-series panel.

Targets determine split membership. Validation and test windows may borrow
history from earlier splits; scalers always use the training time interval.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preparation import PreparedDataset, prepare_table
from .timeline import (
    TimePoint, make_time_point, parse_time, time_offset, validate_timeline, window_starts,
)


class BenchmarkDatasetFactory:
    """Lazily prepare shared arrays, then cache each split's window view."""

    def __init__(self, args):
        self.args = args
        self._endo_list = args.endo_vars
        self._exo_con_list = args.exo_con_vars
        self._effective_exo_dis_dict = dict(args.exo_dis_dict)
        self._prepared_data = None
        self._all_area_ids = None
        self._datasets = {}

    _build_effective_exo_dis_dict = staticmethod(dict)

    def _custom_area_split_enabled(self):
        return bool(getattr(self.args, "enable_custom_area_split", False))

    def _effective_exo_dis_dict_for_run(self):
        return {
            name: size for name, size in self._effective_exo_dis_dict.items()
            if name != "area_id" or not self._custom_area_split_enabled()
        }

    def _selected_area_ids_for_split(self, split):
        if not self._custom_area_split_enabled():
            return None
        fields = {"train": "train_val_area_ids", "val": "train_val_area_ids", "test": "test_area_ids"}
        if split not in fields:
            raise ValueError(f"Unknown split: {split}")
        return list(getattr(self.args, fields[split]))

    def _validate_area_selection(self, all_area_ids):
        if self._custom_area_split_enabled():
            self._all_area_ids = list(all_area_ids)
            for field in ("train_val_area_ids", "test_area_ids"):
                missing = [area for area in getattr(self.args, field) if area not in set(all_area_ids)]
                if missing:
                    raise ValueError(f"{field} contains unknown area_ids: {missing}")

    def _prepared(self):
        if self._prepared_data is None:
            boundaries = {
                f"{split}_start_{part}": getattr(self.args, f"{split}_start_{part}")
                for split in ("train", "val", "test") for part in ("day_index", "time")
            }
            self._prepared_data = MultiAreaTSDataset.prepare_data(
                dataset_path=self.args.dataset_path, **boundaries,
                endo_vars=self._endo_list, exo_con_vars=self._exo_con_list,
                exo_dis_vars=list(self.exo_dis_dict()),
            )
            self._validate_area_selection(self._prepared_data.areas)
        return self._prepared_data

    def _build_dataset(self, split):
        if split not in self._datasets:
            self._datasets[split] = MultiAreaTSDataset(
                split=split, prepared_data=self._prepared(),
                input_len=self.input_len(), output_len=self.output_len(),
                data_stride=self.args.data_stride,
                selected_area_ids=self._selected_area_ids_for_split(split),
            )
        return self._datasets[split]

    def train(self):
        return self._build_dataset("train")

    def val(self):
        return self._build_dataset("val")

    def test(self):
        return self._build_dataset("test")

    def data_summary(self):
        panel = self._prepared()
        fields = ("dataset_start", "dataset_end", "requested_train_start", "val_start", "test_start", "csv_engine")
        return {
            **{name: getattr(panel, name) for name in fields},
            "actual_train_start": panel.train_start,
            "num_time_points": len(panel.time_points), "num_areas": len(panel.areas),
        }

    def load_warnings(self):
        return list(self._prepared().warnings)

    def input_len(self):
        return self.args.input_len

    def output_len(self):
        return self.args.output_len

    def x_dim(self):
        return len(self._endo_list)

    def exo_con_list(self):
        return self._exo_con_list

    def exo_dis_dict(self):
        return self._effective_exo_dis_dict_for_run()

    def custom_area_split_enabled(self):
        return self._custom_area_split_enabled()

    def all_area_ids(self):
        return list(self._prepared().areas)


class MultiAreaTSDataset(Dataset):
    SCENARIO_FEATURES = [
        "weather_factor", "public_holiday", "traditional_festival",
        "western_festival", "large_scale_event_1", "large_scale_event_2",
    ]
    # (batch key, panel attribute, uses forecast interval, tensor dtype)
    _SAMPLE_FIELDS = (
        ("x_endo", "endo", False, torch.float32),
        ("x_exo_con", "exo_con", False, torch.float32),
        ("x_exo_dis", "exo_dis", False, torch.long),
        ("y", "endo", True, torch.float32),
        ("y_exo_con", "exo_con", True, torch.float32),
        ("y_exo_dis", "exo_dis", True, torch.long),
        ("y_scenario_features", "scenario_features", True, torch.float32),
    )

    def __init__(self, split="train", input_len=48 * 7 * 4, output_len=48 * 7,
                 data_stride=48, prepared_data=None, dataset_path=None,
                 train_start_day_index=0, train_start_time="00:00:00",
                 val_start_day_index=1095, val_start_time="00:00:00",
                 test_start_day_index=1277, test_start_time="00:00:00",
                 endo_vars=None, exo_con_vars=None, exo_dis_vars=None,
                 selected_area_ids=None, freq="30min"):
        super().__init__()
        self.input_len, self.output_len, self.data_stride = input_len, output_len, data_stride
        if data_stride <= 0:
            raise ValueError("data_stride must be positive.")
        self.prepared_data = prepared_data or self.prepare_data(
            dataset_path, train_start_day_index, train_start_time,
            val_start_day_index, val_start_time, test_start_day_index,
            test_start_time, endo_vars, exo_con_vars, exo_dis_vars, freq,
        )
        self._attach_panel()
        self.area_indices = self._resolve_area_indices(selected_area_ids)
        self.selected_area_ids = [self._all_area_ids[index] for index in self.area_indices]
        self.A = len(self.area_indices)
        self.window_start_indices = self._build_window_start_indices(split)
        if not self.window_start_indices:
            raise ValueError(f"No available windows for split '{split}'.")
        self.num_windows = len(self.window_start_indices)
        self.split = split
        self._split_summary = self._build_split_summary()

    def _attach_panel(self):
        for name in (
            "time_points", "areas", "train_start", "val_start", "test_start",
            "train_start_idx", "val_start_idx", "test_start_idx", "endo",
            "exo_con", "exo_dis", "scenario_features", "scaler_endo", "scaler_exo_con",
        ):
            setattr(self, name, getattr(self.prepared_data, name))
        for name in ("endo_vars", "exo_con_vars", "exo_dis_vars", "scenario_feature_vars"):
            setattr(self, name, list(getattr(self.prepared_data, name)))
        self.T = len(self.time_points)
        self._all_area_ids = list(self.areas)
        self._scenario_feature_idx = dict(zip(self.scenario_feature_vars, range(len(self.scenario_feature_vars))))

    @classmethod
    def prepare_data(cls, dataset_path, train_start_day_index, train_start_time,
                     val_start_day_index, val_start_time, test_start_day_index,
                     test_start_time, endo_vars, exo_con_vars, exo_dis_vars, freq="30min"):
        return prepare_table(
            dataset_path=dataset_path, freq=freq, scenario_vars=cls.SCENARIO_FEATURES,
            train_start_day_index=train_start_day_index, train_start_time=train_start_time,
            val_start_day_index=val_start_day_index, val_start_time=val_start_time,
            test_start_day_index=test_start_day_index, test_start_time=test_start_time,
            endo_vars=endo_vars, exo_con_vars=exo_con_vars, exo_dis_vars=exo_dis_vars,
        )

    def _build_window_start_indices(self, split):
        return window_starts(
            total=self.T, history=self.input_len, horizon=self.output_len,
            stride=self.data_stride, origin=self.train_start_idx,
            validation=self.val_start_idx, test=self.test_start_idx, split=split,
        )

    def _build_split_summary(self):
        summary = dict(split=self.split, num_windows=self.num_windows,
                       num_samples=len(self), num_areas=self.A, area_ids=list(self.selected_area_ids))
        for label, start in (("first", self.window_start_indices[0]), ("last", self.window_start_indices[-1])):
            target = start + self.input_len
            for key, index in (("input_start", start), ("input_end", target - 1),
                               ("target_start", target), ("target_end", target + self.output_len - 1)):
                summary[f"{label}_{key}"] = self.time_points[index]
        return summary

    def split_summary(self):
        return self._split_summary.copy()

    def _scenario_feature(self, scenario_features, name):
        return scenario_features[..., self._scenario_feature_idx[name]]

    def _any_event(self, features, names):
        masks = [self._scenario_feature(features, name) != 0 for name in names]
        return np.logical_or.reduce(masks).astype(np.int64)

    def weather_events(self, scenario_features):
        return (self._scenario_feature(scenario_features, "weather_factor") > 0).astype(np.int64)

    def holiday_events(self, scenario_features):
        return self._any_event(scenario_features, ("public_holiday", "traditional_festival", "western_festival"))

    def large_scale_events(self, scenario_features):
        return self._any_event(scenario_features, ("large_scale_event_1", "large_scale_event_2"))

    def inverse_transform_endo(self, data):
        tensor_input = isinstance(data, torch.Tensor)
        array = data.detach().cpu().numpy() if tensor_input else data
        restored = self.scaler_endo.inverse_transform(array.reshape(-1, array.shape[-1])).reshape(array.shape)
        return torch.tensor(restored) if tensor_input else restored

    _parse_time = staticmethod(parse_time)
    _make_time_point = staticmethod(make_time_point)
    _time_point_offset = staticmethod(time_offset)
    _sanity_check = staticmethod(validate_timeline)

    def _resolve_area_indices(self, selected_area_ids):
        lookup = {area: index for index, area in enumerate(self._all_area_ids)}
        return [lookup[area] for area in selected_area_ids] if selected_area_ids else list(lookup.values())

    def __len__(self):
        return self.num_windows * self.A

    def __getitem__(self, idx):
        window, position = divmod(idx, self.A)
        area = self.area_indices[position]
        start = self.window_start_indices[window]
        boundary = start + self.input_len
        intervals = (slice(start, boundary), slice(boundary, boundary + self.output_len))
        return {
            key: torch.tensor(getattr(self, source)[intervals[forecast], area], dtype=dtype)
            for key, source, forecast, dtype in self._SAMPLE_FIELDS
        }
