"""Calendar validation and window geometry shared by all dataset splits."""

from dataclasses import dataclass
from datetime import time as Time

import numpy as np
import pandas as pd


@dataclass(frozen=True, order=True)
class TimePoint:
    day_index: int
    time: Time

    def __str__(self):
        return f"day_index={self.day_index}, time={self.time.isoformat()}"


def parse_time(value):
    try:
        parsed = value if isinstance(value, Time) else Time.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid time value {value!r}; expected HH:MM:SS.") from exc
    if parsed.tzinfo is not None:
        raise ValueError("time values must not include a timezone.")
    if parsed.microsecond:
        raise ValueError("time values must use whole-second precision.")
    return parsed


def make_time_point(day_index, time_value):
    if isinstance(day_index, bool) or not isinstance(day_index, (int, np.integer)):
        raise ValueError("day_index split boundaries must be integers.")
    return TimePoint(int(day_index), parse_time(time_value))


def time_offset(point):
    return ((point.day_index * 24 + point.time.hour) * 60 + point.time.minute) * 60 + point.time.second


def validate_timeline(frame, points, freq):
    if not points:
        raise ValueError("Dataset contains no time points.")
    if points[0].day_index != 0:
        raise ValueError("day_index must start at 0.")
    interval = int(pd.Timedelta(freq).total_seconds())
    if interval <= 0:
        raise ValueError("freq must be positive.")
    offsets = np.fromiter(map(time_offset, points), dtype=np.int64, count=len(points))
    if (np.diff(offsets) != interval).any():
        raise ValueError(f"day_index/time is not continuous at frequency {freq}.")
    counts = frame.groupby(["day_index", "time"], sort=False)["area_id"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Inconsistent area count.")


def window_starts(*, total, history, horizon, stride, origin, validation, test, split):
    """Intersect a split's target interval with the original sampling lattice."""
    last = total - history - horizon
    if last < 0:
        raise ValueError("Time length too short.")
    # No candidate windows means the original sampler never inspected the split.
    if origin > last:
        return []
    boundaries = {"train": (origin, validation), "val": (validation, test), "test": (test, total)}
    if split not in boundaries:
        raise ValueError(f"Unknown split: {split}")
    lower, upper = boundaries[split]
    first = max(origin, lower - history)
    first = origin + ((first - origin + stride - 1) // stride) * stride
    last = min(last, upper - history - horizon)
    return list(range(first, last + 1, stride))
