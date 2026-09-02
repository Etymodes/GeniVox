from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from genivox.core.models import TrainingMetric


@dataclass(frozen=True, slots=True)
class MetricLineError:
    line_number: int
    message: str
    raw_line: str


@dataclass(frozen=True, slots=True)
class MetricsReadResult:
    metrics: tuple[TrainingMetric, ...]
    errors: tuple[MetricLineError, ...]


class MetricParseError(ValueError):
    pass


_STEP_KEYS = ("step", "global_step", "iteration", "iter")
_TIMESTAMP_KEYS = ("timestamp", "time", "wall_time")


def parse_metric_line(line: str) -> TrainingMetric:
    """Parse one JSON object into the UI-neutral TrainingMetric model."""

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise MetricParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MetricParseError("metric line must be a JSON object")

    step_value = _first_present(payload, _STEP_KEYS)
    if isinstance(step_value, bool) or not isinstance(step_value, (int, float)):
        raise MetricParseError("metric line requires a numeric step/global_step/iteration")
    if not math.isfinite(float(step_value)) or int(step_value) != step_value:
        raise MetricParseError("step must be a finite integer")

    timestamp_value = _first_present(payload, _TIMESTAMP_KEYS)
    timestamp: float | None = None
    if timestamp_value is not None:
        if isinstance(timestamp_value, bool) or not isinstance(timestamp_value, (int, float)):
            raise MetricParseError("timestamp must be numeric when present")
        timestamp = float(timestamp_value)
        if not math.isfinite(timestamp):
            raise MetricParseError("timestamp must be finite")

    nested_values = payload.get("metrics")
    if nested_values is not None:
        if not isinstance(nested_values, dict):
            raise MetricParseError("metrics must be an object")
        candidate_values: Mapping[str, Any] = nested_values
    else:
        ignored = {*_STEP_KEYS, *_TIMESTAMP_KEYS, "metrics"}
        candidate_values = {key: value for key, value in payload.items() if key not in ignored}

    values: dict[str, float] = {}
    for name, value in candidate_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            values[str(name)] = numeric_value
    if not values:
        raise MetricParseError("metric line contains no finite numeric metric values")

    return TrainingMetric(step=int(step_value), values=values, timestamp=timestamp)


def read_metrics_jsonl(path: str | Path, *, strict: bool = False) -> MetricsReadResult:
    metrics_path = Path(path).expanduser().resolve()
    metrics: list[TrainingMetric] = []
    errors: list[MetricLineError] = []
    with metrics_path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                metrics.append(parse_metric_line(raw_line))
            except MetricParseError as exc:
                if strict:
                    raise MetricParseError(f"{metrics_path}:{line_number}: {exc}") from exc
                errors.append(MetricLineError(line_number, str(exc), raw_line.rstrip("\r\n")))
    return MetricsReadResult(tuple(metrics), tuple(errors))


class JsonlMetricTail:
    """Incrementally read complete JSONL metric records from a growing file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._offset = 0
        self._line_number = 0
        self._file_identity: tuple[int, int] | None = None

    @property
    def offset(self) -> int:
        return self._offset

    def reset(self) -> None:
        self._offset = 0
        self._line_number = 0
        self._file_identity = None

    def poll(self) -> MetricsReadResult:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return MetricsReadResult((), ())

        identity = (stat.st_dev, stat.st_ino)
        if self._file_identity is None:
            self._file_identity = identity
        elif identity != self._file_identity or stat.st_size < self._offset:
            self._offset = 0
            self._line_number = 0
            self._file_identity = identity

        metrics: list[TrainingMetric] = []
        errors: list[MetricLineError] = []
        with self.path.open("rb") as stream:
            stream.seek(self._offset)
            while True:
                line_start = stream.tell()
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.endswith((b"\n", b"\r")):
                    stream.seek(line_start)
                    break
                self._offset = stream.tell()
                self._line_number += 1
                try:
                    line = raw_line.decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    errors.append(MetricLineError(self._line_number, f"invalid UTF-8: {exc}", ""))
                    continue
                if not line.strip():
                    continue
                try:
                    metrics.append(parse_metric_line(line))
                except MetricParseError as exc:
                    errors.append(MetricLineError(self._line_number, str(exc), line.rstrip("\r\n")))
        return MetricsReadResult(tuple(metrics), tuple(errors))


def _first_present(payload: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None
