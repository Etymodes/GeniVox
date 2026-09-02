from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genivox.core.paths import ensure_private_directory, protect_private_file


@dataclass(slots=True)
class HumanRating:
    naturalness: int | None = None
    speaker_similarity: int | None = None
    emotion_match: int | None = None
    pronunciation: int | None = None
    notes: str = ""

    def validate(self) -> None:
        for name in ("naturalness", "speaker_similarity", "emotion_match", "pronunciation"):
            value = getattr(self, name)
            if value is not None and not 1 <= value <= 5:
                raise ValueError(f"{name} must be between 1 and 5")


@dataclass(slots=True)
class ExperimentRecord:
    engine_id: str
    text: str
    audio_path: str
    parameters: dict[str, Any]
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    language_segments: list[dict[str, Any]] = field(default_factory=list)
    objective_metrics: dict[str, float] = field(default_factory=dict)
    rating: HumanRating = field(default_factory=HumanRating)
    provenance: dict[str, Any] = field(default_factory=dict)
    preference: str = "未评价"

    def to_dict(self) -> dict[str, Any]:
        self.rating.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        payload = dict(data)
        payload["rating"] = HumanRating(**payload.get("rating", {}))
        return cls(**payload)


class ExperimentStore:
    """Thread-safe JSONL ledger for reproducible comparisons and human ratings."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def append(self, record: ExperimentRecord) -> None:
        with self._lock:
            ensure_private_directory(self.path.parent)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            protect_private_file(self.path)

    def read_all(self) -> list[ExperimentRecord]:
        with self._lock:
            return self._read_all_unlocked()

    def _read_all_unlocked(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []
        records: list[ExperimentRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(ExperimentRecord.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid experiment record at line {line_number}: {exc}") from exc
        return records

    def update_rating(self, record_id: str, rating: HumanRating) -> None:
        rating.validate()
        with self._lock:
            records = self._read_all_unlocked()
            found = False
            for record in records:
                if record.id == record_id:
                    record.rating = rating
                    found = True
                    break
            if not found:
                raise KeyError(record_id)
            self._rewrite_unlocked(records)

    def update_preference(self, record_id: str, preference: str, notes: str = "") -> None:
        with self._lock:
            records = self._read_all_unlocked()
            found = False
            for record in records:
                if record.id == record_id:
                    record.preference = preference
                    record.rating.notes = notes
                    found = True
                    break
            if not found:
                raise KeyError(record_id)
            self._rewrite_unlocked(records)

    def _rewrite_unlocked(self, records: list[ExperimentRecord]) -> None:
        ensure_private_directory(self.path.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            record.to_dict(), ensure_ascii=False, separators=(",", ":")
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(self.path)
            protect_private_file(self.path)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise
