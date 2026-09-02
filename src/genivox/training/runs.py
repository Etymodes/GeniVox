from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from genivox.core.paths import ensure_private_directory, protect_private_file


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
_ALLOWED_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: _TERMINAL_STATUSES,
    RunStatus.SUCCEEDED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SECRET_KEY_PARTS = ("token", "secret", "password", "credential", "api_key", "apikey")


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    status: RunStatus
    created_at: str
    log_path: Path
    metrics_path: Path
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": _redact_command(self.command),
            "cwd": str(self.cwd),
            "environment": _redact_environment(self.environment),
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "log_path": str(self.log_path),
            "metrics_path": str(self.metrics_path),
            "metadata": _redact_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        return cls(
            run_id=str(payload["run_id"]),
            command=tuple(str(part) for part in payload["command"]),
            cwd=Path(payload["cwd"]),
            environment={str(key): str(value) for key, value in payload.get("environment", {}).items()},
            status=RunStatus(payload["status"]),
            created_at=str(payload["created_at"]),
            started_at=_optional_string(payload.get("started_at")),
            finished_at=_optional_string(payload.get("finished_at")),
            exit_code=_optional_int(payload.get("exit_code")),
            error=_optional_string(payload.get("error")),
            log_path=Path(payload["log_path"]),
            metrics_path=Path(payload["metrics_path"]),
            metadata=dict(payload.get("metadata", {})),
        )


class RunStore:
    """Persist one JSON manifest plus log/metric paths for each local training run."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        ensure_private_directory(self.root)
        self._lock = threading.RLock()

    def create(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        environment: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        run_id: str | None = None,
    ) -> RunManifest:
        selected_run_id = run_id or _new_run_id()
        _validate_run_id(selected_run_id)
        parts = _validate_command(command)
        workdir = Path(cwd).expanduser().resolve()
        if not workdir.is_dir():
            raise NotADirectoryError(workdir)
        env = {str(key): str(value) for key, value in (environment or {}).items()}
        run_metadata = dict(metadata or {})
        json.dumps(run_metadata)

        run_dir = self.run_dir(selected_run_id)
        with self._lock:
            run_dir.mkdir(parents=True, exist_ok=False)
            ensure_private_directory(run_dir)
            manifest = RunManifest(
                run_id=selected_run_id,
                command=parts,
                cwd=workdir,
                environment=env,
                status=RunStatus.CREATED,
                created_at=_now(),
                log_path=run_dir / "training.log",
                metrics_path=run_dir / "metrics.jsonl",
                metadata=run_metadata,
            )
            self.save(manifest)
        return manifest

    def run_dir(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self.root / run_id

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def save(self, manifest: RunManifest) -> None:
        _validate_run_id(manifest.run_id)
        payload = manifest.to_dict()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        run_dir = self.run_dir(manifest.run_id)
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        temporary = run_dir / f".manifest-{uuid.uuid4().hex}.tmp"
        with self._lock:
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(serialized)
                    stream.flush()
                    os.fsync(stream.fileno())
                destination = self.manifest_path(manifest.run_id)
                os.replace(temporary, destination)
                protect_private_file(destination)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, run_id: str) -> RunManifest:
        path = self.manifest_path(run_id)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Run manifest is not a JSON object: {path}")
        manifest = RunManifest.from_dict(payload)
        if manifest.run_id != run_id:
            raise ValueError(f"Run manifest id {manifest.run_id!r} does not match directory {run_id!r}")
        return manifest

    def list_runs(self) -> list[RunManifest]:
        manifests = [
            self.load(path.parent.name) for path in self.root.glob("*/manifest.json") if path.parent.is_dir()
        ]
        return sorted(manifests, key=lambda manifest: manifest.created_at, reverse=True)

    def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        exit_code: int | None = None,
        error: str | None = None,
    ) -> RunManifest:
        with self._lock:
            manifest = self.load(run_id)
            if status is manifest.status:
                return manifest
            if status not in _ALLOWED_TRANSITIONS[manifest.status]:
                raise ValueError(f"Invalid run status transition: {manifest.status.value} -> {status.value}")

            now = _now()
            updated = replace(
                manifest,
                status=status,
                started_at=now if status is RunStatus.RUNNING else manifest.started_at,
                finished_at=now if status in _TERMINAL_STATUSES else manifest.finished_at,
                exit_code=exit_code,
                error=error,
            )
            self.save(updated)
            return updated


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes)):
        raise TypeError("command must be a sequence of arguments, not a shell command string")
    parts = tuple(command)
    if not parts:
        raise ValueError("command must contain an executable")
    if any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("every command argument must be a non-empty string")
    return parts


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-', up to 128 characters")


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in environment.items():
        normalized = key.casefold()
        redacted[key] = "***REDACTED***" if any(part in normalized for part in _SECRET_KEY_PARTS) else value
    return redacted


def _redact_command(command: Sequence[str]) -> list[str]:
    """Redact conventional secret-valued CLI flags in persisted provenance."""

    redacted: list[str] = []
    redact_next = False
    for part in command:
        if redact_next:
            redacted.append("***REDACTED***")
            redact_next = False
            continue
        key, separator, _ = part.partition("=")
        normalized_key = key.lstrip("-/").casefold().replace("-", "_")
        if separator and any(fragment in normalized_key for fragment in _SECRET_KEY_PARTS):
            redacted.append(f"{key}=***REDACTED***")
            continue
        redacted.append(part)
        normalized = part.lstrip("-/").casefold().replace("-", "_")
        redact_next = any(fragment in normalized for fragment in _SECRET_KEY_PARTS)
    return redacted


def _redact_value(value: Any, *, key: str = "") -> Any:
    if any(part in key.casefold() for part in _SECRET_KEY_PARTS):
        return "***REDACTED***"
    if isinstance(value, Mapping):
        return {str(item_key): _redact_value(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
