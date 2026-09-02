from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from genivox.core.models import EngineManifest, EngineTransport
from genivox.core.paths import ensure_private_directory, protect_private_file

from .base import EngineAdapter, EngineConfigurationError
from .gpt_sovits import GptSovitsV2HttpAdapter
from .mock import MockWavAdapter
from .process import JsonProcessAdapter

_ENGINE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class EngineRegistry:
    """Collection of engine manifests loaded from a versioned JSON document."""

    def __init__(self, manifests: list[EngineManifest] | None = None) -> None:
        self._manifests: dict[str, EngineManifest] = {}
        for manifest in manifests or []:
            self.register(manifest)

    @classmethod
    def load(cls, path: str | Path) -> EngineRegistry:
        registry_path = Path(path).resolve()
        try:
            document = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EngineConfigurationError(f"could not read engine registry {registry_path}: {exc}") from exc
        if not isinstance(document, dict):
            raise EngineConfigurationError("engine registry must be a JSON object")
        if document.get("schema_version") != 1:
            raise EngineConfigurationError("engine registry schema_version must be 1")
        records = document.get("engines")
        if not isinstance(records, list):
            raise EngineConfigurationError("engine registry 'engines' must be a JSON array")

        manifests: list[EngineManifest] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise EngineConfigurationError(f"engine registry item {index} must be an object")
            try:
                resolved = _resolve_manifest_paths(record, registry_path.parent)
                manifests.append(EngineManifest.from_dict(resolved))
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineConfigurationError(f"invalid engine registry item {index}: {exc}") from exc
        return cls(manifests)

    def register(self, manifest: EngineManifest) -> None:
        _validate_engine_id(manifest.id)
        if manifest.id in self._manifests:
            raise EngineConfigurationError(f"duplicate engine id: {manifest.id!r}")
        self._manifests[manifest.id] = manifest

    def upsert(self, manifest: EngineManifest) -> None:
        """Add or explicitly replace a manifest by id."""

        _validate_engine_id(manifest.id)
        self._manifests[manifest.id] = manifest

    def remove(self, engine_id: str) -> EngineManifest:
        try:
            return self._manifests.pop(engine_id)
        except KeyError as exc:
            raise EngineConfigurationError(f"unknown engine id: {engine_id!r}") from exc

    def save(self, path: str | Path) -> None:
        """Atomically persist this registry as a schema-versioned JSON file."""

        registry_path = Path(path).expanduser().resolve()
        ensure_private_directory(registry_path.parent)
        payload = {
            "schema_version": 1,
            "engines": [manifest.to_dict() for manifest in self._manifests.values()],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{registry_path.name}.", suffix=".tmp", dir=registry_path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, registry_path)
            protect_private_file(registry_path)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    def __iter__(self) -> Iterator[EngineManifest]:
        return iter(self._manifests.values())

    def list_manifests(self) -> tuple[EngineManifest, ...]:
        return tuple(self._manifests.values())

    def get_manifest(self, engine_id: str) -> EngineManifest:
        try:
            return self._manifests[engine_id]
        except KeyError as exc:
            raise EngineConfigurationError(f"unknown engine id: {engine_id!r}") from exc

    def create_adapter(self, engine_id: str) -> EngineAdapter:
        manifest = self.get_manifest(engine_id)
        if manifest.transport is EngineTransport.MOCK:
            return MockWavAdapter(manifest)
        if manifest.transport is EngineTransport.PROCESS:
            return JsonProcessAdapter(manifest)
        if manifest.transport is EngineTransport.HTTP:
            driver = manifest.metadata.get("adapter")
            if driver == "gpt_sovits_v2":
                return GptSovitsV2HttpAdapter(manifest)
            raise EngineConfigurationError(
                f"HTTP engine {engine_id!r} needs metadata.adapter='gpt_sovits_v2'"
            )
        raise EngineConfigurationError(f"unsupported transport: {manifest.transport}")


def _resolve_manifest_paths(record: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    result = dict(record)
    for field_name in ("root", "python", "checkpoint_dir"):
        value = result.get(field_name)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            result[field_name] = str((base_dir / candidate).resolve())
    return result


def _validate_engine_id(engine_id: str) -> None:
    if not _ENGINE_ID_PATTERN.fullmatch(engine_id) or engine_id.endswith("."):
        raise EngineConfigurationError(
            "engine id must start with a letter or number and contain only letters, "
            "numbers, '.', '_' or '-', up to 128 characters"
        )
    if engine_id.split(".", maxsplit=1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise EngineConfigurationError(f"engine id is a reserved Windows device name: {engine_id!r}")
