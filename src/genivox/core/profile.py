from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from genivox.core.paths import ensure_private_directory, protect_private_file


@dataclass(slots=True)
class ConsentRecord:
    authorized: bool
    scope: str
    recorded_at: str


@dataclass(slots=True)
class SourceRecording:
    path: str
    sha256: str
    transcript: str = ""
    language: str = "und"
    emotion_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        transcript: str = "",
        language: str = "und",
        emotion_label: str | None = None,
    ) -> SourceRecording:
        return cls(
            path=str(path.resolve()),
            sha256=file_sha256(path),
            transcript=transcript,
            language=language,
            emotion_label=emotion_label,
        )


@dataclass(slots=True)
class VoiceProfile:
    id: str
    display_name: str
    consent: ConsentRecord
    schema_version: int = 1
    pronunciation_defaults: dict[str, str] = field(default_factory=dict)
    source_recordings: list[SourceRecording] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    engine_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate_for_training(self) -> list[str]:
        errors: list[str] = []
        if not self.consent.authorized:
            errors.append("Voice profile is not authorized for use.")
        if not self.consent.scope.strip():
            errors.append("Consent scope is empty.")
        if not self.source_recordings:
            errors.append("Voice profile has no source recordings.")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        payload = dict(data)
        payload["consent"] = ConsentRecord(**payload["consent"])
        payload["source_recordings"] = [
            SourceRecording(**item) for item in payload.get("source_recordings", [])
        ]
        return cls(**payload)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_voice_profile(path: Path) -> VoiceProfile:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported voice profile schema: {data.get('schema_version')!r}")
    return VoiceProfile.from_dict(data)


def save_voice_profile(profile: VoiceProfile, path: Path) -> None:
    ensure_private_directory(path.parent)
    payload = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
        protect_private_file(path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
