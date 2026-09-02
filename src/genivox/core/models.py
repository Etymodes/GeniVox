from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Capability(StrEnum):
    VOICE_CLONE = "voice_clone"
    CROSS_LINGUAL = "cross_lingual"
    SPEED = "speed"
    EMOTION_VECTOR = "emotion_vector"
    STYLE_INSTRUCTION = "style_instruction"
    PHONEME_INPUT = "phoneme_input"
    STREAMING = "streaming"
    FINE_TUNE = "fine_tune"


class EngineTransport(StrEnum):
    MOCK = "mock"
    HTTP = "http"
    PROCESS = "process"


@dataclass(slots=True)
class EngineManifest:
    id: str
    name: str
    transport: EngineTransport
    capabilities: list[Capability] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    endpoint: str | None = None
    root: str | None = None
    python: str | None = None
    command: list[str] = field(default_factory=list)
    checkpoint_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("engine id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("engine name must be a non-empty string")
        if not isinstance(self.transport, EngineTransport):
            raise ValueError("engine transport must be an EngineTransport value")
        if not isinstance(self.capabilities, list) or any(
            not isinstance(item, Capability) for item in self.capabilities
        ):
            raise ValueError("engine capabilities must be a list of Capability values")
        if not isinstance(self.languages, list) or any(
            not isinstance(item, str) or not item.strip() for item in self.languages
        ):
            raise ValueError("engine languages must be a list of non-empty strings")
        if not isinstance(self.command, list) or any(
            not isinstance(item, str) or not item for item in self.command
        ):
            raise ValueError("engine command must be a list of non-empty argument strings")
        if not isinstance(self.metadata, dict):
            raise ValueError("engine metadata must be an object")
        for field_name in ("endpoint", "root", "python", "checkpoint_dir"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"engine {field_name} must be a string or null")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineManifest:
        payload = dict(data)
        payload["transport"] = EngineTransport(payload["transport"])
        raw_capabilities = payload.get("capabilities", [])
        if not isinstance(raw_capabilities, list):
            raise ValueError("engine capabilities must be a JSON array")
        payload["capabilities"] = [Capability(item) for item in raw_capabilities]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["transport"] = self.transport.value
        payload["capabilities"] = [item.value for item in self.capabilities]
        return payload


@dataclass(slots=True)
class LanguageSegment:
    text: str
    language: str
    start: int
    end: int
    source: str = "auto"
    confidence: float = 1.0


@dataclass(slots=True)
class ProsodyProfile:
    duration_seconds: float = 0.0
    sample_rate: int = 0
    mean_f0_hz: float | None = None
    f0_std_hz: float | None = None
    f0_min_hz: float | None = None
    f0_max_hz: float | None = None
    rms_dbfs: float | None = None
    voiced_ratio: float | None = None
    acoustic_peak_rate_hz: float | None = None
    emotion: dict[str, float] = field(default_factory=dict)
    style_instruction: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SynthesisRequest:
    text: str
    output_path: Path
    engine_id: str
    language: str = "auto"
    segments: list[LanguageSegment] = field(default_factory=list)
    reference_audio: Path | None = None
    prompt_text: str = ""
    speed: float = 1.0
    emotion: dict[str, float] = field(default_factory=dict)
    style_instruction: str = ""
    seed: int = -1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SynthesisResult:
    output_path: Path
    engine_id: str
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetRecord:
    audio_path: Path
    text: str
    language: str = "und"
    speaker: str = "default"
    emotion: str = "unlabeled"
    duration_seconds: float | None = None


@dataclass(slots=True)
class TrainingMetric:
    step: int
    values: dict[str, float]
    timestamp: float | None = None
