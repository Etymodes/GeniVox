from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class EmotionAnalysisError(RuntimeError):
    """Raised when an explicitly configured emotion provider fails."""


@runtime_checkable
class EmotionAnalyzer(Protocol):
    """Pluggable boundary for a real, separately installed emotion model."""

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> Mapping[str, float]: ...


def _parse_probabilities(payload: object, *, source: str) -> dict[str, float]:
    if isinstance(payload, dict) and "emotion" in payload:
        payload = payload["emotion"]
    if not isinstance(payload, dict):
        raise EmotionAnalysisError(
            f"{source} must contain an emotion object or a direct label-to-probability object"
        )

    probabilities: dict[str, float] = {}
    for raw_label, raw_value in payload.items():
        label = str(raw_label).strip()
        if not label:
            raise EmotionAnalysisError(f"{source} contains an empty emotion label")
        if isinstance(raw_value, bool):
            raise EmotionAnalysisError(f"{source} probability for {label!r} is not numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise EmotionAnalysisError(
                f"{source} probability for {label!r} is not numeric: {raw_value!r}"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EmotionAnalysisError(
                f"{source} probability for {label!r} must be finite and between 0 and 1"
            )
        probabilities[label] = value
    return probabilities


@dataclass(slots=True)
class NoOpEmotionAnalyzer:
    """Honest default: no model means no inferred emotion probabilities."""

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> Mapping[str, float]:
        return {}


@dataclass(slots=True)
class SidecarEmotionAnalyzer:
    """Read reviewed emotion probabilities from ``<audio>.emotion.json``.

    Accepted JSON forms are ``{"emotion": {"calm": 0.8}}`` and the direct
    ``{"calm": 0.8}`` form. Values are intentionally not normalized because
    some emotion models are multi-label classifiers.
    """

    suffix: str = ".emotion.json"
    strict: bool = False

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> Mapping[str, float]:
        if source_path is None:
            raise EmotionAnalysisError("Sidecar emotion analysis requires source_path")
        sidecar_path = source_path.with_name(source_path.name + self.suffix)
        if not sidecar_path.exists():
            if self.strict:
                raise EmotionAnalysisError(f"Emotion sidecar not found: {sidecar_path}")
            return {}
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EmotionAnalysisError(f"Cannot read emotion sidecar {sidecar_path}: {exc}") from exc
        return _parse_probabilities(payload, source=str(sidecar_path))


@dataclass(slots=True)
class ExternalEmotionAnalyzer:
    """Run an optional emotion model behind a dependency-free JSON bridge.

    The configured command receives ``{"audio_path": "..."}`` on standard
    input and must emit either ``{"emotion": {label: probability}}`` or a
    direct label-to-probability object on standard output. This boundary can
    wrap emotion2vec/FunASR, ONNX, or a remote-worker client without making
    any of them a base GeniVox dependency.
    """

    command: Sequence[str]
    timeout_seconds: float = 60.0
    environment: Mapping[str, str] | None = None
    _command: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._command = tuple(str(part) for part in self.command)
        if not self._command:
            raise ValueError("External emotion analyzer command cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("External emotion analyzer timeout must be positive")

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> Mapping[str, float]:
        if source_path is None:
            raise EmotionAnalysisError("External emotion analysis requires source_path")

        child_environment = None
        if self.environment is not None:
            child_environment = os.environ.copy()
            child_environment.update({str(key): str(value) for key, value in self.environment.items()})

        request = json.dumps({"audio_path": str(source_path.resolve())}, ensure_ascii=False)
        try:
            completed = subprocess.run(
                self._command,
                input=request,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                timeout=self.timeout_seconds,
                check=False,
                env=child_environment,
            )
        except FileNotFoundError as exc:
            raise EmotionAnalysisError(
                f"Emotion analyzer executable was not found: {self._command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise EmotionAnalysisError(
                f"Emotion analyzer timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise EmotionAnalysisError(f"Cannot start emotion analyzer: {exc}") from exc
        except UnicodeError as exc:
            raise EmotionAnalysisError("Emotion analyzer output is not valid UTF-8") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
            raise EmotionAnalysisError(
                f"Emotion analyzer exited with code {completed.returncode}: {detail[:1000]}"
            )
        if not completed.stdout.strip():
            raise EmotionAnalysisError("Emotion analyzer returned empty stdout; expected JSON")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EmotionAnalysisError(f"Emotion analyzer returned invalid JSON: {exc}") from exc
        return _parse_probabilities(payload, source="Emotion analyzer output")
