from __future__ import annotations

import json
import math
import subprocess
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

from genivox.core.models import (
    Capability,
    EngineManifest,
    EngineTransport,
    SynthesisRequest,
    SynthesisResult,
)

from .base import EngineAdapter, EngineConfigurationError, EngineExecutionError


class JsonProcessAdapter(EngineAdapter):
    """Run a local engine that accepts one JSON request on stdin and returns JSON on stdout."""

    def __init__(self, manifest: EngineManifest) -> None:
        if manifest.transport is not EngineTransport.PROCESS:
            raise EngineConfigurationError("JsonProcessAdapter requires transport='process'")
        if not manifest.command:
            raise EngineConfigurationError("process manifest requires a command")
        if manifest.metadata.get("trusted_local_code") is not True:
            raise EngineConfigurationError(
                "process engine is not trusted; re-register it and explicitly allow local code"
            )
        super().__init__(manifest)

    @property
    def implemented_capabilities(self) -> frozenset[Capability]:
        # A generic process is a protocol bridge. Its manifest is the explicit capability contract.
        return frozenset(self.manifest.capabilities)

    def _synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        command = [*self.manifest.command]
        if self.manifest.python:
            command.insert(0, self.manifest.python)
        timeout = float(self.manifest.metadata.get("timeout_seconds", 300.0))
        if timeout <= 0:
            raise EngineConfigurationError("process timeout_seconds must be greater than zero")

        try:
            completed = subprocess.run(
                command,
                input=json.dumps(_request_payload(request), ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                cwd=self.manifest.root or None,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineExecutionError(
                f"engine process timed out after {timeout:g} seconds"
            ) from exc
        except OSError as exc:
            raise EngineExecutionError(f"could not start engine process: {exc}") from exc
        except UnicodeError as exc:
            raise EngineExecutionError("engine process output is not valid UTF-8") from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no error output"
            raise EngineExecutionError(
                f"engine process exited with code {completed.returncode}: {detail[:2_000]}"
            )
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise EngineExecutionError("engine process did not return valid JSON") from exc
        if not isinstance(response, dict):
            raise EngineExecutionError("engine process response must be a JSON object")
        if response.get("ok") is False:
            detail = str(response.get("error", "engine reported failure"))
            raise EngineExecutionError(detail[:2_000])

        output_path = Path(response.get("output_path", request.output_path))
        if not output_path.is_absolute() and self.manifest.root:
            output_path = Path(self.manifest.root) / output_path
        requested_output = request.output_path.expanduser().resolve()
        output_path = output_path.expanduser().resolve()
        if output_path != requested_output:
            raise EngineExecutionError(
                "engine process returned an output path different from the requested destination"
            )
        if not output_path.is_file():
            raise EngineExecutionError(f"engine process did not create output audio: {output_path}")
        try:
            with wave.open(str(output_path), "rb") as output_audio:
                sample_rate = output_audio.getframerate()
                frame_count = output_audio.getnframes()
                if output_audio.getcomptype() != "NONE":
                    raise EngineExecutionError("engine process output WAV must be uncompressed PCM")
                if (
                    sample_rate <= 0
                    or output_audio.getnchannels() <= 0
                    or output_audio.getsampwidth() <= 0
                    or frame_count <= 0
                ):
                    raise EngineExecutionError("engine process output WAV contains no playable samples")
                pcm = output_audio.readframes(frame_count)
                expected_bytes = (
                    frame_count * output_audio.getnchannels() * output_audio.getsampwidth()
                )
                if len(pcm) != expected_bytes:
                    raise EngineExecutionError("engine process output WAV is truncated")
                measured_duration = frame_count / sample_rate
        except (EOFError, wave.Error) as exc:
            raise EngineExecutionError(
                f"engine process did not create valid WAV audio: {output_path}"
            ) from exc

        duration = response.get("duration_seconds")
        metadata = response.get("metadata", {})
        if duration is not None and (
            isinstance(duration, bool) or not isinstance(duration, (int, float))
        ):
            raise EngineExecutionError("duration_seconds in process response must be numeric or null")
        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise EngineExecutionError(
                    "duration_seconds in process response must be finite and greater than zero"
                )
            tolerance = max(0.05, measured_duration * 0.02)
            if abs(duration - measured_duration) > tolerance:
                raise EngineExecutionError(
                    "duration_seconds in process response does not match the output WAV"
                )
        if not isinstance(metadata, dict):
            raise EngineExecutionError("metadata in process response must be a JSON object")
        return SynthesisResult(
            output_path=output_path,
            engine_id=self.manifest.id,
            duration_seconds=measured_duration,
            metadata=metadata,
        )


def _request_payload(request: SynthesisRequest) -> dict[str, Any]:
    return {
        "text": request.text,
        "output_path": str(request.output_path),
        "engine_id": request.engine_id,
        "language": request.language,
        "segments": [asdict(segment) for segment in request.segments],
        "reference_audio": str(request.reference_audio) if request.reference_audio else None,
        "prompt_text": request.prompt_text,
        "speed": request.speed,
        "emotion": request.emotion,
        "style_instruction": request.style_instruction,
        "seed": request.seed,
        "extra": request.extra,
    }
