from __future__ import annotations

import os
import tempfile
import wave
from dataclasses import replace
from pathlib import Path

from genivox.core.models import Capability, LanguageSegment, SynthesisRequest, SynthesisResult

from .base import EngineAdapter, EngineExecutionError, InvalidSynthesisRequest


class SynthesisPipeline:
    """Compose explicitly labelled language segments through one engine instance."""

    def __init__(self, adapter: EngineAdapter) -> None:
        self.adapter = adapter

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not request.segments:
            return self.adapter.synthesize(request)
        if request.output_path.suffix.casefold() != ".wav":
            raise InvalidSynthesisRequest("segmented synthesis output must use the .wav extension")
        if len(request.segments) > 256:
            raise InvalidSynthesisRequest("segmented synthesis accepts at most 256 source spans")
        _validate_source_segments(request.text, request.segments)

        segments = _coalesce_neutral_segments(request.segments)
        languages = {segment.language.casefold() for segment in segments}
        if len(languages) > 1:
            self.adapter.require_capabilities(
                (Capability.CROSS_LINGUAL,), context="multi-language segments"
            )
        for segment in segments:
            if not segment.text.strip():
                raise InvalidSynthesisRequest("language segments must not contain empty text")
            if segment.language.casefold() in {"auto", "und", "mixed", "multilingual"}:
                raise InvalidSynthesisRequest(
                    f"segment {segment.start}:{segment.end} needs an explicit language"
                )
            self.adapter._require_language(segment.language)

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="genivox-segments-") as temporary_dir:
            outputs: list[Path] = []
            segment_metadata: list[dict[str, object]] = []
            for index, segment in enumerate(segments):
                segment_path = Path(temporary_dir) / f"{index:04d}.wav"
                segment_request = replace(
                    request,
                    text=segment.text,
                    output_path=segment_path,
                    language=segment.language,
                    segments=[],
                    extra={**request.extra, "text_lang": segment.language},
                )
                result = self.adapter.synthesize(segment_request)
                outputs.append(result.output_path)
                segment_metadata.append(
                    {
                        "language": segment.language,
                        "duration_seconds": result.duration_seconds,
                        "metadata": result.metadata,
                    }
                )

            duration = _concatenate_pcm_wav(outputs, request.output_path)

        return SynthesisResult(
            output_path=request.output_path,
            engine_id=self.adapter.manifest.id,
            duration_seconds=duration,
            metadata={"segment_count": len(segments), "segments": segment_metadata},
        )


def _validate_source_segments(text: str, segments: list[LanguageSegment]) -> None:
    """Ensure recorded source offsets describe the exact text sent to adapters."""

    previous_end = 0
    for segment in segments:
        if (
            isinstance(segment.start, bool)
            or isinstance(segment.end, bool)
            or not isinstance(segment.start, int)
            or not isinstance(segment.end, int)
            or segment.start < previous_end
            or segment.end <= segment.start
            or segment.end > len(text)
        ):
            raise InvalidSynthesisRequest("language segment offsets are invalid or overlap")
        if text[segment.start : segment.end] != segment.text:
            raise InvalidSynthesisRequest(
                "language segment text does not match its offsets in the source text"
            )
        previous_end = segment.end


def _concatenate_pcm_wav(inputs: list[Path], destination: Path) -> float:
    if not inputs:
        raise InvalidSynthesisRequest("at least one WAV segment is required")

    parameters: tuple[int, int, int, str] | None = None
    total_frames = 0
    for path in inputs:
        try:
            with wave.open(str(path), "rb") as source:
                current = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                    source.getcomptype(),
                )
                if current[3] != "NONE":
                    raise EngineExecutionError(f"WAV segment is not uncompressed PCM: {path}")
                if source.getnframes() <= 0:
                    raise EngineExecutionError(f"WAV segment contains no playable samples: {path}")
                frame_count = source.getnframes()
                pcm = source.readframes(frame_count)
                expected_bytes = frame_count * source.getnchannels() * source.getsampwidth()
                if len(pcm) != expected_bytes:
                    raise EngineExecutionError(f"WAV segment is truncated: {path}")
                if parameters is None:
                    parameters = current
                elif current != parameters:
                    raise EngineExecutionError(
                        "WAV segments have incompatible channel, width, rate, or compression parameters"
                    )
        except (OSError, EOFError, wave.Error) as exc:
            raise EngineExecutionError(f"could not read WAV segment {path}: {exc}") from exc

    assert parameters is not None
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with wave.open(str(temporary_path), "wb") as target:
            target.setnchannels(parameters[0])
            target.setsampwidth(parameters[1])
            target.setframerate(parameters[2])
            target.setcomptype(parameters[3], "not compressed")
            for path in inputs:
                with wave.open(str(path), "rb") as source:
                    frame_count = source.getnframes()
                    frames = source.readframes(frame_count)
                    target.writeframes(frames)
                    total_frames += frame_count
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return total_frames / parameters[2]


def _coalesce_neutral_segments(segments: list[LanguageSegment]) -> list[LanguageSegment]:
    """Attach punctuation outside explicit tags to a neighboring spoken segment."""

    spoken: list[LanguageSegment] = []
    leading = ""
    for segment in segments:
        if (
            segment.language == "und"
            and not any(char.isalnum() for char in segment.text)
        ):
            punctuation = segment.text
            if spoken:
                previous = spoken[-1]
                spoken[-1] = replace(previous, text=previous.text + punctuation)
            else:
                leading += punctuation
            continue
        if leading:
            segment = replace(segment, text=leading + segment.text)
            leading = ""
        spoken.append(segment)
    if leading and spoken:
        previous = spoken[-1]
        spoken[-1] = replace(previous, text=previous.text + leading)
    return spoken
