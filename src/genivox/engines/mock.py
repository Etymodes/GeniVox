from __future__ import annotations

import math
import struct
import wave

from genivox.core.models import (
    Capability,
    EngineManifest,
    EngineTransport,
    SynthesisRequest,
    SynthesisResult,
)

from .base import EngineAdapter, EngineConfigurationError


class MockWavAdapter(EngineAdapter):
    """Deterministic WAV generator for exercising the UI without a TTS install."""

    _CAPABILITIES = frozenset({Capability.CROSS_LINGUAL, Capability.SPEED})

    def __init__(self, manifest: EngineManifest) -> None:
        if manifest.transport is not EngineTransport.MOCK:
            raise EngineConfigurationError("MockWavAdapter requires transport='mock'")
        super().__init__(manifest)

    @property
    def implemented_capabilities(self) -> frozenset[Capability]:
        return self._CAPABILITIES

    def _synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        sample_rate = int(self.manifest.metadata.get("sample_rate", 16_000))
        if sample_rate <= 0:
            raise EngineConfigurationError("mock sample_rate must be greater than zero")

        base_duration = max(0.15, min(5.0, len(request.text.strip()) * 0.06))
        duration = base_duration / request.speed
        frame_count = max(1, round(sample_rate * duration))
        frequency = 220.0 + (sum(request.text.encode("utf-8")) % 220)
        amplitude = 2_000

        with wave.open(str(request.output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            frames = bytearray()
            for index in range(frame_count):
                value = round(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
                frames.extend(struct.pack("<h", value))
            output.writeframes(frames)

        return SynthesisResult(
            output_path=request.output_path,
            engine_id=self.manifest.id,
            duration_seconds=frame_count / sample_rate,
            metadata={"backend": "mock", "sample_rate": sample_rate},
        )
