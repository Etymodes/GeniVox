from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

from genivox.core.models import (
    Capability,
    EngineManifest,
    EngineTransport,
    SynthesisRequest,
    SynthesisResult,
)

from .base import EngineAdapter, EngineConfigurationError, EngineExecutionError

_ALLOWED_EXTRA_FIELDS = frozenset(
    {
        "aux_ref_audio_paths",
        "batch_size",
        "batch_threshold",
        "fragment_interval",
        "parallel_infer",
        "prompt_lang",
        "repetition_penalty",
        "sample_steps",
        "split_bucket",
        "super_sampling",
        "temperature",
        "text_lang",
        "text_split_method",
        "top_k",
        "top_p",
    }
)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Keep a loopback request on loopback instead of following a 3xx elsewhere."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


def _open_local_http(
    request: urllib.request.Request, *, timeout: float
) -> object:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _RejectRedirects(),
    )
    return opener.open(request, timeout=timeout)


class GptSovitsV2HttpAdapter(EngineAdapter):
    """Adapter for GPT-SoVITS api_v2.py's JSON ``POST /tts`` endpoint."""

    _CAPABILITIES = frozenset(
        {
            Capability.VOICE_CLONE,
            Capability.CROSS_LINGUAL,
            Capability.SPEED,
            Capability.FINE_TUNE,
        }
    )

    def __init__(self, manifest: EngineManifest) -> None:
        if manifest.transport is not EngineTransport.HTTP:
            raise EngineConfigurationError("GptSovitsV2HttpAdapter requires transport='http'")
        if not manifest.endpoint:
            raise EngineConfigurationError("GPT-SoVITS HTTP manifest requires an endpoint")
        parsed_endpoint = urllib.parse.urlparse(manifest.endpoint)
        if (
            parsed_endpoint.scheme not in {"http", "https"}
            or parsed_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise EngineConfigurationError(
                "GPT-SoVITS endpoint must be a loopback URL in local-only mode"
            )
        super().__init__(manifest)

    @property
    def implemented_capabilities(self) -> frozenset[Capability]:
        return self._CAPABILITIES

    def validate_request(self, request: SynthesisRequest) -> None:
        super().validate_request(request)
        if request.reference_audio is None:
            raise EngineExecutionError("GPT-SoVITS requires a reference audio file")
        prompt_language = request.extra.get("prompt_lang")
        if not isinstance(prompt_language, str) or not prompt_language.strip():
            raise EngineExecutionError("GPT-SoVITS requires a prompt/reference language")

    def _synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        payload = self._build_payload(request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            self.manifest.endpoint,
            data=body,
            headers={"Accept": "audio/wav", "Content-Type": "application/json"},
            method="POST",
        )
        timeout = float(self.manifest.metadata.get("timeout_seconds", 120.0))
        if timeout <= 0:
            raise EngineConfigurationError("HTTP timeout_seconds must be greater than zero")

        try:
            with _open_local_http(http_request, timeout=timeout) as response:
                maximum_bytes = int(
                    self.manifest.metadata.get("max_response_bytes", 512 * 1024 * 1024)
                )
                if maximum_bytes <= 0:
                    raise EngineConfigurationError("max_response_bytes must be greater than zero")
                audio = response.read(maximum_bytes + 1)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1_000).decode("utf-8", errors="replace")
            suffix = f": {detail}" if detail else ""
            raise EngineExecutionError(
                f"GPT-SoVITS returned HTTP {exc.code}{suffix}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EngineExecutionError(f"GPT-SoVITS request failed: {exc}") from exc

        if not audio:
            raise EngineExecutionError("GPT-SoVITS returned an empty audio response")
        if len(audio) > maximum_bytes:
            raise EngineExecutionError(
                f"GPT-SoVITS response exceeds the configured {maximum_bytes} byte limit"
            )
        duration = _wav_duration(audio)
        _atomic_write(request.output_path, audio)
        return SynthesisResult(
            output_path=request.output_path,
            engine_id=self.manifest.id,
            duration_seconds=duration,
            metadata={"backend": "gpt_sovits_v2", "request": payload},
        )

    def _build_payload(self, request: SynthesisRequest) -> dict[str, object]:
        extras = {key: value for key, value in request.extra.items() if key in _ALLOWED_EXTRA_FIELDS}
        text_language = str(extras.pop("text_lang", request.language))
        prompt_language = str(extras.pop("prompt_lang", request.language))
        payload: dict[str, object] = {
            "text": request.text,
            "text_lang": text_language,
            "ref_audio_path": str(request.reference_audio) if request.reference_audio else "",
            "prompt_text": request.prompt_text,
            "prompt_lang": prompt_language,
            "speed_factor": request.speed,
            "seed": request.seed,
            "media_type": "wav",
        }
        payload.update(extras)
        return payload


def _wav_duration(audio: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio), "rb") as source:
            frame_rate = source.getframerate()
            if source.getcomptype() != "NONE":
                raise EngineExecutionError("GPT-SoVITS returned compressed rather than PCM WAV audio")
            if (
                frame_rate <= 0
                or source.getnchannels() <= 0
                or source.getsampwidth() <= 0
                or source.getnframes() <= 0
            ):
                raise EngineExecutionError("GPT-SoVITS returned WAV audio with no playable samples")
            frame_count = source.getnframes()
            pcm = source.readframes(frame_count)
            expected_bytes = frame_count * source.getnchannels() * source.getsampwidth()
            if len(pcm) != expected_bytes:
                raise EngineExecutionError("GPT-SoVITS returned a truncated WAV response")
            return frame_count / frame_rate
    except (EOFError, wave.Error) as exc:
        raise EngineExecutionError("GPT-SoVITS response is not valid WAV audio") from exc


def _atomic_write(destination: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        temporary_path.replace(destination)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
