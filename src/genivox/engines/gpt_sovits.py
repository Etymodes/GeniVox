from __future__ import annotations

import http.client
import io
import itertools
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from enum import StrEnum
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

_OPENAPI_RESPONSE_LIMIT = 1024 * 1024
_WEIGHT_CANDIDATE_LIMIT = 256
_WEIGHT_SCAN_ROOT_LIMIT = 128
_WEIGHT_SCAN_ENTRY_LIMIT = 10_000
_WEIGHT_SCAN_DEPTH_LIMIT = 4
_REQUIRED_TTS_FIELDS = frozenset({"text", "text_lang", "ref_audio_path", "prompt_lang"})
_ALWAYS_SENT_TTS_FIELDS = frozenset(
    {
        "text",
        "text_lang",
        "ref_audio_path",
        "prompt_text",
        "prompt_lang",
        "speed_factor",
        "seed",
        "media_type",
    }
)
_TTS_FIELD_TYPES = {
    "text": "string",
    "text_lang": "string",
    "ref_audio_path": "string",
    "prompt_text": "string",
    "prompt_lang": "string",
    "speed_factor": "number",
    "seed": "integer",
    "media_type": "string",
}
_GPT_WEIGHT_DIRECTORIES = ("GPT_weights*", "GPT_SoVITS/pretrained_models")
_SOVITS_WEIGHT_DIRECTORIES = ("SoVITS_weights*", "GPT_SoVITS/pretrained_models")


class GptSovitsProbeStatus(StrEnum):
    INVALID = "invalid"
    OFFLINE = "offline"
    WRONG_SERVICE = "wrong_service"
    INCOMPATIBLE = "incompatible"
    API_READY = "api_ready"


@dataclass(frozen=True, slots=True)
class GptSovitsProbeResult:
    status: GptSovitsProbeStatus
    message: str
    checked_url: str | None = None
    latency_ms: int | None = None
    synthesis_verified: bool = False

    @property
    def ready(self) -> bool:
        return self.status is GptSovitsProbeStatus.API_READY


@dataclass(frozen=True, slots=True)
class GptSovitsInstallationProbe:
    root: Path | None
    entrypoint: Path | None
    core_module: Path | None
    config: Path | None
    python: Path | None
    gpt_weight_candidates: tuple[Path, ...]
    sovits_weight_candidates: tuple[Path, ...]
    issues: tuple[str, ...]

    @property
    def launch_ready(self) -> bool:
        return not self.issues

    @property
    def weights_detected(self) -> bool:
        return bool(self.gpt_weight_candidates and self.sovits_weight_candidates)


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


def inspect_gpt_sovits_installation(
    root: str | Path | None,
    *,
    python: str | Path | None = None,
) -> GptSovitsInstallationProbe:
    """Inspect an existing checkout without importing or executing model code."""

    if root is None or not str(root).strip():
        return GptSovitsInstallationProbe(
            None, None, None, None, None, (), (), ("root_missing",)
        )
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        return GptSovitsInstallationProbe(
            resolved_root,
            None,
            None,
            None,
            None,
            (),
            (),
            ("root_not_directory",),
        )

    entrypoint = resolved_root / "api_v2.py"
    core_module = resolved_root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
    config = resolved_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    resolved_python = _resolve_gpt_sovits_python(resolved_root, python)
    gpt_weights = _find_weight_candidates(
        resolved_root, _GPT_WEIGHT_DIRECTORIES, ".ckpt"
    )
    sovits_weights = _find_weight_candidates(
        resolved_root, _SOVITS_WEIGHT_DIRECTORIES, ".pth"
    )
    issues: list[str] = []
    if not entrypoint.is_file():
        issues.append("api_v2_missing")
    if not core_module.is_file():
        issues.append("tts_core_missing")
    if not config.is_file():
        issues.append("tts_config_missing")
    if resolved_python is None:
        issues.append("python_missing")
    return GptSovitsInstallationProbe(
        resolved_root,
        entrypoint if entrypoint.is_file() else None,
        core_module if core_module.is_file() else None,
        config if config.is_file() else None,
        resolved_python,
        gpt_weights,
        sovits_weights,
        tuple(issues),
    )


def probe_gpt_sovits_api(
    endpoint: str,
    *,
    timeout: float = 2.5,
    max_response_bytes: int = _OPENAPI_RESPONSE_LIMIT,
) -> GptSovitsProbeResult:
    """Read the loopback OpenAPI document without running inference or changing server state."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("probe timeout must be greater than zero")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes <= 0
    ):
        raise ValueError("probe response limit must be greater than zero")
    try:
        openapi_url = _gpt_sovits_openapi_url(endpoint)
    except (TypeError, ValueError) as exc:
        return GptSovitsProbeResult(GptSovitsProbeStatus.INVALID, str(exc))

    request = urllib.request.Request(
        openapi_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    started = time.monotonic()
    try:
        with _open_local_http(request, timeout=timeout) as response:
            body = response.read(max_response_bytes + 1)
    except urllib.error.HTTPError as exc:
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            f"OpenAPI probe returned HTTP {exc.code}",
            openapi_url,
            started,
        )
    except http.client.InvalidURL as exc:
        return _timed_probe_result(
            GptSovitsProbeStatus.INVALID,
            f"Invalid GPT-SoVITS endpoint: {exc}",
            openapi_url,
            started,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _timed_probe_result(
            GptSovitsProbeStatus.OFFLINE,
            f"GPT-SoVITS service is unreachable: {exc}",
            openapi_url,
            started,
        )
    except http.client.HTTPException as exc:
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            f"The loopback service returned an invalid HTTP response: {exc}",
            openapi_url,
            started,
        )

    if len(body) > max_response_bytes:
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            f"OpenAPI response exceeds {max_response_bytes} bytes",
            openapi_url,
            started,
        )
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            "OpenAPI response is not valid UTF-8 JSON",
            openapi_url,
            started,
        )
    if not isinstance(document, dict):
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            "OpenAPI response is not a JSON object",
            openapi_url,
            started,
        )

    openapi_version = document.get("openapi")
    if not isinstance(openapi_version, str) or re.fullmatch(
        r"3\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", openapi_version
    ) is None:
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            "The response is not an OpenAPI 3 document",
            openapi_url,
            started,
        )

    paths = document.get("paths")
    if not isinstance(paths, dict) or "/tts" not in paths:
        return _timed_probe_result(
            GptSovitsProbeStatus.WRONG_SERVICE,
            "The loopback service does not expose GPT-SoVITS /tts",
            openapi_url,
            started,
        )
    tts_route = paths["/tts"]
    if not isinstance(tts_route, dict) or not isinstance(tts_route.get("post"), dict):
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts route does not expose the required POST operation",
            openapi_url,
            started,
        )
    request_schema = _openapi_request_schema(document, tts_route["post"])
    if request_schema is None:
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema could not be resolved",
            openapi_url,
            started,
        )
    properties = request_schema.get("properties")
    if not isinstance(properties, dict):
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema does not declare object properties",
            openapi_url,
            started,
        )
    property_names = frozenset(str(name) for name in properties)
    missing = sorted(_REQUIRED_TTS_FIELDS - property_names)
    if missing:
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema is missing: " + ", ".join(missing),
            openapi_url,
            started,
        )
    wrong_types = sorted(
        field
        for field, expected_type in _TTS_FIELD_TYPES.items()
        if field in properties
        and not _openapi_property_accepts_type(properties.get(field), expected_type)
    )
    if wrong_types:
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema has incompatible types for: " + ", ".join(wrong_types),
            openapi_url,
            started,
        )
    required = request_schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema has an invalid required-field declaration",
            openapi_url,
            started,
        )
    unknown_required = sorted(set(required) - _ALWAYS_SENT_TTS_FIELDS)
    if unknown_required:
        return _timed_probe_result(
            GptSovitsProbeStatus.INCOMPATIBLE,
            "The /tts request schema requires unsupported fields: "
            + ", ".join(unknown_required),
            openapi_url,
            started,
        )
    if request_schema.get("additionalProperties") is False:
        rejected_fields = sorted(_ALWAYS_SENT_TTS_FIELDS - property_names)
        if rejected_fields:
            return _timed_probe_result(
                GptSovitsProbeStatus.INCOMPATIBLE,
                "The /tts request schema rejects adapter fields: "
                + ", ".join(rejected_fields),
                openapi_url,
                started,
            )
    return _timed_probe_result(
        GptSovitsProbeStatus.API_READY,
        "The endpoint matches the GPT-SoVITS api_v2 request shape; synthesis is unverified",
        openapi_url,
        started,
    )


def _resolve_gpt_sovits_python(
    root: Path, configured: str | Path | None
) -> Path | None:
    if configured is not None and str(configured).strip():
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    candidates = (
        root / "runtime" / "python.exe",
        root / ".venv" / "Scripts" / "python.exe",
        root / "venv" / "Scripts" / "python.exe",
        root / ".venv" / "bin" / "python",
        root / "venv" / "bin" / "python",
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _find_weight_candidates(
    root: Path, directory_patterns: tuple[str, ...], suffix: str
) -> tuple[Path, ...]:
    """Bound inspection to documented weight roots without traversing links outside the checkout."""

    candidates: set[Path] = set()
    pending: list[tuple[Path, int]] = []
    staged_roots = 0
    for pattern in directory_patterns:
        try:
            remaining_roots = _WEIGHT_SCAN_ROOT_LIMIT - staged_roots
            directories = itertools.islice(root.glob(pattern), remaining_roots)
            for directory in directories:
                staged_roots += 1
                if directory.is_symlink():
                    continue
                resolved_directory = directory.resolve()
                if (
                    not resolved_directory.is_relative_to(root)
                    or not resolved_directory.is_dir()
                ):
                    continue
                pending.append((resolved_directory, 0))
        except OSError:
            continue
        if staged_roots >= _WEIGHT_SCAN_ROOT_LIMIT:
            break

    visited_directories: set[Path] = set()
    visited_entries = 0
    while pending and visited_entries < _WEIGHT_SCAN_ENTRY_LIMIT:
        directory, depth = pending.pop()
        if directory in visited_directories:
            continue
        visited_directories.add(directory)
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited_entries += 1
                    if visited_entries > _WEIGHT_SCAN_ENTRY_LIMIT:
                        break
                    if entry.is_symlink():
                        continue
                    candidate = Path(entry.path)
                    if entry.is_file(follow_symlinks=False):
                        if candidate.suffix.casefold() != suffix.casefold():
                            continue
                        resolved_candidate = candidate.resolve()
                        if resolved_candidate.is_relative_to(root):
                            candidates.add(resolved_candidate)
                        if len(candidates) >= _WEIGHT_CANDIDATE_LIMIT:
                            return tuple(sorted(candidates, key=os.fspath))
                    elif (
                        depth < _WEIGHT_SCAN_DEPTH_LIMIT
                        and entry.is_dir(follow_symlinks=False)
                    ):
                        resolved_candidate = candidate.resolve()
                        if resolved_candidate.is_relative_to(root):
                            pending.append((resolved_candidate, depth + 1))
        except OSError:
            continue
    return tuple(sorted(candidates, key=os.fspath))


def _gpt_sovits_openapi_url(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("GPT-SoVITS endpoint is required")
    normalized_endpoint = endpoint.strip()
    if any(ord(character) < 33 or ord(character) == 127 for character in normalized_endpoint):
        raise ValueError("GPT-SoVITS endpoint must not contain whitespace or control characters")
    try:
        parsed = urllib.parse.urlsplit(normalized_endpoint)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"invalid GPT-SoVITS endpoint: {exc}") from exc
    if parsed.scheme not in {"http", "https"} or hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ValueError("GPT-SoVITS probe only accepts a loopback HTTP endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("GPT-SoVITS endpoint must not contain credentials, query, or fragment")
    if not parsed.path.isascii():
        raise ValueError("GPT-SoVITS endpoint path must be ASCII or percent-encoded")
    endpoint_path = parsed.path.rstrip("/")
    if not endpoint_path.endswith("/tts"):
        raise ValueError("GPT-SoVITS endpoint path must end with /tts")
    openapi_path = endpoint_path[: -len("/tts")] + "/openapi.json"
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, openapi_path, "", "")
    )


def _openapi_request_schema(
    document: dict[str, object], post_operation: dict[str, object]
) -> dict[str, object] | None:
    request_body = post_operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None
    json_media = content.get("application/json")
    if not isinstance(json_media, dict):
        return None
    schema = json_media.get("schema")
    if not isinstance(schema, dict):
        return None
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/components/schemas/"):
        components = document.get("components")
        schemas = components.get("schemas") if isinstance(components, dict) else None
        name = reference.removeprefix("#/components/schemas/")
        schema = schemas.get(name) if isinstance(schemas, dict) else None
        if not isinstance(schema, dict):
            return None
    return schema


def _openapi_property_accepts_type(property_schema: object, expected_type: str) -> bool:
    if not isinstance(property_schema, dict):
        return False
    property_type = property_schema.get("type")
    if property_type == expected_type:
        return True
    if isinstance(property_type, list) and expected_type in property_type:
        return True
    alternatives = property_schema.get("anyOf")
    return isinstance(alternatives, list) and any(
        isinstance(alternative, dict) and alternative.get("type") == expected_type
        for alternative in alternatives
    )


def _timed_probe_result(
    status: GptSovitsProbeStatus,
    message: str,
    checked_url: str,
    started: float,
) -> GptSovitsProbeResult:
    elapsed = max(0, round((time.monotonic() - started) * 1000))
    return GptSovitsProbeResult(status, message, checked_url, elapsed)


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
