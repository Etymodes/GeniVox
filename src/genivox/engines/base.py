from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

from genivox.core.models import Capability, EngineManifest, SynthesisRequest, SynthesisResult


class EngineError(RuntimeError):
    """Base exception for engine configuration and execution failures."""


class EngineConfigurationError(EngineError):
    """Raised when an engine manifest cannot be honored by an adapter."""


class InvalidSynthesisRequest(EngineError):
    """Raised when a synthesis request is incomplete or internally inconsistent."""


class UnsupportedCapabilityError(InvalidSynthesisRequest):
    """Raised when a request uses a capability the selected engine does not provide."""


class UnsupportedLanguageError(InvalidSynthesisRequest):
    """Raised when a request names a language the selected engine does not provide."""


class EngineExecutionError(EngineError):
    """Raised when an engine starts but does not produce a valid result."""


class EngineAdapter(ABC):
    """Validated interface shared by all synthesis backends."""

    def __init__(self, manifest: EngineManifest) -> None:
        self.manifest = manifest
        declared = set(manifest.capabilities)
        unsupported = declared.difference(self.implemented_capabilities)
        if unsupported:
            names = ", ".join(sorted(item.value for item in unsupported))
            raise EngineConfigurationError(
                f"adapter {type(self).__name__} does not implement capabilities declared by "
                f"engine {manifest.id!r}: {names}"
            )

    @property
    @abstractmethod
    def implemented_capabilities(self) -> frozenset[Capability]:
        """Capabilities this adapter can actually pass through or apply."""

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.validate_request(request)
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        return self._synthesize(request)

    def validate_request(self, request: SynthesisRequest) -> None:
        if request.engine_id != self.manifest.id:
            raise InvalidSynthesisRequest(
                f"request engine_id {request.engine_id!r} does not match {self.manifest.id!r}"
            )
        if not request.text.strip():
            raise InvalidSynthesisRequest("synthesis text must not be empty")
        if request.output_path.exists() and request.output_path.is_dir():
            raise InvalidSynthesisRequest(f"output path is a directory: {request.output_path}")
        if request.reference_audio is not None and not request.reference_audio.is_file():
            raise InvalidSynthesisRequest(f"reference audio does not exist: {request.reference_audio}")
        if not math.isfinite(request.speed) or request.speed <= 0:
            raise InvalidSynthesisRequest("speed must be finite and greater than zero")
        if isinstance(request.seed, bool) or not isinstance(request.seed, int):
            raise InvalidSynthesisRequest("seed must be an integer")
        for label, value in request.emotion.items():
            if not label.strip():
                raise InvalidSynthesisRequest("emotion labels must not be empty")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidSynthesisRequest(f"emotion value for {label!r} must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise InvalidSynthesisRequest(
                    f"emotion value for {label!r} must be finite and between 0 and 1"
                )

        self._require_language(request.language)

        required: list[tuple[bool, Capability, str]] = [
            (request.reference_audio is not None, Capability.VOICE_CLONE, "reference_audio"),
            (request.speed != 1.0, Capability.SPEED, "speed"),
            (bool(request.emotion), Capability.EMOTION_VECTOR, "emotion"),
            (bool(request.style_instruction.strip()), Capability.STYLE_INSTRUCTION, "style_instruction"),
        ]
        for enabled, capability, field_name in required:
            if enabled:
                self.require_capabilities((capability,), context=field_name)

        prompt_language = request.extra.get("prompt_lang")
        text_language = request.extra.get("text_lang", request.language)
        if not isinstance(text_language, str):
            raise UnsupportedLanguageError("text_lang must be a string")
        self._require_language(text_language)
        if isinstance(prompt_language, str):
            self._require_language(prompt_language)
            if _is_concrete_language(prompt_language) and _is_concrete_language(text_language):
                if prompt_language.casefold() != text_language.casefold():
                    self.require_capabilities(
                        (Capability.CROSS_LINGUAL,), context="prompt_lang/text_lang"
                    )
        elif prompt_language is not None:
            raise UnsupportedLanguageError("prompt_lang must be a string")

    def require_capabilities(
        self, capabilities: Iterable[Capability], *, context: str = "request"
    ) -> None:
        declared = set(self.manifest.capabilities)
        for capability in capabilities:
            if capability not in declared or capability not in self.implemented_capabilities:
                raise UnsupportedCapabilityError(
                    f"engine {self.manifest.id!r} cannot apply {context}: "
                    f"missing capability {capability.value!r}"
                )

    def _require_language(self, language: str) -> None:
        if not language or not language.strip():
            raise UnsupportedLanguageError("language must not be empty")
        if not self.manifest.languages:
            return
        requested = language.casefold()
        available = {item.casefold() for item in self.manifest.languages}
        if requested not in available:
            choices = ", ".join(self.manifest.languages)
            raise UnsupportedLanguageError(
                f"engine {self.manifest.id!r} does not declare language {language!r}; "
                f"available: {choices}"
            )

    @abstractmethod
    def _synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Execute a request after common validation has succeeded."""


def _is_concrete_language(language: str) -> bool:
    return language.casefold() not in {"auto", "und", "mixed", "multilingual"}
