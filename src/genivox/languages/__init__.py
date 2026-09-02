"""Multilingual text routing and optional phonemization frontends."""

from .phonemizer import (
    EspeakNgPhonemizer,
    Phonemizer,
    PhonemizerError,
    PhonemizerUnavailableError,
    UnsupportedLanguageError,
)
from .router import LanguageRouter, segment_text

__all__ = [
    "EspeakNgPhonemizer",
    "LanguageRouter",
    "Phonemizer",
    "PhonemizerError",
    "PhonemizerUnavailableError",
    "UnsupportedLanguageError",
    "segment_text",
]
