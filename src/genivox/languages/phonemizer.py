"""Pluggable phonemizer API and an optional eSpeak NG subprocess frontend."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable


class PhonemizerError(RuntimeError):
    """Base error raised by phonemizer frontends."""


class PhonemizerUnavailableError(PhonemizerError):
    """Raised when an optional phonemizer executable is not installed."""


class UnsupportedLanguageError(PhonemizerError):
    """Raised when a frontend has no mapping for the requested language."""


@runtime_checkable
class Phonemizer(Protocol):
    """Minimal interface implemented by text-to-phoneme frontends."""

    @property
    def available(self) -> bool: ...

    def supports(self, language: str) -> bool: ...

    def phonemize(self, text: str, language: str) -> str: ...


class EspeakNgPhonemizer:
    """Use an installed ``espeak-ng`` executable without a Python dependency."""

    VOICES = {"la": "la", "grc": "grc", "ru": "ru"}

    def __init__(self, executable: str | Path | None = None) -> None:
        configured = executable if executable is not None else os.environ.get("GENIVOX_ESPEAK_PATH")
        self._configured_executable = str(configured) if configured else None
        self._availability: bool | None = None

    @property
    def executable(self) -> str | None:
        if self._configured_executable:
            configured_path = Path(self._configured_executable)
            if configured_path.is_file() and os.access(configured_path, os.X_OK):
                return str(configured_path.resolve())
            return shutil.which(self._configured_executable)
        return shutil.which("espeak-ng") or shutil.which("espeak")

    @property
    def available(self) -> bool:
        executable = self.executable
        if executable is None:
            return False
        if self._availability is None:
            try:
                completed = subprocess.run(
                    [executable, "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                )
                version_output = (completed.stdout + completed.stderr).casefold()
                self._availability = completed.returncode == 0 and "speak" in version_output
            except (OSError, subprocess.TimeoutExpired):
                self._availability = False
        return self._availability

    def supports(self, language: str) -> bool:
        return language.casefold() in self.VOICES

    def phonemize(self, text: str, language: str) -> str:
        language = language.casefold()
        if not self.supports(language):
            raise UnsupportedLanguageError(
                f"eSpeak NG frontend supports {sorted(self.VOICES)}, not {language!r}"
            )
        executable = self.executable
        if executable is None:
            raise PhonemizerUnavailableError(
                "eSpeak NG was not found; install espeak-ng or pass its executable path"
            )
        try:
            completed = subprocess.run(
                [executable, "-q", "--ipa=3", "-v", self.VOICES[language], "--stdin"],
                check=True,
                capture_output=True,
                input=text,
                text=True,
                encoding="utf-8",
                timeout=15,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise PhonemizerError(f"eSpeak NG failed for language {language!r}: {error}") from error
        phonemes = completed.stdout.strip()
        if not phonemes:
            raise PhonemizerError(f"eSpeak NG returned no phonemes for language {language!r}")
        return phonemes
