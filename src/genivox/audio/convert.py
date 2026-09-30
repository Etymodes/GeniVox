"""Decode common reference recordings into a temporary PCM WAV for local engines."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from genivox.audio.wav import UnsupportedWavError, read_pcm_wav

SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg"})
_SAMPLE_RATE = 32_000
_MAX_DURATION_SECONDS = 300.0


class AudioConversionError(ValueError):
    """The selected audio cannot be decoded within the reference-audio limits."""


def ffmpeg_executable() -> str:
    # The package wheel includes FFmpeg on Windows, where it need not be on PATH.
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise AudioConversionError(
            "FFmpeg is unavailable; reinstall GeniVox to restore its decoder"
        ) from None


@contextmanager
def normalized_audio(
    path: str | Path, *, max_duration_seconds: float = 30.0
) -> Iterator[Path]:
    """Yield a readable mono 16-bit PCM WAV, deleting converted files on exit.

    Existing mono PCM16 WAVs are returned as-is. Other supported formats are
    decoded into a private temporary directory. Callers must use the path
    inside the context so a local engine can still read it during synthesis.
    """

    if not math.isfinite(max_duration_seconds) or not 0 < max_duration_seconds <= _MAX_DURATION_SECONDS:
        raise ValueError(
            f"max_duration_seconds must be between 0 and {_MAX_DURATION_SECONDS:g}"
        )
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise AudioConversionError(f"Audio file does not exist: {source}")
    if source.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
        raise AudioConversionError(
            f"Unsupported audio format {source.suffix or '(none)'}; choose WAV, FLAC, MP3, M4A, or OGG"
        )

    if source.suffix.casefold() in {".wav", ".wave"}:
        try:
            audio = read_pcm_wav(source, max_duration_seconds=max_duration_seconds)
            if audio.channels == 1 and audio.sample_width_bytes == 2 and audio.frame_count > 0:
                yield source
                return
        except (UnsupportedWavError, OSError):
            # A floating-point WAV, for example, still decodes through FFmpeg.
            pass

    with tempfile.TemporaryDirectory(prefix="genivox-audio-") as directory:
        converted = Path(directory) / "reference.wav"
        command = [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel", "error",
            "-nostdin",
            "-y",
            "-i", str(source),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-t", f"{max_duration_seconds + 1:.3f}",
            str(converted),
        ]
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=min(120.0, max(30.0, max_duration_seconds * 2)),
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AudioConversionError(f"Could not decode {source.name}: {exc}") from exc
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()[-600:]
            raise AudioConversionError(f"Could not decode {source.name}: {detail or 'FFmpeg failed'}")
        try:
            audio = read_pcm_wav(converted, max_duration_seconds=max_duration_seconds)
        except (UnsupportedWavError, OSError) as exc:
            raise AudioConversionError(f"Invalid or overlong audio in {source.name}: {exc}") from exc
        if audio.frame_count == 0:
            raise AudioConversionError(f"Audio has no playable samples: {source.name}")
        yield converted
