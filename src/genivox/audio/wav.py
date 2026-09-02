from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class UnsupportedWavError(ValueError):
    """Raised when a WAV file is not uncompressed integer PCM."""


@dataclass(slots=True)
class PcmWav:
    """Decoded PCM WAV samples, normalized to approximately ``[-1, 1]``.

    ``samples`` always has shape ``(frames, channels)``. Keeping the channel
    axis avoids surprising callers while :attr:`mono` supplies the mixdown
    used by the lightweight analyzers.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    sample_width_bytes: int
    source_path: Path

    @property
    def frame_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate

    @property
    def mono(self) -> np.ndarray:
        if self.channels == 1:
            return self.samples[:, 0]
        return np.mean(self.samples, axis=1, dtype=np.float64).astype(np.float32)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        octets = np.frombuffer(raw, dtype=np.uint8)
        if octets.size % 3:
            raise UnsupportedWavError("Malformed 24-bit PCM WAV: incomplete sample")
        octets = octets.reshape(-1, 3).astype(np.int32)
        values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8_388_608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    raise UnsupportedWavError(
        f"Unsupported PCM sample width: {sample_width * 8} bits; expected 8, 16, 24, or 32"
    )


def read_pcm_wav(
    path: str | Path,
    *,
    max_duration_seconds: float = 300.0,
    max_pcm_bytes: int = 256 * 1024 * 1024,
) -> PcmWav:
    """Read an uncompressed integer PCM WAV using only ``wave`` and NumPy."""

    if max_duration_seconds <= 0 or max_pcm_bytes <= 0:
        raise ValueError("WAV analysis limits must be greater than zero")
    source_path = Path(path)
    try:
        with wave.open(str(source_path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise UnsupportedWavError(
                    f"Compressed WAV is unsupported ({handle.getcomptype()}): {source_path}"
                )
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            if channels <= 0 or sample_rate <= 0:
                raise UnsupportedWavError(f"Invalid WAV header: {source_path}")
            duration_seconds = frame_count / sample_rate
            if duration_seconds > max_duration_seconds:
                raise UnsupportedWavError(
                    f"WAV is {duration_seconds:.1f}s; analysis limit is "
                    f"{max_duration_seconds:.1f}s"
                )
            declared_bytes = frame_count * channels * sample_width
            if declared_bytes > max_pcm_bytes:
                raise UnsupportedWavError(
                    f"WAV declares {declared_bytes} PCM bytes; analysis limit is {max_pcm_bytes}"
                )
            raw = handle.readframes(frame_count)
    except (EOFError, wave.Error) as exc:
        raise UnsupportedWavError(f"Cannot read PCM WAV {source_path}: {exc}") from exc

    decoded = _decode_pcm(raw, sample_width)
    expected_samples = frame_count * channels
    if decoded.size != expected_samples:
        raise UnsupportedWavError(
            f"Malformed PCM WAV: header declares {expected_samples} samples, read {decoded.size}"
        )
    samples = decoded.reshape(frame_count, channels)
    return PcmWav(samples, sample_rate, channels, sample_width, source_path)
