from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from genivox.audio.emotion import EmotionAnalyzer, NoOpEmotionAnalyzer
from genivox.audio.wav import PcmWav, read_pcm_wav
from genivox.core.models import ProsodyProfile

_DBFS_FLOOR = -120.0


@dataclass(frozen=True, slots=True)
class AudioQualityThresholds:
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 30.0
    min_sample_rate: int = 16_000
    clipping_level: float = 0.995
    max_clipping_ratio: float = 0.001
    silence_threshold_dbfs: float = -45.0
    max_silence_ratio: float = 0.60
    min_snr_proxy_db: float = 12.0


@dataclass(slots=True)
class AudioQualityReport:
    duration_seconds: float
    sample_rate: int
    channels: int
    sample_width_bits: int
    peak_dbfs: float
    rms_dbfs: float
    clipping_ratio: float
    silence_ratio: float
    snr_proxy_db: float | None
    is_acceptable: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FrameProsody:
    """Frame-level acoustic measurements; unvoiced F0 values are ``NaN``."""

    times_seconds: np.ndarray
    rms: np.ndarray
    zero_crossing_rate: np.ndarray
    f0_hz: np.ndarray


def _dbfs(amplitude: float) -> float:
    if amplitude <= 0.0:
        return _DBFS_FLOOR
    return max(_DBFS_FLOOR, 20.0 * math.log10(amplitude))


def _frames(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if samples.size == 0:
        return np.empty((0, frame_size), dtype=np.float32)
    frame_count = max(1, 1 + math.ceil(max(0, samples.size - frame_size) / hop_size))
    padded_size = (frame_count - 1) * hop_size + frame_size
    padded = np.pad(samples, (0, padded_size - samples.size))
    shape = (frame_count, frame_size)
    strides = (padded.strides[0] * hop_size, padded.strides[0])
    return np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides).copy()


def _frame_rms(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    frame_size = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(sample_rate * hop_ms / 1000.0))
    framed = _frames(samples, frame_size, hop_size).astype(np.float64)
    if framed.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sqrt(np.mean(np.square(framed), axis=1))


def check_audio_quality(
    audio: PcmWav,
    thresholds: AudioQualityThresholds | None = None,
) -> AudioQualityReport:
    """Evaluate reference-audio fitness using transparent signal proxies.

    ``snr_proxy_db`` contrasts loud active frames with plausible noise-only
    frames. It remains ``None`` when the clip has no usable quiet region and
    is not a calibrated laboratory SNR measurement.
    """

    limits = thresholds or AudioQualityThresholds()
    all_samples = audio.samples.astype(np.float64).reshape(-1)
    mono = audio.mono.astype(np.float64)
    peak = float(np.max(np.abs(all_samples))) if all_samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(all_samples)))) if all_samples.size else 0.0
    clipping_ratio = (
        float(np.mean(np.abs(all_samples) >= limits.clipping_level)) if all_samples.size else 0.0
    )

    frame_rms = _frame_rms(mono, audio.sample_rate)
    silence_level = 10.0 ** (limits.silence_threshold_dbfs / 20.0)
    silence_ratio = float(np.mean(frame_rms < silence_level)) if frame_rms.size else 1.0

    snr_proxy_db: float | None = None
    maximum_frame_rms = float(np.max(frame_rms)) if frame_rms.size else 0.0
    # Do not report a fake SNR when a clip contains no plausible noise-only
    # frames. A steady clean tone and steady noise are indistinguishable by
    # frame-energy contrast alone.
    noise_candidate_level = max(silence_level, maximum_frame_rms * 0.10)
    noise_candidates = frame_rms[frame_rms <= noise_candidate_level]
    active_candidates = frame_rms[frame_rms > noise_candidate_level]
    if noise_candidates.size >= 2 and active_candidates.size >= 2:
        active_powers = np.sort(np.square(active_candidates))
        group_size = max(1, math.ceil(active_powers.size * 0.2))
        noise_power = max(float(np.mean(np.square(noise_candidates))), 1e-12)
        loud_power = float(np.mean(active_powers[-group_size:]))
        signal_power = max(loud_power - noise_power, 1e-12)
        snr_proxy_db = float(np.clip(10.0 * math.log10(signal_power / noise_power), -120.0, 120.0))

    warnings: list[str] = []
    duration = audio.duration_seconds
    if duration < limits.min_duration_seconds:
        warnings.append(
            f"Audio is too short ({duration:.2f}s); target at least {limits.min_duration_seconds:.2f}s"
        )
    if duration > limits.max_duration_seconds:
        warnings.append(
            f"Audio is long ({duration:.2f}s); trim to at most {limits.max_duration_seconds:.2f}s"
        )
    if audio.sample_rate < limits.min_sample_rate:
        warnings.append(
            f"Sample rate is low ({audio.sample_rate}Hz); target at least {limits.min_sample_rate}Hz"
        )
    if clipping_ratio > limits.max_clipping_ratio:
        warnings.append(
            f"Clipping detected in {clipping_ratio:.2%} of samples; target at most "
            f"{limits.max_clipping_ratio:.2%}"
        )
    if silence_ratio > limits.max_silence_ratio:
        warnings.append(
            f"Silence occupies {silence_ratio:.1%} of frames; target at most "
            f"{limits.max_silence_ratio:.1%}"
        )
    if rms == 0.0:
        warnings.append("Audio contains no measurable signal")
    elif snr_proxy_db is not None and snr_proxy_db < limits.min_snr_proxy_db:
        warnings.append(
            f"Frame-energy SNR proxy is low ({snr_proxy_db:.1f}dB); target at least "
            f"{limits.min_snr_proxy_db:.1f}dB"
        )

    return AudioQualityReport(
        duration_seconds=duration,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        sample_width_bits=audio.sample_width_bytes * 8,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        snr_proxy_db=snr_proxy_db,
        is_acceptable=not warnings,
        warnings=warnings,
    )


def _estimate_f0(
    frame: np.ndarray,
    sample_rate: int,
    *,
    min_f0_hz: float,
    max_f0_hz: float,
    minimum_correlation: float,
) -> float:
    centered = frame.astype(np.float64) - float(np.mean(frame))
    windowed = centered * np.hanning(centered.size)
    energy = float(np.dot(windowed, windowed))
    if energy <= 1e-12:
        return math.nan

    fft_size = 1 << (2 * centered.size - 1).bit_length()
    spectrum = np.fft.rfft(windowed, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[: centered.size]
    autocorrelation /= max(float(autocorrelation[0]), 1e-12)

    minimum_lag = max(1, math.floor(sample_rate / max_f0_hz))
    maximum_lag = min(centered.size - 2, math.ceil(sample_rate / min_f0_hz))
    if minimum_lag > maximum_lag:
        return math.nan
    region = autocorrelation[minimum_lag : maximum_lag + 1]
    offset = int(np.argmax(region))
    lag = minimum_lag + offset
    if float(autocorrelation[lag]) < minimum_correlation:
        return math.nan

    if 1 <= lag < autocorrelation.size - 1:
        left, center, right = autocorrelation[lag - 1 : lag + 2]
        denominator = left - 2.0 * center + right
        if abs(float(denominator)) > 1e-12:
            lag += float(0.5 * (left - right) / denominator)
    return float(sample_rate / lag)


def extract_frame_prosody(
    audio: PcmWav,
    *,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    min_f0_hz: float = 50.0,
    max_f0_hz: float = 500.0,
) -> FrameProsody:
    """Extract RMS, zero-crossing rate, and autocorrelation F0 by frame."""

    if frame_ms <= 0.0 or hop_ms <= 0.0:
        raise ValueError("Frame and hop sizes must be positive")
    if not 0.0 < min_f0_hz < max_f0_hz:
        raise ValueError("F0 bounds must satisfy 0 < min_f0_hz < max_f0_hz")

    frame_size = max(2, round(audio.sample_rate * frame_ms / 1000.0))
    hop_size = max(1, round(audio.sample_rate * hop_ms / 1000.0))
    framed = _frames(audio.mono, frame_size, hop_size)
    if framed.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return FrameProsody(empty, empty.copy(), empty.copy(), empty.copy())

    float_frames = framed.astype(np.float64)
    rms = np.sqrt(np.mean(np.square(float_frames), axis=1))
    signs = np.signbit(float_frames)
    zero_crossing_rate = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    active_floor = max(10.0 ** (-45.0 / 20.0), float(np.percentile(rms, 90)) * 0.10)
    f0 = np.full(rms.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(rms >= active_floor):
        f0[index] = _estimate_f0(
            float_frames[index],
            audio.sample_rate,
            min_f0_hz=min_f0_hz,
            max_f0_hz=max_f0_hz,
            minimum_correlation=0.30,
        )
    times = (np.arange(rms.size, dtype=np.float64) * hop_size + frame_size / 2.0) / audio.sample_rate
    return FrameProsody(times, rms, zero_crossing_rate, f0)


def _estimate_acoustic_peak_rate(frames: FrameProsody, duration_seconds: float) -> float | None:
    if frames.rms.size < 3 or duration_seconds <= 0.0 or float(np.max(frames.rms)) <= 0.0:
        return None
    kernel = np.ones(5, dtype=np.float64) / 5.0
    envelope = np.convolve(frames.rms, kernel, mode="same")
    threshold = max(float(np.percentile(envelope, 35)), float(np.max(envelope)) * 0.15)
    candidates = np.flatnonzero(
        (envelope[1:-1] > envelope[:-2])
        & (envelope[1:-1] >= envelope[2:])
        & (envelope[1:-1] >= threshold)
    ) + 1
    selected: list[int] = []
    minimum_gap = 8  # 80 ms with the default 10 ms hop; only an acoustic rhythm proxy.
    for candidate in candidates:
        if not selected or candidate - selected[-1] >= minimum_gap:
            selected.append(int(candidate))
        elif envelope[candidate] > envelope[selected[-1]]:
            selected[-1] = int(candidate)
    return len(selected) / duration_seconds


def describe_acoustic_style(profile: ProsodyProfile) -> str:
    """Turn measurements into an explainable, non-semantic Chinese description.

    The output deliberately does not infer emotional states from pitch or
    loudness. A high pitch is not evidence of happiness, and a low RMS is not
    evidence of sadness.
    """

    parts = ["声学代理（不是情绪识别）"]
    if profile.rms_dbfs is not None:
        if profile.rms_dbfs < -30.0:
            level = "较轻"
        elif profile.rms_dbfs > -14.0:
            level = "较强"
        else:
            level = "中等"
        parts.append(f"整体响度{level}（RMS {profile.rms_dbfs:.1f} dBFS）")
    if profile.mean_f0_hz is not None:
        pitch = f"平均基频约 {profile.mean_f0_hz:.0f} Hz"
        if profile.f0_std_hz is not None:
            pitch += f"、帧间标准差约 {profile.f0_std_hz:.0f} Hz"
        parts.append(pitch)
    if profile.acoustic_peak_rate_hz is not None:
        rate = profile.acoustic_peak_rate_hz
        if rate < 3.0:
            pace = "节奏峰较疏"
        elif rate > 5.5:
            pace = "节奏峰较密"
        else:
            pace = "节奏峰密度中等"
        parts.append(f"{pace}（约 {rate:.1f} 个声学峰/秒）")
    if profile.voiced_ratio is not None:
        parts.append(f"可测基频帧占比约 {profile.voiced_ratio:.0%}")
    return "；".join(parts) + "。"


def analyze_prosody(
    path: str | Path,
    *,
    emotion_analyzer: EmotionAnalyzer | None = None,
    quality_thresholds: AudioQualityThresholds | None = None,
) -> ProsodyProfile:
    """Analyze one reference WAV and return the shared ``ProsodyProfile`` model."""

    audio = read_pcm_wav(path)
    quality = check_audio_quality(audio, quality_thresholds)
    frames = extract_frame_prosody(audio)
    voiced = frames.f0_hz[np.isfinite(frames.f0_hz)]
    analyzer = emotion_analyzer or NoOpEmotionAnalyzer()
    emotion = dict(
        analyzer.analyze(audio.mono, audio.sample_rate, source_path=audio.source_path)
    )
    warnings = list(quality.warnings)
    if not emotion:
        warnings.append(
            "Emotion was not inferred: configure a sidecar or external model; acoustic proxies are not labels"
        )

    profile = ProsodyProfile(
        duration_seconds=audio.duration_seconds,
        sample_rate=audio.sample_rate,
        mean_f0_hz=float(np.mean(voiced)) if voiced.size else None,
        f0_std_hz=float(np.std(voiced)) if voiced.size else None,
        f0_min_hz=float(np.min(voiced)) if voiced.size else None,
        f0_max_hz=float(np.max(voiced)) if voiced.size else None,
        rms_dbfs=quality.rms_dbfs,
        voiced_ratio=float(voiced.size / frames.f0_hz.size) if frames.f0_hz.size else None,
        acoustic_peak_rate_hz=_estimate_acoustic_peak_rate(
            frames, audio.duration_seconds
        ),
        emotion=emotion,
        warnings=warnings,
    )
    profile.style_instruction = describe_acoustic_style(profile)
    return profile
