from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from genivox.audio import (
    EmotionAnalysisError,
    ExternalEmotionAnalyzer,
    SidecarEmotionAnalyzer,
    analyze_prosody,
    check_audio_quality,
    describe_acoustic_style,
    extract_frame_prosody,
    read_pcm_wav,
)


def _write_pcm16(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    pcm = np.clip(np.rint(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class AudioAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reads_stereo_pcm_and_mixdown(self) -> None:
        path = self.root / "stereo.wav"
        left = np.full(160, 0.25)
        right = np.full(160, -0.25)
        _write_pcm16(path, np.column_stack((left, right)), sample_rate=8_000)

        audio = read_pcm_wav(path)

        self.assertEqual(audio.channels, 2)
        self.assertEqual(audio.samples.shape, (160, 2))
        self.assertAlmostEqual(audio.duration_seconds, 0.02)
        np.testing.assert_allclose(audio.mono, 0.0, atol=1e-4)

    def test_rejects_audio_above_analysis_limits_before_decoding(self) -> None:
        path = self.root / "long.wav"
        _write_pcm16(path, np.zeros(8_000), sample_rate=8_000)

        with self.assertRaisesRegex(ValueError, "analysis limit"):
            read_pcm_wav(path, max_duration_seconds=0.5)
        with self.assertRaisesRegex(ValueError, "PCM bytes"):
            read_pcm_wav(path, max_pcm_bytes=1_000)

    def test_extracts_autocorrelation_f0_and_acoustic_style(self) -> None:
        path = self.root / "tone.wav"
        sample_rate = 16_000
        duration = 2.0
        time = np.arange(round(sample_rate * duration)) / sample_rate
        envelope = 0.35 + 0.15 * np.sin(2.0 * math.pi * 4.0 * time)
        samples = envelope * np.sin(2.0 * math.pi * 220.0 * time)
        _write_pcm16(path, samples, sample_rate)

        audio = read_pcm_wav(path)
        frames = extract_frame_prosody(audio)
        profile = analyze_prosody(path)

        self.assertEqual(frames.rms.shape, frames.zero_crossing_rate.shape)
        self.assertGreater(float(np.nanmedian(frames.f0_hz)), 215.0)
        self.assertLess(float(np.nanmedian(frames.f0_hz)), 225.0)
        self.assertIsNotNone(profile.mean_f0_hz)
        self.assertAlmostEqual(profile.mean_f0_hz or 0.0, 220.0, delta=5.0)
        self.assertEqual(profile.emotion, {})
        self.assertIn("声学代理（不是情绪识别）", describe_acoustic_style(profile))
        self.assertTrue(any("Emotion was not inferred" in item for item in profile.warnings))
        self.assertIsNone(check_audio_quality(audio).snr_proxy_db)

    def test_quality_report_flags_clipping_and_silence(self) -> None:
        path = self.root / "bad.wav"
        samples = np.zeros(32_000)
        samples[24_000:28_000] = 1.0
        _write_pcm16(path, samples)

        report = check_audio_quality(read_pcm_wav(path))

        self.assertFalse(report.is_acceptable)
        self.assertGreater(report.clipping_ratio, 0.10)
        self.assertGreater(report.silence_ratio, 0.60)
        self.assertTrue(any("Clipping" in item for item in report.warnings))
        self.assertTrue(any("Silence" in item for item in report.warnings))

    def test_snr_proxy_requires_and_uses_quiet_regions(self) -> None:
        path = self.root / "room-tone.wav"
        generator = np.random.default_rng(42)
        noise = generator.normal(0.0, 0.001, 32_000)
        time = np.arange(16_000) / 16_000
        noise[8_000:24_000] += 0.15 * np.sin(2.0 * math.pi * 180.0 * time)
        _write_pcm16(path, noise)

        report = check_audio_quality(read_pcm_wav(path))

        self.assertIsNotNone(report.snr_proxy_db)
        self.assertGreater(report.snr_proxy_db or 0.0, 20.0)

    def test_sidecar_emotion_is_explicit_input_not_acoustic_guess(self) -> None:
        path = self.root / "voice.wav"
        _write_pcm16(path, np.zeros(16_000))
        sidecar = path.with_name(path.name + ".emotion.json")
        sidecar.write_text(json.dumps({"emotion": {"calm": 0.8, "happy": 0.15}}), encoding="utf-8")

        profile = analyze_prosody(path, emotion_analyzer=SidecarEmotionAnalyzer(strict=True))

        self.assertEqual(profile.emotion, {"calm": 0.8, "happy": 0.15})
        self.assertFalse(any("Emotion was not inferred" in item for item in profile.warnings))

    def test_external_emotion_json_bridge(self) -> None:
        path = self.root / "voice.wav"
        _write_pcm16(path, np.zeros(16_000))
        bridge = (
            "import json,sys; p=json.load(sys.stdin); "
            "assert p['audio_path'].endswith('voice.wav'); "
            "json.dump({'emotion': {'neutral': 0.9}}, sys.stdout)"
        )
        analyzer = ExternalEmotionAnalyzer([sys.executable, "-c", bridge], timeout_seconds=5)

        profile = analyze_prosody(path, emotion_analyzer=analyzer)

        self.assertEqual(profile.emotion, {"neutral": 0.9})

    def test_external_emotion_bridge_reports_invalid_output(self) -> None:
        path = self.root / "voice.wav"
        _write_pcm16(path, np.zeros(160))
        analyzer = ExternalEmotionAnalyzer([sys.executable, "-c", "print('not-json')"])

        with self.assertRaisesRegex(EmotionAnalysisError, "invalid JSON"):
            analyzer.analyze(np.zeros(160), 16_000, source_path=path)


if __name__ == "__main__":
    unittest.main()
