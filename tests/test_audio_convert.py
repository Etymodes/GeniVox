from __future__ import annotations

import io
import json
import math
import subprocess
import tempfile
import unittest
import urllib.request
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from genivox.audio.convert import AudioConversionError, ffmpeg_executable, normalized_audio
from genivox.audio.wav import read_pcm_wav
from genivox.core.models import Capability, EngineManifest, EngineTransport, SynthesisRequest
from genivox.engines.gpt_sovits import GptSovitsV2HttpAdapter


def _write_tone(path: Path, *, seconds: float = 4.0, channels: int = 1) -> None:
    rate = 32_000
    times = np.arange(int(rate * seconds)) / rate
    tone = (0.2 * np.sin(2 * math.pi * 220 * times) * 32767).astype("<i2")
    samples = np.repeat(tone[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def _encode(source: Path, destination: Path, codec: str) -> None:
    process = subprocess.run(
        [ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(source), "-c:a", codec, str(destination)],
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise AssertionError(process.stderr.decode("utf-8", errors="replace"))


def _response_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 160)
    return buffer.getvalue()


class AudioConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preserves_valid_mono_pcm16_wav(self) -> None:
        source = self.root / "sample.wav"
        _write_tone(source)
        with normalized_audio(source) as normalized:
            self.assertEqual(normalized, source.resolve())
            self.assertEqual(read_pcm_wav(normalized).sample_width_bytes, 2)
        self.assertTrue(source.exists())

    def test_decodes_common_formats_and_cleans_temporary_wav(self) -> None:
        source = self.root / "original.wav"
        _write_tone(source)
        for suffix, codec in ((".m4a", "aac"), (".mp3", "libmp3lame"),
                              (".flac", "flac"), (".ogg", "libvorbis")):
            with self.subTest(suffix=suffix):
                encoded = self.root / f"语音 样本{suffix}"
                _encode(source, encoded, codec)
                with normalized_audio(encoded) as normalized:
                    self.assertTrue(normalized.is_file())
                    self.assertEqual(normalized.suffix, ".wav")
                    self.assertNotEqual(normalized, encoded)
                    decoded = read_pcm_wav(normalized)
                    self.assertEqual((decoded.sample_rate, decoded.channels,
                                      decoded.sample_width_bytes), (32_000, 1, 2))
                    self.assertGreater(decoded.duration_seconds, 3.5)
                    self.assertLess(decoded.duration_seconds, 4.2)
                self.assertFalse(normalized.exists())
                self.assertTrue(encoded.is_file())

    def test_downmixes_stereo_wav(self) -> None:
        source = self.root / "stereo.wav"
        _write_tone(source, channels=2)
        with normalized_audio(source) as normalized:
            self.assertNotEqual(normalized, source)
            self.assertEqual(read_pcm_wav(normalized).channels, 1)
        self.assertFalse(normalized.exists())

    def test_rejects_corrupt_unsupported_and_overlong_audio(self) -> None:
        corrupt = self.root / "bad.m4a"
        corrupt.write_bytes(b"not audio")
        with self.assertRaisesRegex(AudioConversionError, "Could not decode bad.m4a"):
            with normalized_audio(corrupt):
                pass
        unknown = self.root / "recording.txt"
        unknown.write_bytes(b"not audio")
        with self.assertRaisesRegex(AudioConversionError, "Unsupported audio format"):
            with normalized_audio(unknown):
                pass
        long_audio = self.root / "long.wav"
        _write_tone(long_audio, seconds=4.0)
        with self.assertRaisesRegex(AudioConversionError, "overlong"):
            with normalized_audio(long_audio, max_duration_seconds=3):
                pass

    def test_gpt_adapter_keeps_decoded_m4a_alive_until_http_returns(self) -> None:
        source = self.root / "source.wav"
        recording = self.root / "我的录音.m4a"
        _write_tone(source)
        _encode(source, recording, "aac")
        adapter = GptSovitsV2HttpAdapter(EngineManifest(
            id="gpt", name="GPT-SoVITS", transport=EngineTransport.HTTP,
            capabilities=[Capability.VOICE_CLONE], languages=["zh"],
            endpoint="http://127.0.0.1:9880/tts",
        ))
        received: list[Path] = []

        def fake_http(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
            payload = json.loads(request.data or b"{}")
            decoded = Path(payload["ref_audio_path"])
            received.append(decoded)
            self.assertTrue(decoded.is_file())
            self.assertEqual((read_pcm_wav(decoded).channels, read_pcm_wav(decoded).sample_width_bytes),
                             (1, 2))
            return io.BytesIO(_response_wav())

        with patch("genivox.engines.gpt_sovits._open_local_http", side_effect=fake_http):
            result = adapter.synthesize(SynthesisRequest(
                text="你好", output_path=self.root / "output.wav", engine_id="gpt",
                language="zh", reference_audio=recording, prompt_text="我的录音",
                extra={"prompt_lang": "zh"},
            ))
        self.assertEqual(len(received), 1)
        self.assertFalse(received[0].exists())
        self.assertTrue(recording.exists())
        self.assertTrue(result.output_path.exists())
