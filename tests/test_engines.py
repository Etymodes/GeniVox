from __future__ import annotations

import io
import json
import sys
import tempfile
import textwrap
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path
from unittest.mock import patch

import genivox.engines.gpt_sovits as gpt_sovits_module
from genivox.core.models import (
    Capability,
    EngineManifest,
    EngineTransport,
    LanguageSegment,
    SynthesisRequest,
    SynthesisResult,
)
from genivox.engines import (
    EngineAdapter,
    EngineConfigurationError,
    EngineExecutionError,
    EngineRegistry,
    GptSovitsV2HttpAdapter,
    InvalidSynthesisRequest,
    JsonProcessAdapter,
    MockWavAdapter,
    SynthesisPipeline,
    UnsupportedCapabilityError,
    UnsupportedLanguageError,
)


def _manifest(
    *,
    engine_id: str = "test",
    transport: EngineTransport = EngineTransport.MOCK,
    capabilities: list[Capability] | None = None,
    languages: list[str] | None = None,
    **kwargs: object,
) -> EngineManifest:
    return EngineManifest(
        id=engine_id,
        name="Test engine",
        transport=transport,
        capabilities=capabilities or [],
        languages=languages or [],
        **kwargs,
    )


def _write_wav(path: Path, *, frames: int = 80, rate: int = 8_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x00" * frames)


def _wav_bytes(*, frames: int = 80, rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


class EngineValidationTests(unittest.TestCase):
    def test_unsupported_controls_are_not_silently_ignored(self) -> None:
        adapter = MockWavAdapter(
            _manifest(capabilities=[Capability.CROSS_LINGUAL, Capability.SPEED])
        )
        with tempfile.TemporaryDirectory() as directory:
            base = dict(
                text="salve",
                output_path=Path(directory) / "out.wav",
                engine_id="test",
            )
            with self.assertRaisesRegex(UnsupportedCapabilityError, "emotion_vector"):
                adapter.synthesize(SynthesisRequest(**base, emotion={"happy": 0.8}))
            with self.assertRaisesRegex(UnsupportedCapabilityError, "style_instruction"):
                adapter.synthesize(SynthesisRequest(**base, style_instruction="whisper"))

    def test_adapter_rejects_overclaimed_manifest_capability(self) -> None:
        manifest = _manifest(capabilities=[Capability.EMOTION_VECTOR])
        with self.assertRaisesRegex(EngineConfigurationError, "emotion_vector"):
            MockWavAdapter(manifest)

    def test_request_validation_rejects_bad_identity_speed_and_reference(self) -> None:
        adapter = MockWavAdapter(_manifest(capabilities=[Capability.SPEED]))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "out.wav"
            with self.assertRaises(InvalidSynthesisRequest):
                adapter.synthesize(
                    SynthesisRequest(text=" ", output_path=destination, engine_id="test")
                )
            with self.assertRaises(InvalidSynthesisRequest):
                adapter.synthesize(
                    SynthesisRequest(text="x", output_path=destination, engine_id="wrong")
                )
            for speed in (0.0, -1.0):
                with self.assertRaises(InvalidSynthesisRequest):
                    adapter.synthesize(
                        SynthesisRequest(
                            text="x", output_path=destination, engine_id="test", speed=speed
                        )
                    )
            with self.assertRaises(InvalidSynthesisRequest):
                adapter.synthesize(
                    SynthesisRequest(
                        text="x",
                        output_path=destination,
                        engine_id="test",
                        reference_audio=Path(directory) / "missing.wav",
                    )
                )

    def test_language_is_explicitly_validated(self) -> None:
        adapter = MockWavAdapter(
            _manifest(capabilities=[Capability.CROSS_LINGUAL], languages=["la", "grc"])
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(UnsupportedLanguageError, "ru"):
                adapter.synthesize(
                    SynthesisRequest(
                        text="Привет",
                        output_path=Path(directory) / "out.wav",
                        engine_id="test",
                        language="ru",
                    )
                )


class RegistryTests(unittest.TestCase):
    def test_load_registry_and_create_adapters(self) -> None:
        document = {
            "schema_version": 1,
            "engines": [
                {
                    "id": "mock",
                    "name": "Mock",
                    "transport": "mock",
                    "capabilities": ["speed"],
                    "languages": ["en"],
                },
                {
                    "id": "gpt",
                    "name": "GPT-SoVITS",
                    "transport": "http",
                    "capabilities": ["voice_clone", "cross_lingual", "speed"],
                    "languages": ["zh", "en"],
                    "endpoint": "http://127.0.0.1:9880/tts",
                    "checkpoint_dir": "weights",
                    "metadata": {"adapter": "gpt_sovits_v2"},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            registry = EngineRegistry.load(path)

            self.assertEqual([item.id for item in registry], ["mock", "gpt"])
            self.assertIsInstance(registry.create_adapter("mock"), MockWavAdapter)
            self.assertIsInstance(registry.create_adapter("gpt"), GptSovitsV2HttpAdapter)
            self.assertEqual(
                registry.get_manifest("gpt").checkpoint_dir,
                str((Path(directory) / "weights").resolve()),
            )

    def test_registry_rejects_bad_schema_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text('{"schema_version": 2, "engines": []}', encoding="utf-8")
            with self.assertRaisesRegex(EngineConfigurationError, "schema_version"):
                EngineRegistry.load(path)

            record = {"id": "same", "name": "Same", "transport": "mock"}
            path.write_text(
                json.dumps({"schema_version": 1, "engines": [record, record]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EngineConfigurationError, "duplicate"):
                EngineRegistry.load(path)

    def test_registry_wraps_non_utf8_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_bytes(b"\xff\xfe\x00")

            with self.assertRaisesRegex(EngineConfigurationError, "could not read"):
                EngineRegistry.load(path)

    def test_save_upsert_and_remove_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            registry = EngineRegistry()
            registry.upsert(
                _manifest(
                    engine_id="mock-one",
                    capabilities=[Capability.SPEED],
                )
            )
            registry.save(path)

            loaded = EngineRegistry.load(path)
            self.assertEqual(loaded.get_manifest("mock-one").name, "Test engine")
            self.assertEqual(loaded.remove("mock-one").id, "mock-one")
            loaded.save(path)
            self.assertEqual(EngineRegistry.load(path).list_manifests(), ())

    def test_registry_rejects_traversal_like_engine_id(self) -> None:
        registry = EngineRegistry()
        with self.assertRaisesRegex(EngineConfigurationError, "engine id"):
            registry.register(_manifest(engine_id="../outside"))
        with self.assertRaisesRegex(EngineConfigurationError, "reserved Windows"):
            registry.register(_manifest(engine_id="CON"))

    def test_registry_wraps_malformed_path_field_as_configuration_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "engines.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "engines": [
                            {
                                "id": "bad",
                                "name": "Bad",
                                "transport": "mock",
                                "root": 123,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EngineConfigurationError, "item 0"):
                EngineRegistry.load(path)


class MockAdapterTests(unittest.TestCase):
    def test_writes_valid_wav_and_applies_speed(self) -> None:
        adapter = MockWavAdapter(_manifest(capabilities=[Capability.SPEED]))
        with tempfile.TemporaryDirectory() as directory:
            slow = Path(directory) / "nested" / "slow.wav"
            fast = Path(directory) / "fast.wav"
            slow_result = adapter.synthesize(
                SynthesisRequest(text="abcdef", output_path=slow, engine_id="test", speed=0.5)
            )
            fast_result = adapter.synthesize(
                SynthesisRequest(text="abcdef", output_path=fast, engine_id="test", speed=2.0)
            )
            with wave.open(str(slow), "rb") as slow_wav, wave.open(str(fast), "rb") as fast_wav:
                self.assertEqual(slow_wav.getnchannels(), 1)
                self.assertEqual(slow_wav.getsampwidth(), 2)
                self.assertEqual(slow_wav.getframerate(), 16_000)
                self.assertEqual(slow_wav.getnframes(), fast_wav.getnframes() * 4)
            self.assertGreater(slow_result.duration_seconds or 0, fast_result.duration_seconds or 0)


class GptSovitsAdapterTests(unittest.TestCase):
    def _adapter(self) -> GptSovitsV2HttpAdapter:
        return GptSovitsV2HttpAdapter(
            _manifest(
                engine_id="gpt",
                transport=EngineTransport.HTTP,
                capabilities=[Capability.VOICE_CLONE, Capability.CROSS_LINGUAL, Capability.SPEED],
                languages=["zh", "en"],
                endpoint="http://127.0.0.1:9880/tts",
                metadata={"timeout_seconds": 9},
            )
        )

    def test_maps_gpt_sovits_v2_request_fields(self) -> None:
        adapter = self._adapter()
        captured: dict[str, object] = {}

        def fake_urlopen(request: urllib.request.Request, *, timeout: float) -> io.BytesIO:
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data or b"{}")
            captured["timeout"] = timeout
            return io.BytesIO(_wav_bytes())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "我的参考.wav"
            reference.write_bytes(b"reference")
            output = root / "output.wav"
            request = SynthesisRequest(
                text="Hello, κόσμε!",
                output_path=output,
                engine_id="gpt",
                language="en",
                reference_audio=reference,
                prompt_text="你好",
                speed=1.25,
                seed=42,
                extra={"prompt_lang": "zh", "top_k": 7, "unsupported_option": "ignored"},
            )
            with patch("genivox.engines.gpt_sovits._open_local_http", side_effect=fake_urlopen):
                result = adapter.synthesize(request)

            self.assertEqual(captured["url"], "http://127.0.0.1:9880/tts")
            self.assertEqual(captured["timeout"], 9)
            payload = captured["payload"]
            assert isinstance(payload, dict)
            self.assertEqual(payload["ref_audio_path"], str(reference))
            self.assertEqual(payload["prompt_text"], "你好")
            self.assertEqual(payload["prompt_lang"], "zh")
            self.assertEqual(payload["text_lang"], "en")
            self.assertEqual(payload["speed_factor"], 1.25)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["top_k"], 7)
            self.assertNotIn("unsupported_option", payload)
            self.assertEqual(output.read_bytes(), _wav_bytes())
            self.assertEqual(result.output_path, output)

    def test_rejects_emotion_and_style_instead_of_dropping_them(self) -> None:
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            base = dict(
                text="hello",
                output_path=Path(directory) / "out.wav",
                engine_id="gpt",
                language="en",
            )
            with self.assertRaisesRegex(UnsupportedCapabilityError, "emotion_vector"):
                adapter.synthesize(SynthesisRequest(**base, emotion={"happy": 1.0}))
            with self.assertRaisesRegex(UnsupportedCapabilityError, "style_instruction"):
                adapter.synthesize(SynthesisRequest(**base, style_instruction="angry"))

    def test_http_error_is_reported_and_does_not_create_output(self) -> None:
        adapter = self._adapter()
        error = urllib.error.HTTPError(
            adapter.manifest.endpoint or "",
            400,
            "bad request",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"bad language"}'),
        )
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.wav"
            reference.write_bytes(b"reference")
            output = Path(directory) / "out.wav"
            request = SynthesisRequest(
                text="hello",
                output_path=output,
                engine_id="gpt",
                language="en",
                reference_audio=reference,
                extra={"prompt_lang": "en"},
            )
            with patch("genivox.engines.gpt_sovits._open_local_http", side_effect=error):
                with self.assertRaisesRegex(EngineExecutionError, "bad language"):
                    adapter.synthesize(request)
            self.assertFalse(output.exists())

    def test_reference_audio_and_prompt_language_are_required_before_http(self) -> None:
        adapter = self._adapter()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.wav"
            with self.assertRaisesRegex(EngineExecutionError, "reference audio"):
                adapter.synthesize(
                    SynthesisRequest(
                        text="hello",
                        output_path=output,
                        engine_id="gpt",
                        language="en",
                        extra={"prompt_lang": "en"},
                    )
                )

    def test_http_adapter_rejects_non_loopback_registry_endpoint(self) -> None:
        with self.assertRaisesRegex(EngineConfigurationError, "loopback"):
            GptSovitsV2HttpAdapter(
                _manifest(
                    engine_id="gpt",
                    transport=EngineTransport.HTTP,
                    endpoint="https://example.com/tts",
                )
            )

    def test_local_http_opener_ignores_proxies_and_rejects_redirects(self) -> None:
        response = io.BytesIO(_wav_bytes())
        opener = unittest.mock.Mock()
        opener.open.return_value = response
        request = urllib.request.Request("http://127.0.0.1:9880/tts")

        with patch.object(urllib.request, "build_opener", return_value=opener) as build_opener:
            returned = gpt_sovits_module._open_local_http(request, timeout=3.0)

        handlers = build_opener.call_args.args
        self.assertIsInstance(handlers[0], urllib.request.ProxyHandler)
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsInstance(handlers[1], gpt_sovits_module._RejectRedirects)
        opener.open.assert_called_once_with(request, timeout=3.0)
        self.assertIs(returned, response)

    def test_http_response_limit_is_enforced(self) -> None:
        adapter = self._adapter()
        adapter.manifest.metadata["max_response_bytes"] = 16
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            reference.write_bytes(b"reference")
            request = SynthesisRequest(
                text="hello",
                output_path=root / "out.wav",
                engine_id="gpt",
                language="en",
                reference_audio=reference,
                extra={"prompt_lang": "en"},
            )
            with patch(
                "genivox.engines.gpt_sovits._open_local_http",
                return_value=io.BytesIO(_wav_bytes()),
            ):
                with self.assertRaisesRegex(EngineExecutionError, "byte limit"):
                    adapter.synthesize(request)


class ProcessAdapterTests(unittest.TestCase):
    def test_process_adapter_requires_explicit_trust(self) -> None:
        with self.assertRaisesRegex(EngineConfigurationError, "not trusted"):
            JsonProcessAdapter(
                _manifest(
                    engine_id="process",
                    transport=EngineTransport.PROCESS,
                    command=["bridge.py"],
                )
            )

    def test_json_process_round_trip(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import pathlib
            import sys

            request = json.load(sys.stdin)
            output = pathlib.Path(request["output_path"])
            import wave
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(b"\\x00\\x00" * 80)
            print(json.dumps({
                "output_path": str(output),
                "duration_seconds": 0.01,
                "metadata": {"received_text": request["text"]},
            }, ensure_ascii=False))
            """
        )
        manifest = _manifest(
            engine_id="process",
            transport=EngineTransport.PROCESS,
            capabilities=[Capability.EMOTION_VECTOR, Capability.STYLE_INSTRUCTION],
            command=["-c", script],
            python=sys.executable,
            metadata={"timeout_seconds": 2, "trusted_local_code": True},
        )
        adapter = JsonProcessAdapter(manifest)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.wav"
            result = adapter.synthesize(
                SynthesisRequest(
                    text="χαῖρε",
                    output_path=output,
                    engine_id="process",
                    language="grc",
                    emotion={"calm": 0.7},
                    style_instruction="measured recitation",
                )
            )
            self.assertEqual(result.duration_seconds, 0.01)
            self.assertEqual(result.metadata["received_text"], "χαῖρε")
            self.assertTrue(output.is_file())

    def test_timeout_and_nonzero_exit_are_actionable(self) -> None:
        timeout_adapter = JsonProcessAdapter(
            _manifest(
                engine_id="process",
                transport=EngineTransport.PROCESS,
                command=["-c", "import time; time.sleep(1)"],
                python=sys.executable,
                metadata={"timeout_seconds": 0.02, "trusted_local_code": True},
            )
        )
        error_adapter = JsonProcessAdapter(
            _manifest(
                engine_id="process",
                transport=EngineTransport.PROCESS,
                command=["-c", "import sys; print('model missing', file=sys.stderr); sys.exit(3)"],
                python=sys.executable,
                metadata={"trusted_local_code": True},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="test",
                output_path=Path(directory) / "out.wav",
                engine_id="process",
            )
            with self.assertRaisesRegex(EngineExecutionError, "timed out"):
                timeout_adapter.synthesize(request)
            with self.assertRaisesRegex(EngineExecutionError, "model missing"):
                error_adapter.synthesize(request)

    def test_process_must_return_requested_output_path(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import pathlib
            import sys
            import wave

            request = json.load(sys.stdin)
            output = pathlib.Path(request["output_path"]).with_name("other.wav")
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(b"\\x00\\x00" * 80)
            print(json.dumps({"output_path": str(output)}))
            """
        )
        adapter = JsonProcessAdapter(
            _manifest(
                engine_id="process",
                transport=EngineTransport.PROCESS,
                command=["-c", script],
                python=sys.executable,
                metadata={"trusted_local_code": True},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="test",
                output_path=Path(directory) / "requested.wav",
                engine_id="process",
            )
            with self.assertRaisesRegex(EngineExecutionError, "different"):
                adapter.synthesize(request)

    def test_process_duration_must_match_nonempty_pcm_output(self) -> None:
        script = textwrap.dedent(
            """
            import json
            import pathlib
            import sys
            import wave

            request = json.load(sys.stdin)
            output = pathlib.Path(request["output_path"])
            with wave.open(str(output), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(8000)
                wav_file.writeframes(b"\\x00\\x00" * 80)
            print(json.dumps({"output_path": str(output), "duration_seconds": 9.0}))
            """
        )
        adapter = JsonProcessAdapter(
            _manifest(
                engine_id="process",
                transport=EngineTransport.PROCESS,
                command=["-c", script],
                python=sys.executable,
                metadata={"trusted_local_code": True},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="test",
                output_path=Path(directory) / "out.wav",
                engine_id="process",
            )
            with self.assertRaisesRegex(EngineExecutionError, "does not match"):
                adapter.synthesize(request)


class _RecordingWavAdapter(EngineAdapter):
    def __init__(self, manifest: EngineManifest) -> None:
        self.requests: list[SynthesisRequest] = []
        super().__init__(manifest)

    @property
    def implemented_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.VOICE_CLONE, Capability.CROSS_LINGUAL})

    def _synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.requests.append(request)
        _write_wav(request.output_path, frames=80)
        return SynthesisResult(
            output_path=request.output_path,
            engine_id=self.manifest.id,
            duration_seconds=0.01,
        )


class PipelineTests(unittest.TestCase):
    def test_uses_one_adapter_and_reference_for_all_languages_then_joins_pcm(self) -> None:
        manifest = _manifest(
            engine_id="polyglot",
            capabilities=[Capability.VOICE_CLONE, Capability.CROSS_LINGUAL],
            languages=["la", "grc", "ru"],
        )
        adapter = _RecordingWavAdapter(manifest)
        pipeline = SynthesisPipeline(adapter)
        segments = [
            LanguageSegment("Salve.", "la", 0, 6),
            LanguageSegment("Χαῖρε.", "grc", 7, 13),
            LanguageSegment("Привет.", "ru", 14, 21),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "voice.wav"
            _write_wav(reference)
            output = root / "joined.wav"
            result = pipeline.synthesize(
                SynthesisRequest(
                    text="Salve. Χαῖρε. Привет.",
                    output_path=output,
                    engine_id="polyglot",
                    segments=segments,
                    reference_audio=reference,
                )
            )

            self.assertEqual([item.language for item in adapter.requests], ["la", "grc", "ru"])
            self.assertTrue(all(item.reference_audio == reference for item in adapter.requests))
            self.assertEqual(len({id(adapter) for _ in adapter.requests}), 1)
            with wave.open(str(output), "rb") as joined:
                self.assertEqual(joined.getnframes(), 240)
                self.assertEqual(joined.getframerate(), 8_000)
            self.assertEqual(result.duration_seconds, 0.03)

    def test_segment_text_must_match_source_offsets(self) -> None:
        adapter = _RecordingWavAdapter(
            _manifest(engine_id="polyglot", languages=["en"])
        )
        pipeline = SynthesisPipeline(adapter)
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="Text recorded in the ledger",
                output_path=Path(directory) / "out.wav",
                engine_id="polyglot",
                segments=[LanguageSegment("different text", "en", 0, 14)],
            )

            with self.assertRaisesRegex(InvalidSynthesisRequest, "does not match"):
                pipeline.synthesize(request)
            self.assertEqual(adapter.requests, [])

    def test_unsupported_segment_language_fails_without_fallback(self) -> None:
        manifest = _manifest(
            engine_id="polyglot",
            capabilities=[Capability.CROSS_LINGUAL],
            languages=["la", "grc"],
        )
        adapter = _RecordingWavAdapter(manifest)
        pipeline = SynthesisPipeline(adapter)
        with tempfile.TemporaryDirectory() as directory:
            request = SynthesisRequest(
                text="Привет",
                output_path=Path(directory) / "out.wav",
                engine_id="polyglot",
                segments=[LanguageSegment("Привет", "ru", 0, 6)],
            )
            with self.assertRaisesRegex(UnsupportedLanguageError, "ru"):
                pipeline.synthesize(request)
            self.assertEqual(adapter.requests, [])

    def test_punctuation_outside_explicit_tags_is_attached_to_spoken_segments(self) -> None:
        adapter = _RecordingWavAdapter(
            _manifest(
                engine_id="polyglot",
                capabilities=[Capability.CROSS_LINGUAL],
                languages=["la", "grc", "ru"],
            )
        )
        segments = [
            LanguageSegment("Salve", "la", 0, 5, source="explicit"),
            LanguageSegment(": ", "und", 5, 7, source="neutral"),
            LanguageSegment("χαῖρε", "grc", 7, 12, source="explicit"),
            LanguageSegment(". ", "und", 12, 14, source="neutral"),
            LanguageSegment("Привет", "ru", 14, 20, source="explicit"),
            LanguageSegment("!", "und", 20, 21, source="neutral"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            SynthesisPipeline(adapter).synthesize(
                SynthesisRequest(
                    text="Salve: χαῖρε. Привет!",
                    output_path=Path(directory) / "out.wav",
                    engine_id="polyglot",
                    segments=segments,
                )
            )

        self.assertEqual([item.language for item in adapter.requests], ["la", "grc", "ru"])
        self.assertEqual([item.text for item in adapter.requests], ["Salve: ", "χαῖρε. ", "Привет!"])


if __name__ == "__main__":
    unittest.main()
