from __future__ import annotations

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from genivox.controller import (  # noqa: E402
    WorkbenchController,
    _dataset_audio_snapshot,
    _manifest_from_import,
    _resolve_dataset_manifest,
)
from genivox.core.models import (  # noqa: E402
    Capability,
    DatasetRecord,
    EngineManifest,
    EngineTransport,
    ProsodyProfile,
)
from genivox.core.paths import WorkspacePaths  # noqa: E402
from genivox.ui import MainWindow  # noqa: E402
from genivox.ui.theme import apply_theme  # noqa: E402


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        apply_theme(cls.app)

    def test_process_bundle_manifest_is_registered_without_running_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge = {
                "schema_version": 1,
                "id": "custom-local",
                "command": ["bridge.py"],
                "training_command": ["train.py", "--data", "{dataset_path}"],
                "capabilities": ["voice_clone", "cross_lingual", "fine_tune"],
                "languages": ["la", "grc", "ru"],
            }
            (root / "genivox-engine.json").write_text(json.dumps(bridge), encoding="utf-8")

            manifest = _manifest_from_import(
                {
                    "engine_type": "自定义适配器",
                    "name": "Classics",
                    "root": str(root),
                    "transport": "process",
                    "trusted_local_code": True,
                }
            )

            self.assertEqual(manifest.id, "custom-local")
            self.assertEqual(manifest.transport, EngineTransport.PROCESS)
            self.assertEqual(manifest.command, ["bridge.py"])
            self.assertIn(Capability.FINE_TUNE, manifest.capabilities)
            self.assertEqual(manifest.languages, ["la", "grc", "ru"])
            self.assertEqual(
                manifest.metadata["training_command"],
                ["train.py", "--data", "{dataset_path}"],
            )

    def test_controller_backs_up_invalid_registry_and_recovers_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspacePaths(Path(directory) / "workspace").ensure()
            registry_file = workspace.engines / "registry.json"
            registry_file.write_text("{bad json", encoding="utf-8")
            window = MainWindow()

            controller = WorkbenchController(window, workspace)
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            self.assertEqual(
                {manifest.id for manifest in controller.registry},
                {"mock-local", "gpt-sovits-v2-local"},
            )
            self.assertEqual(len(list(workspace.engines.glob("registry.invalid-*.json"))), 1)
            self.assertIn("已备份", window.model_manager_page.status_chip.text())
            window.close()

    def test_gpt_http_import_normalizes_tts_endpoint(self) -> None:
        manifest = _manifest_from_import(
            {
                "engine_type": "GPT-SoVITS",
                "name": "GPT local",
                "transport": "http",
                "endpoint": "http://127.0.0.1:9880",
            }
        )
        self.assertEqual(manifest.endpoint, "http://127.0.0.1:9880/tts")
        self.assertEqual(manifest.metadata["adapter"], "gpt_sovits_v2")

    def test_http_import_is_restricted_to_loopback(self) -> None:
        with self.assertRaisesRegex(ValueError, "本地模式"):
            _manifest_from_import(
                {
                    "engine_type": "GPT-SoVITS",
                    "name": "Remote",
                    "transport": "http",
                    "endpoint": "https://example.com/tts",
                }
            )

    def test_engine_presets_do_not_overclaim_official_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "root": str(root),
                "transport": "process",
                "trusted_local_code": True,
            }
            index = _manifest_from_import(
                {**common, "engine_type": "IndexTTS2.5", "name": "Index"}
            )
            vox = _manifest_from_import(
                {**common, "engine_type": "VoxCPM2", "name": "Vox"}
            )

            self.assertNotIn(Capability.FINE_TUNE, index.capabilities)
            self.assertEqual(index.languages, ["zh", "en", "ja", "es", "ar"])
            self.assertNotIn(Capability.SPEED, vox.capabilities)
            self.assertIn("ru", vox.languages)
            self.assertIn("el", vox.languages)
            self.assertNotIn("grc", vox.languages)

    def test_dataset_directory_resolution_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preferred = root / "manifest.jsonl"
            preferred.write_text("", encoding="utf-8")
            (root / "other.csv").write_text("", encoding="utf-8")
            self.assertEqual(_resolve_dataset_manifest(root), preferred)

    def test_dataset_audio_snapshot_changes_after_audio_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "sample.wav"
            audio.write_bytes(b"first")
            records = [DatasetRecord(audio, "hello")]
            before = _dataset_audio_snapshot(records)

            audio.write_bytes(b"replacement with a different size")

            self.assertNotEqual(_dataset_audio_snapshot(records), before)

    def test_controller_generates_mixed_tagged_mock_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspacePaths(Path(directory) / "workspace")
            window = MainWindow()
            controller = WorkbenchController(window, workspace)
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            result, request = controller._perform_synthesis(
                {
                    "text": "[la]Salve.[/la] [grc]Χαῖρε.[/grc] [ru]Привет.[/ru]",
                    "engine_id": "mock-local",
                    "auto_language": True,
                    "language": "auto",
                    "speed": 1.1,
                    "emotion": {"happy": 0.0},
                    "auto_emotion": False,
                    "style_instruction": "",
                    "seed": 42,
                    "output_directory": str(workspace.outputs),
                }
            )

            self.assertEqual([item.language for item in request.segments], ["la", "grc", "ru"])
            self.assertTrue(result.output_path.is_file())
            with wave.open(str(result.output_path), "rb") as audio:
                self.assertGreater(audio.getnframes(), 0)
            window.close()

    def test_empty_explicit_span_has_no_speakable_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            with self.assertRaisesRegex(ValueError, "可朗读"):
                controller._perform_synthesis(
                    {
                        "text": "[en]   [/en]",
                        "engine_id": "mock-local",
                        "auto_language": True,
                        "language": "auto",
                        "emotion": {},
                        "auto_emotion": False,
                        "style_instruction": "",
                        "output_directory": str(root),
                    }
                )
            window.close()

    def test_custom_engine_languages_populate_fallback_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            controller.registry.upsert(
                EngineManifest(
                    id="custom-uk",
                    name="Custom Ukrainian",
                    transport=EngineTransport.MOCK,
                    languages=["uk", "en"],
                )
            )
            controller._refresh_engine_views()
            combo = window.synthesis_page.engine_combo
            combo.setCurrentIndex(combo.findData("custom-uk"))

            self.assertGreaterEqual(window.synthesis_page.fallback_language.findData("uk"), 0)
            self.assertGreaterEqual(window.synthesis_page.reference_language.findData("uk"), 0)
            window.close()

    def test_edited_ipa_is_kept_in_pronunciation_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow()
            controller = WorkbenchController(
                window, WorkspacePaths(Path(directory) / "workspace")
            )
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            rows = [
                {
                    "text": "Salve",
                    "language": "la",
                    "start": 4,
                    "end": 9,
                    "source": "explicit",
                    "confidence": 1.0,
                    "frontend": "eSpeak-ng IPA 基线",
                    "phonemes": "salwe",
                    "join": "PCM 顺序拼接",
                }
            ]
            controller._language_rows = rows
            controller._language_rows_text = "[la]Salve[/la]"
            page = window.multilingual_page
            page.set_text("[la]Salve[/la]")
            page.set_segments(rows)

            self.assertFalse(page.segment_table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)
            self.assertTrue(page.segment_table.item(0, 5).flags() & Qt.ItemFlag.ItemIsEditable)
            page.segment_table.item(0, 5).setText("ˈsal.weː")
            controller._store_language_plan(page._settings())

            self.assertEqual(
                controller._language_plan["segments"][0]["phonemes"], "ˈsal.weː"
            )
            window.close()

    def test_stale_pronunciation_segments_are_not_attached_to_changed_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow()
            controller = WorkbenchController(
                window, WorkspacePaths(Path(directory) / "workspace")
            )
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            stale = {
                "text": "Salve",
                "language": "la",
                "start": 4,
                "end": 9,
                "source": "explicit",
                "confidence": 1.0,
                "phonemes": "ˈsal.weː",
            }
            controller._language_rows = [stale]
            controller._language_rows_text = "[la]Salve[/la]"

            controller._store_language_plan(
                {"text": "[la]Vale[/la]", "segments": [stale]}
            )

            self.assertEqual(controller._language_plan, {})
            self.assertIn("重新切分", window.multilingual_page.status_label.text())
            window.close()

    def test_generate_button_completes_async_mock_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspacePaths(Path(directory) / "workspace")
            window = MainWindow()
            controller = WorkbenchController(window, workspace)
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            page = window.synthesis_page
            page.engine_combo.setCurrentIndex(page.engine_combo.findData("mock-local"))
            page.set_text("[la]Salve.[/la] [grc]Χαῖρε.[/grc] [ru]Привет.[/ru]")

            page.generate_button.click()
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            self.assertEqual(controller._queue[0]["status"], "完成")
            self.assertTrue(Path(controller._queue[0]["output"]).is_file())
            self.assertEqual(controller._recent_tasks[0]["type"], "合成")
            self.assertEqual(controller._recent_tasks[0]["status"], "完成")
            self.assertEqual(window.overview_page.recent_table.rowCount(), 1)
            window.close()

    def test_experiment_export_and_ledger_keep_candidate_and_request_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = WorkspacePaths(root / "workspace")
            export_directory = root / "reports"
            window = MainWindow()
            controller = WorkbenchController(window, workspace)
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            controller.add_experiment_candidate(
                {
                    "name": "baseline",
                    "engine_id": "mock-local",
                    "checkpoint_path": None,
                    "speed": 1.0,
                    "style_instruction": "",
                    "seed": 11,
                }
            )
            controller.add_experiment_candidate(
                {
                    "name": "faster",
                    "engine_id": "mock-local",
                    "checkpoint_path": None,
                    "speed": 1.2,
                    "style_instruction": "",
                    "seed": 22,
                }
            )
            controller.run_experiment(
                {
                    "text": "[ru]Привет.[/ru]",
                    "reference_audio": None,
                    "output_directory": str(workspace.outputs / "experiments"),
                }
            )
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            self.assertEqual(len(controller._experiment_results), 2)
            first_result = controller._experiment_results[0]
            self.assertEqual(first_result["candidate"]["engine_id"], "mock-local")
            self.assertIsNone(first_result["candidate"]["checkpoint_path"])
            self.assertEqual(first_result["candidate"]["seed"], 11)

            ledger = controller.experiments.read_all()
            self.assertEqual(len(ledger), 2)
            self.assertEqual(ledger[0].parameters["language"], "ru")
            self.assertEqual(ledger[0].parameters["request_extra"]["prompt_lang"], "auto")

            controller.save_experiment_preference(
                {"selected_row": 0, "preference": "最佳", "note": "发音最清晰"}
            )
            persisted = controller.experiments.read_all()[0]
            self.assertEqual(persisted.preference, "最佳")
            self.assertEqual(persisted.rating.notes, "发音最清晰")

            controller.export_experiment({"output_directory": str(export_directory)})
            reports = list(export_directory.glob("experiment-*.json"))
            self.assertEqual(len(reports), 1)
            exported = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(exported["schema_version"], 1)
            self.assertEqual(exported["results"][0]["candidate"], first_result["candidate"])
            window.close()

    def test_mock_engine_rejects_checkpoint_override_instead_of_ignoring_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            with self.assertRaisesRegex(ValueError, "独立进程桥"):
                controller._perform_synthesis(
                    {
                        "text": "[en]Hello[/en]",
                        "engine_id": "mock-local",
                        "checkpoint_path": str(root / "candidate.ckpt"),
                        "auto_language": True,
                        "language": "auto",
                        "emotion": {},
                        "auto_emotion": False,
                        "style_instruction": "",
                        "output_directory": str(root),
                    }
                )
            window.close()

    def test_empty_voice_payload_clears_previous_synthesis_reference(self) -> None:
        window = MainWindow()
        window.synthesis_page.set_reference_audio("old.wav")
        window.synthesis_page.reference_authorized.setChecked(True)

        window._apply_voice_profile({"audio_path": "", "authorized": False})

        self.assertEqual(window.synthesis_page.reference_path.path(), "")
        self.assertFalse(window.synthesis_page.reference_authorized.isChecked())
        window.close()

    def test_changing_voice_audio_invalidates_previous_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_audio = root / "old.wav"
            new_audio = root / "new.wav"
            old_audio.write_bytes(b"old")
            new_audio.write_bytes(b"new")

            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            window.voice_profile_page.set_audio_path(old_audio)
            profile = ProsodyProfile(
                mean_f0_hz=220.0,
                emotion={"calm": 0.8},
                style_instruction="measured",
            )
            controller._last_profile = profile
            controller._last_profile_path = old_audio.resolve()
            window.set_profile(profile)
            window.voice_profile_page.reference_transcript.setText("old transcript")
            window.voice_profile_page.authorized_voice.setChecked(True)

            window.voice_profile_page.set_audio_path(new_audio)

            self.assertIsNone(controller._last_profile)
            self.assertEqual(window.voice_profile_page._profile_payload()["emotion"], {})
            self.assertEqual(window.voice_profile_page.style_text.toPlainText(), "")
            self.assertEqual(window.voice_profile_page.reference_transcript.text(), "")
            self.assertFalse(window.voice_profile_page.authorized_voice.isChecked())
            window.close()

    def test_reference_voice_requires_per_file_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            reference.write_bytes(b"reference")
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            with self.assertRaisesRegex(ValueError, "授权"):
                controller._perform_synthesis(
                    {
                        "text": "[en]Hello[/en]",
                        "engine_id": "gpt-sovits-v2-local",
                        "auto_language": True,
                        "language": "auto",
                        "reference_audio": str(reference),
                        "reference_authorized": False,
                        "emotion": {},
                        "auto_emotion": False,
                        "style_instruction": "",
                        "output_directory": str(root),
                    }
                )
            window.close()

    def test_single_und_text_requires_explicit_language_when_engine_has_no_auto(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = WorkspacePaths(root / "workspace")
            window = MainWindow()
            controller = WorkbenchController(window, workspace)
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            controller.registry.upsert(
                EngineManifest(
                    id="english-only",
                    name="English process bridge",
                    transport=EngineTransport.PROCESS,
                    languages=["en"],
                    command=["unused-bridge"],
                )
            )

            with self.assertRaisesRegex(ValueError, r"\[en\]"):
                controller._perform_synthesis(
                    {
                        "text": "Hello, this is an English sentence.",
                        "engine_id": "english-only",
                        "auto_language": True,
                        "language": "auto",
                        "speed": 1.0,
                        "emotion": {},
                        "auto_emotion": False,
                        "style_instruction": "",
                        "seed": 42,
                        "output_directory": str(workspace.outputs),
                    }
                )
            window.close()

    def test_unmarked_greek_mixed_with_latin_text_requires_historical_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            with self.assertRaisesRegex(ValueError, "历史语言"):
                controller._perform_synthesis(
                    {
                        "text": "χαῖρε hello",
                        "engine_id": "mock-local",
                        "auto_language": True,
                        "language": "auto",
                        "emotion": {},
                        "auto_emotion": False,
                        "style_instruction": "",
                        "output_directory": str(root),
                    }
                )
            window.close()

    def test_ambiguous_cyrillic_requires_explicit_russian_tag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            window = MainWindow()
            controller = WorkbenchController(window, WorkspacePaths(root / "workspace"))
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()

            for text in ("Привіт, світе!", "Здравей, свят!"):
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, r"\[ru\]"):
                    controller._perform_synthesis(
                        {
                            "text": text,
                            "engine_id": "mock-local",
                            "auto_language": True,
                            "language": "auto",
                            "emotion": {},
                            "auto_emotion": False,
                            "style_instruction": "",
                            "output_directory": str(root),
                        }
                    )
            window.close()

    def test_oversized_analysis_invalidates_pending_language_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow()
            controller = WorkbenchController(
                window, WorkspacePaths(Path(directory) / "workspace")
            )
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            controller._language_rows = [{"text": "old"}]
            controller._language_rows_text = "old"
            old_revision = controller._text_analysis_revision

            controller.analyze_text("x" * 20_001)

            self.assertGreater(controller._text_analysis_revision, old_revision)
            self.assertEqual(controller._language_rows, [])
            self.assertEqual(controller._language_rows_text, "")
            window.close()

    def test_text_analysis_rejects_excessive_language_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow()
            controller = WorkbenchController(
                window, WorkspacePaths(Path(directory) / "workspace")
            )
            controller.thread_pool.waitForDone(10_000)
            self.app.processEvents()
            text = "".join(
                f"[{'la' if index % 2 == 0 else 'ru'}]x[/{'la' if index % 2 == 0 else 'ru'}]"
                for index in range(257)
            )

            controller.analyze_text(text)

            self.assertEqual(controller._language_rows, [])
            self.assertIn("256", window.multilingual_page.status_label.text())
            window.close()


if __name__ == "__main__":
    unittest.main()
