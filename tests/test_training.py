from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from genivox.core.models import DatasetRecord
from genivox.training import (
    AuditConfig,
    JsonlMetricTail,
    ManifestParseError,
    MetricParseError,
    RunStatus,
    RunStore,
    TrainingRunner,
    audit_dataset,
    load_dataset_manifest,
    parse_metric_line,
    read_metrics_jsonl,
)


class ManifestTests(unittest.TestCase):
    def test_loads_pipe_csv_and_jsonl_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.wav").touch()
            (root / "b.wav").touch()

            pipe_path = root / "dataset.list"
            pipe_path.write_text(
                "a.wav|songyuan|la|neutral|Salve, munde!\nb.wav|songyuan|grc|excited|χαῖρε κόσμε\n",
                encoding="utf-8",
            )
            pipe_records = load_dataset_manifest(pipe_path)
            self.assertEqual([record.language for record in pipe_records], ["la", "grc"])
            self.assertEqual(pipe_records[0].audio_path, (root / "a.wav").resolve())

            csv_path = root / "dataset.csv"
            csv_path.write_text(
                "wav_path,transcript,lang,speaker_id,style,duration\n"
                'a.wav,"Ave, amice",la,songyuan,calm,1.25\n',
                encoding="utf-8",
            )
            csv_record = load_dataset_manifest(csv_path)[0]
            self.assertEqual(csv_record.text, "Ave, amice")
            self.assertEqual(csv_record.duration_seconds, 1.25)

            jsonl_path = root / "dataset.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "audio": "b.wav",
                        "text": "Привет",
                        "language": "ru",
                        "speaker": "songyuan",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            jsonl_record = load_dataset_manifest(jsonl_path)[0]
            self.assertEqual(jsonl_record.language, "ru")
            self.assertEqual(jsonl_record.emotion, "unlabeled")

    def test_pipe_header_and_parse_error_report_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.psv"
            path.write_text("audio|text|lang\nclip.wav|hello|en\n", encoding="utf-8")
            self.assertEqual(load_dataset_manifest(path)[0].language, "en")

            path.write_text("clip.wav|speaker|en\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestParseError, r":1: pipe row"):
                load_dataset_manifest(path)

    def test_loads_ljspeech_three_column_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "metadata.csv"
            manifest.write_text(
                "LJ001-0001|Raw transcript.|Normalized transcript.\n",
                encoding="utf-8",
            )

            [record] = load_dataset_manifest(manifest)

            self.assertEqual(record.audio_path, (root / "wavs" / "LJ001-0001.wav").resolve())
            self.assertEqual(record.text, "Normalized transcript.")


class AuditTests(unittest.TestCase):
    def test_empty_dataset_is_a_blocking_error(self) -> None:
        result = audit_dataset([])

        self.assertEqual(result.error_count, 1)
        self.assertEqual(result.issues[0].code, "empty_dataset")

    def test_audit_reports_distributions_and_never_mutates_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wav_path = root / "short.wav"
            _write_silent_wav(wav_path, duration_seconds=0.25)
            long_wav_path = root / "long.wav"
            _write_silent_wav(long_wav_path, duration_seconds=40.0)
            missing_path = root / "missing.wav"
            records = [
                DatasetRecord(wav_path, " Salve   MUNDE ", "la", "me", "calm"),
                DatasetRecord(long_wav_path, "salve munde", "la", "me", "calm", 40.0),
                DatasetRecord(missing_path, "χαῖρε", "grc", "me", "excited"),
            ]

            result = audit_dataset(
                records,
                config=AuditConfig(min_duration_seconds=0.5, max_duration_seconds=30),
            )

            self.assertEqual(result.record_count, 3)
            self.assertEqual(result.existing_audio_count, 2)
            self.assertEqual(result.language_counts, {"la": 2, "grc": 1})
            self.assertEqual(result.emotion_counts, {"calm": 2, "excited": 1})
            self.assertEqual(result.duration.known_count, 2)
            self.assertEqual(result.duration.unknown_count, 1)
            self.assertIsNone(
                records[0].duration_seconds,
                "duration probing must not mutate the source record",
            )

            issue_codes = [issue.code for issue in result.issues]
            self.assertIn("too_short", issue_codes)
            self.assertIn("too_long", issue_codes)
            self.assertIn("missing_audio", issue_codes)
            self.assertIn("duplicate_text", issue_codes)
            recommendation_codes = {item.code for item in result.recommendations}
            self.assertIn("resolve_missing_audio", recommendation_codes)
            self.assertIn("review_duration_outliers", recommendation_codes)
            self.assertIn("review_duplicate_text", recommendation_codes)

    def test_audit_blocks_corrupt_wav_even_when_manifest_supplies_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corrupt = Path(directory) / "corrupt.wav"
            corrupt.write_bytes(b"not a wav")

            result = audit_dataset(
                [DatasetRecord(corrupt, "Salve", "la", duration_seconds=2.5)]
            )

            self.assertGreater(result.error_count, 0)
            self.assertIn("invalid_audio", {issue.code for issue in result.issues})
            self.assertEqual(result.duration.known_count, 0)

    def test_audit_rejects_non_positive_declared_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clip.flac"
            audio.write_bytes(b"opaque non-WAV fixture")

            result = audit_dataset(
                [DatasetRecord(audio, "Hello", "en", duration_seconds=-1.0)]
            )

            self.assertIn("invalid_duration", {issue.code for issue in result.issues})

    def test_audit_uses_wav_duration_when_manifest_value_disagrees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "clip.wav"
            _write_silent_wav(audio, duration_seconds=2.0)

            result = audit_dataset(
                [DatasetRecord(audio, "Hello", "en", duration_seconds=0.75)]
            )

            self.assertAlmostEqual(result.duration.total_seconds, 2.0)
            self.assertIn(
                "declared_duration_mismatch", {issue.code for issue in result.issues}
            )


class MetricsTests(unittest.TestCase):
    def test_metric_parser_accepts_nested_and_flat_records(self) -> None:
        nested = parse_metric_line('{"global_step": 7, "metrics": {"loss": 0.25, "note": "x"}}')
        flat = parse_metric_line('{"step": 8, "loss": 0.2, "lr": 0.0001, "epoch": "one"}')
        self.assertEqual(nested.step, 7)
        self.assertEqual(nested.values, {"loss": 0.25})
        self.assertEqual(flat.values, {"loss": 0.2, "lr": 0.0001})
        with self.assertRaises(MetricParseError):
            parse_metric_line('{"loss": 1.0}')

    def test_read_and_incrementally_tail_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"step":1,"loss":1.0}\nnot-json\n', encoding="utf-8")
            loaded = read_metrics_jsonl(path)
            self.assertEqual([metric.step for metric in loaded.metrics], [1])
            self.assertEqual(len(loaded.errors), 1)

            path.write_bytes(b'{"step":1,"loss":1.0}\n{"step":2,"loss":')
            tail = JsonlMetricTail(path)
            first = tail.poll()
            self.assertEqual([metric.step for metric in first.metrics], [1])
            offset_after_first = tail.offset

            with path.open("ab") as stream:
                stream.write(b"0.5}\n")
            second = tail.poll()
            self.assertEqual([metric.step for metric in second.metrics], [2])
            self.assertGreater(tail.offset, offset_after_first)
            self.assertEqual(tail.poll().metrics, ())


class RunTests(unittest.TestCase):
    def test_run_store_persists_status_and_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(root / "runs")
            manifest = store.create(
                [
                    sys.executable,
                    "-c",
                    "print('ok')",
                    "--token",
                    "command-secret",
                    "--api-key=inline-secret",
                ],
                cwd=root,
                environment={"VISIBLE": "yes", "API_TOKEN": "do-not-save"},
                metadata={"nested": {"access_token": "also-do-not-save", "stage": "s1"}},
                run_id="run-1",
            )
            self.assertEqual(manifest.status, RunStatus.CREATED)
            persisted = store.load("run-1")
            self.assertEqual(persisted.environment["VISIBLE"], "yes")
            self.assertEqual(persisted.environment["API_TOKEN"], "***REDACTED***")
            self.assertEqual(
                persisted.command[-3:],
                ("--token", "***REDACTED***", "--api-key=***REDACTED***"),
            )
            self.assertEqual(persisted.metadata["nested"]["access_token"], "***REDACTED***")
            self.assertEqual(persisted.metadata["nested"]["stage"], "s1")
            if os.name == "posix":
                self.assertEqual(store.root.stat().st_mode & 0o777, 0o700)
                self.assertEqual(store.manifest_path("run-1").stat().st_mode & 0o777, 0o600)
            running = store.transition("run-1", RunStatus.RUNNING)
            self.assertIsNotNone(running.started_at)
            finished = store.transition("run-1", RunStatus.SUCCEEDED, exit_code=0)
            self.assertIsNotNone(finished.finished_at)
            self.assertEqual(store.list_runs()[0].status, RunStatus.SUCCEEDED)

    def test_runner_logs_output_passes_environment_and_rejects_shell_strings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = TrainingRunner(RunStore(root / "runs"))
            output: list[str] = []
            handle = runner.start(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "import os; print(os.environ['GENIVOX_TEST_VALUE'], flush=True)",
                ],
                cwd=root,
                environment={"GENIVOX_TEST_VALUE": "χαῖρε Привет"},
                on_output=output.append,
            )
            result = handle.wait(timeout=10)
            self.assertEqual(result.status, RunStatus.SUCCEEDED)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(output, ["χαῖρε Привет"])
            self.assertIn("χαῖρε Привет", result.log_path.read_text(encoding="utf-8"))

            with self.assertRaises(TypeError):
                runner.start("python train.py", cwd=root)  # type: ignore[arg-type]

    def test_runner_persists_valid_metric_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = TrainingRunner(RunStore(root / "runs"))
            handle = runner.start(
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "print('status'); print('{\"step\":3,\"loss\":0.75}')",
                ],
                cwd=root,
            )

            result = handle.wait(timeout=10)

            self.assertEqual(result.status, RunStatus.SUCCEEDED)
            persisted = read_metrics_jsonl(result.metrics_path, strict=True)
            self.assertEqual(persisted.metrics[0].step, 3)
            self.assertEqual(persisted.metrics[0].values["loss"], 0.75)

    @unittest.skipIf(os.name not in {"posix", "nt"}, "subprocess cancellation is platform-specific")
    def test_runner_can_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = TrainingRunner(RunStore(root / "runs"))
            handle = runner.start(
                [sys.executable, "-u", "-c", "import time; print('ready', flush=True); time.sleep(30)"],
                cwd=root,
            )
            result = handle.cancel(grace_seconds=0.5)
            self.assertEqual(result.status, RunStatus.CANCELLED)
            self.assertIsNotNone(result.exit_code)


def _write_silent_wav(path: Path, *, duration_seconds: float, sample_rate: int = 8_000) -> None:
    frame_count = int(duration_seconds * sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)


if __name__ == "__main__":
    unittest.main()
