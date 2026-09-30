"""Local GPT-SoVITS startup owns only its child process."""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from genivox.core.models import EngineManifest, EngineTransport
from genivox.engines.gpt_sovits import GptSovitsProbeResult, GptSovitsProbeStatus
from genivox.services.gpt_sovits_service import (
    GptSovitsServiceError,
    GptSovitsServiceManager,
)


def _installation(root: Path) -> Path:
    (root / "GPT_SoVITS" / "configs").mkdir(parents=True)
    (root / "GPT_SoVITS" / "TTS_infer_pack").mkdir(parents=True)
    (root / "api_v2.py").write_text("# test entrypoint\n", encoding="utf-8")
    (root / "GPT_SoVITS" / "configs" / "tts_infer.yaml").write_text(
        "custom: {}\n", encoding="utf-8"
    )
    (root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").write_text(
        "# test core\n", encoding="utf-8"
    )
    python = root / "python.exe"
    python.touch()
    return python


def _manifest(root: Path, python: Path) -> EngineManifest:
    return EngineManifest(
        id="gpt-local",
        name="GPT-SoVITS local",
        transport=EngineTransport.HTTP,
        endpoint="http://127.0.0.1:9880/tts",
        root=str(root),
        python=str(python),
        metadata={
            "adapter": "gpt_sovits_v2",
            "trusted_local_code": True,
            "auto_start": True,
        },
    )


def _status(status: GptSovitsProbeStatus) -> GptSovitsProbeResult:
    return GptSovitsProbeResult(status, status.value)


class _FakeProcess:
    def __init__(self, *, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False

    def poll(self) -> int | None:
        return self.exit_code

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float) -> int:
        return self.exit_code or 0


class GptSovitsServiceTests(unittest.TestCase):
    def test_launches_real_local_child_and_stops_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "backend"
            _installation(root)
            (root / "api_v2.py").write_text(
                "import argparse\n"
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "parser = argparse.ArgumentParser()\n"
                "parser.add_argument('-a')\n"
                "parser.add_argument('-p', type=int)\n"
                "parser.add_argument('-c')\n"
                "args = parser.parse_args()\n"
                "class Handler(BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        self.send_response(200)\n"
                "        self.end_headers()\n"
                "HTTPServer((args.a, args.p), Handler).serve_forever()\n",
                encoding="utf-8",
            )
            with socket.socket() as available:
                available.bind(("127.0.0.1", 0))
                port = available.getsockname()[1]

            def live_probe(endpoint: str) -> GptSovitsProbeResult:
                try:
                    with urllib.request.urlopen(
                        endpoint.replace("/tts", "/ready"), timeout=0.2
                    ) as response:
                        if response.status == 200:
                            return _status(GptSovitsProbeStatus.API_READY)
                except (urllib.error.URLError, TimeoutError):
                    pass
                return _status(GptSovitsProbeStatus.OFFLINE)

            manifest = _manifest(root, Path(sys.executable))
            manifest.endpoint = f"http://127.0.0.1:{port}/tts"
            service = GptSovitsServiceManager(Path(directory) / "logs", probe=live_probe)
            try:
                self.assertTrue(service.ensure_ready(manifest, timeout_seconds=10).ready)
                child = service._process
                self.assertIsNotNone(child)
                assert child is not None
                self.assertIsNone(child.poll())
            finally:
                service.stop()
            self.assertIsNotNone(child.poll())
            self.assertTrue(service.log_path.is_file())

    def test_reuses_ready_external_service_without_starting_or_stopping_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            service = GptSovitsServiceManager(
                root / "logs", probe=lambda _: _status(GptSovitsProbeStatus.API_READY)
            )
            with patch("genivox.services.gpt_sovits_service.subprocess.Popen") as spawn:
                result = service.ensure_ready(_manifest(python.parent, python))
                service.stop()
            self.assertTrue(result.ready)
            spawn.assert_not_called()
            self.assertIsNone(service.log_path)

    def test_wrong_service_and_missing_consent_cannot_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            manifest = _manifest(python.parent, python)
            service = GptSovitsServiceManager(
                root / "logs", probe=lambda _: _status(GptSovitsProbeStatus.WRONG_SERVICE)
            )
            with patch("genivox.services.gpt_sovits_service.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(GptSovitsServiceError, "wrong_service"):
                    service.ensure_ready(manifest)
                manifest.metadata["trusted_local_code"] = False
                with self.assertRaisesRegex(GptSovitsServiceError, "trusted HTTP"):
                    service.ensure_ready(manifest)
            spawn.assert_not_called()

    def test_launches_once_with_loopback_args_and_stops_owned_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            child = _FakeProcess()
            results = iter(
                [
                    _status(GptSovitsProbeStatus.OFFLINE),
                    _status(GptSovitsProbeStatus.API_READY),
                    _status(GptSovitsProbeStatus.API_READY),
                ]
            )
            service = GptSovitsServiceManager(root / "logs", probe=lambda _: next(results))
            with patch(
                "genivox.services.gpt_sovits_service.subprocess.Popen", return_value=child
            ) as spawn:
                manifest = _manifest(python.parent, python)
                self.assertTrue(service.ensure_ready(manifest).ready)
                self.assertTrue(service.ensure_ready(manifest).ready)
                self.assertFalse(child.terminated)
                service.stop()
            spawn.assert_called_once()
            args, kwargs = spawn.call_args
            self.assertEqual(
                args[0],
                [
                    str(python.resolve()),
                    "api_v2.py",
                    "-a",
                    "127.0.0.1",
                    "-p",
                    "9880",
                    "-c",
                    "GPT_SoVITS/configs/tts_infer.yaml",
                ],
            )
            self.assertEqual(kwargs["cwd"], python.parent.resolve())
            self.assertFalse(kwargs["shell"])
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(service.log_path.is_file())
            self.assertTrue(child.terminated)

    def test_startup_timeout_stops_child_and_reports_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            child = _FakeProcess()
            service = GptSovitsServiceManager(
                root / "logs", probe=lambda _: _status(GptSovitsProbeStatus.OFFLINE)
            )
            with patch(
                "genivox.services.gpt_sovits_service.subprocess.Popen", return_value=child
            ):
                with self.assertRaisesRegex(GptSovitsServiceError, "see .*\\.log"):
                    service.ensure_ready(_manifest(python.parent, python), timeout_seconds=0.02)
            self.assertTrue(child.terminated)

    def test_failed_child_reports_exit_code_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            child = _FakeProcess(exit_code=17)
            service = GptSovitsServiceManager(
                root / "logs", probe=lambda _: _status(GptSovitsProbeStatus.OFFLINE)
            )
            with patch(
                "genivox.services.gpt_sovits_service.subprocess.Popen", return_value=child
            ):
                with self.assertRaisesRegex(GptSovitsServiceError, "exited with code 17; see"):
                    service.ensure_ready(_manifest(python.parent, python))
            self.assertTrue(service.log_path.is_file())

    def test_concurrent_startup_only_spawns_one_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            child = _FakeProcess()
            calls = 0
            calls_lock = threading.Lock()

            def probe(_: str) -> GptSovitsProbeResult:
                nonlocal calls
                with calls_lock:
                    calls += 1
                    return _status(
                        GptSovitsProbeStatus.OFFLINE
                        if calls == 1
                        else GptSovitsProbeStatus.API_READY
                    )

            service = GptSovitsServiceManager(root / "logs", probe=probe)
            manifest = _manifest(python.parent, python)
            outcomes: list[object] = []

            def start() -> None:
                try:
                    outcomes.append(service.ensure_ready(manifest))
                except Exception as exc:
                    outcomes.append(exc)

            with patch(
                "genivox.services.gpt_sovits_service.subprocess.Popen", return_value=child
            ) as spawn:
                threads = [threading.Thread(target=start) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)
                service.stop()
            self.assertEqual(len(outcomes), 2)
            self.assertTrue(all(isinstance(item, GptSovitsProbeResult) for item in outcomes))
            spawn.assert_called_once()
            self.assertTrue(child.terminated)

    def test_stop_interrupts_startup_wait_without_waiting_for_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            child = _FakeProcess()
            probed_after_launch = threading.Event()
            calls = 0

            def probe(_: str) -> GptSovitsProbeResult:
                nonlocal calls
                calls += 1
                if calls > 1:
                    probed_after_launch.set()
                return _status(GptSovitsProbeStatus.OFFLINE)

            service = GptSovitsServiceManager(root / "logs", probe=probe)
            outcomes: list[Exception] = []

            def start() -> None:
                try:
                    service.ensure_ready(_manifest(python.parent, python), timeout_seconds=90)
                except Exception as exc:
                    outcomes.append(exc)

            with patch(
                "genivox.services.gpt_sovits_service.subprocess.Popen", return_value=child
            ):
                worker = threading.Thread(target=start)
                worker.start()
                self.assertTrue(probed_after_launch.wait(timeout=5))
                service.stop()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertTrue(child.terminated)
            self.assertEqual(len(outcomes), 1)
            self.assertIsInstance(outcomes[0], GptSovitsServiceError)

    def test_rejects_non_default_loopback_scheme_or_portless_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = _installation(root / "backend")
            manifest = _manifest(python.parent, python)
            service = GptSovitsServiceManager(root / "logs")
            for endpoint in (
                "http://127.0.0.1/tts",
                "http://127.0.0.1:0/tts",
                "http://localhost:9880/tts",
                "https://127.0.0.1:9880/tts",
                "http://127.0.0.1:9880/tts?secret=1",
            ):
                manifest.endpoint = endpoint
                with self.subTest(endpoint=endpoint):
                    with self.assertRaises(GptSovitsServiceError):
                        service.ensure_ready(manifest)
