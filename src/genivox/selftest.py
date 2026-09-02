"""Offline installation self-test for the base GeniVox workbench."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from genivox import __version__
from genivox.audio.wav import read_pcm_wav
from genivox.core.config import load_or_create_engine_registry
from genivox.core.models import SynthesisRequest
from genivox.core.paths import WorkspacePaths

_EXPECTED_PYTHON = (3, 11)
_EXPECTED_POINTER_BITS = 64
_MOCK_ENGINE_ID = "mock-local"

CheckResult = dict[str, Any]
_qt_application: Any | None = None


def _runtime_python_version() -> tuple[int, int]:
    return sys.version_info[:2]


def _runtime_pointer_bits() -> int:
    return struct.calcsize("P") * 8


def _python_version_check() -> CheckResult:
    runtime_version = _runtime_python_version()
    actual = ".".join(str(part) for part in runtime_version)
    expected = ".".join(str(part) for part in _EXPECTED_PYTHON)
    return {
        "actual": actual,
        "expected": expected,
        "name": "python_version",
        "status": "pass" if runtime_version == _EXPECTED_PYTHON else "fail",
    }


def _python_architecture_check() -> CheckResult:
    actual = _runtime_pointer_bits()
    return {
        "actual_bits": actual,
        "expected_bits": _EXPECTED_POINTER_BITS,
        "name": "python_architecture",
        "status": "pass" if actual == _EXPECTED_POINTER_BITS else "fail",
    }


def _package_version_check() -> CheckResult:
    try:
        installed_version: str | None = metadata.version("genivox")
    except metadata.PackageNotFoundError:
        installed_version = None
    return {
        "installed": installed_version,
        "name": "package_version",
        "source": __version__,
        "status": "pass" if installed_version == __version__ else "fail",
    }


def _qt_runtime_check() -> CheckResult:
    from PySide6 import __version__ as pyside_version
    from PySide6.QtCore import qVersion
    from PySide6.QtWidgets import QApplication

    from genivox.ui import MainWindow

    global _qt_application
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    _qt_application = application

    window = MainWindow()
    try:
        window.close()
        application.processEvents()
    finally:
        window.deleteLater()
        application.processEvents()

    return {
        "name": "qt_runtime",
        "platform": application.platformName(),
        "pyside_version": pyside_version,
        "qt_version": qVersion(),
        "status": "pass",
    }


def _mock_synthesis_check() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="genivox-selftest-") as temporary_directory:
        workspace = WorkspacePaths(Path(temporary_directory) / "workspace").ensure()
        registry = load_or_create_engine_registry(workspace)
        adapter = registry.create_adapter(_MOCK_ENGINE_ID)
        output_path = workspace.outputs / "selftest.wav"
        result = adapter.synthesize(
            SynthesisRequest(
                text="GeniVox offline self-test",
                output_path=output_path,
                engine_id=_MOCK_ENGINE_ID,
            )
        )
        pcm = read_pcm_wav(result.output_path)
        if pcm.frame_count <= 0 or pcm.samples.size <= 0:
            raise ValueError("mock adapter produced an empty PCM WAV")
        return {
            "channels": pcm.channels,
            "engine_id": result.engine_id,
            "frame_count": pcm.frame_count,
            "name": "mock_synthesis",
            "sample_rate": pcm.sample_rate,
            "sample_width_bits": pcm.sample_width_bytes * 8,
            "status": "pass",
        }


def _run_guarded(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception as exc:
        return {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "name": name,
            "status": "fail",
        }


def run_selftest() -> dict[str, Any]:
    """Run deterministic base checks without accessing models, GPUs, or the network."""

    checks = [
        _run_guarded("python_version", _python_version_check),
        _run_guarded("python_architecture", _python_architecture_check),
        _run_guarded("package_version", _package_version_check),
        _run_guarded("qt_runtime", _qt_runtime_check),
        _run_guarded("mock_synthesis", _mock_synthesis_check),
    ]
    passed = all(check["status"] == "pass" for check in checks)
    return {
        "checks": checks,
        "schema_version": 1,
        "status": "pass" if passed else "fail",
    }


def main() -> int:
    report = run_selftest()
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
