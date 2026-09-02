from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from genivox import __version__, selftest


class SelfTestTests(unittest.TestCase):
    def _passing_environment(self):
        return (
            patch.object(selftest, "_runtime_python_version", return_value=(3, 11)),
            patch.object(selftest, "_runtime_pointer_bits", return_value=64),
            patch.object(selftest.metadata, "version", return_value=__version__),
        )

    def test_run_selftest_uses_temporary_workspace_and_validates_pcm_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            default_workspace = Path(directory) / "must-not-be-created"
            version_patch, architecture_patch, metadata_patch = self._passing_environment()
            with (
                patch.dict(
                    os.environ,
                    {
                        "GENIVOX_WORKSPACE": str(default_workspace),
                        "QT_QPA_PLATFORM": "offscreen",
                    },
                ),
                version_patch,
                architecture_patch,
                metadata_patch,
            ):
                report = selftest.run_selftest()

            self.assertEqual(report["status"], "pass")
            self.assertEqual(
                [check["name"] for check in report["checks"]],
                [
                    "python_version",
                    "python_architecture",
                    "package_version",
                    "qt_runtime",
                    "mock_synthesis",
                ],
            )
            qt_check = report["checks"][-2]
            self.assertEqual(qt_check["status"], "pass")
            self.assertTrue(qt_check["pyside_version"])
            self.assertTrue(qt_check["qt_version"])
            mock_check = report["checks"][-1]
            self.assertEqual(mock_check["engine_id"], "mock-local")
            self.assertGreater(mock_check["frame_count"], 0)
            self.assertEqual(mock_check["sample_width_bits"], 16)
            self.assertFalse(default_workspace.exists())

    def test_main_writes_stable_json_and_returns_zero(self) -> None:
        version_patch, architecture_patch, metadata_patch = self._passing_environment()
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}),
            version_patch,
            architecture_patch,
            metadata_patch,
            redirect_stdout(output),
        ):
            exit_code = selftest.main()

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            output.getvalue(),
            json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )

    def test_python_or_metadata_mismatch_returns_nonzero(self) -> None:
        with (
            patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}),
            patch.object(selftest, "_runtime_python_version", return_value=(3, 12)),
            patch.object(selftest, "_runtime_pointer_bits", return_value=64),
            patch.object(selftest.metadata, "version", return_value="9.9.9"),
            redirect_stdout(io.StringIO()),
        ):
            exit_code = selftest.main()

        self.assertEqual(exit_code, 1)

    def test_mock_failure_is_structured_and_returns_nonzero(self) -> None:
        version_patch, architecture_patch, metadata_patch = self._passing_environment()
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"QT_QPA_PLATFORM": "offscreen"}),
            version_patch,
            architecture_patch,
            metadata_patch,
            patch.object(selftest, "read_pcm_wav", side_effect=ValueError("invalid wav")),
            redirect_stdout(output),
        ):
            exit_code = selftest.main()

        report = json.loads(output.getvalue())
        mock_check = report["checks"][-1]
        self.assertEqual(exit_code, 1)
        self.assertEqual(mock_check["status"], "fail")
        self.assertEqual(mock_check["error_type"], "ValueError")
        self.assertEqual(mock_check["error"], "invalid wav")

    def test_qt_failure_is_structured_in_the_declared_order(self) -> None:
        version_patch, architecture_patch, metadata_patch = self._passing_environment()
        with (
            version_patch,
            architecture_patch,
            metadata_patch,
            patch.object(selftest, "_qt_runtime_check", side_effect=RuntimeError("qt unavailable")),
        ):
            report = selftest.run_selftest()

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["checks"][3], {
            "error": "qt unavailable",
            "error_type": "RuntimeError",
            "name": "qt_runtime",
            "status": "fail",
        })
        self.assertEqual(report["checks"][4]["name"], "mock_synthesis")
        self.assertEqual(report["checks"][4]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
