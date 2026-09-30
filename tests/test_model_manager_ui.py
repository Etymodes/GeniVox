"""Model manager form behavior around optional local service startup."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from genivox.ui.pages.models import ModelManagerPage  # noqa: E402


class ModelManagerUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.page = ModelManagerPage()

    def tearDown(self) -> None:
        self.page.close()

    def test_auto_start_requires_gpt_http_root_python_and_trust(self) -> None:
        self.assertTrue(self.page.auto_start.isEnabled())
        self.page.transport.setCurrentText("独立进程")
        self.assertFalse(self.page.auto_start.isEnabled())
        self.page.transport.setCurrentText("HTTP 服务")
        self.page.engine_type.setCurrentText("IndexTTS2.5")
        self.assertFalse(self.page.auto_start.isEnabled())
        self.page.engine_type.setCurrentText("GPT-SoVITS")

        emitted: list[dict[str, object]] = []
        self.page.import_requested.connect(emitted.append)
        self.page.auto_start.setChecked(True)
        self.page.import_button.click()
        self.assertEqual(emitted, [])
        self.assertIn("源码目录和 Python", self.page.import_feedback.text())

        self.page.engine_root.set_path("C:/GPT-SoVITS")
        self.page.python_path.set_path("C:/GPT-SoVITS/runtime/python.exe")
        self.page.import_button.click()
        self.assertEqual(emitted, [])
        self.assertIn("请确认允许启动", self.page.import_feedback.text())

        self.page.trust_local_code.setChecked(True)
        self.page.import_button.click()
        self.assertEqual(len(emitted), 1)
        self.assertTrue(emitted[0]["auto_start"])
        self.assertIsNone(emitted[0]["edit_engine_id"])
        self.page.python_path.set_path("C:/GPT-SoVITS/runtime2/python.exe")
        self.assertFalse(self.page.trust_local_code.isChecked())

    def test_configure_builtin_gpt_service_reuses_existing_id(self) -> None:
        self.page.set_engines(
            [
                {
                    "id": "gpt-sovits-v2-local",
                    "name": "GPT-SoVITS 本地 API",
                    "engine_type": "gpt-sovits-v2-local",
                    "transport": "http",
                    "root": "",
                    "python": "",
                    "endpoint": "http://127.0.0.1:9880/tts",
                    "checkpoint_dir": "C:/missing/gpt-sovits",
                    "auto_start": False,
                    "trusted_local_code": False,
                }
            ]
        )
        self.page.models_table.setCurrentCell(0, 0)
        self.assertTrue(self.page.configure_button.isEnabled())
        self.page.configure_button.click()
        self.assertEqual(self.page.import_payload()["edit_engine_id"], "gpt-sovits-v2-local")
        self.assertEqual(self.page.import_button.text(), "保存服务配置")
        self.assertEqual(self.page.checkpoint_path.path(), "")

        self.page.engine_root.set_path("C:/GPT-SoVITS")
        self.page.python_path.set_path("C:/GPT-SoVITS/runtime/python.exe")
        self.page.auto_start.setChecked(True)
        self.page.trust_local_code.setChecked(True)
        emitted: list[dict[str, object]] = []
        self.page.import_requested.connect(emitted.append)
        self.page.import_button.click()
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["edit_engine_id"], "gpt-sovits-v2-local")
        self.assertEqual(emitted[0]["endpoint"], "http://127.0.0.1:9880/tts")
        self.assertTrue(emitted[0]["auto_start"])


if __name__ == "__main__":
    unittest.main()
