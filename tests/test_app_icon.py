from __future__ import annotations

import os
import struct
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QIcon, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from genivox.app import create_application  # noqa: E402
from genivox.core.paths import WorkspacePaths  # noqa: E402
from genivox.ui.resources import APP_ICON_RESOURCE, load_app_icon  # noqa: E402


class AppIconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_icon_resource_decodes_to_a_renderable_pixmap(self) -> None:
        self.assertTrue(APP_ICON_RESOURCE.is_file())
        image = QImage.fromData(APP_ICON_RESOURCE.read_bytes())
        self.assertFalse(image.isNull())

        icon = load_app_icon()
        self.assertFalse(icon.isNull())
        self.assertFalse(icon.pixmap(32, 32).isNull())

    def test_windows_shortcut_icon_contains_standard_sizes(self) -> None:
        icon_data = files("genivox").joinpath("assets", "genivox-app-icon.ico").read_bytes()
        reserved, image_type, image_count = struct.unpack_from("<HHH", icon_data)
        self.assertEqual((reserved, image_type, image_count), (0, 1, 6))

        sizes = []
        for index in range(image_count):
            entry_offset = 6 + (index * 16)
            width, height = struct.unpack_from("<BB", icon_data, entry_offset)
            byte_count, image_offset = struct.unpack_from("<II", icon_data, entry_offset + 8)
            self.assertGreaterEqual(image_offset, 6 + (image_count * 16))
            self.assertLessEqual(image_offset + byte_count, len(icon_data))
            sizes.append((width or 256, height or 256))
        self.assertEqual(sizes, [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])

    def test_application_window_and_brand_use_packaged_icon(self) -> None:
        self.app.setWindowIcon(QIcon())
        with tempfile.TemporaryDirectory() as directory:
            workspace = WorkspacePaths(Path(directory) / "workspace")
            with patch("genivox.app.WorkbenchController"):
                app, window, _ = create_application([], workspace=workspace)
            try:
                self.assertFalse(app.windowIcon().isNull())
                self.assertFalse(window.windowIcon().isNull())
                self.assertEqual(window.app_mark.text(), "")
                pixmap = window.app_mark.pixmap()
                self.assertIsNotNone(pixmap)
                assert pixmap is not None
                self.assertFalse(pixmap.isNull())
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
