"""Application entry point for the local GeniVox desktop workbench."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from genivox.controller import WorkbenchController
from genivox.core.paths import WorkspacePaths, default_workspace
from genivox.ui import MainWindow
from genivox.ui.theme import apply_theme


def create_application(
    argv: Sequence[str] | None = None,
    *,
    workspace: WorkspacePaths | None = None,
) -> tuple[QApplication, MainWindow, WorkbenchController]:
    app = QApplication.instance() or QApplication(list(argv or []))
    app.setApplicationName("GeniVox")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Etymodes")
    apply_theme(app)
    window = MainWindow()
    controller = WorkbenchController(window, workspace or default_workspace())
    return app, window, controller


def main() -> int:
    app, window, controller = create_application(sys.argv)
    window._genivox_controller = controller  # type: ignore[attr-defined]
    window.show()
    return app.exec()
