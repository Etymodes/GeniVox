"""Packaged visual resources for the GeniVox desktop interface."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtGui import QIcon, QImage, QPixmap

APP_ICON_RESOURCE = files("genivox").joinpath("assets", "genivox-app-icon.png")


def load_app_icon() -> QIcon:
    """Decode the packaged application icon without relying on a filesystem path."""

    image = QImage.fromData(APP_ICON_RESOURCE.read_bytes())
    if image.isNull():
        raise RuntimeError("The packaged GeniVox application icon is invalid.")
    return QIcon(QPixmap.fromImage(image))
