"""Small, dependency-free widgets shared by the desktop pages."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.theme import COLORS


class Card(QFrame):
    """A consistently styled container with an optional title and subtitle."""

    def __init__(
        self,
        title: str = "",
        subtitle: str = "",
        *,
        margins: tuple[int, int, int, int] = (16, 14, 16, 16),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(*margins)
        self.content_layout.setSpacing(11)

        if title:
            title_label = QLabel(title)
            title_label.setObjectName("sectionTitle")
            self.content_layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("muted")
            subtitle_label.setWordWrap(True)
            self.content_layout.addWidget(subtitle_label)


class PageHeader(QWidget):
    """Shared page heading with an optional action area."""

    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        subtitle_label.setWordWrap(True)
        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        layout.addLayout(text_layout, 1)

        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        layout.addLayout(self.actions)


class MetricCard(QFrame):
    """Compact metric display that can be updated by a controller."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        detail: str = "尚未检测",
        accent: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumWidth(135)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)

        self.label = QLabel(label)
        self.label.setObjectName("metricLabel")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        if accent:
            self.value.setStyleSheet(f"color: {accent};")
        self.detail = QLabel(detail)
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)

        layout.addWidget(self.label)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def set_metric(self, value: object, detail: str | None = None) -> None:
        self.value.setText(str(value))
        if detail is not None:
            self.detail.setText(detail)


class PathField(QWidget):
    """A line edit and a local filesystem picker."""

    path_changed = Signal(str)

    def __init__(
        self,
        placeholder: str,
        *,
        mode: str = "file",
        file_filter: str = "所有文件 (*.*)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.button = QPushButton("浏览…")
        self.button.setFixedWidth(76)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

        if self.mode == "file_or_directory":
            menu = QMenu(self.button)
            menu.addAction("选择文件…", self._choose_file)
            menu.addAction("选择目录…", self._choose_directory)
            self.button.setMenu(menu)
        else:
            self.button.clicked.connect(self._choose_path)
        self.edit.textChanged.connect(self.path_changed)

    def _start_path(self) -> str:
        current = self.edit.text().strip()
        if not current:
            return ""
        path = Path(current)
        if path.is_file():
            return str(path.parent)
        return current if path.is_dir() else ""

    def _choose_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择文件", self._start_path(), self.file_filter
        )
        if selected:
            self.edit.setText(selected)

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择目录", self._start_path())
        if selected:
            self.edit.setText(selected)

    def _choose_path(self) -> None:
        if self.mode == "directory":
            self._choose_directory()
            return
        elif self.mode == "save":
            selected, _ = QFileDialog.getSaveFileName(
                self, "选择保存位置", self._start_path(), self.file_filter
            )
        else:
            self._choose_file()
            return
        if selected:
            self.edit.setText(selected)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str | Path) -> None:
        self.edit.setText(str(path))


class EmptyTable(QTableWidget):
    """Table widget that paints a useful empty-state message."""

    def __init__(self, empty_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.empty_text = empty_text
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        if self.rowCount() != 0:
            return
        painter = QPainter(self.viewport())
        painter.setPen(QColor(COLORS["text_muted"]))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)


class WaveformWidget(QWidget):
    """Lightweight waveform preview; accepts normalized samples from a controller."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._samples: list[float] = []
        self.setMinimumHeight(155)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_samples(self, samples: Iterable[float]) -> None:
        values = [float(value) for value in samples]
        peak = max((abs(value) for value in values), default=1.0) or 1.0
        self._samples = [max(-1.0, min(1.0, value / peak)) for value in values]
        self.update()

    def clear(self) -> None:
        self._samples.clear()
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setBrush(QColor("#0D1520"))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRoundedRect(rect, 7, 7)

        mid_y = rect.center().y()
        painter.setPen(QPen(QColor("#23364A"), 1))
        painter.drawLine(QPointF(rect.left() + 10, mid_y), QPointF(rect.right() - 10, mid_y))
        if not self._samples:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "导入或录制声音后显示波形")
            return

        samples = self._samples
        usable_width = max(1.0, rect.width() - 20)
        usable_height = max(1.0, rect.height() - 24)
        points: list[QPointF] = []
        for index, value in enumerate(samples):
            x = rect.left() + 10 + index * usable_width / max(1, len(samples) - 1)
            y = mid_y - value * usable_height * 0.46
            points.append(QPointF(x, y))
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.setPen(QPen(QColor(COLORS["cyan"]), 1.5))
        painter.drawPath(path)


class LossChartWidget(QWidget):
    """Small painted multi-series training chart without a plotting dependency."""

    SERIES_COLORS = (
        COLORS["primary"],
        COLORS["orange"],
        COLORS["cyan"],
        COLORS["purple"],
        COLORS["green"],
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: dict[str, list[tuple[int, float]]] = {}
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def append_metric(self, step: int, values: Mapping[str, float]) -> None:
        for name, raw_value in values.items():
            value = float(raw_value)
            if not math.isfinite(value):
                continue
            points = self._series.setdefault(str(name), [])
            points.append((int(step), value))
            if len(points) > 2000:
                del points[: len(points) - 2000]
        self.update()

    def set_metrics(self, metrics: Iterable[tuple[int, Mapping[str, float]]]) -> None:
        self._series.clear()
        for step, values in metrics:
            for name, raw_value in values.items():
                value = float(raw_value)
                if math.isfinite(value):
                    self._series.setdefault(str(name), []).append((int(step), value))
        self.update()

    def clear(self) -> None:
        self._series.clear()
        self.update()

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        full_rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        painter.setBrush(QColor("#0D1520"))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRoundedRect(full_rect, 7, 7)

        chart = full_rect.adjusted(48, 20, -18, -36)
        painter.setFont(QFont(painter.font().family(), 8))
        painter.setPen(QPen(QColor("#263A50"), 1))
        for index in range(5):
            y = chart.top() + index * chart.height() / 4
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        points = [point for series in self._series.values() for point in series]
        if not points:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(chart, Qt.AlignmentFlag.AlignCenter, "训练开始后在此绘制 loss / validation loss")
            return

        min_step = min(step for step, _ in points)
        max_step = max(step for step, _ in points)
        min_value = min(value for _, value in points)
        max_value = max(value for _, value in points)
        if min_step == max_step:
            max_step += 1
        if math.isclose(min_value, max_value):
            margin = max(abs(min_value) * 0.1, 0.1)
            min_value -= margin
            max_value += margin

        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawText(QRectF(2, chart.top() - 6, 43, 18), Qt.AlignmentFlag.AlignRight, f"{max_value:.3g}")
        painter.drawText(
            QRectF(2, chart.bottom() - 10, 43, 18),
            Qt.AlignmentFlag.AlignRight,
            f"{min_value:.3g}",
        )
        painter.drawText(
            QRectF(chart.left(), chart.bottom() + 7, chart.width(), 18),
            Qt.AlignmentFlag.AlignLeft,
            str(min_step),
        )
        painter.drawText(
            QRectF(chart.left(), chart.bottom() + 7, chart.width(), 18),
            Qt.AlignmentFlag.AlignRight,
            str(max_step),
        )

        for series_index, (name, series) in enumerate(self._series.items()):
            color = QColor(self.SERIES_COLORS[series_index % len(self.SERIES_COLORS)])
            path = QPainterPath()
            for index, (step, value) in enumerate(series):
                x = chart.left() + (step - min_step) / (max_step - min_step) * chart.width()
                y = chart.bottom() - (value - min_value) / (max_value - min_value) * chart.height()
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(color, 1.8))
            painter.drawPath(path)

            legend_x = chart.left() + series_index * 112
            painter.fillRect(QRectF(legend_x, 5, 12, 3), color)
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(QRectF(legend_x + 17, 0, 90, 14), name)
