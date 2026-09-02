"""Overview page for local engines, hardware, and recent activity."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.theme import COLORS
from genivox.ui.widgets import Card, EmptyTable, MetricCard, PageHeader


def _value(item: object, key: str, default: object = "") -> object:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class OverviewPage(QWidget):
    """At-a-glance system and workflow state."""

    navigate_requested = Signal(str)
    refresh_requested = Signal()
    open_workspace_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(16)

        header = PageHeader("概览", "本地语音模型、设备资源与当前任务状态")
        refresh_button = QPushButton("刷新状态")
        refresh_button.clicked.connect(self.refresh_requested)
        header.actions.addWidget(refresh_button)
        root.addWidget(header)

        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.gpu_card = MetricCard("GPU", "未检测", "等待系统探测", COLORS["primary"])
        self.vram_card = MetricCard("显存", "—", "已用 / 总量", COLORS["cyan"])
        self.engine_card = MetricCard("可用后端", "0", "尚未注册", COLORS["green"])
        self.task_card = MetricCard("运行任务", "0", "队列空闲", COLORS["orange"])
        for card in (self.gpu_card, self.vram_card, self.engine_card, self.task_card):
            metrics.addWidget(card, 1)
        root.addLayout(metrics)

        center = QGridLayout()
        center.setHorizontalSpacing(12)
        center.setVerticalSpacing(12)

        readiness_card = Card("引擎就绪状态", "探测本地运行环境、权重与核心能力")
        self.engine_table = EmptyTable("尚未扫描到本地 TTS 后端")
        self.engine_table.setColumnCount(5)
        self.engine_table.setHorizontalHeaderLabels(["后端", "版本", "设备", "状态", "能力"])
        self.engine_table.horizontalHeader().setStretchLastSection(True)
        self.engine_table.setMinimumHeight(220)
        readiness_card.content_layout.addWidget(self.engine_table)
        center.addWidget(readiness_card, 0, 0, 2, 1)

        quick_card = Card("快速开始", "常用工作流入口")
        actions = (
            ("创建一次语音", "synthesis"),
            ("分析我的声音", "voice_profile"),
            ("审计训练数据", "training"),
            ("导入本地模型", "models"),
        )
        for label, route in actions:
            button = QPushButton(label)
            if route == "synthesis":
                button.setObjectName("primaryButton")
            button.clicked.connect(lambda checked=False, route=route: self.navigate_requested.emit(route))
            quick_card.content_layout.addWidget(button)
        quick_card.content_layout.addStretch(1)
        center.addWidget(quick_card, 0, 1)

        workspace_card = Card("项目工作区")
        self.workspace_path = QLabel("尚未配置")
        self.workspace_path.setObjectName("muted")
        self.workspace_path.setWordWrap(True)
        open_button = QPushButton("打开目录")
        open_button.clicked.connect(self.open_workspace_requested)
        workspace_card.content_layout.addWidget(self.workspace_path)
        workspace_card.content_layout.addWidget(open_button)
        center.addWidget(workspace_card, 1, 1)
        center.setColumnStretch(0, 3)
        center.setColumnStretch(1, 1)
        root.addLayout(center, 1)

        recent_card = Card("最近任务", "合成、分析和训练任务会汇总在这里")
        self.recent_table = EmptyTable("尚无任务记录 · 可从“合成工作台”开始")
        self.recent_table.setColumnCount(5)
        self.recent_table.setHorizontalHeaderLabels(["类型", "名称", "后端", "状态", "时间"])
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.setMinimumHeight(150)
        recent_card.content_layout.addWidget(self.recent_table)
        root.addWidget(recent_card)

    def set_status(self, status: Mapping[str, Any]) -> None:
        """Update hardware and task status with controller-provided values."""

        self.gpu_card.set_metric(status.get("gpu", "未检测"), str(status.get("device", "本地设备")))
        used = status.get("vram_used_gb")
        total = status.get("vram_total_gb")
        if isinstance(used, (int, float)) and isinstance(total, (int, float)):
            vram_value = f"{used:.1f} / {total:.1f} GB"
        else:
            vram_value = "—"
        self.vram_card.set_metric(vram_value, "已用 / 总量")
        self.task_card.set_metric(status.get("active_tasks", 0), str(status.get("task_detail", "队列空闲")))
        if workspace := status.get("workspace"):
            self.workspace_path.setText(str(workspace))

    def set_engines(self, engines: Iterable[object]) -> None:
        rows = list(engines)
        self.engine_table.setRowCount(len(rows))
        ready_count = 0
        for row, engine in enumerate(rows):
            raw_status = str(_value(engine, "status", "未知"))
            is_ready = raw_status.lower() in {"ready", "available", "就绪", "可用"}
            if is_ready:
                ready_count += 1
            capabilities = _value(engine, "capabilities", [])
            if not isinstance(capabilities, str):
                capabilities = "、".join(getattr(value, "value", str(value)) for value in capabilities)
            values = (
                _value(engine, "name", _value(engine, "id", "未命名")),
                _value(engine, "version", "—"),
                _value(engine, "device", "—"),
                raw_status,
                capabilities or "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 3:
                    item.setForeground(QColor(COLORS["green"] if is_ready else COLORS["text_muted"]))
                self.engine_table.setItem(row, column, item)
        self.engine_card.set_metric(ready_count, f"共注册 {len(rows)} 个")
        self.engine_table.resizeColumnsToContents()

    def set_recent_tasks(self, tasks: Iterable[Mapping[str, Any]]) -> None:
        rows = list(tasks)
        active_count = sum(str(task.get("status", "")) == "运行中" for task in rows)
        self.task_card.set_metric(
            active_count,
            "有任务运行" if active_count else "队列空闲",
        )
        self.recent_table.setRowCount(len(rows))
        keys = ("type", "name", "engine", "status", "time")
        for row, task in enumerate(rows):
            for column, key in enumerate(keys):
                self.recent_table.setItem(row, column, QTableWidgetItem(str(task.get(key, "—"))))
        self.recent_table.resizeColumnsToContents()
