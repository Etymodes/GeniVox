"""Local engine, runtime, and checkpoint management page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.widgets import Card, EmptyTable, PageHeader, PathField

CAPABILITY_COLUMNS = (
    ("voice_clone", "声音克隆"),
    ("cross_lingual", "跨语言"),
    ("speed", "语速"),
    ("emotion_vector", "情绪向量"),
    ("style_instruction", "风格指令"),
    ("phoneme_input", "音素输入"),
    ("fine_tune", "微调"),
)


def _read(item: object, key: str, default: object = "") -> object:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class ModelManagerPage(QWidget):
    """Register local model installations without executing model code in the UI."""

    scan_requested = Signal()
    verify_environment_requested = Signal(dict)
    probe_requested = Signal(str)
    import_requested = Signal(dict)
    remove_requested = Signal(str)
    activate_requested = Signal(str)
    open_root_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configuration_revision = 0
        self._status_owner = "initial"
        self._editing_engine_id = ""
        self._registered_engines: dict[str, dict[str, Any]] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader("模型管理", "登记本地代码、Python 环境和权重；模型运行由独立适配器负责")
        scan_button = QPushButton("刷新本机状态")
        scan_button.clicked.connect(self.scan_requested)
        header.actions.addWidget(scan_button)
        self.status_chip = QLabel("尚未扫描")
        self.status_chip.setObjectName("chip")
        header.actions.addWidget(self.status_chip)
        root.addWidget(header)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        import_card = Card("导入本地模型或微调权重", "可以引用现有目录，默认不复制大型权重")
        form = QFormLayout()
        self.engine_type = QComboBox()
        self.engine_type.addItems(
            ["GPT-SoVITS", "IndexTTS2.5", "VoxCPM2", "Qwen3-TTS", "Fish Speech / S2 Pro", "自定义适配器"]
        )
        self.display_name = QLineEdit()
        self.display_name.setPlaceholderText("显示名称；留空则使用后端名称")
        self.engine_root = PathField("后端源码或安装目录", mode="directory")
        self.python_path = PathField(
            "Python 可执行文件 / Conda 环境",
            file_filter="Python (python.exe python);;所有文件 (*.*)",
        )
        self.checkpoint_path = PathField(
            "基础模型目录或 checkpoint 文件", mode="file_or_directory"
        )
        self.transport = QComboBox()
        self.transport.addItems(["独立进程", "HTTP 服务", "内置测试后端"])
        self.transport.setCurrentText("HTTP 服务")
        self.endpoint = QLineEdit()
        self.endpoint.setText("http://127.0.0.1:9880/tts")
        self.endpoint.setPlaceholderText("HTTP 模式，例如 http://127.0.0.1:9880/tts")
        form.addRow("后端类型", self.engine_type)
        form.addRow("名称", self.display_name)
        form.addRow("源码目录", self.engine_root)
        form.addRow("Python 环境", self.python_path)
        form.addRow("权重目录", self.checkpoint_path)
        form.addRow("连接方式", self.transport)
        form.addRow("服务地址", self.endpoint)
        import_card.content_layout.addLayout(form)
        self.reference_existing = QCheckBox("v0.1 仅引用现有目录（不复制权重）")
        self.reference_existing.setChecked(True)
        self.reference_existing.setEnabled(False)
        self.reference_existing.setToolTip("复制、下载和重定位模型文件将在后续导入向导中实现")
        self.trust_local_code = QCheckBox("允许启动此目录中的本地模型代码")
        self.trust_local_code.setToolTip(
            "进程会继承当前用户权限；更换源码目录、Python 或连接方式后必须重新确认"
        )
        self.auto_start = QCheckBox("随 GeniVox 启动本地 GPT-SoVITS 服务")
        self.auto_start.setToolTip(
            "仅适用于已确认可信、配置了源码目录及 Python 的 GPT-SoVITS HTTP 服务；"
            "应用启动时检查本机服务，离线时才尝试启动"
        )
        self.engine_root.path_changed.connect(self._invalidate_code_trust)
        self.python_path.path_changed.connect(self._invalidate_code_trust)
        self.transport.currentIndexChanged.connect(self._invalidate_code_trust)
        import_card.content_layout.addWidget(self.reference_existing)
        import_card.content_layout.addWidget(self.trust_local_code)
        import_card.content_layout.addWidget(self.auto_start)
        import_actions = QHBoxLayout()
        self.verify_button = QPushButton("检查安装与服务")
        self.verify_button.clicked.connect(
            lambda: self.verify_environment_requested.emit(self.import_payload())
        )
        self.import_button = QPushButton("登记模型")
        self.import_button.setObjectName("primaryButton")
        self.import_button.clicked.connect(self._emit_import)
        self.new_registration_button = QPushButton("新建登记")
        self.new_registration_button.setEnabled(False)
        self.new_registration_button.clicked.connect(self._reset_registration_form)
        import_actions.addWidget(self.verify_button)
        import_actions.addStretch(1)
        import_actions.addWidget(self.new_registration_button)
        import_actions.addWidget(self.import_button)
        import_card.content_layout.addLayout(import_actions)
        self.import_feedback = QLabel("模型不会在登记时加载进显存")
        self.import_feedback.setObjectName("muted")
        self.import_feedback.setWordWrap(True)
        import_card.content_layout.addWidget(self.import_feedback)
        self.engine_type.currentIndexChanged.connect(self._mark_configuration_changed)
        self.display_name.textChanged.connect(self._mark_configuration_changed)
        self.engine_root.path_changed.connect(self._mark_configuration_changed)
        self.python_path.path_changed.connect(self._mark_configuration_changed)
        self.checkpoint_path.path_changed.connect(self._mark_configuration_changed)
        self.transport.currentIndexChanged.connect(self._mark_configuration_changed)
        self.endpoint.textChanged.connect(self._mark_configuration_changed)
        self.trust_local_code.toggled.connect(self._mark_configuration_changed)
        self.auto_start.toggled.connect(self._mark_configuration_changed)
        self.engine_type.currentIndexChanged.connect(self._sync_auto_start_eligibility)
        self.transport.currentIndexChanged.connect(self._sync_auto_start_eligibility)
        self._sync_auto_start_eligibility()
        top_row.addWidget(import_card, 3)

        env_card = Card("默认本地环境", "供新登记模型预填，不会修改系统环境变量")
        env_form = QFormLayout()
        self.workspace_path = PathField("GeniVox 工作目录", mode="directory")
        self.cache_path = PathField("模型缓存目录", mode="directory")
        for display_path in (self.workspace_path, self.cache_path):
            display_path.edit.setReadOnly(True)
            display_path.button.setEnabled(False)
        self.default_device = QComboBox()
        self.default_device.addItems(["cuda:0", "cpu", "自动"])
        self.default_precision = QComboBox()
        self.default_precision.addItems(["bf16", "fp16", "fp32", "int8 / 量化"])
        env_form.addRow("工作区", self.workspace_path)
        env_form.addRow("缓存", self.cache_path)
        env_form.addRow("默认设备", self.default_device)
        env_form.addRow("默认精度", self.default_precision)
        env_card.content_layout.addLayout(env_form)
        self.environment_summary = QLabel("GPU、CUDA、FFmpeg 与磁盘状态会在扫描后显示")
        self.environment_summary.setObjectName("muted")
        self.environment_summary.setWordWrap(True)
        env_card.content_layout.addWidget(self.environment_summary)
        env_card.content_layout.addStretch(1)
        top_row.addWidget(env_card, 2)
        root.addLayout(top_row)

        tabs = QTabWidget()
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        models_layout.setContentsMargins(10, 10, 10, 10)
        self.models_table = EmptyTable("没有已登记模型 · 可从上方引用已有模型目录")
        self.models_table.setColumnCount(8)
        self.models_table.setHorizontalHeaderLabels(
            ["名称", "后端", "版本", "设备", "权重", "连接", "状态", "当前"]
        )
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.setMinimumHeight(210)
        self.models_table.currentCellChanged.connect(self._update_configure_button)
        models_layout.addWidget(self.models_table)
        model_actions = QHBoxLayout()
        open_button = QPushButton("打开源码目录")
        open_button.clicked.connect(self._emit_open)
        self.probe_button = QPushButton("检查已选服务")
        self.probe_button.clicked.connect(self._emit_probe)
        self.configure_button = QPushButton("配置已选服务")
        self.configure_button.setEnabled(False)
        self.configure_button.clicked.connect(self._configure_selected_service)
        activate_button = QPushButton("选作默认")
        activate_button.clicked.connect(self._emit_activate)
        remove_button = QPushButton("移除登记")
        remove_button.setObjectName("dangerButton")
        remove_button.clicked.connect(self._emit_remove)
        model_actions.addWidget(open_button)
        model_actions.addWidget(self.probe_button)
        model_actions.addWidget(self.configure_button)
        model_actions.addWidget(activate_button)
        model_actions.addStretch(1)
        model_actions.addWidget(remove_button)
        models_layout.addLayout(model_actions)
        tabs.addTab(models_tab, "已登记模型")

        capability_tab = QWidget()
        capability_layout = QVBoxLayout(capability_tab)
        capability_layout.setContentsMargins(10, 10, 10, 10)
        self.capability_table = EmptyTable("登记模型后显示适配器报告的能力矩阵")
        self.capability_table.setColumnCount(2 + len(CAPABILITY_COLUMNS))
        self.capability_table.setHorizontalHeaderLabels(
            ["模型", "语言"] + [label for _, label in CAPABILITY_COLUMNS]
        )
        self.capability_table.horizontalHeader().setStretchLastSection(True)
        self.capability_table.setMinimumHeight(245)
        capability_layout.addWidget(self.capability_table)
        note = QLabel("“—”表示适配器未报告，不等于模型必然不支持。能力验证应由后端探测任务完成。")
        note.setObjectName("muted")
        capability_layout.addWidget(note)
        tabs.addTab(capability_tab, "能力矩阵")
        root.addWidget(tabs, 1)

    def import_payload(self) -> dict[str, Any]:
        transport_map = {"独立进程": "process", "HTTP 服务": "http", "内置测试后端": "mock"}
        return {
            "engine_type": self.engine_type.currentText(),
            "name": self.display_name.text().strip() or self.engine_type.currentText(),
            "root": self.engine_root.path() or None,
            "python": self.python_path.path() or None,
            "checkpoint_dir": self.checkpoint_path.path() or None,
            "transport": transport_map[self.transport.currentText()],
            "endpoint": self.endpoint.text().strip() or None,
            "reference_existing": self.reference_existing.isChecked(),
            "trusted_local_code": self.trust_local_code.isChecked(),
            "auto_start": self.auto_start.isChecked(),
            "edit_engine_id": self._editing_engine_id or None,
            "device": self.default_device.currentText(),
            "precision": self.default_precision.currentText(),
        }

    def _emit_import(self) -> None:
        payload = self.import_payload()
        if payload["transport"] == "http" and not payload["endpoint"]:
            self.import_feedback.setText("HTTP 服务模式需要填写服务地址。")
            return
        if payload["transport"] == "process" and not payload["root"]:
            self.import_feedback.setText("独立进程模式需要填写后端源码或安装目录。")
            return
        if payload["auto_start"] and not (
            payload["engine_type"] == "GPT-SoVITS" and payload["transport"] == "http"
        ):
            self.import_feedback.setText("仅 GPT-SoVITS HTTP 服务支持随应用启动。")
            return
        if payload["auto_start"] and not (payload["root"] and payload["python"]):
            self.import_feedback.setText("自动启动需要填写 GPT-SoVITS 源码目录和 Python 可执行文件。")
            return
        if (
            payload["transport"] == "process" or payload["root"] or payload["auto_start"]
        ) and not payload["trusted_local_code"]:
            self.import_feedback.setText("请确认允许启动所选目录中的本地模型代码。")
            return
        self.import_feedback.setText(
            "等待控制器验证并保存…" if self._editing_engine_id else "等待控制器验证并登记…"
        )
        self.import_requested.emit(payload)

    def _invalidate_code_trust(self, *_: object) -> None:
        self.trust_local_code.setChecked(False)

    def _sync_auto_start_eligibility(self, *_: object) -> None:
        eligible = (
            self.engine_type.currentText() == "GPT-SoVITS"
            and self.transport.currentText() == "HTTP 服务"
        )
        if not eligible:
            self.auto_start.setChecked(False)
        self.auto_start.setEnabled(eligible)

    def _update_configure_button(self, *_: object) -> None:
        engine_id = self._selected_engine_id()
        engine = self._registered_engines.get(engine_id, {})
        transport = getattr(engine.get("transport"), "value", engine.get("transport"))
        self.configure_button.setEnabled(
            transport == "http" and (
                engine.get("engine_type") == "GPT-SoVITS" or engine_id == "gpt-sovits-v2-local"
            )
        )

    def _configure_selected_service(self) -> None:
        engine_id = self._selected_engine_id()
        engine = self._registered_engines.get(engine_id)
        if not self.configure_button.isEnabled() or engine is None:
            return
        self._editing_engine_id = engine_id
        self.engine_type.setCurrentText("GPT-SoVITS")
        self.display_name.setText(str(engine.get("name") or ""))
        self.engine_root.set_path(str(engine.get("root") or ""))
        self.python_path.set_path(str(engine.get("python") or ""))
        checkpoint = str(engine.get("checkpoint_dir") or "")
        if engine_id == "gpt-sovits-v2-local" and checkpoint and not Path(checkpoint).exists():
            checkpoint = ""  # Built-in placeholder is not a requirement for external GPT weights.
        self.checkpoint_path.set_path(checkpoint)
        self.transport.setCurrentText("HTTP 服务")
        self.endpoint.setText(str(engine.get("endpoint") or ""))
        self.auto_start.setChecked(bool(engine.get("auto_start", False)))
        # Restore only the trust previously recorded for exactly these paths.
        # Any later path edit clears the checkbox via _invalidate_code_trust.
        self.trust_local_code.setChecked(bool(engine.get("trusted_local_code", False)))
        self.engine_type.setEnabled(False)
        self.transport.setEnabled(False)
        self.import_button.setText("保存服务配置")
        self.new_registration_button.setEnabled(True)
        self.import_feedback.setText("已载入服务配置；修改源码目录或 Python 后请重新确认允许启动本地代码")

    def _reset_registration_form(self) -> None:
        self._editing_engine_id = ""
        self.engine_type.setEnabled(True)
        self.transport.setEnabled(True)
        self.import_button.setText("登记模型")
        self.new_registration_button.setEnabled(False)
        self.engine_root.set_path("")
        self.python_path.set_path("")
        self.checkpoint_path.set_path("")
        self.trust_local_code.setChecked(False)
        self.auto_start.setChecked(False)
        self.display_name.clear()
        self.import_feedback.setText("填写新模型配置后登记")

    def _mark_configuration_changed(self, *_: object) -> None:
        self._configuration_revision += 1
        message = "配置已更改；请重新检查安装与服务"
        self.import_feedback.setText(message)
        if self._status_owner == "form":
            self.status_chip.setText(message)

    @property
    def configuration_revision(self) -> int:
        return self._configuration_revision

    def _selected_engine_id(self) -> str:
        row = self.models_table.currentRow()
        if row < 0:
            return ""
        item = self.models_table.item(row, 0)
        return str(item.data(256) or "") if item else ""

    def _emit_open(self) -> None:
        row = self.models_table.currentRow()
        item = self.models_table.item(row, 0) if row >= 0 else None
        root = str(item.data(257) or "") if item else ""
        self.open_root_requested.emit(root)

    def _emit_activate(self) -> None:
        if engine_id := self._selected_engine_id():
            self.activate_requested.emit(engine_id)

    def _emit_probe(self) -> None:
        if engine_id := self._selected_engine_id():
            self.probe_requested.emit(engine_id)
        else:
            self.set_service_status("请先选择一个已登记的 GPT-SoVITS HTTP 服务")

    def _emit_remove(self) -> None:
        if engine_id := self._selected_engine_id():
            self.remove_requested.emit(engine_id)

    def set_engines(self, engines: Iterable[object]) -> None:
        rows = list(engines)
        self._registered_engines = {}
        self.models_table.setRowCount(len(rows))
        self.capability_table.setRowCount(len(rows))
        for row, engine in enumerate(rows):
            engine_id = str(_read(engine, "id", _read(engine, "name", "")))
            name = str(_read(engine, "name", engine_id or "未命名"))
            root = str(_read(engine, "root", "") or "")
            checkpoint = _read(engine, "checkpoint_dir", "—") or "—"
            transport = getattr(_read(engine, "transport", "—"), "value", _read(engine, "transport", "—"))
            self._registered_engines[engine_id] = {
                "name": name,
                "engine_type": _read(engine, "engine_type", ""),
                "root": root,
                "python": _read(engine, "python", "") or "",
                "checkpoint_dir": _read(engine, "checkpoint_dir", "") or "",
                "transport": transport,
                "endpoint": _read(engine, "endpoint", "") or "",
                "auto_start": _read(engine, "auto_start", False),
                "trusted_local_code": _read(engine, "trusted_local_code", False),
            }
            values = (
                name,
                _read(engine, "engine_type", engine_id),
                _read(engine, "version", "—"),
                _read(engine, "device", "—"),
                checkpoint,
                transport,
                _read(engine, "status", "未知"),
                "是" if _read(engine, "active", False) else "",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(256, engine_id)
                    item.setData(257, root)
                self.models_table.setItem(row, column, item)

            languages = _read(engine, "languages", [])
            capabilities = _read(engine, "capabilities", [])
            capability_values = {getattr(value, "value", str(value)) for value in capabilities}
            self.capability_table.setItem(row, 0, QTableWidgetItem(name))
            language_text = languages if isinstance(languages, str) else "、".join(map(str, languages))
            self.capability_table.setItem(row, 1, QTableWidgetItem(language_text or "—"))
            for offset, (key, _) in enumerate(CAPABILITY_COLUMNS, start=2):
                reported = key in capability_values
                self.capability_table.setItem(row, offset, QTableWidgetItem("✓" if reported else "—"))
        self.models_table.resizeColumnsToContents()
        self.capability_table.resizeColumnsToContents()
        self._update_configure_button()

    def set_environment(self, environment: Mapping[str, Any]) -> None:
        parts = [
            f"GPU：{environment.get('gpu', '未检测')}",
            f"CUDA：{environment.get('cuda', '未检测')}",
            f"FFmpeg：{environment.get('ffmpeg', '未检测')}",
            f"可用磁盘：{environment.get('disk_free', '—')}",
        ]
        self.environment_summary.setText("\n".join(parts))
        if path := environment.get("workspace"):
            self.workspace_path.set_path(str(path))
        if path := environment.get("cache"):
            self.cache_path.set_path(str(path))

    def set_status(self, text: str, *, busy: bool = False) -> None:
        del busy
        self._status_owner = "form"
        self.status_chip.setText(text)
        self.import_feedback.setText(text)

    def set_service_status(self, text: str) -> None:
        self._status_owner = "service"
        self.status_chip.setText(text)
        self.status_chip.setToolTip(text)
        self.import_feedback.setText(text)

    def set_model_operation_busy(self, busy: bool) -> None:
        self.verify_button.setEnabled(not busy)
        self.probe_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
