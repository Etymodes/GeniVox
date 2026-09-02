"""Dataset audit, fine-tuning configuration, and training monitor page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.theme import COLORS
from genivox.ui.widgets import Card, EmptyTable, LossChartWidget, MetricCard, PageHeader, PathField


class TrainingPage(QWidget):
    """UI-only training surface; workers are supplied by the controller."""

    audit_requested = Signal(str)
    prepare_requested = Signal(dict)
    start_requested = Signal(dict)
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    open_run_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._audit_busy = False
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader("数据与训练", "审计语料分布、配置少样本微调，并实时观察训练指标")
        self.status_chip = QLabel("未开始")
        self.status_chip.setObjectName("chip")
        header.actions.addWidget(self.status_chip)
        root.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "数据审计")
        self.tabs.addTab(self._build_training_tab(), "训练配置与监控")
        root.addWidget(self.tabs, 1)

    def _build_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        source_card = Card("训练数据源", "支持读取控制器提供的 LJSpeech、JSONL 或 GPT-SoVITS 标注格式")
        source_row = QHBoxLayout()
        self.dataset_path = PathField("选择包含标注文件的语料目录", mode="directory")
        self.dataset_format = QComboBox()
        self.dataset_format.addItem("按扩展名与列结构自动识别")
        self.dataset_format.setEnabled(False)
        self.dataset_format.setToolTip("v0.1 自动识别 JSONL、CSV、GPT-SoVITS 与 LJSpeech 标注")
        self.audit_button = QPushButton("扫描与审计")
        self.audit_button.setObjectName("primaryButton")
        self.audit_button.clicked.connect(self._emit_audit)
        source_row.addWidget(self.dataset_path, 1)
        source_row.addWidget(self.dataset_format)
        source_row.addWidget(self.audit_button)
        source_card.content_layout.addLayout(source_row)
        self.audit_status = QLabel("选择数据后检查音频、文本、语言、时长和类别平衡")
        self.audit_status.setObjectName("muted")
        source_card.content_layout.addWidget(self.audit_status)
        layout.addWidget(source_card)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.utterance_card = MetricCard("总记录", "—", "条目")
        self.hours_card = MetricCard("总时长", "—", "小时", COLORS["cyan"])
        self.language_card = MetricCard("语言", "—", "已识别类别", COLORS["primary"])
        self.speaker_card = MetricCard("说话人", "—", "已识别身份", COLORS["purple"])
        self.issue_card = MetricCard("需要处理", "—", "错误 / 警告", COLORS["orange"])
        for card in (
            self.utterance_card,
            self.hours_card,
            self.language_card,
            self.speaker_card,
            self.issue_card,
        ):
            stats.addWidget(card, 1)
        layout.addLayout(stats)

        tables = QHBoxLayout()
        tables.setSpacing(12)
        distribution_card = Card(
            "数据分布与引导", "v0.1 显示观测分布；目标需由你或训练后端配置"
        )
        self.distribution_table = EmptyTable("审计完成后显示语言、情绪、说话人与时长分布")
        self.distribution_table.setColumnCount(7)
        self.distribution_table.setHorizontalHeaderLabels(
            ["维度", "类别", "样本", "时长", "占比", "目标", "建议"]
        )
        self.distribution_table.horizontalHeader().setStretchLastSection(True)
        self.distribution_table.setMinimumHeight(265)
        distribution_card.content_layout.addWidget(self.distribution_table)
        tables.addWidget(distribution_card, 3)

        issues_card = Card("质量问题", "保留定位信息，修复动作由数据层执行")
        self.issue_table = EmptyTable("未发现问题，或尚未运行审计")
        self.issue_table.setColumnCount(4)
        self.issue_table.setHorizontalHeaderLabels(["级别", "记录定位", "问题", "建议"])
        self.issue_table.horizontalHeader().setStretchLastSection(True)
        self.issue_table.setMinimumHeight(265)
        issues_card.content_layout.addWidget(self.issue_table)
        tables.addWidget(issues_card, 2)
        layout.addLayout(tables, 1)

        prepare_row = QHBoxLayout()
        self.deduplicate = QCheckBox("去除完全重复")
        self.deduplicate.setChecked(True)
        self.deduplicate.setEnabled(False)
        self.create_validation = QCheckBox("分层划分验证集")
        self.create_validation.setChecked(True)
        self.create_validation.setEnabled(False)
        self.balance_sampler = QCheckBox("训练时使用分布感知采样")
        self.balance_sampler.setEnabled(False)
        prepare_button = QPushButton("训练清单生成（v0.2）")
        prepare_button.setEnabled(False)
        prepare_button.setToolTip("v0.1 只读审计原始数据，不会生成或改写训练清单")
        prepare_button.clicked.connect(self._emit_prepare)
        prepare_row.addWidget(self.deduplicate)
        prepare_row.addWidget(self.create_validation)
        prepare_row.addWidget(self.balance_sampler)
        prepare_row.addStretch(1)
        prepare_row.addWidget(prepare_button)
        layout.addLayout(prepare_row)
        return page

    def _build_training_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        config_card = Card("微调配置", "参数只在点击开始后组成请求，不在 UI 线程启动训练")
        config_form = QFormLayout()
        config_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.training_engine = QComboBox()
        self.training_engine.addItem("未发现可微调后端", "")
        self.base_model_path = PathField(
            "基础模型目录或 checkpoint 文件", mode="file_or_directory"
        )
        self.run_output_path = PathField("训练输出目录", mode="directory")
        self.train_stage = QComboBox()
        self.train_stage.addItems(
            ["推荐：声音克隆微调", "声学模型 / S2", "语义模型 / S1", "LoRA / Adapter", "全量微调"]
        )
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda:0", "cpu", "自动"])
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["bf16", "fp16", "fp32"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 256)
        self.batch_size.setValue(4)
        self.gradient_accumulation = QSpinBox()
        self.gradient_accumulation.setRange(1, 128)
        self.gradient_accumulation.setValue(4)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 10_000)
        self.epochs.setValue(20)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(7)
        self.learning_rate.setRange(0.0000001, 1.0)
        self.learning_rate.setValue(0.0001)
        self.learning_rate.setSingleStep(0.00001)
        self.validation_interval = QSpinBox()
        self.validation_interval.setRange(1, 100_000)
        self.validation_interval.setValue(100)
        self.max_duration = QDoubleSpinBox()
        self.max_duration.setRange(1.0, 120.0)
        self.max_duration.setValue(15.0)
        self.max_duration.setSuffix(" s")
        config_form.addRow("训练后端", self.training_engine)
        config_form.addRow("基础权重", self.base_model_path)
        config_form.addRow("输出位置", self.run_output_path)
        config_form.addRow("训练阶段", self.train_stage)
        config_form.addRow("设备", self.device_combo)
        config_form.addRow("精度", self.precision_combo)
        config_form.addRow("批大小", self.batch_size)
        config_form.addRow("梯度累积", self.gradient_accumulation)
        config_form.addRow("轮次", self.epochs)
        config_form.addRow("学习率", self.learning_rate)
        config_form.addRow("验证间隔", self.validation_interval)
        config_form.addRow("单条最长", self.max_duration)
        config_card.content_layout.addLayout(config_form)

        self.freeze_speaker = QCheckBox("冻结声纹编码器")
        self.freeze_speaker.setChecked(True)
        self.mixed_precision = QCheckBox("启用混合精度")
        self.mixed_precision.setChecked(True)
        self.early_stopping = QCheckBox("验证集连续恶化时早停")
        self.early_stopping.setChecked(True)
        config_card.content_layout.addWidget(self.freeze_speaker)
        config_card.content_layout.addWidget(self.mixed_precision)
        config_card.content_layout.addWidget(self.early_stopping)
        config_card.content_layout.addStretch(1)
        layout.addWidget(config_card, 2)

        monitor_column = QVBoxLayout()
        monitor_column.setSpacing(12)
        chart_card = Card("训练曲线", "最多保留最近 2000 个点；controller 可持续调用 append_metric")
        self.loss_chart = LossChartWidget()
        chart_card.content_layout.addWidget(self.loss_chart, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0.0% · 等待训练")
        chart_card.content_layout.addWidget(self.progress_bar)
        monitor_column.addWidget(chart_card, 3)

        run_card = Card("运行控制")
        run_controls = QHBoxLayout()
        self.start_button = QPushButton("开始微调")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._emit_start)
        self.pause_button = QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.setToolTip("通用进程桥暂不支持安全暂停/恢复")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._toggle_pause)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setObjectName("dangerButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.open_run_button = QPushButton("打开运行目录")
        self.open_run_button.clicked.connect(
            lambda: self.open_run_requested.emit(self.run_output_path.path())
        )
        run_controls.addWidget(self.start_button)
        run_controls.addWidget(self.pause_button)
        run_controls.addWidget(self.cancel_button)
        run_controls.addStretch(1)
        run_controls.addWidget(self.open_run_button)
        run_card.content_layout.addLayout(run_controls)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("训练日志会显示在这里")
        self.log_view.setMaximumHeight(125)
        run_card.content_layout.addWidget(self.log_view)
        monitor_column.addWidget(run_card, 2)
        layout.addLayout(monitor_column, 5)
        return page

    def _emit_audit(self) -> None:
        path = self.dataset_path.path()
        if not path:
            self.audit_status.setText("请先选择语料目录。")
            return
        self.audit_status.setText("等待数据层扫描…")
        self.audit_requested.emit(path)

    def _emit_prepare(self) -> None:
        self.prepare_requested.emit(
            {
                "dataset_path": self.dataset_path.path(),
                "format": self.dataset_format.currentText(),
                "deduplicate": self.deduplicate.isChecked(),
                "create_validation": self.create_validation.isChecked(),
                "distribution_aware_sampler": self.balance_sampler.isChecked(),
            }
        )

    def _emit_start(self) -> None:
        payload = self.training_payload()
        if not payload["dataset_path"]:
            self.set_status("缺少训练数据目录")
            self.tabs.setCurrentIndex(0)
            return
        if not payload["engine_id"]:
            self.set_status("缺少可微调后端")
            return
        self.start_requested.emit(payload)

    def _toggle_pause(self, paused: bool) -> None:
        self.pause_button.setText("继续" if paused else "暂停")
        if paused:
            self.pause_requested.emit()
        else:
            self.resume_requested.emit()

    def training_payload(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path.path(),
            "dataset_format": self.dataset_format.currentText(),
            "engine_id": str(self.training_engine.currentData() or ""),
            "base_model_path": self.base_model_path.path() or None,
            "output_path": self.run_output_path.path() or None,
            "stage": self.train_stage.currentText(),
            "device": self.device_combo.currentText(),
            "precision": self.precision_combo.currentText(),
            "batch_size": self.batch_size.value(),
            "gradient_accumulation": self.gradient_accumulation.value(),
            "epochs": self.epochs.value(),
            "learning_rate": self.learning_rate.value(),
            "validation_interval": self.validation_interval.value(),
            "max_duration_seconds": self.max_duration.value(),
            "freeze_speaker_encoder": self.freeze_speaker.isChecked(),
            "mixed_precision": self.mixed_precision.isChecked(),
            "early_stopping": self.early_stopping.isChecked(),
        }

    def set_engines(self, engines: Iterable[object]) -> None:
        current = self.training_engine.currentData()
        self.training_engine.clear()
        for engine in engines:
            if isinstance(engine, Mapping):
                engine_id = str(engine.get("id", engine.get("name", "")))
                label = str(engine.get("name", engine_id))
                capabilities = engine.get("capabilities", [])
            else:
                engine_id = str(getattr(engine, "id", getattr(engine, "name", "")))
                label = str(getattr(engine, "name", engine_id))
                capabilities = getattr(engine, "capabilities", [])
            capability_values = {getattr(item, "value", str(item)) for item in capabilities}
            if engine_id and "fine_tune" in capability_values:
                self.training_engine.addItem(label, engine_id)
        if self.training_engine.count() == 0:
            self.training_engine.addItem("未发现可微调后端", "")
        elif current:
            index = self.training_engine.findData(current)
            if index >= 0:
                self.training_engine.setCurrentIndex(index)

    def set_dataset_report(self, report: Mapping[str, Any]) -> None:
        summary = report.get("summary", report)
        utterances = summary.get("utterances", summary.get("records", 0))
        hours = summary.get("hours")
        if hours is None and isinstance(summary.get("duration_seconds"), (int, float)):
            hours = float(summary["duration_seconds"]) / 3600
        self.utterance_card.set_metric(utterances, "清单中的总记录")
        self.hours_card.set_metric(f"{float(hours):.2f}" if isinstance(hours, (int, float)) else "—", "小时")
        self.language_card.set_metric(summary.get("languages", "—"), "语言类别")
        self.speaker_card.set_metric(summary.get("speakers", "—"), "说话人类别")

        distribution = list(report.get("distribution", []))
        self.distribution_table.setRowCount(len(distribution))
        for row, item in enumerate(distribution):
            ratio = item.get("ratio", 0.0)
            ratio_number = float(ratio) if isinstance(ratio, (int, float)) else 0.0
            values = (
                item.get("dimension", "—"),
                item.get("category", "—"),
                item.get("count", "—"),
                item.get("duration", item.get("duration_seconds", "—")),
                f"{ratio_number:.1%}",
                item.get("target", "—"),
                item.get("guidance", "保持"),
            )
            for column, value in enumerate(values):
                self.distribution_table.setItem(row, column, QTableWidgetItem(str(value)))

        issues = list(report.get("issues", []))
        self.issue_table.setRowCount(len(issues))
        error_count = sum(1 for issue in issues if str(issue.get("level", "")).lower() == "error")
        self.issue_card.set_metric(len(issues), f"{error_count} 个错误")
        for row, issue in enumerate(issues):
            values = (
                issue.get("level", "warning"),
                issue.get("location", issue.get("path", "—")),
                issue.get("message", "—"),
                issue.get("guidance", "人工复核"),
            )
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                if column == 0:
                    color = COLORS["red"] if str(value).lower() == "error" else COLORS["orange"]
                    table_item.setForeground(QColor(color))
                self.issue_table.setItem(row, column, table_item)
        self.audit_status.setText(str(report.get("status", "审计完成，可复核分布与问题清单。")))
        self.distribution_table.resizeColumnsToContents()
        self.issue_table.resizeColumnsToContents()

    def append_metric(
        self,
        metric: object,
        values: Mapping[str, float] | None = None,
    ) -> None:
        if values is None:
            if isinstance(metric, Mapping):
                step = int(metric.get("step", 0))
                metric_values = metric.get("values", {})
            else:
                step = int(getattr(metric, "step", 0))
                metric_values = getattr(metric, "values", {})
        else:
            step = int(metric)
            metric_values = values
        if isinstance(metric_values, Mapping):
            self.loss_chart.append_metric(step, metric_values)

    def append_log(self, text: str) -> None:
        self.log_view.append(text)

    def reset_run_display(self) -> None:
        self.loss_chart.clear()
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0.0% · 正在启动")

    def set_audit_busy(self, busy: bool) -> None:
        self._audit_busy = busy
        self.audit_button.setEnabled(not busy)
        self.dataset_path.setEnabled(not busy)
        self.start_button.setEnabled(not busy)

    def set_status(
        self,
        text: str,
        *,
        progress: float | None = None,
        busy: bool | None = None,
        paused: bool = False,
    ) -> None:
        self.status_chip.setText(text)
        if progress is not None:
            normalized = max(0.0, min(1.0, float(progress)))
            self.progress_bar.setValue(round(normalized * 1000))
            self.progress_bar.setFormat(f"{normalized:.1%} · {text}")
        if busy is not None:
            self.start_button.setEnabled(not busy and not self._audit_busy)
            self.audit_button.setEnabled(not busy and not self._audit_busy)
            self.pause_button.setEnabled(False)
            self.cancel_button.setEnabled(busy)
            if not busy and self.pause_button.isChecked():
                self.pause_button.blockSignals(True)
                self.pause_button.setChecked(False)
                self.pause_button.setText("暂停")
                self.pause_button.blockSignals(False)
        if paused != self.pause_button.isChecked() and self.pause_button.isEnabled():
            self.pause_button.blockSignals(True)
            self.pause_button.setChecked(paused)
            self.pause_button.setText("继续" if paused else "暂停")
            self.pause_button.blockSignals(False)
