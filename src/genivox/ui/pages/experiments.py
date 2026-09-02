"""Side-by-side synthesis experiment and evaluation page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.widgets import Card, EmptyTable, PageHeader, PathField


class ExperimentPage(QWidget):
    """Collect comparable generation candidates and display evaluation results."""

    add_candidate_requested = Signal(dict)
    remove_candidates_requested = Signal(list)
    run_requested = Signal(dict)
    cancel_requested = Signal()
    play_requested = Signal(str)
    preference_requested = Signal(dict)
    export_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine_options: dict[str, dict[str, Any]] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader("实验对比", "固定文本与参考声音，比较模型、权重和控制参数")
        self.status_chip = QLabel("等待实验")
        self.status_chip.setObjectName("chip")
        header.actions.addWidget(self.status_chip)
        root.addWidget(header)

        setup_row = QHBoxLayout()
        setup_row.setSpacing(12)
        scenario_card = Card("固定实验条件", "所有候选共享这些输入，保证结果可比较")
        self.test_text = QTextEdit()
        self.test_text.setPlaceholderText("输入包含目标语言、停顿和情绪变化的评测文本")
        self.test_text.setMaximumHeight(90)
        scenario_card.content_layout.addWidget(self.test_text)
        self.reference_audio = PathField(
            "固定参考声音",
            file_filter="音频文件 (*.wav *.flac *.mp3 *.m4a);;所有文件 (*.*)",
        )
        self.reference_authorized = QCheckBox("我确认有权使用此固定参考声音")
        self.reference_authorized.setToolTip("更换参考文件后需要重新确认")
        self.reference_audio.path_changed.connect(
            lambda _: self.reference_authorized.setChecked(False)
        )
        self.experiment_output = PathField("实验输出目录", mode="directory")
        self.reference_transcript = QLineEdit()
        self.reference_transcript.setPlaceholderText("固定参考声音的精确转写（GPT-SoVITS 建议填写）")
        self.reference_language = QComboBox()
        for label, language in (
            ("参考语种：自动", "auto"),
            ("中文", "zh"),
            ("English", "en"),
            ("日本語", "ja"),
            ("한국어", "ko"),
            ("粤语", "yue"),
            ("Latina", "la"),
            ("古希腊语", "grc"),
            ("现代希腊语", "el"),
            ("Русский", "ru"),
        ):
            self.reference_language.addItem(label, language)
        scenario_card.content_layout.addWidget(self.reference_audio)
        scenario_card.content_layout.addWidget(self.reference_authorized)
        scenario_card.content_layout.addWidget(self.reference_transcript)
        scenario_card.content_layout.addWidget(self.reference_language)
        scenario_card.content_layout.addWidget(self.experiment_output)
        setup_row.addWidget(scenario_card, 3)

        candidate_card = Card("添加候选")
        form = QFormLayout()
        self.candidate_name = QLineEdit()
        self.candidate_name.setPlaceholderText("例如：GPT-SoVITS / neutral")
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("未发现后端", "")
        self.engine_combo.currentIndexChanged.connect(self._update_candidate_controls)
        self.checkpoint_path = PathField(
            "可选：权重目录或 checkpoint 文件", mode="file_or_directory"
        )
        self.checkpoint_trusted = QCheckBox("我确认候选权重来自可信来源")
        self.checkpoint_trusted.setToolTip("更换权重路径后需要重新确认；模型文件可能包含可执行载荷")
        self.checkpoint_path.path_changed.connect(
            lambda _: self.checkpoint_trusted.setChecked(False)
        )
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setValue(1.0)
        self.speed.setSingleStep(0.05)
        self.style = QLineEdit()
        self.style.setPlaceholderText("自然、庄重、慢速……")
        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_147_483_647)
        self.seed.setSpecialValueText("随机")
        self.seed.setValue(42)
        form.addRow("候选名称", self.candidate_name)
        form.addRow("后端", self.engine_combo)
        form.addRow("权重", self.checkpoint_path)
        form.addRow("权重信任", self.checkpoint_trusted)
        form.addRow("语速", self.speed)
        form.addRow("风格", self.style)
        form.addRow("种子", self.seed)
        candidate_card.content_layout.addLayout(form)
        self.add_button = QPushButton("添加到实验")
        self.add_button.setObjectName("primaryButton")
        self.add_button.clicked.connect(self._emit_add)
        candidate_card.content_layout.addWidget(self.add_button)
        setup_row.addWidget(candidate_card, 2)
        root.addLayout(setup_row)

        candidates_card = Card("实验矩阵")
        self.candidate_table = EmptyTable("尚无候选 · 至少添加两个配置以进行横向比较")
        self.candidate_table.setColumnCount(7)
        self.candidate_table.setHorizontalHeaderLabels(
            ["候选", "后端", "权重", "语速", "风格", "种子", "状态"]
        )
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.setMinimumHeight(155)
        candidates_card.content_layout.addWidget(self.candidate_table)
        matrix_actions = QHBoxLayout()
        self.remove_button = QPushButton("移除所选")
        self.remove_button.clicked.connect(self._emit_remove)
        self.cancel_button = QPushButton("停止实验")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.run_button = QPushButton("运行全部候选")
        self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(self._emit_run)
        matrix_actions.addWidget(self.remove_button)
        matrix_actions.addStretch(1)
        matrix_actions.addWidget(self.cancel_button)
        matrix_actions.addWidget(self.run_button)
        candidates_card.content_layout.addLayout(matrix_actions)
        root.addWidget(candidates_card)

        results_card = Card("评测结果", "客观指标只提供线索；最终保留人工 A/B 偏好")
        self.results_table = EmptyTable("运行实验后显示 WER、声纹、情绪、时长误差和实时率")
        self.results_table.setColumnCount(9)
        self.results_table.setHorizontalHeaderLabels(
            ["候选", "WER↓", "声纹↑", "情绪↑", "时长误差↓", "RTF↓", "输出", "试听", "偏好"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMinimumHeight(190)
        results_card.content_layout.addWidget(self.results_table)
        result_actions = QHBoxLayout()
        self.preference_note = QLineEdit()
        self.preference_note.setPlaceholderText("记录听感：发音、音色漂移、韵律或伪影")
        save_preference = QPushButton("保存偏好")
        save_preference.clicked.connect(self._emit_preference)
        export_button = QPushButton("导出报告")
        export_button.clicked.connect(self._emit_export)
        result_actions.addWidget(self.preference_note, 1)
        result_actions.addWidget(save_preference)
        result_actions.addWidget(export_button)
        results_card.content_layout.addLayout(result_actions)
        root.addWidget(results_card, 1)

    def _candidate_payload(self) -> dict[str, Any]:
        return {
            "name": self.candidate_name.text().strip(),
            "engine_id": str(self.engine_combo.currentData() or ""),
            "checkpoint_path": self.checkpoint_path.path() or None
            if self.checkpoint_path.isEnabled()
            else None,
            "checkpoint_trusted": self.checkpoint_trusted.isChecked()
            if self.checkpoint_trusted.isEnabled()
            else False,
            "speed": self.speed.value() if self.speed.isEnabled() else 1.0,
            "style_instruction": self.style.text().strip() if self.style.isEnabled() else "",
            "seed": self.seed.value(),
        }

    def _emit_add(self) -> None:
        payload = self._candidate_payload()
        if not payload["engine_id"]:
            self.status_chip.setText("缺少后端")
            return
        if payload["checkpoint_path"] and not payload["checkpoint_trusted"]:
            self.status_chip.setText("请确认候选权重来自可信来源")
            return
        self.add_candidate_requested.emit(payload)

    def _emit_remove(self) -> None:
        rows = sorted({index.row() for index in self.candidate_table.selectedIndexes()})
        self.remove_candidates_requested.emit(rows)

    def _emit_run(self) -> None:
        if not self.test_text.toPlainText().strip():
            self.status_chip.setText("缺少评测文本")
            return
        if self.reference_audio.path() and not self.reference_authorized.isChecked():
            self.status_chip.setText("请确认你有权使用固定参考声音")
            return
        self.run_requested.emit(
            {
                "text": self.test_text.toPlainText().strip(),
                "reference_audio": self.reference_audio.path() or None,
                "reference_authorized": self.reference_authorized.isChecked(),
                "reference_transcript": self.reference_transcript.text().strip(),
                "reference_language": str(self.reference_language.currentData() or "auto"),
                "output_directory": self.experiment_output.path() or None,
            }
        )

    def _emit_preference(self) -> None:
        selected_row = self.results_table.currentRow()
        preference = "未评价"
        if selected_row >= 0:
            widget = self.results_table.cellWidget(selected_row, 8)
            if isinstance(widget, QComboBox):
                preference = widget.currentText()
        self.preference_requested.emit(
            {
                "selected_row": selected_row,
                "preference": preference,
                "note": self.preference_note.text().strip(),
            }
        )

    def _emit_export(self) -> None:
        self.export_requested.emit({"output_directory": self.experiment_output.path() or None})

    def set_engines(self, engines: Iterable[object]) -> None:
        self._engine_options = {}
        self.engine_combo.clear()
        for engine in engines:
            if isinstance(engine, Mapping):
                engine_id = str(engine.get("id", engine.get("name", "")))
                name = str(engine.get("name", engine_id))
                capabilities = {
                    getattr(item, "value", str(item))
                    for item in engine.get("capabilities", [])
                }
                transport = getattr(engine.get("transport"), "value", engine.get("transport", ""))
            else:
                engine_id = str(getattr(engine, "id", getattr(engine, "name", "")))
                name = str(getattr(engine, "name", engine_id))
                capabilities = {
                    getattr(item, "value", str(item))
                    for item in getattr(engine, "capabilities", [])
                }
                transport = getattr(getattr(engine, "transport", ""), "value", "")
            if engine_id:
                self.engine_combo.addItem(name, engine_id)
                self._engine_options[engine_id] = {
                    "capabilities": capabilities,
                    "transport": str(transport),
                }
        if self.engine_combo.count() == 0:
            self.engine_combo.addItem("未发现后端", "")
        self._update_candidate_controls()

    def _update_candidate_controls(self, *_: object) -> None:
        engine_id = str(self.engine_combo.currentData() or "")
        options = self._engine_options.get(engine_id, {})
        capabilities = set(options.get("capabilities", set()))
        process_transport = options.get("transport") == "process"
        self.speed.setEnabled("speed" in capabilities)
        self.style.setEnabled("style_instruction" in capabilities)
        self.checkpoint_path.setEnabled(process_transport)
        self.checkpoint_trusted.setEnabled(process_transport)
        if not process_transport:
            self.checkpoint_trusted.setChecked(False)

    def set_candidates(self, candidates: Iterable[Mapping[str, Any]]) -> None:
        rows = list(candidates)
        self.candidate_table.setRowCount(len(rows))
        keys = ("name", "engine", "checkpoint", "speed", "style", "seed", "status")
        for row, candidate in enumerate(rows):
            for column, key in enumerate(keys):
                self.candidate_table.setItem(row, column, QTableWidgetItem(str(candidate.get(key, "—"))))
        self.candidate_table.resizeColumnsToContents()

    def set_results(self, results: Iterable[Mapping[str, Any]]) -> None:
        rows = list(results)
        self.results_table.setRowCount(len(rows))
        keys = ("name", "wer", "speaker_similarity", "emotion_match", "duration_error", "rtf", "output")
        for row, result in enumerate(rows):
            error = str(result.get("error", ""))
            for column, key in enumerate(keys):
                value = result.get(key, "—")
                if key == "output" and error:
                    value = f"失败：{error}"
                item = QTableWidgetItem(str(value))
                if error:
                    item.setToolTip(error)
                self.results_table.setItem(row, column, item)
            output = str(result.get("output", ""))
            play_button = QPushButton("播放")
            play_button.setEnabled(bool(output))
            play_button.clicked.connect(lambda checked=False, path=output: self.play_requested.emit(path))
            self.results_table.setCellWidget(row, 7, play_button)
            preference = QComboBox()
            preference.addItems(["未评价", "最佳", "可用", "较差"])
            preference.setCurrentText(str(result.get("preference", "未评价")))
            self.results_table.setCellWidget(row, 8, preference)
        self.results_table.resizeColumnsToContents()

    def set_status(self, text: str, *, busy: bool = False) -> None:
        self.status_chip.setText(text)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        for widget in (
            self.test_text,
            self.reference_audio,
            self.reference_authorized,
            self.reference_transcript,
            self.reference_language,
            self.experiment_output,
            self.candidate_name,
            self.engine_combo,
            self.checkpoint_path,
            self.checkpoint_trusted,
            self.speed,
            self.style,
            self.seed,
            self.add_button,
            self.remove_button,
        ):
            widget.setEnabled(not busy)
