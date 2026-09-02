"""Interactive multi-engine speech synthesis workbench."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.widgets import Card, EmptyTable, PageHeader, PathField

EMOTIONS = (
    ("happy", "喜悦"),
    ("angry", "愤怒"),
    ("sad", "悲伤"),
    ("afraid", "恐惧"),
    ("disgusted", "厌恶"),
    ("melancholic", "忧郁"),
    ("surprised", "惊讶"),
    ("calm", "平静 / 中性"),
)

CAPABILITY_LABELS = {
    "voice_clone": "声音克隆",
    "cross_lingual": "同声纹跨语言",
    "speed": "原生语速",
    "emotion_vector": "情绪向量",
    "style_instruction": "风格指令",
    "phoneme_input": "音素输入",
    "streaming": "流式输出",
    "fine_tune": "微调",
}

LANGUAGE_LABELS = {
    "zh": "中文",
    "en": "English",
    "ja": "日本語",
    "ko": "한국어",
    "yue": "粤语",
    "la": "Latina",
    "grc": "古希腊语 / Ἑλληνική",
    "el": "现代希腊语 / Ελληνικά",
    "ru": "Русский",
    "es": "Español",
    "ar": "العربية",
}


def _engine_label(engine: object) -> tuple[str, str]:
    if isinstance(engine, str):
        return engine, engine
    if isinstance(engine, Mapping):
        engine_id = str(engine.get("id", engine.get("name", "")))
        return engine_id, str(engine.get("name", engine_id))
    engine_id = str(getattr(engine, "id", getattr(engine, "name", "")))
    return engine_id, str(getattr(engine, "name", engine_id))


def _segment_value(segment: object, key: str, default: object = "") -> object:
    if isinstance(segment, Mapping):
        return segment.get(key, default)
    return getattr(segment, key, default)


class SynthesisPage(QWidget):
    """Collects synthesis parameters and emits immutable request payloads."""

    generate_requested = Signal(dict)
    analyze_text_requested = Signal(str)
    reference_changed = Signal(str)
    preview_reference_requested = Signal(str)
    stop_requested = Signal()
    queue_clear_requested = Signal()
    open_output_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader("合成工作台", "用同一声音连续朗读多语言文本，并比较不同 TTS 后端")
        self.status_chip = QLabel("空闲")
        self.status_chip.setObjectName("chip")
        header.actions.addWidget(self.status_chip)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        input_card = Card("文本与声音", "语言可自动分段；无需为每段手动切换模型")
        text_toolbar = QHBoxLayout()
        self.auto_language = QCheckBox("自动识别并切分多语言")
        self.auto_language.setChecked(True)
        self.fallback_language = QComboBox()
        for label, language in (
            ("默认：自动", "auto"),
            ("中文", "zh"),
            ("English", "en"),
            ("日本語", "ja"),
            ("한국어", "ko"),
            ("Latina", "la"),
            ("古希腊语 / Ἑλληνική", "grc"),
            ("现代希腊语 / Ελληνικά", "el"),
            ("Русский", "ru"),
        ):
            self.fallback_language.addItem(label, language)
        self.analyze_button = QPushButton("分析文本")
        self.analyze_button.clicked.connect(self._emit_text_analysis)
        text_toolbar.addWidget(self.auto_language)
        text_toolbar.addStretch(1)
        text_toolbar.addWidget(self.fallback_language)
        text_toolbar.addWidget(self.analyze_button)
        input_card.content_layout.addLayout(text_toolbar)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "例如：[la]Salvē, amīce.[/la] [grc]Χαῖρε, ὦ φίλε.[/grc] "
            "[ru]Привет, мой друг.[/ru] 你好，朋友。"
        )
        self.text_edit.setMinimumHeight(180)
        input_card.content_layout.addWidget(self.text_edit, 1)

        reference_label = QLabel("声音参考")
        reference_label.setObjectName("sectionTitle")
        input_card.content_layout.addWidget(reference_label)
        reference_row = QHBoxLayout()
        self.reference_path = PathField(
            "3–30 秒干净人声 WAV / FLAC / MP3",
            file_filter="音频文件 (*.wav *.flac *.mp3 *.m4a *.ogg);;所有文件 (*.*)",
        )
        self.reference_path.path_changed.connect(self.reference_changed)
        self.reference_path.path_changed.connect(
            lambda _: self.reference_authorized.setChecked(False)
        )
        self.preview_reference_button = QPushButton("试听")
        self.preview_reference_button.clicked.connect(
            lambda: self.preview_reference_requested.emit(self.reference_path.path())
        )
        reference_row.addWidget(self.reference_path, 1)
        reference_row.addWidget(self.preview_reference_button)
        input_card.content_layout.addLayout(reference_row)

        self.reference_authorized = QCheckBox("我确认有权使用此参考声音进行合成")
        self.reference_authorized.setToolTip("更换参考文件后需要重新确认")
        input_card.content_layout.addWidget(self.reference_authorized)

        self.reference_transcript = QLineEdit()
        self.reference_transcript.setPlaceholderText("参考音频对应文本（精确克隆模式可填写）")
        transcript_row = QHBoxLayout()
        transcript_row.addWidget(self.reference_transcript, 1)
        self.reference_language = QComboBox()
        for label, language in (
            ("参考语种：自动", "auto"),
            ("中文", "zh"),
            ("English", "en"),
            ("日本語", "ja"),
            ("한국어", "ko"),
            ("粤语", "yue"),
            ("Latina", "la"),
            ("古希腊语 / Ἑλληνική", "grc"),
            ("现代希腊语 / Ελληνικά", "el"),
            ("Русский", "ru"),
            ("Español", "es"),
            ("العربية", "ar"),
        ):
            self.reference_language.addItem(label, language)
        transcript_row.addWidget(self.reference_language)
        input_card.content_layout.addLayout(transcript_row)

        segment_label = QLabel("语言分段预览")
        segment_label.setObjectName("sectionTitle")
        input_card.content_layout.addWidget(segment_label)
        self.segment_table = EmptyTable("点击“分析文本”预览语言边界和发音路径")
        self.segment_table.setColumnCount(4)
        self.segment_table.setHorizontalHeaderLabels(["片段", "语言", "置信度", "前端"])
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.setMinimumHeight(128)
        input_card.content_layout.addWidget(self.segment_table)
        splitter.addWidget(input_card)

        controls_card = Card("生成参数", "控制项会随所选后端能力启用")
        engine_form = QFormLayout()
        engine_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("未发现可用后端", "")
        self.capability_label = QLabel("扫描或导入模型后可生成")
        self.capability_label.setObjectName("muted")
        self.capability_label.setWordWrap(True)
        engine_form.addRow("TTS 后端", self.engine_combo)
        engine_form.addRow("可用控制", self.capability_label)
        controls_card.content_layout.addLayout(engine_form)

        speed_row = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(100)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix(" ×")
        self.speed_spin.setFixedWidth(92)
        self.speed_slider.valueChanged.connect(lambda value: self.speed_spin.setValue(value / 100))
        self.speed_spin.valueChanged.connect(lambda value: self.speed_slider.setValue(round(value * 100)))
        speed_row.addWidget(QLabel("语速"))
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_spin)
        controls_card.content_layout.addLayout(speed_row)

        emotion_title = QHBoxLayout()
        emotion_title.addWidget(QLabel("情绪向量"))
        self.auto_emotion = QCheckBox("从文本推断")
        emotion_title.addStretch(1)
        emotion_title.addWidget(self.auto_emotion)
        controls_card.content_layout.addLayout(emotion_title)

        emotion_grid = QGridLayout()
        emotion_grid.setHorizontalSpacing(10)
        emotion_grid.setVerticalSpacing(4)
        self.emotion_sliders: dict[str, QSlider] = {}
        self.emotion_values: dict[str, QLabel] = {}
        for index, (key, label) in enumerate(EMOTIONS):
            column_group = index // 4
            row = index % 4
            base_column = column_group * 3
            name_label = QLabel(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            value_label = QLabel("0.00")
            value_label.setObjectName("muted")
            value_label.setFixedWidth(31)
            slider.valueChanged.connect(
                lambda value, target=value_label: target.setText(f"{value / 100:.2f}")
            )
            emotion_grid.addWidget(name_label, row, base_column)
            emotion_grid.addWidget(slider, row, base_column + 1)
            emotion_grid.addWidget(value_label, row, base_column + 2)
            self.emotion_sliders[key] = slider
            self.emotion_values[key] = value_label
        emotion_grid.setColumnStretch(1, 1)
        emotion_grid.setColumnStretch(4, 1)
        controls_card.content_layout.addLayout(emotion_grid)

        self.style_instruction = QLineEdit()
        self.style_instruction.setPlaceholderText("风格说明：沉稳、稍快、像讲述古代史诗一样……")
        controls_card.content_layout.addWidget(self.style_instruction)

        option_form = QFormLayout()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("随机")
        self.output_directory = PathField("输出目录", mode="directory")
        option_form.addRow("随机种子", self.seed_spin)
        option_form.addRow("输出目录", self.output_directory)
        controls_card.content_layout.addLayout(option_form)

        self.validation_label = QLabel("")
        self.validation_label.setObjectName("muted")
        self.validation_label.setWordWrap(True)
        controls_card.content_layout.addWidget(self.validation_label)

        generate_row = QHBoxLayout()
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip("一次性 HTTP/进程请求暂不支持安全中断")
        self.stop_button.clicked.connect(self.stop_requested)
        self.generate_button = QPushButton("开始生成")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self._emit_generate)
        generate_row.addWidget(self.stop_button)
        generate_row.addStretch(1)
        generate_row.addWidget(self.generate_button)
        controls_card.content_layout.addLayout(generate_row)
        controls_card.content_layout.addStretch(1)
        splitter.addWidget(controls_card)
        splitter.setSizes([720, 520])
        root.addWidget(splitter, 1)

        queue_card = Card("生成任务记录")
        queue_toolbar = QHBoxLayout()
        self.queue_summary = QLabel("0 个任务")
        self.queue_summary.setObjectName("muted")
        clear_button = QPushButton("清空已完成")
        clear_button.clicked.connect(self.queue_clear_requested)
        open_button = QPushButton("打开输出目录")
        open_button.clicked.connect(lambda: self.open_output_requested.emit(self.output_directory.path()))
        queue_toolbar.addWidget(self.queue_summary)
        queue_toolbar.addStretch(1)
        queue_toolbar.addWidget(clear_button)
        queue_toolbar.addWidget(open_button)
        queue_card.content_layout.addLayout(queue_toolbar)
        self.queue_table = EmptyTable("暂无任务 · 填写文本后开始生成")
        self.queue_table.setColumnCount(6)
        self.queue_table.setHorizontalHeaderLabels(["任务", "后端", "语言", "时长", "状态", "输出"])
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.setMaximumHeight(185)
        queue_card.content_layout.addWidget(self.queue_table)
        root.addWidget(queue_card)

    def _emit_text_analysis(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.validation_label.setText("请先输入需要分析的文本。")
            return
        self.validation_label.clear()
        self.analyze_text_requested.emit(text)

    def _emit_generate(self) -> None:
        payload = self.request_payload()
        if not payload["text"]:
            self.validation_label.setText("文本不能为空。")
            return
        if not payload["engine_id"]:
            self.validation_label.setText("请先在“模型管理”中导入并启用一个后端。")
            return
        if payload["reference_audio"] and not payload["reference_authorized"]:
            self.validation_label.setText("请确认你有权使用所选参考声音。")
            return
        self.validation_label.clear()
        self.generate_requested.emit(payload)

    def request_payload(self) -> dict[str, Any]:
        language_data = self.fallback_language.currentData()
        return {
            "text": self.text_edit.toPlainText().strip(),
            "engine_id": str(self.engine_combo.currentData() or ""),
            "reference_audio": (
                self.reference_path.path() or None if self.reference_path.isEnabled() else None
            ),
            "reference_authorized": self.reference_authorized.isChecked(),
            "reference_transcript": self.reference_transcript.text().strip(),
            "reference_language": str(self.reference_language.currentData() or "auto"),
            "auto_language": self.auto_language.isChecked(),
            "language": str(language_data or "auto"),
            "speed": self.speed_spin.value() if self.speed_spin.isEnabled() else 1.0,
            "emotion": (
                {key: slider.value() / 100 for key, slider in self.emotion_sliders.items()}
                if next(iter(self.emotion_sliders.values())).isEnabled()
                else {}
            ),
            "auto_emotion": self.auto_emotion.isChecked(),
            "style_instruction": (
                self.style_instruction.text().strip()
                if self.style_instruction.isEnabled()
                else ""
            ),
            "seed": self.seed_spin.value(),
            "output_directory": self.output_directory.path() or None,
        }

    def set_engines(self, engines: Iterable[object]) -> None:
        current = self.engine_combo.currentData()
        active_engine = ""
        self.engine_combo.clear()
        for engine in engines:
            engine_id, label = _engine_label(engine)
            if engine_id:
                self.engine_combo.addItem(label, engine_id)
                if isinstance(engine, Mapping) and bool(engine.get("active", False)):
                    active_engine = engine_id
        if self.engine_combo.count() == 0:
            self.engine_combo.addItem("未发现可用后端", "")
        else:
            preferred = current or active_engine
            index = self.engine_combo.findData(preferred)
            if index >= 0:
                self.engine_combo.setCurrentIndex(index)

    def set_engine_capabilities(self, capabilities: Iterable[str]) -> None:
        ordered_values = [getattr(value, "value", str(value)) for value in capabilities]
        values = set(ordered_values)
        labels = [CAPABILITY_LABELS.get(value, value) for value in ordered_values]
        self.capability_label.setText("、".join(labels) if labels else "该后端未报告可控能力")
        supports_speed = "speed" in values
        self.speed_slider.setEnabled(supports_speed)
        self.speed_spin.setEnabled(supports_speed)
        supports_emotion = "emotion_vector" in values
        for slider in self.emotion_sliders.values():
            slider.setEnabled(supports_emotion)
        self.auto_emotion.blockSignals(True)
        self.auto_emotion.setChecked(False)
        self.auto_emotion.setEnabled(False)
        self.auto_emotion.setToolTip("v0.1 尚未接入文本情绪分类；可手动设置兼容后端的控制项")
        self.auto_emotion.blockSignals(False)
        self.style_instruction.setEnabled("style_instruction" in values)
        supports_clone = "voice_clone" in values
        self.reference_path.setEnabled(supports_clone)
        self.reference_authorized.setEnabled(supports_clone)
        if not supports_clone:
            self.reference_authorized.setChecked(False)
        self.reference_transcript.setEnabled(supports_clone)
        self.reference_language.setEnabled(supports_clone)
        self.preview_reference_button.setEnabled(supports_clone)

    def set_engine_languages(self, languages: Iterable[str]) -> None:
        target = str(self.fallback_language.currentData() or "auto")
        reference = str(self.reference_language.currentData() or "auto")
        normalized = list(dict.fromkeys(str(language).casefold() for language in languages))
        for combo, automatic_label, current in (
            (self.fallback_language, "默认：自动", target),
            (self.reference_language, "参考语种：自动", reference),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(automatic_label, "auto")
            for language in normalized:
                combo.addItem(LANGUAGE_LABELS.get(language, language), language)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def set_segments(self, segments: Iterable[object]) -> None:
        rows = list(segments)
        self.segment_table.setRowCount(len(rows))
        for row, segment in enumerate(rows):
            confidence = _segment_value(segment, "confidence", 0.0)
            confidence_text = (
                f"{float(confidence):.0%}"
                if isinstance(confidence, (int, float))
                else str(confidence)
            )
            values = (
                _segment_value(segment, "text", ""),
                _segment_value(segment, "language", "und"),
                confidence_text,
                _segment_value(
                    segment,
                    "frontend",
                    _segment_value(segment, "source", "auto"),
                ),
            )
            for column, value in enumerate(values):
                self.segment_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.segment_table.resizeColumnsToContents()

    def set_queue(self, jobs: Iterable[Mapping[str, Any]]) -> None:
        rows = list(jobs)
        self.queue_table.setRowCount(len(rows))
        keys = ("name", "engine", "language", "duration", "status", "output")
        for row, job in enumerate(rows):
            for column, key in enumerate(keys):
                self.queue_table.setItem(row, column, QTableWidgetItem(str(job.get(key, "—"))))
        self.queue_summary.setText(f"{len(rows)} 个任务")
        self.queue_table.resizeColumnsToContents()

    def set_status(self, text: str, *, busy: bool = False) -> None:
        self.status_chip.setText(text)
        self.generate_button.setEnabled(not busy)
        self.stop_button.setEnabled(False)

    def set_text(self, text: str) -> None:
        self.text_edit.setPlainText(text)

    def set_reference_audio(self, path: str | Path) -> None:
        self.reference_path.set_path(path)
