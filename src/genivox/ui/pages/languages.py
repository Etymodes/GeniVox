"""Mixed-language segmentation and pronunciation planning page."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.widgets import Card, EmptyTable, PageHeader


def _value(item: object, key: str, default: object = "") -> object:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


class MultilingualPage(QWidget):
    """Configure per-segment pronunciation while retaining one speaker identity."""

    analyze_text_requested = Signal(str)
    preview_requested = Signal(dict)
    use_in_synthesis_requested = Signal(dict)
    lexicon_import_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._segments: list[dict[str, Any]] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader("多语发音", "自动识别句内语言边界，为古典语言指定可复现的发音方案")
        self.status_label = QLabel("等待文本")
        self.status_label.setObjectName("chip")
        header.actions.addWidget(self.status_label)
        root.addWidget(header)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        text_card = Card("混合语言文本", "同一个声音画像贯穿全部片段，不必按语言更换说话人模型")
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "例如：[la]Caesar dixit[/la]: [grc]Ἀνερρίφθω κύβος[/grc]. "
            "[ru]Затем он перешёл Рубикон[/ru]。"
        )
        self.text_edit.setMinimumHeight(145)
        text_card.content_layout.addWidget(self.text_edit)
        text_controls = QHBoxLayout()
        self.auto_detect = QCheckBox("自动语言识别")
        self.auto_detect.setChecked(True)
        self.auto_detect.setEnabled(False)
        self.auto_detect.setToolTip("分析阶段固定启用；显式语言标签始终优先")
        self.keep_voice = QCheckBox("锁定同一声纹")
        self.keep_voice.setChecked(True)
        self.keep_voice.setEnabled(False)
        self.keep_voice.setToolTip("v0.1 混读管线固定复用同一后端和参考声音")
        analyze_button = QPushButton("切分并标注")
        analyze_button.setObjectName("primaryButton")
        analyze_button.clicked.connect(self._emit_analysis)
        text_controls.addWidget(self.auto_detect)
        text_controls.addWidget(self.keep_voice)
        text_controls.addStretch(1)
        text_controls.addWidget(analyze_button)
        text_card.content_layout.addLayout(text_controls)
        top_row.addWidget(text_card, 3)

        pronunciation_card = Card(
            "古典语言发音计划（元数据）",
            "v0.1 选项会随计划保存，但不会改写 eSpeak IPA；只有兼容 process bridge 才能应用",
        )
        pronunciation_form = QFormLayout()
        self.latin_profile = QComboBox()
        self.latin_profile.addItems(["古典拉丁语（公元前1世纪重构）", "教会拉丁语", "自定义词典"])
        self.greek_profile = QComboBox()
        self.greek_profile.addItems(
            ["阿提卡古希腊语（公元前5世纪）", "通用希腊语（公元1世纪）", "伊拉斯谟式", "现代希腊语"]
        )
        self.russian_stress = QComboBox()
        self.russian_stress.addItems(["自动重音词典", "文本显式重音优先", "仅规则推断"])
        self.numeral_mode = QComboBox()
        self.numeral_mode.addItems(["按片段语言展开", "保持原字符", "逐字符读取"])
        pronunciation_form.addRow("拉丁语", self.latin_profile)
        pronunciation_form.addRow("古希腊语", self.greek_profile)
        pronunciation_form.addRow("俄语重音", self.russian_stress)
        pronunciation_form.addRow("数字与符号", self.numeral_mode)
        pronunciation_card.content_layout.addLayout(pronunciation_form)
        top_row.addWidget(pronunciation_card, 2)
        root.addLayout(top_row)

        segment_card = Card(
            "片段与音素预览",
            "eSpeak-ng 仅作基线；可双击 IPA 列校订，并传给兼容的 process bridge",
        )
        self.segment_table = EmptyTable("尚未切分文本")
        self.segment_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.SelectedClicked
        )
        self.segment_table.setColumnCount(8)
        self.segment_table.setHorizontalHeaderLabels(
            ["序号", "文本片段", "语言", "置信度", "发音前端", "IPA 预览", "声纹", "连接方式"]
        )
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.setMinimumHeight(210)
        segment_card.content_layout.addWidget(self.segment_table)
        segment_actions = QHBoxLayout()
        self.crossfade = QComboBox()
        self.crossfade.addItem("v0.1：PCM 顺序拼接（无交叉淡化）")
        self.crossfade.setEnabled(False)
        self.crossfade.setToolTip("交叉淡化与响度匹配计划在后续音频拼接层实现")
        preview_button = QPushButton("试听所选片段")
        preview_button.setEnabled(False)
        preview_button.setToolTip("片段试听需要可取消的流式后端桥，计划在后续版本接入")
        preview_button.clicked.connect(self._emit_preview)
        use_button = QPushButton("发送到合成工作台")
        use_button.setObjectName("primaryButton")
        use_button.clicked.connect(self._emit_use)
        segment_actions.addWidget(self.crossfade)
        segment_actions.addStretch(1)
        segment_actions.addWidget(preview_button)
        segment_actions.addWidget(use_button)
        segment_card.content_layout.addLayout(segment_actions)
        root.addWidget(segment_card, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        lexicon_card = Card(
            "发音覆盖词典",
            "规则会随计划传给自定义 process bridge；内置后端暂不直接应用",
        )
        self.lexicon_table = QTableWidget(0, 5)
        self.lexicon_table.setHorizontalHeaderLabels(["原词", "语言", "IPA / 音素", "重音", "来源"])
        self.lexicon_table.horizontalHeader().setStretchLastSection(True)
        self.lexicon_table.verticalHeader().setVisible(False)
        self.lexicon_table.setAlternatingRowColors(True)
        self.lexicon_table.setMinimumHeight(145)
        lexicon_card.content_layout.addWidget(self.lexicon_table)
        lexicon_actions = QHBoxLayout()
        add_button = QPushButton("新增规则")
        add_button.clicked.connect(self.add_lexicon_row)
        import_button = QPushButton("导入词典")
        import_button.setEnabled(False)
        import_button.setToolTip("v0.1 可直接在表格录入；文件导入将在后续版本实现")
        import_button.clicked.connect(self.lexicon_import_requested)
        remove_button = QPushButton("删除所选")
        remove_button.clicked.connect(self.remove_selected_lexicon_rows)
        lexicon_actions.addWidget(add_button)
        lexicon_actions.addWidget(import_button)
        lexicon_actions.addWidget(remove_button)
        lexicon_actions.addStretch(1)
        lexicon_card.content_layout.addLayout(lexicon_actions)
        bottom_row.addWidget(lexicon_card, 3)

        support_card = Card("后端语言覆盖")
        self.support_table = EmptyTable("导入模型后显示语言与发音能力")
        self.support_table.setColumnCount(4)
        self.support_table.setHorizontalHeaderLabels(["后端", "语言", "音素输入", "同声纹跨语"])
        self.support_table.horizontalHeader().setStretchLastSection(True)
        self.support_table.setMinimumHeight(145)
        support_card.content_layout.addWidget(self.support_table)
        bottom_row.addWidget(support_card, 2)
        root.addLayout(bottom_row)

    def _emit_analysis(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.status_label.setText("文本为空")
            return
        self.status_label.setText("等待分析")
        self.analyze_text_requested.emit(text)

    def _settings(self) -> dict[str, Any]:
        return {
            "text": self.text_edit.toPlainText().strip(),
            "auto_detect": self.auto_detect.isChecked(),
            "keep_voice": self.keep_voice.isChecked(),
            "latin_profile": self.latin_profile.currentText(),
            "greek_profile": self.greek_profile.currentText(),
            "russian_stress": self.russian_stress.currentText(),
            "numeral_mode": self.numeral_mode.currentText(),
            "crossfade": self.crossfade.currentText(),
            "lexicon": self.lexicon_entries(),
            "segments": self.segment_entries(),
        }

    def _emit_preview(self) -> None:
        payload = self._settings()
        payload["selected_row"] = self.segment_table.currentRow()
        self.preview_requested.emit(payload)

    def _emit_use(self) -> None:
        self.use_in_synthesis_requested.emit(self._settings())

    def add_lexicon_row(self) -> None:
        row = self.lexicon_table.rowCount()
        self.lexicon_table.insertRow(row)
        defaults = ("", "Latina", "", "", "人工")
        for column, value in enumerate(defaults):
            self.lexicon_table.setItem(row, column, QTableWidgetItem(value))
        self.lexicon_table.setCurrentCell(row, 0)
        self.lexicon_table.editItem(self.lexicon_table.item(row, 0))

    def remove_selected_lexicon_rows(self) -> None:
        rows = sorted({index.row() for index in self.lexicon_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.lexicon_table.removeRow(row)

    def lexicon_entries(self) -> list[dict[str, str]]:
        keys = ("word", "language", "phonemes", "stress", "source")
        entries: list[dict[str, str]] = []
        for row in range(self.lexicon_table.rowCount()):
            entry: dict[str, str] = {}
            for column, key in enumerate(keys):
                item = self.lexicon_table.item(row, column)
                entry[key] = item.text().strip() if item else ""
            entries.append(entry)
        return entries

    def segment_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for row, segment in enumerate(self._segments):
            entry = dict(segment)
            phoneme_item = self.segment_table.item(row, 5)
            phonemes = phoneme_item.text().strip() if phoneme_item else ""
            entry["phonemes"] = "" if phonemes == "—" else phonemes
            entries.append(entry)
        return entries

    def set_segments(self, segments: Iterable[object]) -> None:
        rows = list(segments)
        self._segments = [
            {
                "text": str(_value(segment, "text", "")),
                "language": str(_value(segment, "language", "und")),
                "start": int(_value(segment, "start", 0)),
                "end": int(_value(segment, "end", 0)),
                "source": str(_value(segment, "source", "auto")),
                "confidence": float(_value(segment, "confidence", 0.0)),
                "frontend": str(_value(segment, "frontend", "auto")),
                "phonemes": str(_value(segment, "phonemes", "—")),
                "join": str(_value(segment, "join", "PCM 顺序拼接")),
            }
            for segment in rows
        ]
        self.segment_table.setRowCount(len(rows))
        for row, segment in enumerate(rows):
            confidence = _value(segment, "confidence", 0.0)
            confidence_text = (
                f"{float(confidence):.0%}"
                if isinstance(confidence, (int, float))
                else str(confidence)
            )
            values = (
                row + 1,
                _value(segment, "text", ""),
                _value(segment, "language", "und"),
                confidence_text,
                _value(segment, "frontend", _value(segment, "source", "auto")),
                _value(segment, "phonemes", "—"),
                "锁定" if self.keep_voice.isChecked() else "自动",
                _value(segment, "join", "PCM 顺序拼接"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column != 5:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.segment_table.setItem(row, column, item)
        self.segment_table.resizeColumnsToContents()
        self.status_label.setText(f"已识别 {len(rows)} 段")

    def set_engines(self, engines: Iterable[Mapping[str, Any]]) -> None:
        rows = list(engines)
        self.support_table.setRowCount(len(rows))
        for row, engine in enumerate(rows):
            languages = engine.get("languages", [])
            capabilities = {
                getattr(item, "value", str(item)) for item in engine.get("capabilities", [])
            }
            values = (
                engine.get("name", engine.get("id", "—")),
                "、".join(map(str, languages)) if not isinstance(languages, str) else languages,
                "是" if "phoneme_input" in capabilities else "否",
                "是" if "cross_lingual" in capabilities else "否",
            )
            for column, value in enumerate(values):
                self.support_table.setItem(row, column, QTableWidgetItem(str(value)))
        self.support_table.resizeColumnsToContents()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_text(self, text: str) -> None:
        self.text_edit.setPlainText(text)
