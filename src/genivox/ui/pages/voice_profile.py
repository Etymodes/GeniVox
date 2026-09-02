"""Voice-reference analysis and reusable voice profile page."""

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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.theme import COLORS
from genivox.ui.widgets import Card, EmptyTable, MetricCard, PageHeader, PathField, WaveformWidget


def _read(data: object, key: str, default: object = None) -> object:
    if isinstance(data, Mapping):
        return data.get(key, default)
    return getattr(data, key, default)


def _number(value: object, suffix: str, digits: int = 1) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}{suffix}"


class VoiceProfilePage(QWidget):
    """Build a measurable voice profile from imported or recorded audio."""

    import_requested = Signal(str)
    recording_requested = Signal(bool)
    analyze_requested = Signal(str)
    save_profile_requested = Signal(dict)
    apply_to_synthesis_requested = Signal(dict)
    play_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._emotion: dict[str, float] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(14)

        header = PageHeader(
            "声音画像",
            "测量基频、节奏峰和能量；只有配置独立模型后才显示情绪概率",
        )
        self.record_button = QPushButton("开始录音")
        self.record_button.setObjectName("recordButton")
        self.record_button.setCheckable(True)
        self.record_button.setEnabled(False)
        self.record_button.setText("录音（v0.2）")
        self.record_button.setToolTip("当前版本请导入本地 PCM WAV")
        self.record_button.toggled.connect(self._toggle_recording)
        header.actions.addWidget(self.record_button)
        root.addWidget(header)

        source_card = Card("参考声音", "建议使用 10–30 秒、单人、无混响和背景音乐的录音")
        source_row = QHBoxLayout()
        self.audio_path = PathField(
            "选择未压缩 PCM WAV；其他格式请先用 FFmpeg 转换",
            file_filter="PCM WAV (*.wav *.wave);;所有文件 (*.*)",
        )
        self.import_button = QPushButton("导入")
        self.import_button.clicked.connect(self._emit_import)
        self.play_button = QPushButton("试听")
        self.play_button.clicked.connect(lambda: self.play_requested.emit(self.audio_path.path()))
        self.analyze_button = QPushButton("智能分析")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self._emit_analysis)
        source_row.addWidget(self.audio_path, 1)
        source_row.addWidget(self.import_button)
        source_row.addWidget(self.play_button)
        source_row.addWidget(self.analyze_button)
        source_card.content_layout.addLayout(source_row)
        self.source_status = QLabel("未载入声音")
        self.source_status.setObjectName("muted")
        source_card.content_layout.addWidget(self.source_status)
        self.reference_transcript = QLineEdit()
        self.reference_transcript.setPlaceholderText("可选：参考录音的精确转写，会写入声音画像")
        source_card.content_layout.addWidget(self.reference_transcript)
        root.addWidget(source_card)

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        signal_card = Card("声学轮廓", "波形由音频分析任务异步返回")
        self.waveform = WaveformWidget()
        signal_card.content_layout.addWidget(self.waveform, 1)
        metric_row = QHBoxLayout()
        metric_row.setSpacing(8)
        self.duration_card = MetricCard("文件时长", "—", "未裁剪 / 秒")
        self.pitch_card = MetricCard("平均基频", "—", "F0 / Hz", COLORS["cyan"])
        self.rate_card = MetricCard("节奏峰", "—", "声学峰 / 秒", COLORS["primary"])
        self.energy_card = MetricCard("平均能量", "—", "RMS / dBFS", COLORS["orange"])
        for card in (self.duration_card, self.pitch_card, self.rate_card, self.energy_card):
            metric_row.addWidget(card, 1)
        signal_card.content_layout.addLayout(metric_row)
        main_row.addWidget(signal_card, 3)

        profile_card = Card(
            "分析出的表达方式",
            "模型概率只读；风格说明可校正，再映射到兼容的合成后端",
        )
        self.dominant_emotion = QLabel("尚未分析")
        self.dominant_emotion.setStyleSheet("font-size: 21px; font-weight: 700;")
        profile_card.content_layout.addWidget(self.dominant_emotion)
        self.emotion_table = EmptyTable("等待情绪分析")
        self.emotion_table.setColumnCount(2)
        self.emotion_table.setHorizontalHeaderLabels(["情绪", "概率 / 强度"])
        self.emotion_table.horizontalHeader().setStretchLastSection(True)
        self.emotion_table.setMinimumHeight(185)
        profile_card.content_layout.addWidget(self.emotion_table)
        self.style_text = QTextEdit()
        self.style_text.setPlaceholderText("声学规则生成的说明，可人工修订；不是情绪识别")
        self.style_text.setMaximumHeight(78)
        profile_card.content_layout.addWidget(self.style_text)
        main_row.addWidget(profile_card, 2)
        root.addLayout(main_row, 1)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)
        settings_card = Card("分析策略")
        settings_form = QFormLayout()
        self.language_hint = QComboBox()
        self.language_hint.addItems(
            [
                "自动",
                "中文",
                "English",
                "Latina",
                "古希腊语 / Ἑλληνική",
                "现代希腊语 / Ελληνικά",
                "Русский",
                "日本語",
            ]
        )
        self.emotion_model = QComboBox()
        self.emotion_model.addItem("外部 emotion2vec（需配置环境变量）")
        self.emotion_model.setEnabled(False)
        self.trim_silence = QCheckBox("后续：忽略首尾静音")
        self.trim_silence.setEnabled(False)
        self.normalize_audio = QCheckBox("后续：仅为分析做响度归一化")
        self.normalize_audio.setEnabled(False)
        settings_form.addRow("样本语言", self.language_hint)
        settings_form.addRow("情绪分析器", self.emotion_model)
        settings_form.addRow("预处理", self.trim_silence)
        settings_form.addRow("", self.normalize_audio)
        settings_card.content_layout.addLayout(settings_form)
        bottom_row.addWidget(settings_card, 1)

        library_card = Card(
            "当前画像样本（v0.1 单样本）",
            "保存当前参考录音的 JSON 画像；加载与多样本聚合将在后续版本接入",
        )
        self.samples_table = EmptyTable("尚未向当前画像添加声音样本")
        self.samples_table.setColumnCount(5)
        self.samples_table.setHorizontalHeaderLabels(["文件", "语言", "时长", "质量", "用途"])
        self.samples_table.horizontalHeader().setStretchLastSection(True)
        self.samples_table.setMinimumHeight(145)
        library_card.content_layout.addWidget(self.samples_table)
        self.authorized_voice = QCheckBox("我确认拥有此声音的克隆、合成与训练授权")
        self.authorized_voice.setToolTip("声音画像会把本次确认及时间写入本地授权记录")
        library_card.content_layout.addWidget(self.authorized_voice)
        action_row = QHBoxLayout()
        self.profile_name = QLineEdit()
        self.profile_name.setPlaceholderText("画像名称，例如：松原-自然叙述")
        save_button = QPushButton("保存画像")
        save_button.clicked.connect(self._emit_save)
        apply_button = QPushButton("用于合成")
        apply_button.setObjectName("primaryButton")
        apply_button.clicked.connect(self._emit_apply)
        action_row.addWidget(self.profile_name, 1)
        action_row.addWidget(save_button)
        action_row.addWidget(apply_button)
        library_card.content_layout.addLayout(action_row)
        bottom_row.addWidget(library_card, 2)
        root.addLayout(bottom_row)

    def _toggle_recording(self, active: bool) -> None:
        self.record_button.setText("停止录音" if active else "开始录音")
        self.recording_requested.emit(active)

    def _emit_import(self) -> None:
        path = self.audio_path.path()
        if path:
            self.source_status.setText("等待导入…")
            self.import_requested.emit(path)
        else:
            self.source_status.setText("请先选择声音文件。")

    def _emit_analysis(self) -> None:
        path = self.audio_path.path()
        if path:
            self.source_status.setText("等待分析任务…")
            self.analyze_requested.emit(path)
        else:
            self.source_status.setText("请先选择或录制声音。")

    def _profile_payload(self) -> dict[str, Any]:
        return {
            "name": self.profile_name.text().strip(),
            "audio_path": self.audio_path.path(),
            "transcript": self.reference_transcript.text().strip(),
            "language_hint": self.language_hint.currentText(),
            "style_instruction": self.style_text.toPlainText().strip(),
            "emotion": dict(self._emotion),
            "trim_silence": self.trim_silence.isChecked(),
            "normalize_for_analysis": self.normalize_audio.isChecked(),
            "authorized": self.authorized_voice.isChecked(),
        }

    def _emit_save(self) -> None:
        self.save_profile_requested.emit(self._profile_payload())

    def _emit_apply(self) -> None:
        self.apply_to_synthesis_requested.emit(self._profile_payload())

    def set_profile(self, profile: object) -> None:
        duration = _read(profile, "duration_seconds")
        mean_f0 = _read(profile, "mean_f0_hz")
        rate = _read(profile, "acoustic_peak_rate_hz")
        energy = _read(profile, "rms_dbfs")
        pitch_min = _read(profile, "f0_min_hz")
        pitch_max = _read(profile, "f0_max_hz")

        self.duration_card.set_metric(_number(duration, " s"), "有效声音时长")
        pitch_detail = "F0 / Hz"
        if pitch_min is not None and pitch_max is not None:
            pitch_detail = f"范围 {_number(pitch_min, '')}–{_number(pitch_max, '')} Hz"
        self.pitch_card.set_metric(_number(mean_f0, " Hz"), pitch_detail)
        self.rate_card.set_metric(_number(rate, ""), "估计声学峰 / 秒（非转写音节数）")
        self.energy_card.set_metric(_number(energy, " dB"), "RMS / dBFS")

        emotion = _read(profile, "emotion", {}) or {}
        if isinstance(emotion, Mapping):
            self._emotion = {str(name): float(value) for name, value in emotion.items()}
            sorted_emotions = sorted(emotion.items(), key=lambda item: float(item[1]), reverse=True)
            self.emotion_table.setRowCount(len(sorted_emotions))
            for row, (name, value) in enumerate(sorted_emotions):
                self.emotion_table.setItem(row, 0, QTableWidgetItem(str(name)))
                self.emotion_table.setItem(row, 1, QTableWidgetItem(f"{float(value):.1%}"))
            self.dominant_emotion.setText(str(sorted_emotions[0][0]) if sorted_emotions else "未识别")

        self.style_text.setPlainText(str(_read(profile, "style_instruction", "") or ""))
        warnings = _read(profile, "warnings", []) or []
        status = "分析完成" if not warnings else "分析完成 · " + "；".join(map(str, warnings))
        self.source_status.setText(status)

    def clear_analysis(self, status: str = "分析已清除") -> None:
        self._emotion = {}
        self.duration_card.set_metric("—", "有效声音时长")
        self.pitch_card.set_metric("—", "F0 / Hz")
        self.rate_card.set_metric("—", "估计声学峰 / 秒（非转写音节数）")
        self.energy_card.set_metric("—", "RMS / dBFS")
        self.dominant_emotion.setText("尚未分析")
        self.emotion_table.setRowCount(0)
        self.style_text.clear()
        self.reference_transcript.clear()
        self.waveform.set_samples([])
        self.source_status.setText(status)

    def set_waveform(self, samples: Iterable[float]) -> None:
        self.waveform.set_samples(samples)

    def set_samples(self, samples: Iterable[Mapping[str, Any]]) -> None:
        rows = list(samples)
        self.samples_table.setRowCount(len(rows))
        keys = ("file", "language", "duration", "quality", "purpose")
        for row, sample in enumerate(rows):
            for column, key in enumerate(keys):
                self.samples_table.setItem(row, column, QTableWidgetItem(str(sample.get(key, "—"))))

    def set_status(self, text: str, *, busy: bool = False) -> None:
        self.source_status.setText(text)
        self.analyze_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy)

    def set_audio_path(self, path: str | Path) -> None:
        self.audio_path.set_path(path)
