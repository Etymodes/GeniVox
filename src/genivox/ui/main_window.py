"""Main desktop window and page navigation for GeniVox."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from genivox.ui.pages import (
    ExperimentPage,
    ModelManagerPage,
    MultilingualPage,
    OverviewPage,
    SynthesisPage,
    TrainingPage,
    VoiceProfilePage,
)
from genivox.ui.theme import COLORS, STYLE_SHEET

NAVIGATION = (
    ("overview", "概览"),
    ("synthesis", "合成工作台"),
    ("voice_profile", "声音画像"),
    ("languages", "多语发音"),
    ("training", "数据与训练"),
    ("experiments", "实验对比"),
    ("models", "模型管理"),
)


class MainWindow(QMainWindow):
    """Top-level UI facade used by the application controller."""

    page_changed = Signal(str)
    close_requested = Signal()

    synthesis_requested = Signal(dict)
    text_analysis_requested = Signal(str)
    voice_analysis_requested = Signal(str)
    dataset_audit_requested = Signal(str)
    training_start_requested = Signal(dict)
    training_cancel_requested = Signal()
    model_import_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GeniVox · 本地多语声音实验台")
        self.setMinimumSize(800, 480)
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 800)
        else:
            available = screen.availableGeometry()
            self.resize(
                min(1480, max(800, int(available.width() * 0.94))),
                min(940, max(480, int(available.height() * 0.94))),
            )
        self.setStyleSheet(STYLE_SHEET)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 14)
        sidebar_layout.setSpacing(12)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = QLabel("GV")
        mark.setObjectName("appMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(38, 38)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        name = QLabel("GeniVox")
        name.setObjectName("appName")
        tagline = QLabel("LOCAL VOICE LAB")
        tagline.setObjectName("muted")
        tagline.setStyleSheet("font-size: 9px; letter-spacing: 1px;")
        brand_text.addWidget(name)
        brand_text.addWidget(tagline)
        brand.addWidget(mark)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        sidebar_layout.addLayout(brand)

        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for index, (key, label) in enumerate(NAVIGATION):
            item = QListWidgetItem(f"{index + 1:02d}   {label}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setSizeHint(item.sizeHint().expandedTo(item.sizeHint()))
            self.navigation.addItem(item)
        self.navigation.currentRowChanged.connect(self._on_navigation_changed)
        sidebar_layout.addWidget(self.navigation, 1)

        local_status = QFrame()
        local_status.setObjectName("panel")
        local_layout = QVBoxLayout(local_status)
        local_layout.setContentsMargins(12, 10, 12, 10)
        local_layout.setSpacing(3)
        local_title = QLabel("●  本地模式")
        local_title.setStyleSheet(f"color: {COLORS['green']}; font-weight: 650;")
        local_note = QLabel("工作区默认保存在本机")
        local_note.setObjectName("muted")
        local_note.setWordWrap(True)
        local_note.setToolTip("第三方模型桥以当前用户权限运行，请先审查其联网与文件访问行为")
        local_layout.addWidget(local_title)
        local_layout.addWidget(local_note)
        sidebar_layout.addWidget(local_status)
        version = QLabel("GeniVox 0.1.0")
        version.setObjectName("muted")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(version)
        central_layout.addWidget(sidebar)

        main_column = QVBoxLayout()
        main_column.setContentsMargins(0, 0, 0, 0)
        main_column.setSpacing(0)
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(50)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(24, 0, 24, 0)
        self.breadcrumb = QLabel("GeniVox  /  概览")
        self.breadcrumb.setObjectName("muted")
        self.controller_status = QLabel("控制器未连接")
        self.controller_status.setObjectName("chip")
        top_layout.addWidget(self.breadcrumb)
        top_layout.addStretch(1)
        top_layout.addWidget(self.controller_status)
        main_column.addWidget(top_bar)

        self.page_stack = QStackedWidget()
        self.page_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.overview_page = OverviewPage()
        self.synthesis_page = SynthesisPage()
        self.voice_profile_page = VoiceProfilePage()
        self.multilingual_page = MultilingualPage()
        self.training_page = TrainingPage()
        self.experiment_page = ExperimentPage()
        self.model_manager_page = ModelManagerPage()
        self.pages: dict[str, QWidget] = {
            "overview": self.overview_page,
            "synthesis": self.synthesis_page,
            "voice_profile": self.voice_profile_page,
            "languages": self.multilingual_page,
            "training": self.training_page,
            "experiments": self.experiment_page,
            "models": self.model_manager_page,
        }
        self.page_containers: dict[str, QScrollArea] = {}
        for key, _ in NAVIGATION:
            container = QScrollArea()
            container.setFrameShape(QFrame.Shape.NoFrame)
            container.setWidgetResizable(True)
            container.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            container.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            container.setWidget(self.pages[key])
            self.page_containers[key] = container
            self.page_stack.addWidget(container)
        main_column.addWidget(self.page_stack, 1)
        central_layout.addLayout(main_column, 1)

        self._connect_pages()
        self.navigation.setCurrentRow(0)

    def _connect_pages(self) -> None:
        self.overview_page.navigate_requested.connect(self.show_page)
        self.synthesis_page.generate_requested.connect(self.synthesis_requested)
        self.synthesis_page.analyze_text_requested.connect(self.text_analysis_requested)
        self.multilingual_page.analyze_text_requested.connect(self.text_analysis_requested)
        self.voice_profile_page.analyze_requested.connect(self.voice_analysis_requested)
        self.training_page.audit_requested.connect(self.dataset_audit_requested)
        self.training_page.start_requested.connect(self.training_start_requested)
        self.training_page.cancel_requested.connect(self.training_cancel_requested)
        self.model_manager_page.import_requested.connect(self.model_import_requested)

        self.multilingual_page.use_in_synthesis_requested.connect(self._apply_language_plan)
        self.voice_profile_page.apply_to_synthesis_requested.connect(self._apply_voice_profile)

    def _on_navigation_changed(self, index: int) -> None:
        if not 0 <= index < len(NAVIGATION):
            return
        key, label = NAVIGATION[index]
        self.page_stack.setCurrentWidget(self.page_containers[key])
        self.breadcrumb.setText(f"GeniVox  /  {label}")
        self.page_changed.emit(key)

    def show_page(self, key: str) -> None:
        for index, (route, _) in enumerate(NAVIGATION):
            if route == key:
                self.navigation.setCurrentRow(index)
                return
        raise KeyError(f"Unknown GeniVox page: {key}")

    def _apply_language_plan(self, payload: Mapping[str, Any]) -> None:
        self.synthesis_page.set_text(str(payload.get("text", "")))
        self.synthesis_page.auto_language.setChecked(bool(payload.get("auto_detect", True)))
        self.show_page("synthesis")

    def _apply_voice_profile(self, payload: Mapping[str, Any]) -> None:
        for slider in self.synthesis_page.emotion_sliders.values():
            slider.setValue(0)
        path = payload.get("audio_path")
        self.synthesis_page.set_reference_audio(str(path or ""))
        self.synthesis_page.reference_authorized.setChecked(
            bool(path and payload.get("authorized", False))
        )
        self.synthesis_page.reference_transcript.setText(str(payload.get("transcript", "")))
        language_codes = {
            "自动": "auto",
            "中文": "zh",
            "English": "en",
            "Latina": "la",
            "古希腊语 / Ἑλληνική": "grc",
            "现代希腊语 / Ελληνικά": "el",
            "Русский": "ru",
            "日本語": "ja",
        }
        reference_language = language_codes.get(str(payload.get("language_hint", "自动")), "auto")
        reference_index = self.synthesis_page.reference_language.findData(reference_language)
        if reference_index >= 0:
            self.synthesis_page.reference_language.setCurrentIndex(reference_index)
        if style := payload.get("style_instruction"):
            self.synthesis_page.style_instruction.setText(str(style))
        else:
            self.synthesis_page.style_instruction.clear()
        emotion_aliases = {
            "fearful": "afraid",
            "fear": "afraid",
            "disgust": "disgusted",
            "surprise": "surprised",
            "neutral": "calm",
        }
        emotion = payload.get("emotion", {})
        unmapped_emotions: list[str] = []
        mapped_emotions: list[str] = []
        if isinstance(emotion, Mapping):
            for label, value in emotion.items():
                target = emotion_aliases.get(str(label), str(label))
                slider = self.synthesis_page.emotion_sliders.get(target)
                if slider is not None:
                    slider.setValue(round(max(0.0, min(1.0, float(value))) * 100))
                    if target != str(label):
                        mapped_emotions.append(f"{label}→{target}")
                elif float(value) > 0.0:
                    unmapped_emotions.append(str(label))
        notices: list[str] = []
        if emotion and not next(iter(self.synthesis_page.emotion_sliders.values())).isEnabled():
            notices.append("当前后端不支持情绪向量；画像概率不会用于合成")
        if unmapped_emotions:
            notices.append("未映射的情绪标签：" + "、".join(unmapped_emotions))
        if mapped_emotions:
            notices.append("近似标签映射：" + "、".join(mapped_emotions))
        self.synthesis_page.validation_label.setText("；".join(notices))
        self.show_page("synthesis")

    def set_engines(self, engines: Iterable[object]) -> None:
        """Push one engine snapshot into all pages that consume it."""

        snapshot = list(engines)
        runnable = [
            engine
            for engine in snapshot
            if not isinstance(engine, Mapping) or bool(engine.get("runnable", True))
        ]
        self.overview_page.set_engines(snapshot)
        self.synthesis_page.set_engines(runnable)
        self.training_page.set_engines(runnable)
        self.experiment_page.set_engines(runnable)
        self.model_manager_page.set_engines(snapshot)

    def set_segments(self, segments: Iterable[object]) -> None:
        snapshot = list(segments)
        self.synthesis_page.set_segments(snapshot)
        self.multilingual_page.set_segments(snapshot)

    def set_profile(self, profile: object) -> None:
        self.voice_profile_page.set_profile(profile)

    def set_dataset_report(self, report: Mapping[str, Any]) -> None:
        self.training_page.set_dataset_report(report)

    def append_metric(self, metric: object, values: Mapping[str, float] | None = None) -> None:
        self.training_page.append_metric(metric, values)

    def set_status(self, text: str, *, connected: bool = True) -> None:
        self.controller_status.setText(text)
        color = COLORS["green"] if connected else COLORS["orange"]
        self.controller_status.setStyleSheet(
            f"color: {color}; background: #153039; border: 1px solid #24515C; "
            "border-radius: 9px; padding: 2px 8px;"
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.close_requested.emit()
        super().closeEvent(event)
