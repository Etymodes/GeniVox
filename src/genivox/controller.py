"""Qt controller that connects the workbench UI to local, testable services."""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThreadPool, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from genivox.audio import (
    ExternalEmotionAnalyzer,
    NoOpEmotionAnalyzer,
    analyze_prosody,
    read_pcm_wav,
)
from genivox.core.config import (
    default_engine_registry,
    load_or_create_engine_registry,
    registry_path,
)
from genivox.core.models import (
    Capability,
    EngineManifest,
    EngineTransport,
    LanguageSegment,
    SynthesisRequest,
    SynthesisResult,
)
from genivox.core.paths import WorkspacePaths, ensure_private_directory, protect_private_file
from genivox.core.profile import (
    ConsentRecord,
    SourceRecording,
    VoiceProfile,
    file_sha256,
    save_voice_profile,
)
from genivox.engines import EngineConfigurationError, EngineRegistry, SynthesisPipeline
from genivox.experiments import ExperimentRecord, ExperimentStore
from genivox.languages import EspeakNgPhonemizer, LanguageRouter
from genivox.services.workers import FunctionWorker
from genivox.system import probe_system
from genivox.training import (
    DatasetAudit,
    MetricParseError,
    RunStore,
    TrainingProcess,
    TrainingRunner,
    audit_dataset,
    load_dataset_manifest,
    parse_metric_line,
)
from genivox.ui import MainWindow

_LANGUAGE_LABELS = {
    "默认：自动": "auto",
    "自动": "auto",
    "中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
    "Latina": "la",
    "古希腊语 / Ἑλληνική": "grc",
    "现代希腊语 / Ελληνικά": "el",
    "Русский": "ru",
}

_PROCESS_PRESETS: dict[str, tuple[list[Capability], list[str]]] = {
    "IndexTTS2.5": (
        [
            Capability.VOICE_CLONE,
            Capability.CROSS_LINGUAL,
            Capability.SPEED,
            Capability.EMOTION_VECTOR,
            Capability.STYLE_INSTRUCTION,
        ],
        ["zh", "en", "ja", "es", "ar"],
    ),
    "VoxCPM2": (
        [
            Capability.VOICE_CLONE,
            Capability.CROSS_LINGUAL,
            Capability.STYLE_INSTRUCTION,
            Capability.FINE_TUNE,
        ],
        [
            "ar",
            "my",
            "zh",
            "da",
            "nl",
            "en",
            "fi",
            "fr",
            "de",
            "el",
            "he",
            "hi",
            "id",
            "it",
            "ja",
            "km",
            "ko",
            "lo",
            "ms",
            "no",
            "pl",
            "pt",
            "ru",
            "es",
            "sw",
            "sv",
            "tl",
            "th",
            "tr",
            "vi",
        ],
    ),
}

_MAX_SYNTHESIS_CHARACTERS = 20_000
_MAX_LANGUAGE_SEGMENTS = 256


class WorkbenchController(QObject):
    """Own application state while pages remain passive Qt widgets."""

    training_log_received = Signal(str)
    training_metric_received = Signal(object)
    training_finished = Signal(object)

    def __init__(
        self,
        window: MainWindow,
        workspace: WorkspacePaths,
        *,
        thread_pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(window)
        self.window = window
        self.workspace = workspace.ensure()
        registry_warning = ""
        try:
            self.registry: EngineRegistry = load_or_create_engine_registry(self.workspace)
        except EngineConfigurationError as exc:
            damaged_path = registry_path(self.workspace)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            backup_path = damaged_path.with_name(f"registry.invalid-{timestamp}.json")
            suffix = 2
            while backup_path.exists():
                backup_path = damaged_path.with_name(
                    f"registry.invalid-{timestamp}-{suffix}.json"
                )
                suffix += 1
            damaged_path.replace(backup_path)
            self.registry = default_engine_registry(self.workspace)
            self.registry.save(damaged_path)
            registry_warning = f"引擎注册表无效，已备份为 {backup_path.name} 并恢复默认项：{exc}"
        self.router = LanguageRouter()
        self.thread_pool = thread_pool or QThreadPool.globalInstance()
        self.experiments = ExperimentStore(self.workspace.runs / "experiments.jsonl")
        self.training_runner = TrainingRunner(RunStore(self.workspace.runs / "training"))
        self._workers: set[FunctionWorker] = set()
        self._segments: list[LanguageSegment] = []
        self._queue: list[dict[str, Any]] = []
        self._recent_tasks: list[dict[str, Any]] = []
        self._candidates: list[dict[str, Any]] = []
        self._experiment_results: list[dict[str, Any]] = []
        self._experiment_cancel = threading.Event()
        self._active_training: TrainingProcess | None = None
        self._active_training_task: dict[str, Any] | None = None
        self._last_training_run_path: Path | None = None
        self._last_profile: Any = None
        self._last_profile_path: Path | None = None
        self._selected_voice_path: Path | None = None
        self._voice_analysis_revision = 0
        self._language_plan: dict[str, Any] = {}
        self._language_rows: list[dict[str, Any]] = []
        self._language_rows_text = ""
        self._text_analysis_revision = 0
        self._last_dataset_audit: DatasetAudit | None = None
        self._last_audited_manifest: Path | None = None
        self._last_audited_manifest_sha256: str | None = None
        self._last_audited_audio_snapshot: tuple[tuple[str, int, int], ...] | None = None

        self._connect()
        self._refresh_engine_views()
        self._set_default_paths()
        if registry_warning:
            self.window.model_manager_page.set_status(registry_warning)
            self.window.set_status("注册表已安全恢复", connected=False)
        else:
            self.window.set_status("本地控制器已连接")
        self.refresh_system()

    def _connect(self) -> None:
        self.window.synthesis_requested.connect(self.synthesize)
        self.window.text_analysis_requested.connect(self.analyze_text)
        self.window.voice_analysis_requested.connect(self.analyze_voice)
        self.window.dataset_audit_requested.connect(self.audit_dataset_path)
        self.window.training_start_requested.connect(self.start_training)
        self.window.training_cancel_requested.connect(self.cancel_training)
        self.window.model_import_requested.connect(self.import_model)
        self.window.close_requested.connect(self.close)

        overview = self.window.overview_page
        overview.refresh_requested.connect(self.refresh_system)
        overview.open_workspace_requested.connect(lambda: self.open_path(self.workspace.root))

        synthesis = self.window.synthesis_page
        synthesis.engine_combo.currentIndexChanged.connect(self._update_synthesis_capabilities)
        synthesis.open_output_requested.connect(
            lambda path: self.open_path(Path(path) if path else self.workspace.outputs)
        )
        synthesis.preview_reference_requested.connect(self.open_path)
        synthesis.stop_requested.connect(self._explain_synthesis_stop)
        synthesis.queue_clear_requested.connect(self._clear_completed_queue)

        voice = self.window.voice_profile_page
        voice.audio_path.path_changed.connect(self._on_voice_audio_changed)
        voice.import_requested.connect(self._import_voice_reference)
        voice.play_requested.connect(self.open_path)
        voice.save_profile_requested.connect(self.save_voice_profile)
        voice.recording_requested.connect(self._explain_recording)

        languages = self.window.multilingual_page
        languages.preview_requested.connect(self._preview_language_segment)
        languages.use_in_synthesis_requested.connect(self._store_language_plan)
        languages.lexicon_import_requested.connect(
            lambda: languages.set_status("v0.1 请在表格中录入规则；词典文件导入尚未实现")
        )

        training = self.window.training_page
        training.prepare_requested.connect(self._explain_dataset_preparation)
        training.pause_requested.connect(self._explain_training_pause)
        training.resume_requested.connect(self._explain_training_pause)
        training.open_run_requested.connect(self._open_training_run)

        models = self.window.model_manager_page
        models.scan_requested.connect(self.refresh_system)
        models.verify_environment_requested.connect(self.verify_model_environment)
        models.remove_requested.connect(self.remove_model)
        models.activate_requested.connect(self.activate_model)
        models.open_root_requested.connect(self.open_path)

        experiment = self.window.experiment_page
        experiment.add_candidate_requested.connect(self.add_experiment_candidate)
        experiment.remove_candidates_requested.connect(self.remove_experiment_candidates)
        experiment.run_requested.connect(self.run_experiment)
        experiment.cancel_requested.connect(self.cancel_experiment)
        experiment.play_requested.connect(self.open_path)
        experiment.preference_requested.connect(self.save_experiment_preference)
        experiment.export_requested.connect(self.export_experiment)

        self.training_log_received.connect(self._on_training_log)
        self.training_metric_received.connect(self.window.append_metric)
        self.training_finished.connect(self._on_training_finished)

    def _set_default_paths(self) -> None:
        self.window.synthesis_page.output_directory.set_path(self.workspace.outputs)
        self.window.model_manager_page.set_environment(
            {
                "workspace": str(self.workspace.root),
                "cache": str(self.workspace.models),
                "gpu": "等待探测",
                "cuda": "由各模型环境探测",
                "ffmpeg": "等待探测",
                "disk_free": "—",
            }
        )

    def _run_async(
        self,
        function: Callable[[], Any],
        *,
        succeeded: Callable[[Any], None],
        failed: Callable[[str], None],
        finished: Callable[[], None] | None = None,
    ) -> None:
        worker = FunctionWorker(function)
        self._workers.add(worker)
        worker.signals.succeeded.connect(succeeded)
        worker.signals.failed.connect(failed)
        if finished is not None:
            worker.signals.finished.connect(finished)
        worker.signals.finished.connect(lambda: self._workers.discard(worker))
        self.thread_pool.start(worker)

    def refresh_system(self) -> None:
        self.window.set_status("正在探测本机环境…")

        def applied(info: object) -> None:
            gpu_list = getattr(info, "gpus", [])
            gpu = gpu_list[0] if gpu_list else None
            tools = getattr(info, "tools", {})
            gpu_name = getattr(gpu, "name", "未检测到 NVIDIA GPU")
            total_mib = getattr(gpu, "memory_total_mib", None)
            total_gib = total_mib / 1024 if isinstance(total_mib, int) else None
            self.window.overview_page.set_status(
                {
                    "gpu": gpu_name,
                    "device": f"Python {getattr(info, 'python_version', '—')}",
                    "vram_total_gb": total_gib,
                    "workspace": str(self.workspace.root),
                    "active_tasks": sum(
                        item.get("status") == "运行中" for item in self._recent_tasks
                    ),
                }
            )
            self.window.model_manager_page.set_environment(
                {
                    "gpu": gpu_name,
                    "cuda": "由各隔离模型环境验证",
                    "ffmpeg": "可用" if tools.get("ffmpeg") else "未找到",
                    "disk_free": f"{getattr(info, 'workspace_free_gib', 0):.1f} GiB",
                    "workspace": str(self.workspace.root),
                    "cache": str(self.workspace.models),
                }
            )
            self._refresh_engine_views()
            self.window.set_status("环境探测完成")

        self._run_async(
            lambda: probe_system(self.workspace.root),
            succeeded=applied,
            failed=lambda error: self.window.set_status(error, connected=False),
        )

    def _engine_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for manifest in self.registry:
            metadata = manifest.metadata
            readiness_issues: list[str] = []
            if manifest.transport is EngineTransport.MOCK:
                status = "就绪"
                runnable = True
            elif manifest.transport is EngineTransport.PROCESS:
                if not manifest.command:
                    readiness_issues.append("缺少桥接命令")
                if manifest.root and not Path(manifest.root).is_dir():
                    readiness_issues.append("源码目录不存在")
                if manifest.python and not Path(manifest.python).is_file():
                    readiness_issues.append("Python 不存在")
                if manifest.checkpoint_dir and not Path(manifest.checkpoint_dir).exists():
                    readiness_issues.append("权重路径不存在")
                runnable = not readiness_issues
                status = "；".join(readiness_issues) if readiness_issues else "已登记，未执行探测"
            else:
                runnable = bool(manifest.endpoint)
                status = "端点未探测；生成时连接" if runnable else "缺少 HTTP 端点"
            rows.append(
                {
                    "id": manifest.id,
                    "name": manifest.name,
                    "engine_type": metadata.get("engine_type", manifest.id),
                    "version": metadata.get("version", "—"),
                    "device": metadata.get("device", "独立环境"),
                    "root": manifest.root,
                    "checkpoint_dir": manifest.checkpoint_dir,
                    "transport": manifest.transport,
                    "runnable": runnable,
                    "status": status,
                    "active": bool(metadata.get("active", False)),
                    "capabilities": manifest.capabilities,
                    "languages": manifest.languages,
                }
            )
        return rows

    def _refresh_engine_views(self) -> None:
        rows = self._engine_rows()
        self.window.set_engines(rows)
        self.window.multilingual_page.set_engines(rows)
        self._update_synthesis_capabilities()

    def _update_synthesis_capabilities(self, *_: object) -> None:
        engine_id = str(self.window.synthesis_page.engine_combo.currentData() or "")
        if not engine_id:
            self.window.synthesis_page.set_engine_capabilities([])
            self.window.synthesis_page.set_engine_languages([])
            return
        try:
            manifest = self.registry.get_manifest(engine_id)
        except EngineConfigurationError:
            self.window.synthesis_page.set_engine_capabilities([])
            self.window.synthesis_page.set_engine_languages([])
            return
        self.window.synthesis_page.set_engine_capabilities(
            [item.value for item in manifest.capabilities]
        )
        self.window.synthesis_page.set_engine_languages(manifest.languages)

    def analyze_text(self, text: str) -> None:
        self._text_analysis_revision += 1
        revision = self._text_analysis_revision
        self._language_plan = {}
        if len(text) > _MAX_SYNTHESIS_CHARACTERS:
            self._segments = []
            self._language_rows = []
            self._language_rows_text = ""
            self.window.synthesis_page.set_segments([])
            self.window.multilingual_page.set_segments([])
            message = f"文本超过 {_MAX_SYNTHESIS_CHARACTERS} 字符；请拆分后分析"
            self.window.synthesis_page.validation_label.setText(message)
            self.window.multilingual_page.set_status(message)
            return
        segments = self.router.segment(text)
        if len(segments) > _MAX_LANGUAGE_SEGMENTS:
            self._segments = []
            self._language_rows = []
            self._language_rows_text = ""
            self.window.synthesis_page.set_segments([])
            self.window.multilingual_page.set_segments([])
            message = (
                f"文本切分为 {len(segments)} 段，超过 {_MAX_LANGUAGE_SEGMENTS} 段上限；"
                "请拆分文本"
            )
            self.window.synthesis_page.validation_label.setText(message)
            self.window.multilingual_page.set_status(message)
            return
        self._segments = segments
        self.window.synthesis_page.set_segments(segments)
        self._language_rows = _language_display_rows(segments)
        self._language_rows_text = text
        self.window.multilingual_page.set_segments(self._language_rows)
        unresolved = [
            segment
            for segment in segments
            if segment.language == "und" and any(char.isalnum() for char in segment.text)
        ]
        ambiguous_scripts = [
            segment
            for segment in segments
            if segment.source == "auto"
            and segment.language in {"ru", "zh"}
            and segment.confidence < 0.8
        ]
        message = f"已切分 {len(segments)} 个片段"
        if unresolved:
            if any(segment.source == "greek-script" for segment in unresolved):
                message += "；希腊字母需用 [grc]（古希腊语）或 [el]（现代希腊语）确认"
            else:
                message += f"；{len(unresolved)} 段需用 [en]/[la] 等标签确认"
        if ambiguous_scripts:
            message += "；西里尔字母或纯汉字段需用 [ru]、[zh] 或 [ja] 确认"
        self.window.synthesis_page.validation_label.setText(message)
        self.window.multilingual_page.set_status(message)

        frontend = EspeakNgPhonemizer()
        supported = [
            segment
            for segment in segments
            if frontend.supports(segment.language) and segment.text.strip()
        ]
        if not supported:
            return
        if not frontend.available:
            missing_message = (
                message
                + "；未找到 eSpeak-ng，IPA 预览已跳过（可设置 GENIVOX_ESPEAK_PATH）"
            )
            self.window.synthesis_page.validation_label.setText(missing_message)
            self.window.multilingual_page.set_status(missing_message)
            return

        def work() -> list[dict[str, Any]]:
            rows = _language_display_rows(segments)
            for row, segment in zip(rows, segments, strict=True):
                if frontend.supports(segment.language):
                    row["frontend"] = "eSpeak-ng IPA 基线"
                    row["phonemes"] = frontend.phonemize(segment.text, segment.language)
            return rows

        def succeeded(rows: list[dict[str, Any]]) -> None:
            if revision != self._text_analysis_revision:
                return
            current_rows = self.window.multilingual_page.segment_entries()
            for row, current, original in zip(
                rows, current_rows, self._language_rows, strict=False
            ):
                current_phonemes = str(current.get("phonemes", ""))
                original_phonemes = str(original.get("phonemes", ""))
                if current_phonemes and current_phonemes != original_phonemes:
                    row["phonemes"] = current_phonemes
            self._language_rows = rows
            self.window.multilingual_page.set_segments(rows)
            self.window.multilingual_page.set_status(message + "；IPA 基线已生成")

        def failed(error: str) -> None:
            if revision == self._text_analysis_revision:
                self.window.multilingual_page.set_status(
                    f"{message}；eSpeak-ng 预览失败：{error}"
                )

        self._run_async(
            work,
            succeeded=succeeded,
            failed=failed,
        )

    def synthesize(self, payload: Mapping[str, Any]) -> None:
        job_id = uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "name": f"生成 {job_id}",
            "engine": payload.get("engine_id", "—"),
            "language": payload.get("language", "auto"),
            "duration": "—",
            "status": "运行中",
            "output": "—",
        }
        self._queue.append(job)
        recent_task = self._add_recent_task(
            "合成", job["name"], str(job["engine"]), "运行中"
        )
        self.window.synthesis_page.set_queue(self._queue)
        self.window.synthesis_page.set_status("正在生成…", busy=True)

        def succeeded(result: tuple[SynthesisResult, SynthesisRequest]) -> None:
            output, request = result
            job["duration"] = (
                f"{output.duration_seconds:.2f} s"
                if output.duration_seconds is not None
                else "未知"
            )
            job["status"] = "完成"
            self._update_recent_task(recent_task, "完成")
            job["output"] = str(output.output_path)
            self.window.synthesis_page.set_queue(self._queue)
            self._record_experiment(output, request)
            self.window.synthesis_page.set_status("生成完成", busy=False)

        def failed(error: str) -> None:
            job["status"] = "失败"
            self._update_recent_task(recent_task, "失败")
            job["output"] = error
            self.window.synthesis_page.set_queue(self._queue)
            self.window.synthesis_page.set_status("生成失败", busy=False)
            self.window.synthesis_page.validation_label.setText(error)

        self._run_async(
            lambda: self._perform_synthesis(payload),
            succeeded=succeeded,
            failed=failed,
            finished=lambda: self.window.synthesis_page.set_status(
                "空闲" if job["status"] != "运行中" else "任务结束", busy=False
            ),
        )

    def _perform_synthesis(
        self,
        payload: Mapping[str, Any],
        *,
        output_path: Path | None = None,
    ) -> tuple[SynthesisResult, SynthesisRequest]:
        engine_id = str(payload.get("engine_id", ""))
        manifest = self.registry.get_manifest(engine_id)
        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("合成文本不能为空")
        if len(text) > _MAX_SYNTHESIS_CHARACTERS:
            raise ValueError(
                f"单次合成最多 {_MAX_SYNTHESIS_CHARACTERS} 个字符；请分批生成以控制本地资源"
            )
        if bool(payload.get("auto_emotion", False)):
            raise ValueError("尚未配置文本情绪模型；请关闭“从文本推断”或配置引擎桥")

        language = _language_code(payload.get("language", "auto"))
        segments: list[LanguageSegment] = []
        request_text = text
        if bool(payload.get("auto_language", True)):
            routed = [segment for segment in self.router.segment(text) if segment.text.strip()]
            if not routed or not any(
                any(character.isalnum() for character in segment.text) for segment in routed
            ):
                raise ValueError("文本没有可朗读的字母或数字")
            fallback = language if language not in {"auto", "und"} else None
            if fallback:
                routed = [
                    replace(segment, language=fallback, source="fallback", confidence=1.0)
                    if segment.language == "und"
                    or (segment.language == "la" and segment.source == "heuristic")
                    or (
                        segment.source == "auto"
                        and segment.language in {"ru", "zh"}
                        and segment.confidence < 0.8
                    )
                    else segment
                    for segment in routed
                ]
            historical_ambiguity = [
                segment
                for segment in routed
                if segment.source == "greek-script"
                or (segment.language == "la" and segment.source == "heuristic")
            ]
            if historical_ambiguity:
                raise ValueError(
                    "历史语言不能仅凭文字自动选制式；请用 [la]、[grc] 或 [el] 显式标注"
                )
            script_ambiguity = [
                segment
                for segment in routed
                if segment.source == "auto"
                and segment.language in {"ru", "zh"}
                and segment.confidence < 0.8
            ]
            if script_ambiguity:
                raise ValueError(
                    "文字系统无法唯一确定语言：西里尔字母需确认是否为 [ru]，"
                    "纯汉字段需用 [zh] 或 [ja] 标注"
                )
            unresolved = [
                segment
                for segment in routed
                if segment.language == "und" and any(char.isalnum() for char in segment.text)
            ]
            if unresolved and len(routed) > 1:
                sample = unresolved[0].text.strip()[:40]
                raise ValueError(
                    f"混合文本中有未确认语言片段 {sample!r}；请用 [en]…[/en] 或 [la]…[/la] 标注"
                )
            if len(routed) == 1:
                segment = routed[0]
                if segment.source == "explicit":
                    request_text = segment.text
                    language = segment.language
                elif segment.language != "und":
                    language = segment.language
                elif manifest.languages and "auto" not in manifest.languages:
                    raise ValueError(
                        f"{manifest.name} 要求明确语种；请选择 English 等回退语种，"
                        "或用 [en]…[/en] 显式标注"
                    )
            elif routed:
                if len(routed) > _MAX_LANGUAGE_SEGMENTS:
                    raise ValueError(
                        f"单次合成最多 {_MAX_LANGUAGE_SEGMENTS} 个语言片段；请拆分文本"
                    )
                segments = routed
                language = "auto"

        output = output_path or _new_audio_path(
            Path(str(payload.get("output_directory") or self.workspace.outputs)),
            engine_id,
        )
        reference_value = payload.get("reference_audio")
        reference = Path(str(reference_value)).expanduser().resolve() if reference_value else None
        if reference is not None and not bool(payload.get("reference_authorized", False)):
            raise ValueError("使用参考声音前必须确认你拥有克隆与合成授权")
        emotion = {
            str(label): float(value)
            for label, value in dict(payload.get("emotion", {})).items()
            if float(value) > 0.0
        }
        prompt_language = _language_code(
            payload.get("reference_language", manifest.metadata.get("prompt_language", "auto"))
        )
        checkpoint_override = payload.get("checkpoint_path")
        if checkpoint_override and manifest.transport is not EngineTransport.PROCESS:
            raise ValueError(
                "只有独立进程桥可在单次请求中覆盖 checkpoint；"
                "HTTP 或 Mock 后端请登记独立实例"
            )
        if (
            checkpoint_override
            and manifest.transport is EngineTransport.PROCESS
            and not bool(payload.get("checkpoint_trusted", False))
        ):
            raise ValueError("覆盖默认权重前必须确认该 checkpoint 来自可信来源")
        extra: dict[str, Any] = {}
        if reference is not None:
            if manifest.languages and prompt_language not in manifest.languages:
                raise ValueError(
                    f"{manifest.name} 不支持参考语种 {prompt_language!r}；请明确选择其支持的语种"
                )
            extra["prompt_lang"] = prompt_language
        elif not manifest.languages or prompt_language in manifest.languages:
            extra["prompt_lang"] = prompt_language
        checkpoint_path = checkpoint_override or (
            manifest.checkpoint_dir if manifest.transport is EngineTransport.PROCESS else None
        )
        if checkpoint_path:
            extra["checkpoint_path"] = str(checkpoint_path)
        if (
            self._language_plan.get("text") == text
            and Capability.PHONEME_INPUT in manifest.capabilities
        ):
            extra["pronunciation_plan"] = self._language_plan
        request = SynthesisRequest(
            text=request_text,
            output_path=output,
            engine_id=engine_id,
            language=language,
            segments=segments,
            reference_audio=reference,
            prompt_text=str(payload.get("reference_transcript", "")),
            speed=float(payload.get("speed", 1.0)),
            emotion=emotion,
            style_instruction=str(payload.get("style_instruction", "")),
            seed=int(payload.get("seed", -1)),
            extra=extra,
        )
        adapter = self.registry.create_adapter(engine_id)
        result = SynthesisPipeline(adapter).synthesize(request)
        protect_private_file(result.output_path)
        return result, request

    def _record_experiment(
        self, result: SynthesisResult, request: SynthesisRequest
    ) -> ExperimentRecord:
        record = ExperimentRecord(
            engine_id=request.engine_id,
            text=request.text,
            audio_path=str(result.output_path),
            parameters={
                "language": request.language,
                "speed": request.speed,
                "emotion": request.emotion,
                "style_instruction": request.style_instruction,
                "seed": request.seed,
                "reference_audio": str(request.reference_audio)
                if request.reference_audio
                else None,
                "reference_transcript": request.prompt_text,
                "request_extra": _redact_mapping(request.extra),
            },
            language_segments=[asdict(segment) for segment in request.segments],
            provenance={"backend_metadata": _redact_mapping(result.metadata)},
        )
        self.experiments.append(record)
        return record

    def analyze_voice(self, path: str) -> None:
        self.window.voice_profile_page.set_status("正在分析声学与情绪信息…", busy=True)
        audio_path = Path(path).expanduser().resolve()
        recent_task = self._add_recent_task("分析", audio_path.name, "本地声学分析", "运行中")
        self._voice_analysis_revision += 1
        revision = self._voice_analysis_revision

        def analyze() -> tuple[Any, list[float]]:
            analyzer = _configured_emotion_analyzer()
            profile = analyze_prosody(audio_path, emotion_analyzer=analyzer)
            audio = read_pcm_wav(audio_path)
            mono = audio.mono
            stride = max(1, len(mono) // 1_500)
            return profile, np.asarray(mono[::stride], dtype=float).tolist()

        def succeeded(result: tuple[Any, list[float]]) -> None:
            current_path = self.window.voice_profile_page.audio_path.path()
            current_audio = Path(current_path).expanduser().resolve() if current_path else None
            if revision != self._voice_analysis_revision or current_audio != audio_path:
                self._update_recent_task(recent_task, "已忽略")
                return
            profile, samples = result
            self._last_profile = profile
            self._last_profile_path = audio_path
            self.window.set_profile(profile)
            self.window.voice_profile_page.set_waveform(samples)
            self.window.voice_profile_page.set_status("分析完成", busy=False)
            self._update_recent_task(recent_task, "完成")

        def failed(error: str) -> None:
            current_path = self.window.voice_profile_page.audio_path.path()
            current_audio = Path(current_path).expanduser().resolve() if current_path else None
            if revision != self._voice_analysis_revision or current_audio != audio_path:
                self._update_recent_task(recent_task, "已忽略")
                return
            self._update_recent_task(recent_task, "失败")
            self.window.voice_profile_page.set_status(error, busy=False)

        def finished() -> None:
            if revision == self._voice_analysis_revision:
                self.window.voice_profile_page.analyze_button.setEnabled(True)

        self._run_async(
            analyze,
            succeeded=succeeded,
            failed=failed,
            finished=finished,
        )

    def _on_voice_audio_changed(self, path: str) -> None:
        self._voice_analysis_revision += 1
        selected = Path(path).expanduser().resolve() if path else None
        previous = self._selected_voice_path
        self._selected_voice_path = selected
        self.window.voice_profile_page.authorized_voice.setChecked(False)
        if previous is not None and selected != previous:
            self._last_profile = None
            self._last_profile_path = None
            message = "参考录音已更换；旧分析已清除，请重新点击“智能分析”"
            self.window.voice_profile_page.clear_analysis(message)
            self.window.voice_profile_page.set_status(message, busy=False)

    def save_voice_profile(self, payload: Mapping[str, Any]) -> None:
        audio_value = str(payload.get("audio_path", ""))
        name = str(payload.get("name", "")).strip()
        if not audio_value or not Path(audio_value).is_file():
            self.window.voice_profile_page.set_status("请先选择有效的参考录音")
            return
        if not name:
            self.window.voice_profile_page.set_status("请填写声音画像名称")
            return
        if not bool(payload.get("authorized", False)):
            self.window.voice_profile_page.set_status("保存前请确认你拥有该声音的使用授权")
            return

        selected_audio = Path(audio_value).expanduser().resolve()
        if self._last_profile is not None and self._last_profile_path != selected_audio:
            self.window.voice_profile_page.set_status(
                "所选录音与最近分析的录音不同；请重新点击“智能分析”后再保存"
            )
            return

        name_slug = _slug(name)
        identifier = f"voice-{name_slug}" if name_slug else f"voice-{uuid.uuid4().hex[:8]}"
        base_identifier = identifier
        version = 2
        while (self.workspace.profiles / f"{identifier}.json").exists():
            identifier = f"{base_identifier}-{version}"
            version += 1
        recording = SourceRecording.from_path(
            selected_audio,
            transcript=str(payload.get("transcript", "")),
            language=_language_code(payload.get("language_hint", "und")),
        )
        analysis = self._last_profile.to_dict() if self._last_profile is not None else {}
        analysis["style_instruction"] = str(
            payload.get("style_instruction", analysis.get("style_instruction", ""))
        )
        if isinstance(payload.get("emotion"), Mapping):
            analysis["emotion"] = dict(payload["emotion"])
        profile = VoiceProfile(
            id=identifier,
            display_name=name,
            consent=ConsentRecord(
                authorized=True,
                scope="User confirmed authorized local synthesis and training",
                recorded_at=datetime.now(UTC).isoformat(),
            ),
            source_recordings=[recording],
            analysis=analysis,
            pronunciation_defaults={
                "la": "restored-classical",
                "grc": "reconstructed-attic",
                "ru": "standard-russian",
            },
        )
        destination = self.workspace.profiles / f"{identifier}.json"
        try:
            save_voice_profile(profile, destination)
        except (OSError, TypeError, ValueError) as exc:
            self.window.voice_profile_page.set_status(f"画像保存失败：{exc}")
            return
        suffix = "（同名画像已自动版本化）" if identifier != base_identifier else ""
        self.window.voice_profile_page.set_status(f"画像已保存：{destination}{suffix}")

    def audit_dataset_path(self, path: str) -> None:
        self._last_dataset_audit = None
        self._last_audited_manifest = None
        self._last_audited_manifest_sha256 = None
        self._last_audited_audio_snapshot = None
        self.window.training_page.set_audit_busy(True)
        self.window.training_page.set_status("正在审计数据…")

        def work() -> tuple[Path, str, tuple[tuple[str, int, int], ...], DatasetAudit]:
            manifest_path = _resolve_dataset_manifest(Path(path))
            records = load_dataset_manifest(manifest_path)
            audit = audit_dataset(records)
            return (
                manifest_path,
                file_sha256(manifest_path),
                _dataset_audio_snapshot(records),
                audit,
            )

        def succeeded(
            result: tuple[Path, str, tuple[tuple[str, int, int], ...], DatasetAudit]
        ) -> None:
            manifest_path, manifest_sha256, audio_snapshot, audit = result
            self._last_dataset_audit = audit
            self._last_audited_manifest = manifest_path
            self._last_audited_manifest_sha256 = manifest_sha256
            self._last_audited_audio_snapshot = audio_snapshot
            self.window.set_dataset_report(_audit_report(manifest_path, audit))
            self.window.training_page.set_status("数据审计完成")

        def failed(error: str) -> None:
            self.window.training_page.set_status(error)

        self._run_async(
            work,
            succeeded=succeeded,
            failed=failed,
            finished=lambda: self.window.training_page.set_audit_busy(False),
        )

    def start_training(self, payload: Mapping[str, Any]) -> None:
        if self._active_training and self._active_training.is_running:
            self.window.training_page.set_status("已有训练任务正在运行")
            return
        try:
            engine_id = str(payload.get("engine_id", ""))
            manifest = self.registry.get_manifest(engine_id)
            if Capability.FINE_TUNE not in manifest.capabilities:
                raise ValueError(f"后端 {manifest.name} 未声明微调能力")
            if manifest.metadata.get("trusted_local_code") is not True:
                raise ValueError("训练命令尚未被信任；请重新登记并确认允许启动本地模型代码")
            command_template = manifest.metadata.get("training_command")
            if not _is_string_sequence(command_template):
                raise ValueError(
                    "该后端尚未配置训练命令；请在 genivox-engine.json 中提供 training_command 参数数组"
                )
            cwd = Path(manifest.root or ".").expanduser().resolve()
            values = {key: str(value) for key, value in payload.items()}
            dataset_manifest = _resolve_dataset_manifest(Path(values["dataset_path"]))
            if (
                self._last_dataset_audit is None
                or self._last_audited_manifest != dataset_manifest
            ):
                raise ValueError("开始训练前必须先扫描并审计当前数据清单")
            if self._last_audited_manifest_sha256 != file_sha256(dataset_manifest):
                self._last_dataset_audit = None
                self._last_audited_manifest_sha256 = None
                raise ValueError("数据清单在审计后发生变化；请重新扫描")
            current_audio_snapshot = _dataset_audio_snapshot(
                load_dataset_manifest(dataset_manifest)
            )
            if self._last_audited_audio_snapshot != current_audio_snapshot:
                self._last_dataset_audit = None
                self._last_audited_audio_snapshot = None
                raise ValueError("数据音频在审计后发生变化；请重新扫描")
            if self._last_dataset_audit.record_count == 0:
                raise ValueError("数据清单为空，不能开始训练")
            if self._last_dataset_audit.error_count:
                raise ValueError(
                    f"数据审计仍有 {self._last_dataset_audit.error_count} 个错误；请先修复"
                )
            values["dataset_path"] = str(dataset_manifest)
            values["output_path"] = str(payload.get("output_path") or self.workspace.models / engine_id)
            values["base_model_path"] = str(
                payload.get("base_model_path") or manifest.checkpoint_dir or ""
            )
            command = [part.format(**values) for part in command_template]
            if manifest.python:
                command.insert(0, manifest.python)
            self.window.training_page.reset_run_display()
            self._active_training = self.training_runner.start(
                command,
                cwd=cwd,
                metadata={"engine_id": engine_id, "configuration": dict(payload)},
                on_output=self.training_log_received.emit,
                on_finished=self.training_finished.emit,
            )
            self._last_training_run_path = self.training_runner.store.run_dir(
                self._active_training.run_id
            )
            self._active_training_task = self._add_recent_task(
                "训练",
                self._active_training.run_id,
                manifest.name,
                "运行中",
            )
        except Exception as exc:
            self.window.training_page.set_status(f"无法启动训练：{exc}", busy=False)
            return
        self.window.training_page.set_status(
            f"训练已启动 · {self._active_training.run_id}", progress=0.0, busy=True
        )

    def _on_training_log(self, line: str) -> None:
        self.window.training_page.append_log(line)
        if line.lstrip().startswith("{"):
            try:
                self.training_metric_received.emit(parse_metric_line(line))
            except MetricParseError:
                pass

    def _on_training_finished(self, manifest: object) -> None:
        status = getattr(getattr(manifest, "status", None), "value", "finished")
        error = getattr(manifest, "error", None)
        message = f"训练结束：{status}" + (f" · {error}" if error else "")
        progress = 1.0 if status == "succeeded" else None
        self.window.training_page.set_status(message, progress=progress, busy=False)
        if progress is None:
            self.window.training_page.progress_bar.setFormat(message)
        if self._active_training_task is not None:
            status_labels = {
                "succeeded": "完成",
                "failed": "失败",
                "cancelled": "已取消",
            }
            self._update_recent_task(
                self._active_training_task, status_labels.get(status, status)
            )
            self._active_training_task = None
        self._active_training = None

    def cancel_training(self) -> None:
        if not self._active_training or not self._active_training.is_running:
            self.window.training_page.set_status("没有可取消的训练任务", busy=False)
            return
        process = self._active_training
        self.window.training_page.set_status("正在取消训练…", busy=True)
        self._run_async(
            process.cancel,
            succeeded=lambda _: None,
            failed=lambda error: self.window.training_page.set_status(error, busy=False),
        )

    def _open_training_run(self, configured_output: str) -> None:
        target = self._last_training_run_path
        if target is None and configured_output:
            target = Path(configured_output)
        if target is None:
            target = self.workspace.runs / "training"
        self.open_path(target)

    def verify_model_environment(self, payload: Mapping[str, Any]) -> None:
        messages: list[str] = []
        root = payload.get("root")
        python = payload.get("python")
        checkpoint = payload.get("checkpoint_dir")
        if root:
            messages.append("源码目录有效" if Path(str(root)).is_dir() else "源码目录不存在")
        if python:
            messages.append("Python 有效" if Path(str(python)).is_file() else "Python 不存在")
        if checkpoint:
            messages.append("权重路径有效" if Path(str(checkpoint)).exists() else "权重路径不存在")
        if payload.get("transport") == "http":
            messages.append("HTTP 地址仅登记；生成时才进行连接")
        self.window.model_manager_page.set_status("；".join(messages) or "未提供可验证路径")

    def import_model(self, payload: Mapping[str, Any]) -> None:
        if not bool(payload.get("reference_existing", True)):
            self.window.model_manager_page.set_status(
                "v0.1 只登记现有路径，不复制大型权重；请勾选“引用现有目录”"
            )
            return
        try:
            manifest = _manifest_from_import(payload)
            self.registry.register(manifest)
            self.registry.save(registry_path(self.workspace))
        except Exception as exc:
            self.window.model_manager_page.set_status(f"登记失败：{exc}")
            return
        self._refresh_engine_views()
        detail = "登记完成"
        if manifest.transport is EngineTransport.PROCESS and not manifest.command:
            detail += "；需在源码目录添加 genivox-engine.json 后重新登记"
        if (
            manifest.metadata.get("engine_type") == "VoxCPM2"
            and Capability.FINE_TUNE in manifest.capabilities
        ):
            detail += "；8 GB 显存不承诺本机微调，先按官方显存要求评估"
        self.window.model_manager_page.set_status(detail)

    def remove_model(self, engine_id: str) -> None:
        if engine_id in {"mock-local", "gpt-sovits-v2-local"}:
            self.window.model_manager_page.set_status("内置登记可编辑 registry.json，不从界面删除")
            return
        try:
            self.registry.remove(engine_id)
            self.registry.save(registry_path(self.workspace))
        except EngineConfigurationError as exc:
            self.window.model_manager_page.set_status(str(exc))
            return
        self._refresh_engine_views()
        self.window.model_manager_page.set_status("已移除登记；未删除任何模型文件")

    def activate_model(self, engine_id: str) -> None:
        try:
            self.registry.get_manifest(engine_id)
        except EngineConfigurationError as exc:
            self.window.model_manager_page.set_status(str(exc))
            return
        for manifest in self.registry:
            manifest.metadata["active"] = manifest.id == engine_id
        self.registry.save(registry_path(self.workspace))
        self._refresh_engine_views()
        index = self.window.synthesis_page.engine_combo.findData(engine_id)
        if index >= 0:
            self.window.synthesis_page.engine_combo.setCurrentIndex(index)
        self.window.model_manager_page.set_status(
            "已选作默认合成登记项；此操作不会加载或切换后端进程中的权重"
        )

    def add_experiment_candidate(self, payload: Mapping[str, Any]) -> None:
        candidate = dict(payload)
        if not str(candidate.get("name", "")).strip():
            candidate["name"] = f"候选 {len(self._candidates) + 1}"
        candidate["status"] = "等待"
        self._candidates.append(candidate)
        self.window.experiment_page.set_candidates(_candidate_rows(self._candidates))

    def remove_experiment_candidates(self, rows: Sequence[int]) -> None:
        for row in sorted({int(item) for item in rows}, reverse=True):
            if 0 <= row < len(self._candidates):
                del self._candidates[row]
        self.window.experiment_page.set_candidates(_candidate_rows(self._candidates))

    def run_experiment(self, payload: Mapping[str, Any]) -> None:
        if len(self._candidates) < 2:
            self.window.experiment_page.set_status("至少需要两个候选")
            return
        self._experiment_cancel.clear()
        self.window.experiment_page.set_status("正在依次生成候选…", busy=True)
        candidates = [dict(candidate) for candidate in self._candidates]

        def work() -> list[dict[str, Any]]:
            results: list[dict[str, Any]] = []
            root = Path(str(payload.get("output_directory") or self.workspace.outputs / "experiments"))
            for index, candidate in enumerate(candidates):
                if self._experiment_cancel.is_set():
                    break
                request_payload = {
                    "text": payload["text"],
                    "engine_id": candidate["engine_id"],
                    "reference_audio": payload.get("reference_audio"),
                    "reference_authorized": payload.get("reference_authorized", False),
                    "reference_transcript": payload.get("reference_transcript", ""),
                    "reference_language": payload.get("reference_language", "auto"),
                    "auto_language": True,
                    "language": "auto",
                    "speed": candidate.get("speed", 1.0),
                    "emotion": {},
                    "auto_emotion": False,
                    "style_instruction": candidate.get("style_instruction", ""),
                    "seed": candidate.get("seed", -1),
                    "checkpoint_path": candidate.get("checkpoint_path"),
                    "checkpoint_trusted": candidate.get("checkpoint_trusted", False),
                    "output_directory": str(root),
                }
                name = _slug(str(candidate.get("name", "candidate"))) or f"candidate-{index + 1}"
                destination = _new_audio_path(root, name)
                try:
                    result, request = self._perform_synthesis(
                        request_payload, output_path=destination
                    )
                    record = self._record_experiment(result, request)
                    results.append(
                        {
                            "name": candidate.get("name", name),
                            "candidate": _candidate_snapshot(candidate),
                            "request": _request_snapshot(request),
                            "experiment_record_id": record.id,
                            "wer": "未配置 ASR",
                            "speaker_similarity": "未配置",
                            "emotion_match": "未配置",
                            "duration_error": "—",
                            "rtf": "—",
                            "output": str(result.output_path),
                            "preference": "未评价",
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "name": candidate.get("name", name),
                            "candidate": _candidate_snapshot(candidate),
                            "request": _redact_mapping(request_payload),
                            "wer": "—",
                            "speaker_similarity": "—",
                            "emotion_match": "—",
                            "duration_error": "—",
                            "rtf": "—",
                            "output": "",
                            "preference": "失败",
                            "error": str(exc),
                        }
                    )
            return results

        def succeeded(results: list[dict[str, Any]]) -> None:
            self._experiment_results = results
            self.window.experiment_page.set_results(results)
            failures = sum(not result.get("output") for result in results)
            cancelled = self._experiment_cancel.is_set()
            self.window.experiment_page.set_status(
                (
                    "实验已停止"
                    if cancelled
                    else f"实验完成 · {len(results) - failures} 成功 / {failures} 失败"
                ),
                busy=False,
            )

        self._run_async(
            work,
            succeeded=succeeded,
            failed=lambda error: self.window.experiment_page.set_status(error, busy=False),
            finished=lambda: self.window.experiment_page.run_button.setEnabled(True),
        )

    def save_experiment_preference(self, payload: Mapping[str, Any]) -> None:
        row = int(payload.get("selected_row", -1))
        if not 0 <= row < len(self._experiment_results):
            self.window.experiment_page.set_status("请先选择一个结果行")
            return
        note = str(payload.get("note", ""))
        preference = str(payload.get("preference", "未评价"))
        record_id = str(self._experiment_results[row].get("experiment_record_id", ""))
        if not record_id:
            self.window.experiment_page.set_status("该结果没有可写回的实验记录")
            return
        try:
            self.experiments.update_preference(record_id, preference, note)
        except (KeyError, OSError, ValueError) as exc:
            self.window.experiment_page.set_status(f"保存偏好失败：{exc}")
            return
        self._experiment_results[row]["note"] = note
        self._experiment_results[row]["preference"] = preference
        self.window.experiment_page.set_results(self._experiment_results)
        self.window.experiment_page.set_status("听感备注已保存在当前实验报告中")

    def cancel_experiment(self) -> None:
        self._experiment_cancel.set()
        self.window.experiment_page.set_status("将在当前候选完成后停止…", busy=True)

    def export_experiment(self, payload: Mapping[str, Any]) -> None:
        root = Path(str(payload.get("output_directory") or self.workspace.outputs / "experiments"))
        ensure_private_directory(root)
        exported_at = datetime.now(UTC)
        destination = root / (
            f"experiment-{exported_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{uuid.uuid4().hex[:6]}.json"
        )
        report = {
            "schema_version": 1,
            "exported_at": exported_at.isoformat(),
            "results": self._experiment_results,
        }
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        protect_private_file(destination)
        self.window.experiment_page.set_status(f"报告已导出：{destination}")

    def open_path(self, raw_path: str | Path) -> None:
        if not raw_path:
            return
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            self.window.set_status(f"路径不存在：{path}", connected=False)
            return
        allowed_media = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg"}
        if path.is_file() and path.suffix.casefold() not in allowed_media:
            self.window.set_status(
                f"出于安全考虑，只能从工作台打开目录或音频文件：{path.name}",
                connected=False,
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _clear_completed_queue(self) -> None:
        self._queue = [item for item in self._queue if item.get("status") == "运行中"]
        self.window.synthesis_page.set_queue(self._queue)

    def _add_recent_task(
        self, task_type: str, name: str, engine: str, status: str
    ) -> dict[str, Any]:
        task = {
            "type": task_type,
            "name": name,
            "engine": engine,
            "status": status,
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._recent_tasks.insert(0, task)
        del self._recent_tasks[50:]
        self.window.overview_page.set_recent_tasks(self._recent_tasks)
        return task

    def _update_recent_task(self, task: dict[str, Any], status: str) -> None:
        task["status"] = status
        self.window.overview_page.set_recent_tasks(self._recent_tasks)

    def _explain_synthesis_stop(self) -> None:
        self.window.synthesis_page.validation_label.setText(
            "当前一次性 HTTP/进程适配器无法安全中断；训练任务可取消，流式后端将在后续接入。"
        )

    def _explain_recording(self, active: bool) -> None:
        if active:
            self.window.voice_profile_page.record_button.setChecked(False)
            self.window.voice_profile_page.set_status("v0.1 尚未接入麦克风；请先导入本地 PCM WAV")

    def _import_voice_reference(self, path: str) -> None:
        selected = Path(path).expanduser().resolve()
        if not selected.is_file():
            self.window.voice_profile_page.set_status("所选参考录音不存在")
            return
        self.window.voice_profile_page.set_audio_path(selected)
        self.window.voice_profile_page.set_status(
            f"已载入 {selected.name}；点击“智能分析”提取声学特征"
        )

    def _explain_dataset_preparation(self, _: Mapping[str, Any]) -> None:
        self.window.training_page.set_status("v0.1 的数据准备为只读审计，不会自动改写原始语料")

    def _explain_training_pause(self) -> None:
        self.window.training_page.set_status("通用训练进程暂不支持跨平台暂停；可以安全取消")

    def _preview_language_segment(self, _: Mapping[str, Any]) -> None:
        self.window.multilingual_page.set_status("片段试听需先选择并配置可支持该语言的后端")

    def _store_language_plan(self, payload: Mapping[str, Any]) -> None:
        source_text = str(payload.get("text", ""))
        if source_text != self._language_rows_text:
            self._language_plan = {}
            message = "文本已在分析后修改；请重新切分，再发送发音计划"
            self.window.multilingual_page.set_status(message)
            self.window.synthesis_page.validation_label.setText(message)
            return
        edited_segments = payload.get("segments", [])
        if not isinstance(edited_segments, Sequence) or isinstance(
            edited_segments, (str, bytes)
        ):
            edited_segments = []
        for edited in edited_segments:
            if not isinstance(edited, Mapping):
                self._language_plan = {}
                self.window.multilingual_page.set_status("发音计划格式无效；请重新分析文本")
                return
            start = edited.get("start")
            end = edited.get("end")
            segment_text = str(edited.get("text", ""))
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > len(source_text)
                or source_text[start:end] != segment_text
            ):
                self._language_plan = {}
                message = "文本已在分析后修改；请重新切分，再发送发音计划"
                self.window.multilingual_page.set_status(message)
                self.window.synthesis_page.validation_label.setText(message)
                return

        self._language_plan = dict(payload)
        rows = [dict(row) for row in self._language_rows]
        for row, edited in zip(rows, edited_segments, strict=False):
            row["phonemes"] = str(edited.get("phonemes", row.get("phonemes", "")))
        self._language_plan["segments"] = rows
        self.window.multilingual_page.set_status(
            "发音计划已附加到合成请求；仅声明 phoneme_input 的 process bridge 会应用"
        )
        engine_id = str(self.window.synthesis_page.engine_combo.currentData() or "")
        if engine_id:
            manifest = self.registry.get_manifest(engine_id)
            if Capability.PHONEME_INPUT not in manifest.capabilities:
                self.window.synthesis_page.validation_label.setText(
                    f"{manifest.name} 不支持音素计划；合成时将仅使用其原生文本前端"
                )

    def close(self) -> None:
        if self._active_training and self._active_training.is_running:
            process = self._active_training
            threading.Thread(target=process.cancel, daemon=False).start()


def _configured_emotion_analyzer() -> ExternalEmotionAnalyzer | NoOpEmotionAnalyzer:
    raw = os.environ.get("GENIVOX_EMOTION_COMMAND_JSON", "").strip()
    if not raw:
        return NoOpEmotionAnalyzer()
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"GENIVOX_EMOTION_COMMAND_JSON 不是有效 JSON：{exc}") from exc
    if not _is_string_sequence(command):
        raise ValueError("GENIVOX_EMOTION_COMMAND_JSON 必须是字符串参数数组")
    return ExternalEmotionAnalyzer(command)


def _language_code(value: object) -> str:
    text = str(value or "auto")
    return _LANGUAGE_LABELS.get(text, text.casefold())


def _new_audio_path(root: Path, stem: str) -> Path:
    root = root.expanduser().resolve()
    ensure_private_directory(root)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return root / f"{_slug(stem) or 'speech'}-{timestamp}-{uuid.uuid4().hex[:6]}.wav"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.").lower()
    return normalized[:80]


def _resolve_dataset_manifest(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    preferred = (
        "metadata.jsonl",
        "manifest.jsonl",
        "metadata.csv",
        "metadata.txt",
        "train.list",
    )
    for name in preferred:
        candidate = path / name
        if candidate.is_file():
            return candidate
    candidates = sorted(
        candidate
        for pattern in ("*.jsonl", "*.csv", "*.list", "*.txt", "*.pipe", "*.psv")
        for candidate in path.glob(pattern)
        if candidate.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"目录中没有可识别的 manifest：{path}")
    raise ValueError(f"目录中有多个 manifest，请直接选择一个文件：{path}")


def _dataset_audio_snapshot(records: Sequence[Any]) -> tuple[tuple[str, int, int], ...]:
    """Capture inexpensive file identity used to invalidate a completed audit."""

    snapshot: list[tuple[str, int, int]] = []
    for record in records:
        path = Path(record.audio_path).expanduser().resolve()
        try:
            stat = path.stat()
            snapshot.append((str(path), stat.st_size, stat.st_mtime_ns))
        except OSError:
            snapshot.append((str(path), -1, -1))
    return tuple(snapshot)


def _audit_report(path: Path, audit: DatasetAudit) -> dict[str, Any]:
    distribution: list[dict[str, Any]] = []
    for dimension, counts in (
        ("语言", audit.language_counts),
        ("情绪标签", audit.emotion_counts),
        ("说话人", audit.speaker_counts),
        ("时长", audit.duration.histogram),
    ):
        total = sum(counts.values())
        for category, count in counts.items():
            distribution.append(
                {
                    "dimension": dimension,
                    "category": category,
                    "count": count,
                    "duration": "—",
                    "ratio": count / total if total else 0.0,
                    "target": "未配置",
                    "guidance": "复核少数类别" if total and count / total < 0.05 else "保持",
                }
            )
    issues = [
        {
            "level": issue.severity.value,
            "location": (
                ", ".join(f"记录 #{index + 1}" for index in issue.record_indices)
                or str(path)
            ),
            "message": issue.message,
            "guidance": "先修复错误" if issue.severity.value == "error" else "人工复核",
        }
        for issue in audit.issues
    ]
    issues.extend(
        {
            "level": recommendation.priority.value,
            "location": "全局",
            "message": recommendation.title,
            "guidance": recommendation.guidance,
        }
        for recommendation in audit.recommendations
    )
    return {
        "summary": {
            "records": audit.record_count,
            "duration_seconds": audit.duration.total_seconds,
            "languages": len(audit.language_counts),
            "speakers": len(audit.speaker_counts),
        },
        "distribution": distribution,
        "issues": issues,
        "status": f"已审计 {path.name}；原始数据未被修改",
    }


def _manifest_from_import(payload: Mapping[str, Any]) -> EngineManifest:
    engine_type = str(payload.get("engine_type", "自定义适配器"))
    name = str(payload.get("name", engine_type)).strip() or engine_type
    transport = EngineTransport(str(payload.get("transport", "process")))
    root_value = payload.get("root")
    root = Path(str(root_value)).expanduser().resolve() if root_value else None
    python_value = payload.get("python")
    python = Path(str(python_value)).expanduser().resolve() if python_value else None
    checkpoint_value = payload.get("checkpoint_dir")
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve() if checkpoint_value else None
    if root is not None and not root.is_dir():
        raise NotADirectoryError(root)
    if python is not None and not python.is_file():
        raise FileNotFoundError(python)
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    configuration: dict[str, Any] = {}
    if root is not None and (root / "genivox-engine.json").is_file():
        configuration = json.loads((root / "genivox-engine.json").read_text(encoding="utf-8"))
        if not isinstance(configuration, dict):
            raise ValueError("genivox-engine.json 顶层必须是 JSON 对象")
        if configuration.get("schema_version") != 1:
            raise ValueError("genivox-engine.json schema_version 必须为 1")

    preset_capabilities, preset_languages = _PROCESS_PRESETS.get(engine_type, ([], []))
    raw_capabilities = configuration.get("capabilities", preset_capabilities)
    raw_languages = configuration.get("languages", preset_languages)
    if not _is_string_sequence(raw_capabilities):
        raise ValueError("genivox-engine.json capabilities 必须是字符串数组")
    if not _is_string_sequence(raw_languages):
        raise ValueError("genivox-engine.json languages 必须是字符串数组")
    capabilities = [Capability(item) for item in raw_capabilities]
    languages = [str(item) for item in raw_languages]
    command = configuration.get("command", [])
    if not _is_string_sequence(command):
        raise ValueError("genivox-engine.json command 必须是字符串参数数组")
    raw_metadata = configuration.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("genivox-engine.json metadata 必须是 JSON 对象")
    metadata = dict(raw_metadata)
    trusted_local_code = bool(payload.get("trusted_local_code", False))
    metadata.update(
        {
            "engine_type": engine_type,
            "device": payload.get("device", "auto"),
            "precision": payload.get("precision", "auto"),
            "trusted_local_code": trusted_local_code,
            "status": "已登记",
        }
    )
    if training_command := configuration.get("training_command"):
        if not _is_string_sequence(training_command):
            raise ValueError("genivox-engine.json training_command 必须是字符串参数数组")
        if not trusted_local_code:
            raise ValueError("登记训练命令前必须确认允许启动所选目录中的本地代码")
        metadata["training_command"] = list(training_command)

    endpoint = str(payload.get("endpoint") or "") or None
    if transport is EngineTransport.HTTP:
        if engine_type != "GPT-SoVITS":
            raise ValueError("v0.1 的内置 HTTP 适配器仅支持 GPT-SoVITS；其他后端请选择独立进程桥")
        if not endpoint:
            raise ValueError("GPT-SoVITS HTTP 登记需要 endpoint")
        parsed_endpoint = urllib.parse.urlparse(endpoint)
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if (
            parsed_endpoint.scheme not in {"http", "https"}
            or parsed_endpoint.hostname not in local_hosts
        ):
            raise ValueError("本地模式只接受 localhost、127.0.0.1 或 ::1 的 HTTP 服务")
        if not endpoint.rstrip("/").endswith("/tts"):
            endpoint = endpoint.rstrip("/") + "/tts"
        capabilities = [Capability.VOICE_CLONE, Capability.CROSS_LINGUAL, Capability.SPEED]
        if "training_command" in metadata:
            capabilities.append(Capability.FINE_TUNE)
        languages = ["auto", "zh", "en", "ja", "ko", "yue"]
        metadata["adapter"] = "gpt_sovits_v2"
    elif transport is EngineTransport.MOCK:
        capabilities = [Capability.CROSS_LINGUAL, Capability.SPEED]
        metadata["sample_rate"] = 16_000

    if transport is EngineTransport.PROCESS and not trusted_local_code:
        raise ValueError("独立进程桥必须明确确认允许启动所选目录中的本地代码")

    identifier = _slug(str(configuration.get("id", name))) or f"engine-{uuid.uuid4().hex[:8]}"
    return EngineManifest(
        id=identifier,
        name=name,
        transport=transport,
        capabilities=capabilities,
        languages=languages,
        endpoint=endpoint,
        root=str(root) if root else None,
        python=str(python) if python else None,
        command=list(command),
        checkpoint_dir=str(checkpoint) if checkpoint else None,
        metadata=metadata,
    )


def _is_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, str) and item for item in value
    )


def _language_display_rows(segments: Sequence[LanguageSegment]) -> list[dict[str, Any]]:
    frontend = EspeakNgPhonemizer()
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if frontend.supports(segment.language):
            frontend_name = "eSpeak-ng 可用" if frontend.available else "eSpeak-ng 未安装"
        else:
            frontend_name = segment.source
        rows.append(
            {
                **asdict(segment),
                "frontend": frontend_name,
                "phonemes": "—",
                "join": "PCM 顺序拼接",
            }
        )
    return rows


def _candidate_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.get("name", "—"),
            "engine": item.get("engine_id", "—"),
            "checkpoint": item.get("checkpoint_path", "默认"),
            "speed": item.get("speed", 1.0),
            "style": item.get("style_instruction", "—"),
            "seed": item.get("seed", -1),
            "status": item.get("status", "等待"),
        }
        for item in candidates
    ]


def _candidate_snapshot(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep exactly the candidate controls needed to reproduce one comparison."""

    return {
        "name": str(candidate.get("name", "")),
        "engine_id": str(candidate.get("engine_id", "")),
        "checkpoint_path": candidate.get("checkpoint_path"),
        "checkpoint_trusted": bool(candidate.get("checkpoint_trusted", False)),
        "speed": float(candidate.get("speed", 1.0)),
        "style_instruction": str(candidate.get("style_instruction", "")),
        "seed": int(candidate.get("seed", -1)),
    }


def _request_snapshot(request: SynthesisRequest) -> dict[str, Any]:
    return {
        "text": request.text,
        "engine_id": request.engine_id,
        "language": request.language,
        "language_segments": [asdict(segment) for segment in request.segments],
        "reference_audio": str(request.reference_audio) if request.reference_audio else None,
        "reference_transcript": request.prompt_text,
        "speed": request.speed,
        "emotion": dict(request.emotion),
        "style_instruction": request.style_instruction,
        "seed": request.seed,
        "extra": _redact_mapping(request.extra),
    }


def _redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    sensitive_fragments = ("token", "password", "secret", "api_key", "apikey")
    result: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        if any(fragment in key.casefold() for fragment in sensitive_fragments):
            result[key] = "<redacted>"
        elif isinstance(item, Mapping):
            result[key] = _redact_mapping(item)
        elif isinstance(item, Path):
            result[key] = str(item)
        elif isinstance(item, (list, tuple)):
            result[key] = [
                _redact_mapping(element) if isinstance(element, Mapping) else element
                for element in item
            ]
        else:
            result[key] = item
    return result
