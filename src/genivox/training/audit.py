from __future__ import annotations

import math
import statistics
import wave
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from genivox.core.models import DatasetRecord


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class AuditConfig:
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    histogram_edges_seconds: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 20.0, 30.0)
    probe_wav_duration: bool = True

    def __post_init__(self) -> None:
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be greater than min_duration_seconds")
        if not self.histogram_edges_seconds:
            raise ValueError("histogram edges must not be empty")
        if any(edge <= 0 for edge in self.histogram_edges_seconds):
            raise ValueError("histogram edges must be positive")
        if tuple(sorted(set(self.histogram_edges_seconds))) != self.histogram_edges_seconds:
            raise ValueError("histogram edges must be strictly increasing")


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    severity: IssueSeverity
    message: str
    record_indices: tuple[int, ...] = ()
    value: str | float | None = None


@dataclass(frozen=True, slots=True)
class AuditRecommendation:
    code: str
    priority: IssueSeverity
    title: str
    guidance: str
    evidence: str


@dataclass(frozen=True, slots=True)
class DurationDistribution:
    known_count: int
    unknown_count: int
    total_seconds: float
    minimum_seconds: float | None
    maximum_seconds: float | None
    mean_seconds: float | None
    median_seconds: float | None
    p95_seconds: float | None
    histogram: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    record_count: int
    existing_audio_count: int
    language_counts: dict[str, int]
    emotion_counts: dict[str, int]
    speaker_counts: dict[str, int]
    duration: DurationDistribution
    issues: tuple[AuditIssue, ...]
    recommendations: tuple[AuditRecommendation, ...]

    @property
    def error_count(self) -> int:
        return sum(issue.severity is IssueSeverity.ERROR for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity is IssueSeverity.WARNING for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_dataset(
    records: Iterable[DatasetRecord],
    *,
    config: AuditConfig | None = None,
) -> DatasetAudit:
    """Audit records and return evidence-based guidance without mutating the dataset."""

    selected_config = config or AuditConfig()
    dataset = list(records)
    issues: list[AuditIssue] = []
    if not dataset:
        issues.append(
            AuditIssue(
                code="empty_dataset",
                severity=IssueSeverity.ERROR,
                message="Dataset manifest contains no records",
            )
        )
    durations: list[float] = []
    missing_duration_indices: list[int] = []
    existing_audio_count = 0

    text_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset):
        text_groups[_normalize_text(record.text)].append(index)

        if not record.audio_path.is_file():
            issues.append(
                AuditIssue(
                    code="missing_audio",
                    severity=IssueSeverity.ERROR,
                    message=f"Audio file does not exist: {record.audio_path}",
                    record_indices=(index,),
                    value=str(record.audio_path),
                )
            )
            duration = record.duration_seconds
        else:
            existing_audio_count += 1
            duration = record.duration_seconds
            if selected_config.probe_wav_duration and record.audio_path.suffix.casefold() in {
                ".wav",
                ".wave",
            }:
                probed_duration = _probe_wav_duration(record.audio_path)
                if probed_duration is None:
                    issues.append(
                        AuditIssue(
                            code="invalid_audio",
                            severity=IssueSeverity.ERROR,
                            message=f"WAV audio is empty, truncated, or unreadable: {record.audio_path}",
                            record_indices=(index,),
                            value=str(record.audio_path),
                        )
                    )
                    duration = None
                else:
                    if duration is not None and math.isfinite(duration):
                        tolerance = max(0.05, probed_duration * 0.02)
                        if abs(duration - probed_duration) > tolerance:
                            issues.append(
                                AuditIssue(
                                    code="declared_duration_mismatch",
                                    severity=IssueSeverity.WARNING,
                                    message=(
                                        f"Declared duration {duration:.3f}s differs from WAV "
                                        f"duration {probed_duration:.3f}s"
                                    ),
                                    record_indices=(index,),
                                    value=duration,
                                )
                            )
                    duration = probed_duration

        if duration is None or not math.isfinite(duration):
            missing_duration_indices.append(index)
            continue
        if duration <= 0:
            issues.append(
                AuditIssue(
                    code="invalid_duration",
                    severity=IssueSeverity.ERROR,
                    message=f"Duration must be positive: {duration!r}",
                    record_indices=(index,),
                    value=duration,
                )
            )
            missing_duration_indices.append(index)
            continue
        durations.append(duration)
        if duration < selected_config.min_duration_seconds:
            issues.append(
                AuditIssue(
                    code="too_short",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Duration {duration:.3f}s is below the configured minimum "
                        f"of {selected_config.min_duration_seconds:.3f}s"
                    ),
                    record_indices=(index,),
                    value=duration,
                )
            )
        elif duration > selected_config.max_duration_seconds:
            issues.append(
                AuditIssue(
                    code="too_long",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Duration {duration:.3f}s exceeds the configured maximum "
                        f"of {selected_config.max_duration_seconds:.3f}s"
                    ),
                    record_indices=(index,),
                    value=duration,
                )
            )

    duplicate_groups = [tuple(indices) for text, indices in text_groups.items() if text and len(indices) > 1]
    for indices in duplicate_groups:
        issues.append(
            AuditIssue(
                code="duplicate_text",
                severity=IssueSeverity.WARNING,
                message=f"Normalized transcript is repeated in {len(indices)} records",
                record_indices=indices,
                value=dataset[indices[0]].text,
            )
        )

    if missing_duration_indices:
        issues.append(
            AuditIssue(
                code="unknown_duration",
                severity=IssueSeverity.INFO,
                message=f"Duration is unavailable for {len(missing_duration_indices)} records",
                record_indices=tuple(missing_duration_indices),
            )
        )

    language_counts = _sorted_counts(record.language or "und" for record in dataset)
    emotion_counts = _sorted_counts(record.emotion or "unlabeled" for record in dataset)
    speaker_counts = _sorted_counts(record.speaker or "default" for record in dataset)
    duration_distribution = _duration_distribution(
        durations,
        unknown_count=len(missing_duration_indices),
        edges=selected_config.histogram_edges_seconds,
    )
    recommendations = _build_recommendations(
        dataset_size=len(dataset),
        issues=issues,
        language_counts=language_counts,
        emotion_counts=emotion_counts,
        unknown_duration_count=len(missing_duration_indices),
    )
    return DatasetAudit(
        record_count=len(dataset),
        existing_audio_count=existing_audio_count,
        language_counts=language_counts,
        emotion_counts=emotion_counts,
        speaker_counts=speaker_counts,
        duration=duration_distribution,
        issues=tuple(issues),
        recommendations=tuple(recommendations),
    )


def _probe_wav_duration(path: Path) -> float | None:
    if path.suffix.casefold() not in {".wav", ".wave"}:
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if frame_rate <= 0 or frame_count <= 0 or channels <= 0 or sample_width <= 0:
                return None
            pcm = wav_file.readframes(frame_count)
            if len(pcm) != frame_count * channels * sample_width:
                return None
            return frame_count / frame_rate
    except (EOFError, OSError, wave.Error):
        return None


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sorted_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _duration_distribution(
    durations: list[float],
    *,
    unknown_count: int,
    edges: tuple[float, ...],
) -> DurationDistribution:
    histogram = _histogram(durations, edges)
    if not durations:
        return DurationDistribution(
            known_count=0,
            unknown_count=unknown_count,
            total_seconds=0.0,
            minimum_seconds=None,
            maximum_seconds=None,
            mean_seconds=None,
            median_seconds=None,
            p95_seconds=None,
            histogram=histogram,
        )
    ordered = sorted(durations)
    return DurationDistribution(
        known_count=len(durations),
        unknown_count=unknown_count,
        total_seconds=sum(durations),
        minimum_seconds=ordered[0],
        maximum_seconds=ordered[-1],
        mean_seconds=statistics.fmean(ordered),
        median_seconds=statistics.median(ordered),
        p95_seconds=_percentile(ordered, 0.95),
        histogram=histogram,
    )


def _histogram(values: list[float], edges: tuple[float, ...]) -> dict[str, int]:
    labels = [f"<{_number(edge)}s" for edge in edges[:1]]
    labels.extend(
        f"{_number(left)}–<{_number(right)}s" for left, right in zip(edges[:-1], edges[1:], strict=True)
    )
    labels.append(f">={_number(edges[-1])}s")
    counts = {label: 0 for label in labels}
    for value in values:
        bucket = len(edges)
        for index, edge in enumerate(edges):
            if value < edge:
                bucket = index
                break
        counts[labels[bucket]] += 1
    return counts


def _number(value: float) -> str:
    return f"{value:g}"


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _build_recommendations(
    *,
    dataset_size: int,
    issues: list[AuditIssue],
    language_counts: dict[str, int],
    emotion_counts: dict[str, int],
    unknown_duration_count: int,
) -> list[AuditRecommendation]:
    recommendations: list[AuditRecommendation] = []
    counts = Counter(issue.code for issue in issues)
    if counts["missing_audio"]:
        recommendations.append(
            AuditRecommendation(
                code="resolve_missing_audio",
                priority=IssueSeverity.ERROR,
                title="Resolve missing audio paths",
                guidance="Repair the manifest paths or explicitly exclude the rows after reviewing them.",
                evidence=f"{counts['missing_audio']} record(s) point to files that are not present.",
            )
        )
    if counts["too_short"] or counts["too_long"]:
        recommendations.append(
            AuditRecommendation(
                code="review_duration_outliers",
                priority=IssueSeverity.WARNING,
                title="Review duration outliers",
                guidance=(
                    "Listen before deciding: trim leading silence, merge fragments, or segment long clips "
                    "manually. "
                    "No rows were changed by this audit."
                ),
                evidence=(
                    f"{counts['too_short']} short and {counts['too_long']} long record(s) exceed the "
                    "selected limits."
                ),
            )
        )
    if counts["duplicate_text"]:
        recommendations.append(
            AuditRecommendation(
                code="review_duplicate_text",
                priority=IssueSeverity.WARNING,
                title="Review repeated transcripts",
                guidance=(
                    "Keep intentional alternate takes; otherwise decide manually which recording is cleaner. "
                    "Repeated text was not deduplicated automatically."
                ),
                evidence=f"{counts['duplicate_text']} normalized transcript group(s) are repeated.",
            )
        )
    if counts["unknown_duration"]:
        recommendations.append(
            AuditRecommendation(
                code="measure_unknown_duration",
                priority=IssueSeverity.INFO,
                title="Measure unknown clip durations",
                guidance=(
                    "Add durations during preprocessing so batch sizing and distribution plots are reliable."
                ),
                evidence=f"Duration is unavailable for {unknown_duration_count} record(s).",
            )
        )

    if dataset_size >= 10:
        language_recommendation = _imbalance_recommendation("language", language_counts, dataset_size)
        if language_recommendation is not None:
            recommendations.append(language_recommendation)
        emotion_recommendation = _imbalance_recommendation("emotion", emotion_counts, dataset_size)
        if emotion_recommendation is not None:
            recommendations.append(emotion_recommendation)
    return recommendations


def _imbalance_recommendation(
    dimension: str,
    counts: dict[str, int],
    dataset_size: int,
) -> AuditRecommendation | None:
    if len(counts) < 2:
        return None
    largest_name, largest_count = next(iter(counts.items()))
    share = largest_count / dataset_size
    if share < 0.70:
        return None
    return AuditRecommendation(
        code=f"review_{dimension}_balance",
        priority=IssueSeverity.INFO,
        title=f"Review {dimension} balance",
        guidance=(
            f"Check whether the intended model should prioritize {largest_name!r}; if not, collect or "
            "sample more "
            f"examples from underrepresented {dimension} groups."
        ),
        evidence=f"{largest_name!r} contains {largest_count}/{dataset_size} records ({share:.1%}).",
    )
