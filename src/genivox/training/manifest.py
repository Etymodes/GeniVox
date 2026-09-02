from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from genivox.core.models import DatasetRecord


class ManifestFormat(StrEnum):
    PIPE = "pipe"
    CSV = "csv"
    JSONL = "jsonl"


class ManifestParseError(ValueError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        super().__init__(f"{path}:{line_number}: {message}")


_ALIASES = {
    "audio_path": ("audio_path", "audio", "wav_path", "wav", "file", "path"),
    "text": ("text", "transcript", "sentence"),
    "language": ("language", "lang"),
    "speaker": ("speaker", "speaker_id", "speaker_name"),
    "emotion": ("emotion", "style"),
    "duration_seconds": ("duration_seconds", "duration", "duration_sec"),
}


def load_dataset_manifest(
    path: str | Path,
    *,
    format: ManifestFormat | str | None = None,
    audio_root: str | Path | None = None,
) -> list[DatasetRecord]:
    """Parse a dataset manifest without changing any source files.

    Pipe manifests accept a named header, or the common positional layouts
    ``audio|text``, LJSpeech's ``id|text|normalized_text``,
    ``audio|speaker|language|text``, and ``audio|speaker|language|emotion|text``.
    CSV and JSONL use named fields;
    common aliases such as ``wav_path``, ``transcript`` and ``lang`` are accepted.
    """

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    selected_format = _detect_format(manifest_path) if format is None else ManifestFormat(format)
    root = Path(audio_root).expanduser().resolve() if audio_root is not None else manifest_path.parent

    if selected_format is ManifestFormat.PIPE:
        rows = _read_pipe_rows(manifest_path)
    elif selected_format is ManifestFormat.CSV:
        rows = _read_csv_rows(manifest_path)
    else:
        rows = _read_jsonl_rows(manifest_path)

    return [_record_from_mapping(manifest_path, line_number, row, root) for line_number, row in rows]


def _detect_format(path: Path) -> ManifestFormat:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig") as stream:
            first_content_line = next((line for line in stream if line.strip()), "")
        if first_content_line.count("|") >= 2:
            return ManifestFormat.PIPE
        return ManifestFormat.CSV
    if suffix in {".jsonl", ".ndjson"}:
        return ManifestFormat.JSONL
    if suffix in {".list", ".txt", ".pipe", ".psv"}:
        return ManifestFormat.PIPE
    raise ValueError(f"Cannot infer manifest format from {path.name!r}; pass format explicitly")


def _read_pipe_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    parsed_rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            try:
                values = next(csv.reader([raw_line], delimiter="|", skipinitialspace=False))
            except csv.Error as exc:
                raise ManifestParseError(path, line_number, str(exc)) from exc
            parsed_rows.append((line_number, [value.strip() for value in values]))

    if not parsed_rows:
        return []

    first_line, first_values = parsed_rows[0]
    del first_line
    if _looks_like_header(first_values):
        headers = [_canonical_name(value) for value in first_values]
        _validate_required_headers(path, parsed_rows[0][0], headers)
        return [
            (line_number, _zip_row(path, line_number, headers, values))
            for line_number, values in parsed_rows[1:]
        ]

    layouts = {
        2: ("audio_path", "text"),
        4: ("audio_path", "speaker", "language", "text"),
        5: ("audio_path", "speaker", "language", "emotion", "text"),
    }
    result: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, values in parsed_rows:
        if len(values) == 3:
            utterance_id, transcript, normalized_transcript = values
            if not utterance_id or Path(utterance_id).suffix:
                raise ManifestParseError(
                    path,
                    line_number,
                    "pipe row with 3 fields must use an extension-free LJSpeech utterance id",
                )
            result.append(
                (
                    line_number,
                    {
                        "audio_path": str(Path("wavs") / f"{utterance_id}.wav"),
                        "text": normalized_transcript or transcript,
                    },
                )
            )
            continue
        headers = layouts.get(len(values))
        if headers is None:
            raise ManifestParseError(
                path,
                line_number,
                "pipe row must contain 2, 3, 4, or 5 fields, or start with a named header",
            )
        result.append((line_number, dict(zip(headers, values, strict=True))))
    return result


def _read_csv_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        try:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                return []
            headers = [_canonical_name(value or "") for value in reader.fieldnames]
            _validate_required_headers(path, 1, headers)
            result: list[tuple[int, Mapping[str, Any]]] = []
            for line_number, raw_row in enumerate(reader, 2):
                values = [raw_row.get(name) for name in reader.fieldnames]
                if None in raw_row:
                    raise ManifestParseError(path, line_number, "row has more columns than the header")
                result.append((line_number, dict(zip(headers, values, strict=True))))
            return result
        except csv.Error as exc:
            raise ManifestParseError(path, reader.line_num, str(exc)) from exc


def _read_jsonl_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    result: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ManifestParseError(path, line_number, f"invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ManifestParseError(path, line_number, "JSONL row must be an object")
            result.append((line_number, {_canonical_name(str(key)): value for key, value in row.items()}))
    return result


def _record_from_mapping(
    manifest_path: Path,
    line_number: int,
    raw_row: Mapping[str, Any],
    audio_root: Path,
) -> DatasetRecord:
    row = {_canonical_name(str(key)): value for key, value in raw_row.items()}
    audio_value = _string_value(row.get("audio_path"))
    text = _string_value(row.get("text"))
    if not audio_value:
        raise ManifestParseError(manifest_path, line_number, "audio path is empty")
    if not text:
        raise ManifestParseError(manifest_path, line_number, "text is empty")

    audio_path = Path(audio_value).expanduser()
    if not audio_path.is_absolute():
        audio_path = audio_root / audio_path

    try:
        duration_seconds = _optional_duration(row.get("duration_seconds"))
    except (TypeError, ValueError) as exc:
        raise ManifestParseError(manifest_path, line_number, str(exc)) from exc

    return DatasetRecord(
        audio_path=audio_path.resolve(),
        text=text,
        language=_string_value(row.get("language")) or "und",
        speaker=_string_value(row.get("speaker")) or "default",
        emotion=_string_value(row.get("emotion")) or "unlabeled",
        duration_seconds=duration_seconds,
    )


def _canonical_name(name: str) -> str:
    normalized = name.strip().casefold().replace("-", "_").replace(" ", "_")
    for canonical, aliases in _ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def _looks_like_header(values: Iterable[str]) -> bool:
    names = {_canonical_name(value) for value in values}
    return "audio_path" in names and "text" in names


def _validate_required_headers(path: Path, line_number: int, headers: Iterable[str]) -> None:
    names = list(headers)
    missing = {"audio_path", "text"}.difference(names)
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ManifestParseError(path, line_number, f"missing required field(s): {missing_fields}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ManifestParseError(path, line_number, f"duplicate field(s): {', '.join(duplicates)}")


def _zip_row(path: Path, line_number: int, headers: list[str], values: list[str]) -> Mapping[str, Any]:
    if len(values) != len(headers):
        raise ManifestParseError(path, line_number, "row column count does not match the header")
    return dict(zip(headers, values, strict=True))


def _string_value(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _optional_duration(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    duration = float(value)
    if duration < 0:
        raise ValueError("duration must be non-negative")
    return duration
