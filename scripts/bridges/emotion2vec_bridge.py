#!/usr/bin/env python3
"""One-shot JSON bridge from GeniVox to an optional FunASR emotion2vec model.

The process reads one ``{"audio_path": "..."}`` object from stdin. Successful
stdout is one JSON object; diagnostics and third-party progress output go to
stderr so callers can parse stdout safely.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


class BridgeError(RuntimeError):
    """A user-facing bridge failure."""


_LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "生气": "angry",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "厌恶": "disgusted",
    "fearful": "fearful",
    "fear": "fearful",
    "恐惧": "fearful",
    "happy": "happy",
    "happiness": "happy",
    "开心": "happy",
    "neutral": "neutral",
    "中立": "neutral",
    "other": "other",
    "其他": "other",
    "sad": "sad",
    "sadness": "sad",
    "难过": "sad",
    "surprised": "surprised",
    "surprise": "surprised",
    "吃惊": "surprised",
    "unknown": "unknown",
    "unk": "unknown",
    "<unk>": "unknown",
    "未知": "unknown",
}


def _canonical_label(raw_label: object) -> tuple[str, bool]:
    raw = str(raw_label).strip()
    lowered = raw.casefold()
    candidates = [lowered]
    for separator in ("/", "|", ":"):
        candidates.extend(part.strip() for part in lowered.split(separator))

    for candidate in candidates:
        if candidate in _LABEL_ALIASES:
            return _LABEL_ALIASES[candidate], True

    display = raw if raw else "<empty>"
    return f"unknown:{display}", False


def _as_list(value: object, *, field: str) -> list[object]:
    if isinstance(value, (str, bytes)):
        raise BridgeError(f"FunASR result field {field!r} must be a list, not text")
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BridgeError(f"FunASR result field {field!r} is not iterable") from exc


def _find_label_scores(result: object) -> tuple[list[object], list[object]]:
    """Accept the direct dict and list-of-dicts shapes used by FunASR releases."""

    if isinstance(result, Mapping):
        if "labels" in result and "scores" in result:
            return (
                _as_list(result["labels"], field="labels"),
                _as_list(result["scores"], field="scores"),
            )
        for key in ("result", "results", "output"):
            if key in result:
                try:
                    return _find_label_scores(result[key])
                except BridgeError:
                    pass

    if not isinstance(result, (str, bytes, Mapping)):
        try:
            items = list(result)  # type: ignore[arg-type]
        except TypeError:
            items = []
        for item in items:
            try:
                return _find_label_scores(item)
            except BridgeError:
                pass

    raise BridgeError("FunASR result did not contain compatible labels and scores fields")


def _extract_probabilities(result: object) -> tuple[dict[str, float], list[str]]:
    labels, scores = _find_label_scores(result)
    if len(labels) != len(scores):
        raise BridgeError(
            f"FunASR returned {len(labels)} labels but {len(scores)} scores"
        )

    probabilities: dict[str, float] = {}
    unknown_labels: list[str] = []
    for raw_label, raw_score in zip(labels, scores, strict=True):
        if isinstance(raw_score, bool):
            raise BridgeError(f"Emotion score for {raw_label!r} is not numeric")
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise BridgeError(f"Emotion score for {raw_label!r} is not numeric") from exc
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise BridgeError(
                f"Emotion score for {raw_label!r} must be finite and between 0 and 1"
            )

        label, recognized = _canonical_label(raw_label)
        if label in probabilities:
            raise BridgeError(f"Multiple FunASR labels normalize to {label!r}")
        probabilities[label] = score
        if not recognized:
            unknown_labels.append(str(raw_label))

    return probabilities, unknown_labels


def _load_auto_model() -> Any:
    try:
        # Some dependencies announce themselves with print(); keep stdout machine-readable.
        with redirect_stdout(sys.stderr):
            module = importlib.import_module("funasr")
    except (ImportError, ModuleNotFoundError) as exc:
        raise BridgeError(
            "FunASR is not installed in this bridge environment. "
            "Install it explicitly with: python -m pip install funasr modelscope"
        ) from exc

    auto_model = getattr(module, "AutoModel", None)
    if auto_model is None:
        raise BridgeError(
            "The installed FunASR package does not expose AutoModel; upgrade it with: "
            "python -m pip install --upgrade funasr"
        )
    return auto_model


def analyze_audio(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    hub: str,
    auto_model_cls: Any | None = None,
) -> dict[str, object]:
    """Run one file and return the JSON-serializable bridge response."""

    resolved_audio = audio_path.expanduser().resolve()
    if not resolved_audio.is_file():
        raise BridgeError(f"Audio file does not exist or is not a file: {resolved_audio}")

    model_factory = auto_model_cls or _load_auto_model()
    try:
        # Redirect Python-level library chatter; the only stdout write is in main().
        with redirect_stdout(sys.stderr):
            model = model_factory(model=model_name, device=device, hub=hub)
            result = model.generate(
                input=str(resolved_audio),
                granularity="utterance",
                extract_embedding=False,
            )
    except Exception as exc:
        raise BridgeError(f"FunASR emotion analysis failed: {exc}") from exc

    probabilities, unknown_labels = _extract_probabilities(result)
    return {
        "emotion": probabilities,
        "meta": {
            "model": model_name,
            "device": device,
            "hub": hub,
            "unknown_labels": unknown_labels,
            "interpretation": (
                "Model-inferred emotion probabilities; they are uncertain estimates, not facts."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read {audio_path} JSON from stdin and emit emotion2vec JSON to stdout."
    )
    parser.add_argument("--model", default="iic/emotion2vec_plus_large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hub", default="ms")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw_request = sys.stdin.read()
        try:
            request = json.loads(raw_request)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"stdin is not valid JSON: {exc}") from exc
        if not isinstance(request, Mapping):
            raise BridgeError("stdin JSON must be an object containing audio_path")
        raw_audio_path = request.get("audio_path")
        if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
            raise BridgeError("stdin JSON must contain a non-empty string audio_path")

        response = analyze_audio(
            Path(raw_audio_path),
            model_name=args.model,
            device=args.device,
            hub=args.hub,
        )
    except BridgeError as exc:
        print(f"emotion2vec bridge error: {exc}", file=sys.stderr)
        return 2

    json.dump(response, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
