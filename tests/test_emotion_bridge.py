from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from scripts.bridges import emotion2vec_bridge as bridge


class _FakeAutoModel:
    init_kwargs: dict[str, object] = {}
    generate_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_kwargs = kwargs
        print("fake model initialization log")

    def generate(self, **kwargs: object) -> list[dict[str, object]]:
        self.__class__.generate_kwargs = kwargs
        print("fake model inference log")
        return [
            {
                "labels": ["生气/angry", "中立/neutral", "custom-calm"],
                "scores": [0.1, 0.8, 0.1],
            }
        ]


def test_extracts_common_and_unknown_labels() -> None:
    probabilities, unknown = bridge._extract_probabilities(
        {"labels": ["happy", "<unk>", "new-label"], "scores": [0.7, 0.2, 0.1]}
    )

    assert probabilities == {
        "happy": 0.7,
        "unknown": 0.2,
        "unknown:new-label": 0.1,
    }
    assert unknown == ["new-label"]


def test_rejects_mismatched_result() -> None:
    with pytest.raises(bridge.BridgeError, match="2 labels but 1 scores"):
        bridge._extract_probabilities({"labels": ["happy", "sad"], "scores": [0.5]})


def test_analyze_audio_redirects_model_chatter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF-test-placeholder")

    response = bridge.analyze_audio(
        audio_path,
        model_name="local/emotion2vec",
        device="cpu",
        hub="local",
        auto_model_cls=_FakeAutoModel,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fake model initialization log" in captured.err
    assert response["emotion"] == {
        "angry": 0.1,
        "neutral": 0.8,
        "unknown:custom-calm": 0.1,
    }
    assert response["meta"]["unknown_labels"] == ["custom-calm"]  # type: ignore[index]
    assert _FakeAutoModel.init_kwargs == {
        "model": "local/emotion2vec",
        "device": "cpu",
        "hub": "local",
    }


def test_main_emits_only_json_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF-test-placeholder")
    monkeypatch.setattr(bridge, "_load_auto_model", lambda: _FakeAutoModel)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"audio_path": str(audio_path)})))

    exit_code = bridge.main(["--model", "test-model", "--device", "cpu", "--hub", "local"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["emotion"]["neutral"] == 0.8
    assert "fake model inference log" in captured.err


def test_missing_funasr_has_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(bridge.importlib, "import_module", missing_module)
    with pytest.raises(bridge.BridgeError, match="python -m pip install funasr modelscope"):
        bridge._load_auto_model()


def test_main_reports_missing_funasr_on_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"RIFF-test-placeholder")

    def missing_model() -> object:
        raise bridge.BridgeError(
            "FunASR is not installed; python -m pip install funasr modelscope"
        )

    monkeypatch.setattr(bridge, "_load_auto_model", missing_model)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"audio_path": str(audio_path)})))

    exit_code = bridge.main(["--device", "cpu"])

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == ""
    assert "python -m pip install funasr modelscope" in captured.err
