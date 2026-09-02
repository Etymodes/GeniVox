from __future__ import annotations

import http.client
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import genivox.engines.gpt_sovits as gpt_sovits_module
from genivox.engines.gpt_sovits import (
    GptSovitsProbeStatus,
    inspect_gpt_sovits_installation,
    probe_gpt_sovits_api,
)


def _openapi_document(*, properties: set[str] | None = None) -> bytes:
    fields = (
        properties
        if properties is not None
        else {"text", "text_lang", "ref_audio_path", "prompt_lang"}
    )
    document = {
        "openapi": "3.1.0",
        "paths": {
            "/tts": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/TTS_Request"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "TTS_Request": {
                    "properties": {name: {"type": "string"} for name in fields}
                }
            }
        },
    }
    return json.dumps(document).encode("utf-8")


def _create_installation(root: Path) -> None:
    (root / "GPT_SoVITS" / "configs").mkdir(parents=True)
    (root / "GPT_SoVITS" / "TTS_infer_pack").mkdir(parents=True)
    (root / "runtime").mkdir()
    (root / "api_v2.py").write_text("# entrypoint\n", encoding="utf-8")
    (root / "GPT_SoVITS" / "configs" / "tts_infer.yaml").write_text(
        "custom: {}\n", encoding="utf-8"
    )
    (root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").write_text(
        "# package marker\n", encoding="utf-8"
    )
    (root / "runtime" / "python.exe").write_bytes(b"")


class GptSovitsInstallationProbeTests(unittest.TestCase):
    def test_recognizes_installation_but_empty_weight_directory_is_not_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_installation(root)
            pretrained = root / "GPT_SoVITS" / "pretrained_models"
            pretrained.mkdir()
            (pretrained / ".gitignore").write_text("*\n", encoding="utf-8")

            result = inspect_gpt_sovits_installation(root)

            self.assertTrue(result.launch_ready)
            self.assertEqual(result.entrypoint, (root / "api_v2.py").resolve())
            self.assertEqual(
                result.core_module,
                (root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py").resolve(),
            )
            self.assertEqual(
                result.config,
                (root / "GPT_SoVITS" / "configs" / "tts_infer.yaml").resolve(),
            )
            self.assertEqual(result.python, (root / "runtime" / "python.exe").resolve())
            self.assertEqual(result.gpt_weight_candidates, ())
            self.assertEqual(result.sovits_weight_candidates, ())
            self.assertFalse(result.weights_detected)

    def test_detects_gpt_and_sovits_weight_files_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_installation(root)
            weights = root / "GPT_SoVITS" / "pretrained_models" / "voice"
            weights.mkdir(parents=True)
            gpt_weight = weights / "speaker.ckpt"
            sovits_weight = weights / "speaker.pth"
            gpt_weight.write_bytes(b"gpt")
            sovits_weight.write_bytes(b"sovits")

            result = inspect_gpt_sovits_installation(root)

            self.assertEqual(result.gpt_weight_candidates, (gpt_weight.resolve(),))
            self.assertEqual(result.sovits_weight_candidates, (sovits_weight.resolve(),))
            self.assertTrue(result.weights_detected)

    def test_weight_scan_does_not_follow_a_matching_directory_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "install"
            outside = Path(directory) / "outside"
            _create_installation(root)
            outside.mkdir()
            (outside / "external.ckpt").write_bytes(b"outside")
            link = root / "GPT_weights_external"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks are unavailable on this platform")

            result = inspect_gpt_sovits_installation(root)

            self.assertEqual(result.gpt_weight_candidates, ())

    def test_weight_candidate_count_has_a_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_installation(root)
            weights = root / "GPT_weights_bulk"
            weights.mkdir()
            for index in range(257):
                (weights / f"{index:03}.ckpt").touch()

            result = inspect_gpt_sovits_installation(root)

            self.assertEqual(len(result.gpt_weight_candidates), 256)

    def test_weight_root_staging_has_a_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _create_installation(root)
            weights = root / "GPT_weights_repeated"
            weights.mkdir()
            yielded = 0

            def repeated_glob(_: Path, pattern: str):
                nonlocal yielded
                del pattern
                for _ in range(20_000):
                    yielded += 1
                    yield weights

            with patch.object(Path, "glob", new=repeated_glob):
                gpt_sovits_module._find_weight_candidates(
                    root, ("GPT_weights*", "GPT_SoVITS/pretrained_models"), ".ckpt"
                )

            self.assertEqual(yielded, 128)


class GptSovitsApiProbeTests(unittest.TestCase):
    def test_valid_openapi_contract_is_ready_without_synthesis(self) -> None:
        response = io.BytesIO(_openapi_document())

        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ) as opener:
            result = probe_gpt_sovits_api(
                "http://127.0.0.1:9880/tts", timeout=1.25
            )

        self.assertEqual(result.status, GptSovitsProbeStatus.API_READY)
        self.assertTrue(result.ready)
        self.assertFalse(result.synthesis_verified)
        self.assertEqual(result.checked_url, "http://127.0.0.1:9880/openapi.json")
        request = opener.call_args.args[0]
        self.assertIsInstance(request, urllib.request.Request)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, result.checked_url)
        self.assertEqual(opener.call_args.kwargs["timeout"], 1.25)

    def test_non_loopback_endpoint_is_invalid_without_network_access(self) -> None:
        with patch("genivox.engines.gpt_sovits._open_local_http") as opener:
            result = probe_gpt_sovits_api("https://example.com/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.INVALID)
        self.assertFalse(result.ready)
        self.assertIsNone(result.checked_url)
        opener.assert_not_called()

    def test_url_error_is_reported_as_offline(self) -> None:
        with patch(
            "genivox.engines.gpt_sovits._open_local_http",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = probe_gpt_sovits_api("http://localhost:9880/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.OFFLINE)
        self.assertFalse(result.ready)
        self.assertEqual(result.checked_url, "http://localhost:9880/openapi.json")
        self.assertIn("connection refused", result.message)

    def test_missing_required_contract_field_is_incompatible(self) -> None:
        response = io.BytesIO(
            _openapi_document(
                properties={"text", "text_lang", "ref_audio_path"}
            )
        )

        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.INCOMPATIBLE)
        self.assertFalse(result.ready)
        self.assertIn("prompt_lang", result.message)

    def test_non_string_field_or_unknown_required_field_is_incompatible(self) -> None:
        wrong_type = json.loads(_openapi_document())
        schema = wrong_type["components"]["schemas"]["TTS_Request"]
        schema["properties"]["text"] = {"type": "integer"}
        response = io.BytesIO(json.dumps(wrong_type).encode("utf-8"))
        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")
        self.assertEqual(result.status, GptSovitsProbeStatus.INCOMPATIBLE)

        unknown_required = json.loads(_openapi_document())
        unknown_required["components"]["schemas"]["TTS_Request"]["required"] = [
            "secret"
        ]
        response = io.BytesIO(json.dumps(unknown_required).encode("utf-8"))
        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")
        self.assertEqual(result.status, GptSovitsProbeStatus.INCOMPATIBLE)

    def test_missing_openapi_marker_is_wrong_service(self) -> None:
        document = json.loads(_openapi_document())
        del document["openapi"]
        response = io.BytesIO(json.dumps(document).encode("utf-8"))

        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.WRONG_SERVICE)

    def test_malformed_openapi_version_is_wrong_service(self) -> None:
        document = json.loads(_openapi_document())
        document["openapi"] = "3.evil"
        response = io.BytesIO(json.dumps(document).encode("utf-8"))

        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.WRONG_SERVICE)

    def test_closed_schema_or_wrong_adapter_field_type_is_incompatible(self) -> None:
        closed = json.loads(_openapi_document())
        closed_schema = closed["components"]["schemas"]["TTS_Request"]
        closed_schema["additionalProperties"] = False
        response = io.BytesIO(json.dumps(closed).encode("utf-8"))
        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")
        self.assertEqual(result.status, GptSovitsProbeStatus.INCOMPATIBLE)

        wrong_speed = json.loads(_openapi_document())
        wrong_speed_schema = wrong_speed["components"]["schemas"]["TTS_Request"]
        wrong_speed_schema["properties"]["speed_factor"] = {"type": "string"}
        wrong_speed_schema["required"] = ["speed_factor"]
        response = io.BytesIO(json.dumps(wrong_speed).encode("utf-8"))
        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")
        self.assertEqual(result.status, GptSovitsProbeStatus.INCOMPATIBLE)

    def test_control_characters_are_invalid_without_network_access(self) -> None:
        with patch("genivox.engines.gpt_sovits._open_local_http") as opener:
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/bad path/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.INVALID)
        opener.assert_not_called()

    def test_truncated_http_body_is_wrong_service(self) -> None:
        with patch(
            "genivox.engines.gpt_sovits._open_local_http",
            side_effect=http.client.IncompleteRead(b"{}", 10),
        ):
            result = probe_gpt_sovits_api("http://127.0.0.1:9880/tts")

        self.assertEqual(result.status, GptSovitsProbeStatus.WRONG_SERVICE)

    def test_openapi_response_limit_is_enforced(self) -> None:
        response = io.BytesIO(b"x" * 17)

        with patch(
            "genivox.engines.gpt_sovits._open_local_http", return_value=response
        ):
            result = probe_gpt_sovits_api(
                "http://127.0.0.1:9880/tts", max_response_bytes=16
            )

        self.assertEqual(result.status, GptSovitsProbeStatus.WRONG_SERVICE)
        self.assertFalse(result.ready)
        self.assertIn("exceeds 16 bytes", result.message)


if __name__ == "__main__":
    unittest.main()
