from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from genivox.core.config import load_or_create_engine_registry, registry_path
from genivox.core.models import Capability, EngineManifest, EngineTransport
from genivox.core.paths import WorkspacePaths
from genivox.core.profile import (
    ConsentRecord,
    SourceRecording,
    VoiceProfile,
    load_voice_profile,
    save_voice_profile,
)


class CoreModelsTests(unittest.TestCase):
    def test_engine_manifest_round_trip(self) -> None:
        manifest = EngineManifest(
            id="gpt-sovits-local",
            name="GPT-SoVITS",
            transport=EngineTransport.HTTP,
            endpoint="http://127.0.0.1:9880",
            capabilities=[Capability.VOICE_CLONE, Capability.SPEED],
            languages=["zh", "en", "ja"],
        )
        self.assertEqual(EngineManifest.from_dict(manifest.to_dict()), manifest)

    def test_engine_manifest_rejects_string_command_and_non_string_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "command"):
            EngineManifest.from_dict(
                {
                    "id": "bad",
                    "name": "Bad",
                    "transport": "process",
                    "command": "bridge.py",
                }
            )
        with self.assertRaisesRegex(ValueError, "checkpoint_dir"):
            EngineManifest(
                id="bad-path",
                name="Bad path",
                transport=EngineTransport.MOCK,
                checkpoint_dir=123,  # type: ignore[arg-type]
            )

    def test_workspace_ensure_creates_only_named_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "voice-workspace"
            paths = WorkspacePaths(root).ensure()
            self.assertTrue(paths.engines.is_dir())
            self.assertTrue(paths.outputs.is_dir())
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {"engines", "models", "datasets", "runs", "outputs", "profiles"},
            )

    def test_default_registry_is_persisted_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = WorkspacePaths(Path(temporary_directory) / "workspace").ensure()
            registry = load_or_create_engine_registry(paths)
            self.assertTrue(registry_path(paths).is_file())
            self.assertEqual(
                {manifest.id for manifest in registry},
                {"mock-local", "gpt-sovits-v2-local"},
            )
            self.assertEqual(
                {manifest.id for manifest in load_or_create_engine_registry(paths)},
                {"mock-local", "gpt-sovits-v2-local"},
            )


class VoiceProfileTests(unittest.TestCase):
    def test_profile_save_load_and_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording_path = root / "sample.wav"
            recording_path.write_bytes(b"authorized test recording")
            recording = SourceRecording.from_path(recording_path, transcript="Salve", language="la")
            profile = VoiceProfile(
                id="speaker-1",
                display_name="Speaker One",
                consent=ConsentRecord(True, "local research", "2026-09-02"),
                source_recordings=[recording],
            )

            profile_path = root / "profile.json"
            save_voice_profile(profile, profile_path)
            loaded = load_voice_profile(profile_path)

            self.assertEqual(loaded.id, profile.id)
            self.assertEqual(loaded.source_recordings[0].sha256, recording.sha256)
            self.assertEqual(loaded.validate_for_training(), [])

    def test_training_validation_requires_consent_and_recordings(self) -> None:
        profile = VoiceProfile(
            id="speaker-2",
            display_name="Speaker Two",
            consent=ConsentRecord(False, "", "2026-09-02"),
        )
        self.assertEqual(len(profile.validate_for_training()), 3)


if __name__ == "__main__":
    unittest.main()
