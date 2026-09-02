"""Workspace-backed application configuration defaults."""

from __future__ import annotations

from pathlib import Path

from genivox.core.models import Capability, EngineManifest, EngineTransport
from genivox.core.paths import WorkspacePaths
from genivox.engines import EngineRegistry


def default_engine_registry(workspace: WorkspacePaths) -> EngineRegistry:
    """Return useful local defaults without downloading or executing model code."""

    return EngineRegistry(
        [
            EngineManifest(
                id="mock-local",
                name="Mock WAV（功能自检）",
                transport=EngineTransport.MOCK,
                capabilities=[Capability.CROSS_LINGUAL, Capability.SPEED],
                languages=[],
                metadata={"sample_rate": 16_000, "status": "ready"},
            ),
            EngineManifest(
                id="gpt-sovits-v2-local",
                name="GPT-SoVITS 本地 API",
                transport=EngineTransport.HTTP,
                capabilities=[
                    Capability.VOICE_CLONE,
                    Capability.CROSS_LINGUAL,
                    Capability.SPEED,
                ],
                languages=["auto", "zh", "en", "ja", "ko", "yue"],
                endpoint="http://127.0.0.1:9880/tts",
                checkpoint_dir=str(workspace.models / "gpt-sovits"),
                metadata={
                    "adapter": "gpt_sovits_v2",
                    "api_version": "v2",
                    "model_version": "unverified",
                    "timeout_seconds": 120,
                    "status": "已登记",
                },
            ),
        ]
    )


def registry_path(workspace: WorkspacePaths) -> Path:
    return workspace.engines / "registry.json"


def load_or_create_engine_registry(workspace: WorkspacePaths) -> EngineRegistry:
    """Load the user's registry, creating only a small manifest file when absent."""

    path = registry_path(workspace)
    if path.exists():
        return EngineRegistry.load(path)
    registry = default_engine_registry(workspace)
    registry.save(path)
    return registry
