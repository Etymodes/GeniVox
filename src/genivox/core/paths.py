from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def ensure_private_directory(path: Path) -> Path:
    """Create a user-data directory and tighten POSIX permissions when available."""

    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        path.chmod(0o700)
    return path


def protect_private_file(path: Path) -> Path:
    """Limit a persisted voice/workbench file to its POSIX owner."""

    if os.name == "posix":
        path.chmod(0o600)
    return path


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path

    @property
    def engines(self) -> Path:
        return self.root / "engines"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    def ensure(self) -> WorkspacePaths:
        children = (
            self.root,
            self.engines,
            self.models,
            self.datasets,
            self.runs,
            self.outputs,
            self.profiles,
        )
        for path in children:
            ensure_private_directory(path)
        return self


def default_workspace() -> WorkspacePaths:
    override = os.environ.get("GENIVOX_WORKSPACE")
    if override:
        return WorkspacePaths(Path(override).expanduser().resolve())
    return WorkspacePaths((Path.home() / "GeniVoxWorkspace").resolve())
