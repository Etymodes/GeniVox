from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from genivox.languages import EspeakNgPhonemizer


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    memory_total_mib: int | None = None
    driver_version: str | None = None


@dataclass(frozen=True, slots=True)
class SystemInfo:
    operating_system: str
    python_version: str
    machine: str
    workspace_free_gib: float
    gpus: list[GpuInfo] = field(default_factory=list)
    tools: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_system(workspace: Path) -> SystemInfo:
    workspace.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(workspace)
    return SystemInfo(
        operating_system=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        machine=platform.machine(),
        workspace_free_gib=round(usage.free / (1024**3), 1),
        gpus=_probe_nvidia_gpus(),
        tools={
            "ffmpeg": shutil.which("ffmpeg") is not None,
            "espeak-ng": EspeakNgPhonemizer().available,
            "git": shutil.which("git") is not None,
        },
    )


def _probe_nvidia_gpus() -> list[GpuInfo]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    return _parse_nvidia_smi(completed.stdout)


def _parse_nvidia_smi(output: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=2)]
        if len(fields) != 3:
            continue
        try:
            memory = int(fields[1])
        except ValueError:
            memory = None
        gpus.append(GpuInfo(name=fields[0], memory_total_mib=memory, driver_version=fields[2] or None))
    return gpus


def runtime_executable() -> str:
    """Return the exact interpreter used by the desktop process."""

    return sys.executable
