"""Start and stop an explicitly trusted local GPT-SoVITS API process."""

from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from genivox.core.models import EngineManifest, EngineTransport
from genivox.core.paths import ensure_private_directory, protect_private_file
from genivox.engines.gpt_sovits import (
    GptSovitsProbeResult,
    GptSovitsProbeStatus,
    inspect_gpt_sovits_installation,
    probe_gpt_sovits_api,
)


class GptSovitsServiceError(RuntimeError):
    """A configured local service could not be started or verified."""


class GptSovitsServiceManager:
    """Manage one child process; never terminate an already running external API."""

    def __init__(
        self,
        log_dir: Path,
        *,
        probe: Callable[[str], GptSovitsProbeResult] = probe_gpt_sovits_api,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_path: Path | None = None
        self._probe = probe
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._owned_endpoint: str | None = None

    def ensure_ready(
        self, manifest: EngineManifest, *, timeout_seconds: float = 90.0
    ) -> GptSovitsProbeResult:
        """Reuse a ready service or launch a trusted installation in the caller's worker thread."""

        if not 0 < timeout_seconds <= 600:
            raise ValueError("service startup timeout must be between 0 and 600 seconds")
        with self._lock:
            if self._stopping.is_set():
                raise GptSovitsServiceError("GeniVox is closing; cannot start the model service")
            endpoint, port = _validated_endpoint(manifest)
            if self._process is not None and self._owned_endpoint != endpoint:
                if self._process.poll() is None:
                    raise GptSovitsServiceError(
                        "A different GPT-SoVITS service is already managed by GeniVox"
                    )
                self._process = None
                self._owned_endpoint = None

            initial = self._probe(endpoint)
            if initial.ready:
                return initial
            if initial.status is not GptSovitsProbeStatus.OFFLINE:
                raise GptSovitsServiceError(
                    f"Will not launch GPT-SoVITS: {initial.status.value}: {initial.message}"
                )

            if self._process is None or self._process.poll() is not None:
                installation = inspect_gpt_sovits_installation(
                    manifest.root, python=manifest.python
                )
                if not installation.launch_ready or installation.python is None:
                    issues = ", ".join(installation.issues)
                    raise GptSovitsServiceError(
                        f"GPT-SoVITS installation cannot start: {issues}"
                    )
                self._start(installation.root, installation.python, endpoint, port)

            deadline = time.monotonic() + timeout_seconds
            while True:
                if self._stopping.is_set():
                    self._stop_owned()
                    raise GptSovitsServiceError("GeniVox closed while starting GPT-SoVITS")
                if self._process is None:
                    raise GptSovitsServiceError("GPT-SoVITS process was not started")
                exit_code = self._process.poll()
                if exit_code is not None:
                    self._stop_owned()
                    raise GptSovitsServiceError(
                        f"GPT-SoVITS exited with code {exit_code}; see {self.log_path}"
                    )
                result = self._probe(endpoint)
                if result.ready:
                    return result
                if result.status is not GptSovitsProbeStatus.OFFLINE:
                    self._stop_owned()
                    raise GptSovitsServiceError(
                        "GPT-SoVITS endpoint responded with an incompatible service: "
                        f"{result.message}; see {self.log_path}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._stop_owned()
                    raise GptSovitsServiceError(
                        f"GPT-SoVITS did not become ready within {timeout_seconds:g}s; "
                        f"see {self.log_path}"
                    )
                self._stopping.wait(min(0.5, remaining))

    def stop(self) -> None:
        """Stop only the process launched by this instance."""

        self._stopping.set()
        with self._lock:
            self._stop_owned()

    def _start(self, root: Path | None, python: Path, endpoint: str, port: int) -> None:
        if root is None:
            raise GptSovitsServiceError("GPT-SoVITS source directory is missing")
        ensure_private_directory(self.log_dir)
        name = f"gpt-sovits-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.log"
        self.log_path = self.log_dir / name
        command = [
            str(python),
            "api_v2.py",
            "-a",
            "127.0.0.1",
            "-p",
            str(port),
            "-c",
            "GPT_SoVITS/configs/tts_infer.yaml",
        ]
        try:
            with self.log_path.open("xb") as log:
                protect_private_file(self.log_path)
                self._process = subprocess.Popen(
                    command,
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                    ),
                )
        except OSError as exc:
            raise GptSovitsServiceError(
                f"Cannot start GPT-SoVITS: {exc}; see {self.log_path}"
            ) from exc
        self._owned_endpoint = endpoint

    def _stop_owned(self) -> None:
        process = self._process
        self._process = None
        self._owned_endpoint = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        except OSError:
            pass


def _validated_endpoint(manifest: EngineManifest) -> tuple[str, int]:
    if (
        manifest.transport is not EngineTransport.HTTP
        or manifest.metadata.get("adapter") != "gpt_sovits_v2"
        or manifest.metadata.get("auto_start") is not True
        or manifest.metadata.get("trusted_local_code") is not True
        or not manifest.root
        or not manifest.python
        or not manifest.endpoint
    ):
        raise GptSovitsServiceError(
            "GPT-SoVITS auto start requires a trusted HTTP registration with "
            "source directory, Python executable, and explicit auto start"
        )
    endpoint = manifest.endpoint
    if any(ord(char) < 33 or ord(char) == 127 for char in endpoint):
        raise GptSovitsServiceError("Invalid GPT-SoVITS service address")
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise GptSovitsServiceError(f"Invalid GPT-SoVITS service address: {exc}") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/tts"
        or port is None
        or port == 0
    ):
        raise GptSovitsServiceError(
            "Auto start requires http://127.0.0.1:<port>/tts with no credentials"
        )
    return endpoint, port
