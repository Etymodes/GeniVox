from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from genivox.core.paths import ensure_private_directory, protect_private_file
from genivox.training.metrics import MetricParseError, parse_metric_line
from genivox.training.runs import RunManifest, RunStatus, RunStore

OutputCallback = Callable[[str], None]
FinishedCallback = Callable[[RunManifest], None]


class TrainingProcess:
    def __init__(
        self,
        process: subprocess.Popen[str],
        manifest: RunManifest,
        store: RunStore,
        log_stream: TextIO,
        metric_stream: TextIO,
        *,
        on_output: OutputCallback | None,
        on_finished: FinishedCallback | None,
    ) -> None:
        self._process = process
        self._initial_manifest = manifest
        self._store = store
        self._log_stream = log_stream
        self._metric_stream = metric_stream
        self._on_output = on_output
        self._on_finished = on_finished
        self._cancel_requested = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._pump_output,
            name=f"genivox-training-{manifest.run_id}",
            daemon=True,
        )

    @property
    def run_id(self) -> str:
        return self._initial_manifest.run_id

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def is_running(self) -> bool:
        return not self._done.is_set()

    @property
    def manifest(self) -> RunManifest:
        return self._store.load(self.run_id)

    def start_monitoring(self) -> None:
        self._thread.start()

    def wait(self, timeout: float | None = None) -> RunManifest:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired(list(self._initial_manifest.command), timeout)
        return self._store.load(self.run_id)

    def cancel(self, *, grace_seconds: float = 3.0) -> RunManifest:
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be non-negative")
        if self._done.is_set() or self._process.poll() is not None:
            return self.wait()

        self._cancel_requested.set()
        _terminate(self._process)
        try:
            self._process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            _kill(self._process)
        return self.wait(timeout=max(grace_seconds, 1.0) + 5.0)

    def _pump_output(self) -> None:
        terminal_manifest: RunManifest | None = None
        try:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                self._log_stream.write(line)
                self._log_stream.flush()
                try:
                    parse_metric_line(line)
                except MetricParseError:
                    pass
                else:
                    self._metric_stream.write(line)
                    self._metric_stream.flush()
                if self._on_output is not None:
                    try:
                        self._on_output(line.rstrip("\r\n"))
                    except Exception:
                        pass
            return_code = self._process.wait()
            if self._cancel_requested.is_set():
                status = RunStatus.CANCELLED
                error = "Training run cancelled by user"
            elif return_code == 0:
                status = RunStatus.SUCCEEDED
                error = None
            else:
                status = RunStatus.FAILED
                error = f"Training process exited with code {return_code}"
            terminal_manifest = self._store.transition(
                self.run_id,
                status,
                exit_code=return_code,
                error=error,
            )
        except Exception as exc:
            if self._process.poll() is None:
                _kill(self._process)
                self._process.wait()
            current = self._store.load(self.run_id)
            if current.status not in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
                terminal_manifest = self._store.transition(
                    self.run_id,
                    RunStatus.FAILED,
                    exit_code=self._process.returncode,
                    error=f"Training monitor failed: {exc}",
                )
        finally:
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._log_stream.close()
            self._metric_stream.close()
            self._done.set()
            if terminal_manifest is not None and self._on_finished is not None:
                try:
                    self._on_finished(terminal_manifest)
                except Exception:
                    pass


class TrainingRunner:
    """Launch local training commands without a shell or Qt dependency."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path,
        environment: Mapping[str, str] | None = None,
        metadata: Mapping[str, object] | None = None,
        run_id: str | None = None,
        on_output: OutputCallback | None = None,
        on_finished: FinishedCallback | None = None,
    ) -> TrainingProcess:
        env_overrides = {str(key): str(value) for key, value in (environment or {}).items()}
        manifest = self.store.create(
            command,
            cwd=cwd,
            environment=env_overrides,
            metadata=metadata,
            run_id=run_id,
        )
        ensure_private_directory(manifest.log_path.parent)
        log_stream = manifest.log_path.open("a", encoding="utf-8", newline="")
        metric_stream = manifest.metrics_path.open("a", encoding="utf-8", newline="")
        protect_private_file(manifest.log_path)
        protect_private_file(manifest.metrics_path)
        process_environment = os.environ.copy()
        # Training logs are a UTF-8 protocol too.  Python otherwise writes
        # redirected stdio with the active Windows code page.
        process_environment["PYTHONUTF8"] = "1"
        process_environment["PYTHONIOENCODING"] = "utf-8"
        process_environment.update(env_overrides)

        popen_options: dict[str, object] = {}
        if os.name == "posix":
            popen_options["start_new_session"] = True
        elif os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            process = subprocess.Popen(
                list(manifest.command),
                cwd=manifest.cwd,
                env=process_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                **popen_options,
            )
        except Exception as exc:
            log_stream.write(f"GeniVox could not start the training process: {exc}\n")
            log_stream.close()
            metric_stream.close()
            self.store.transition(manifest.run_id, RunStatus.FAILED, error=str(exc))
            raise

        running_manifest = self.store.transition(manifest.run_id, RunStatus.RUNNING)
        handle = TrainingProcess(
            process,
            running_manifest,
            self.store,
            log_stream,
            metric_stream,
            on_output=on_output,
            on_finished=on_finished,
        )
        handle.start_monitoring()
        return handle


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    elif os.name == "nt" and _taskkill(process.pid, force=False):
        return
    process.terminate()


def _kill(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    elif os.name == "nt" and _taskkill(process.pid, force=True):
        return
    process.kill()


def _taskkill(pid: int, *, force: bool) -> bool:
    command = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        command.append("/F")
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
