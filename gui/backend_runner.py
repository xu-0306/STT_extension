from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6 import QtCore


class BackendRunner(QtCore.QThread):
    status = QtCore.Signal(str)
    started = QtCore.Signal()
    stopped = QtCore.Signal()
    error = QtCore.Signal(str)

    def __init__(self, config_path: str, host: str, port: int) -> None:
        super().__init__()
        self.config_path = config_path
        self.host = host
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._project_root = Path(__file__).resolve().parents[1]

    def _terminate_process(self) -> None:
        if not self._process or self._process.poll() is not None:
            return
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def run(self) -> None:
        try:
            self._stop_requested = False
            env = os.environ.copy()
            env["STT_CONFIG_PATH"] = self.config_path
            cmd = [
                sys.executable,
                "-m",
                "backend.server",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--config",
                self.config_path,
            ]
            creationflags = 0
            if sys.platform.startswith("win"):
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                env=env,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            self.status.emit("Starting")
            self.started.emit()
            exit_code = None
            while True:
                if self._process.poll() is not None:
                    exit_code = self._process.returncode
                    break
                if self._stop_requested:
                    self._terminate_process()
                    exit_code = self._process.returncode
                    break
                time.sleep(0.1)
            if exit_code not in (None, 0) and not self._stop_requested:
                self.error.emit(
                    "Backend failed to start (port in use or dependency error)"
                )
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self._process = None
            self.status.emit("Stopped")
            self.stopped.emit()

    def stop(self) -> None:
        self._stop_requested = True
        self._terminate_process()
