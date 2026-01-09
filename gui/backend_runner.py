from __future__ import annotations

import os
from typing import Optional

from PySide6 import QtCore
import uvicorn


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
        self._server: Optional[uvicorn.Server] = None

    def run(self) -> None:
        try:
            os.environ["STT_CONFIG_PATH"] = self.config_path
            try:
                from backend import server as backend_server
            except Exception as exc:
                self.error.emit(f"Backend import failed: {type(exc).__name__}: {exc}")
                return
            config = uvicorn.Config(
                backend_server.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False,
            )
            self._server = uvicorn.Server(config)
            self.status.emit("Starting")
            self.started.emit()
            self._server.run()
            if self._server and not getattr(self._server, "started", False):
                self.error.emit(
                    "Backend failed to start (port in use or dependency error)"
                )
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.status.emit("Stopped")
            self.stopped.emit()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
