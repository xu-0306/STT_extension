from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore

from backend import model_manager


class ModelDownloadWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    status = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, url: str, target_path: Path) -> None:
        super().__init__()
        self.url = url
        self.target_path = target_path
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            total_bytes_holder = {"total": 0}

            def progress_cb(downloaded: int, total: int | None) -> None:
                total_bytes_holder["total"] = int(total or 0)
                self.progress.emit(downloaded, total_bytes_holder["total"])

            def cancel_cb() -> bool:
                return self._cancelled

            model_manager.download_model(
                self.url,
                self.target_path,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            self.finished.emit(True, "")
        except Exception as exc:
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
