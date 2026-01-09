from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from PySide6 import QtCore

CHUNK_SIZE = 1024 * 256


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
        tmp_path = self.target_path.with_suffix(self.target_path.suffix + ".part")
        try:
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(self.url) as response:
                total = response.headers.get("Content-Length")
                total_bytes = int(total) if total and total.isdigit() else 0
                downloaded = 0
                with tmp_path.open("wb") as handle:
                    while True:
                        if self._cancelled:
                            raise RuntimeError("Download cancelled")
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total_bytes)
            os.replace(tmp_path, self.target_path)
            self.finished.emit(True, "")
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            self.finished.emit(False, f"{type(exc).__name__}: {exc}")
