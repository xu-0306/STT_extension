from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui import config_manager
from gui.backend_runner import BackendRunner
from gui.download_worker import ModelDownloadWorker
from backend.model_manager import WHISPER_MODEL_URLS, model_filename

MODEL_OPTIONS = [
    ("tiny", "Tiny"),
    ("base", "Base"),
    ("small", "Small"),
    ("medium", "Medium"),
    ("large-v3", "Large v3"),
]
SUBTITLE_MAX_CHARS_DEFAULT = 260
SUBTITLE_MAX_SENTENCES_DEFAULT = 2
SUBTITLE_MAX_SENTENCES_CJK = 1
SUBTITLE_HISTORY_LINES_DEFAULT = 2
SUBTITLE_SHOW_PARTIAL_DEFAULT = True
STALL_TIMEOUT_DEFAULT = 15
STALL_CHECK_INTERVAL_DEFAULT = 5


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("STT Backend Control")
        self.setMinimumSize(900, 640)

        self.config_path = config_manager.ensure_user_config()
        self.config: Dict[str, Any] = config_manager.load_config(self.config_path)

        self.backend_runner: Optional[BackendRunner] = None
        self.download_worker: Optional[ModelDownloadWorker] = None
        self.tray: Optional[QtWidgets.QSystemTrayIcon] = None
        self.pending_start = False
        self.restart_pending = False
        self.force_close = False
        self.backend_failed = False
        self.last_backend_error: Optional[str] = None

        self._build_ui()
        self._apply_styles()
        self._load_config_into_ui(self.config)
        self._setup_tray()
        self._update_ws_url()
        self._update_buttons()

        QtCore.QTimer.singleShot(0, self._auto_start_if_valid)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(18, 18, 18, 12)
        layout.setSpacing(12)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_server_tab(), "Server")
        self.tabs.addTab(self._build_stt_tab(), "STT")
        self.tabs.addTab(self._build_tuning_tab(), "Tuning")
        layout.addWidget(self.tabs)

        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self.save_btn = QtWidgets.QPushButton("Save defaults")
        self.reset_btn = QtWidgets.QPushButton("Reset")
        footer.addWidget(self.reset_btn)
        footer.addWidget(self.save_btn)
        layout.addLayout(footer)

        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Idle")

        self.save_btn.clicked.connect(self._save_defaults)
        self.reset_btn.clicked.connect(self._reset_defaults)

        self.setCentralWidget(central)

    def _build_server_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(12)

        server_group = QtWidgets.QGroupBox("Server")
        form = QtWidgets.QFormLayout(server_group)
        form.setLabelAlignment(QtCore.Qt.AlignLeft)
        form.setFormAlignment(QtCore.Qt.AlignTop)

        self.host_input = QtWidgets.QLineEdit()
        self.port_input = QtWidgets.QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setMaximumWidth(140)

        ws_container = QtWidgets.QWidget()
        ws_layout = QtWidgets.QHBoxLayout(ws_container)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        self.ws_url_input = QtWidgets.QLineEdit()
        self.ws_url_input.setReadOnly(True)
        self.copy_ws_btn = QtWidgets.QPushButton("Copy")
        ws_layout.addWidget(self.ws_url_input, 1)
        ws_layout.addWidget(self.copy_ws_btn)

        form.addRow("Host", self.host_input)
        form.addRow("Port", self.port_input)
        form.addRow("WebSocket URL", ws_container)

        layout.addWidget(server_group)

        control_group = QtWidgets.QGroupBox("Controls")
        control_layout = QtWidgets.QHBoxLayout(control_group)
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.restart_btn = QtWidgets.QPushButton("Restart")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.restart_btn)
        control_layout.addStretch(1)

        self.server_status = QtWidgets.QLabel("Idle")
        self.server_status.setProperty("status", "idle")
        control_layout.addWidget(self.server_status)

        layout.addWidget(control_group)
        layout.addStretch(1)

        self.host_input.textChanged.connect(self._update_ws_url)
        self.port_input.valueChanged.connect(self._update_ws_url)
        self.copy_ws_btn.clicked.connect(self._copy_ws_url)
        self.start_btn.clicked.connect(self._start_backend)
        self.stop_btn.clicked.connect(self._stop_backend)
        self.restart_btn.clicked.connect(self._restart_backend)

        return tab

    def _build_stt_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(12)

        model_group = QtWidgets.QGroupBox("Model downloads")
        form = QtWidgets.QFormLayout(model_group)

        self.model_select_combo = QtWidgets.QComboBox()
        self.model_refresh_btn = QtWidgets.QPushButton("Refresh")
        self.model_download_btn = QtWidgets.QPushButton("Download")
        model_select_row = QtWidgets.QWidget()
        model_select_layout = QtWidgets.QHBoxLayout(model_select_row)
        model_select_layout.setContentsMargins(0, 0, 0, 0)
        model_select_layout.addWidget(self.model_select_combo, 1)
        model_select_layout.addWidget(self.model_refresh_btn)
        model_select_layout.addWidget(self.model_download_btn)

        self.model_cache_input = QtWidgets.QLineEdit()
        self.model_cache_btn = QtWidgets.QPushButton("Browse")
        model_cache_row = QtWidgets.QWidget()
        model_cache_layout = QtWidgets.QHBoxLayout(model_cache_row)
        model_cache_layout.setContentsMargins(0, 0, 0, 0)
        model_cache_layout.addWidget(self.model_cache_input, 1)
        model_cache_layout.addWidget(self.model_cache_btn)

        self.download_progress = QtWidgets.QProgressBar()
        self.download_progress.setValue(0)
        self.download_progress.setTextVisible(True)
        self.download_progress.hide()
        self.download_label = QtWidgets.QLabel("")

        form.addRow("Model to download", model_select_row)
        form.addRow("Model directory", model_cache_row)
        form.addRow("Download", self.download_progress)
        form.addRow("", self.download_label)

        layout.addWidget(model_group)
        layout.addStretch(1)

        self.model_cache_btn.clicked.connect(self._browse_model_dir)
        self.model_refresh_btn.clicked.connect(self._refresh_model_list)
        self.model_download_btn.clicked.connect(self._download_selected_model)
        self.model_cache_input.editingFinished.connect(self._refresh_model_list)

        return tab

    def _build_tuning_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(12)

        cleanup_group = QtWidgets.QGroupBox("Subtitle cleanup")
        cleanup_form = QtWidgets.QFormLayout(cleanup_group)
        self.subtitle_max_chars_input = QtWidgets.QSpinBox()
        self.subtitle_max_chars_input.setRange(60, 2000)
        self.subtitle_max_chars_input.setValue(SUBTITLE_MAX_CHARS_DEFAULT)
        self.subtitle_max_sentences_default_input = QtWidgets.QSpinBox()
        self.subtitle_max_sentences_default_input.setRange(1, 6)
        self.subtitle_max_sentences_default_input.setValue(SUBTITLE_MAX_SENTENCES_DEFAULT)
        self.subtitle_max_sentences_cjk_input = QtWidgets.QSpinBox()
        self.subtitle_max_sentences_cjk_input.setRange(1, 6)
        self.subtitle_max_sentences_cjk_input.setValue(SUBTITLE_MAX_SENTENCES_CJK)
        self.subtitle_history_lines_input = QtWidgets.QSpinBox()
        self.subtitle_history_lines_input.setRange(1, 4)
        self.subtitle_history_lines_input.setValue(SUBTITLE_HISTORY_LINES_DEFAULT)
        self.subtitle_show_partial_check = QtWidgets.QCheckBox("Show partial line")
        cleanup_form.addRow("Max chars", self.subtitle_max_chars_input)
        cleanup_form.addRow(
            "Max sentences (Latin)", self.subtitle_max_sentences_default_input
        )
        cleanup_form.addRow(
            "Max sentences (CJK)", self.subtitle_max_sentences_cjk_input
        )
        cleanup_form.addRow("History lines", self.subtitle_history_lines_input)
        cleanup_form.addRow("", self.subtitle_show_partial_check)
        layout.addWidget(cleanup_group)

        stability_group = QtWidgets.QGroupBox("STT stability")
        stability_form = QtWidgets.QFormLayout(stability_group)
        self.stall_timeout_input = QtWidgets.QSpinBox()
        self.stall_timeout_input.setRange(0, 300)
        self.stall_timeout_input.setValue(STALL_TIMEOUT_DEFAULT)
        self.stall_check_interval_input = QtWidgets.QSpinBox()
        self.stall_check_interval_input.setRange(0, 300)
        self.stall_check_interval_input.setValue(STALL_CHECK_INTERVAL_DEFAULT)
        stability_form.addRow("Stall timeout (sec)", self.stall_timeout_input)
        stability_form.addRow(
            "Stall check interval (sec)", self.stall_check_interval_input
        )
        layout.addWidget(stability_group)
        layout.addStretch(1)

        return tab

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
              background-color: #0f1115;
              color: #e6edf3;
              font-family: "Segoe UI";
              font-size: 12px;
            }
            QTabWidget::pane {
              border: 1px solid #202632;
              border-radius: 10px;
              padding: 6px;
            }
            QTabBar::tab {
              background: #151a21;
              padding: 8px 14px;
              margin-right: 4px;
              border-top-left-radius: 8px;
              border-top-right-radius: 8px;
            }
            QTabBar::tab:selected {
              background: #1c2330;
              color: #ffffff;
            }
            QGroupBox {
              border: 1px solid #222a36;
              border-radius: 10px;
              margin-top: 10px;
            }
            QGroupBox::title {
              subcontrol-origin: margin;
              left: 12px;
              padding: 0 6px;
              color: #9fb2c7;
            }
            QLineEdit, QComboBox, QSpinBox {
              background: #151a21;
              border: 1px solid #2b3442;
              border-radius: 8px;
              padding: 6px 8px;
            }
            QLineEdit:read-only {
              background: #10151c;
            }
            QPushButton {
              background: #2563eb;
              color: #ffffff;
              border: none;
              border-radius: 8px;
              padding: 6px 14px;
            }
            QPushButton:hover {
              background: #3b82f6;
            }
            QPushButton:disabled {
              background: #3a3f4a;
              color: #9aa4b2;
            }
            QProgressBar {
              border: 1px solid #2b3442;
              border-radius: 6px;
              background: #0e131a;
              text-align: center;
              height: 12px;
            }
            QProgressBar::chunk {
              background: #22c55e;
              border-radius: 6px;
            }
            QLabel[status="idle"] {
              color: #94a3b8;
            }
            QLabel[status="running"] {
              color: #22c55e;
            }
            QLabel[status="error"] {
              color: #f97316;
            }
            """
        )

    def _setup_tray(self) -> None:
        icon = self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        self.tray = QtWidgets.QSystemTrayIcon(icon, self)
        menu = QtWidgets.QMenu()

        open_action = menu.addAction("Open")
        start_action = menu.addAction("Start")
        stop_action = menu.addAction("Stop")
        restart_action = menu.addAction("Restart")
        menu.addSeparator()
        quit_action = menu.addAction("Quit")

        open_action.triggered.connect(self.showNormal)
        start_action.triggered.connect(self._start_backend)
        stop_action.triggered.connect(self._stop_backend)
        restart_action.triggered.connect(self._restart_backend)
        quit_action.triggered.connect(self._quit_from_tray)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.showNormal()
            self.raise_()
            self.activateWindow()

    def _quit_from_tray(self) -> None:
        self.force_close = True
        self._stop_backend()
        QtWidgets.QApplication.quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.force_close:
            event.accept()
            return
        event.accept()
        self._stop_backend()
        QtWidgets.QApplication.quit()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if (
            event.type() == QtCore.QEvent.WindowStateChange
            and self.isMinimized()
            and self.tray
            and self.tray.isVisible()
        ):
            QtCore.QTimer.singleShot(0, self.hide)
            self.tray.showMessage(
                "STT Backend",
                "Minimized to tray",
                QtWidgets.QSystemTrayIcon.Information,
                1500,
            )
        super().changeEvent(event)

    def _update_ws_url(self) -> None:
        host = self.host_input.text().strip() or "127.0.0.1"
        port = self.port_input.value()
        self.ws_url_input.setText(f"ws://{host}:{port}/asr")

    def _copy_ws_url(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.ws_url_input.text())
        self.status_bar.showMessage("WebSocket URL copied", 2000)

    def _update_buttons(self) -> None:
        running = self.backend_runner is not None and self.backend_runner.isRunning()
        downloading = self.download_worker is not None and self.download_worker.isRunning()
        self.start_btn.setEnabled(not running and not downloading)
        self.restart_btn.setEnabled(not downloading)
        self.stop_btn.setEnabled(running)
        if hasattr(self, "model_download_btn"):
            self.model_download_btn.setEnabled(not downloading)

    def _set_status(self, text: str, status_type: str = "idle") -> None:
        if self.backend_failed and status_type != "error":
            return
        self.server_status.setText(text)
        self.server_status.setProperty("status", status_type)
        self.server_status.style().unpolish(self.server_status)
        self.server_status.style().polish(self.server_status)
        self.status_bar.showMessage(text)
        if self.tray:
            self.tray.setToolTip(text)

    def _auto_start_if_valid(self) -> None:
        if not self._validate_server_settings(show_error=False):
            self._set_status("Invalid config", "error")
            return
        self._start_backend()

    def _validate_server_settings(self, show_error: bool = True) -> bool:
        host = self.host_input.text().strip()
        port = self.port_input.value()
        if not host:
            self.host_input.setText("127.0.0.1")
            self._update_ws_url()
        if port < 1 or port > 65535:
            if show_error:
                self._set_status("Port must be 1-65535", "error")
            return False
        return True

    def _resolve_model_target(self) -> tuple[Path, str]:
        model_name = self.model_select_combo.currentData() or "medium"
        cache_dir = self._get_model_dir()
        return cache_dir / model_filename(model_name), model_name

    def _start_download(self, target_path: Path, model_name: str) -> None:
        url = WHISPER_MODEL_URLS.get(model_name)
        if not url:
            self._set_status("Unknown model size", "error")
            return
        self.download_worker = ModelDownloadWorker(url, target_path)
        self.download_worker.progress.connect(self._on_download_progress)
        self.download_worker.finished.connect(self._on_download_finished)
        self.download_progress.show()
        self.download_progress.setValue(0)
        self.download_label.setText(f"Downloading {model_name}...")
        self.download_worker.start()
        self._update_buttons()

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            percent = int(downloaded / total * 100)
            self.download_progress.setValue(percent)
            self.download_label.setText(f"{percent}% ({downloaded // (1024*1024)} MB / {total // (1024*1024)} MB)")
        else:
            self.download_progress.setRange(0, 0)
            self.download_label.setText(f"{downloaded // (1024*1024)} MB")

    def _on_download_finished(self, ok: bool, message: str) -> None:
        if self.download_worker:
            self.download_worker.deleteLater()
        self.download_worker = None
        self.download_progress.setRange(0, 100)
        if ok:
            self.download_progress.setValue(100)
            self.download_label.setText("Download complete")
            if self.pending_start:
                self.pending_start = False
                self._start_backend()
        else:
            self.download_label.setText(message or "Download failed")
            self._set_status("Download failed", "error")
            self.pending_start = False
        self._update_buttons()

    def _ensure_model_ready(self) -> bool:
        target_path, model_name = self._resolve_model_target()
        is_standard = model_name in WHISPER_MODEL_URLS
        if target_path.exists():
            return True
        if not is_standard:
            QtWidgets.QMessageBox.warning(
                self,
                "Model missing",
                "Selected model file was not found in the directory.",
            )
            self._set_status("Model missing", "error")
            return False
        reply = QtWidgets.QMessageBox.question(
            self,
            "Model missing",
            f"{model_name}.pt was not found. Download it now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            self._set_status("Model missing", "error")
            return False
        self.pending_start = True
        self._start_download(target_path, model_name)
        return False

    def _start_backend(self) -> None:
        self.backend_failed = False
        self.last_backend_error = None
        try:
            if self.backend_runner and self.backend_runner.isRunning():
                self._set_status("Already running", "running")
                return
            if not self._validate_server_settings():
                return
            self._save_config_to_disk()
            if not self._ensure_model_ready():
                return
            host = self.host_input.text().strip()
            port = self.port_input.value()
            self.backend_runner = BackendRunner(str(self.config_path), host, port)
            self.backend_runner.status.connect(self._set_status)
            self.backend_runner.error.connect(self._on_backend_error)
            self.backend_runner.started.connect(
                lambda: self._set_status("Running", "running")
            )
            self.backend_runner.stopped.connect(self._on_backend_stopped)
            self.backend_runner.start()
            self._set_status("Starting", "running")
            self._update_buttons()
        except Exception as exc:
            self._on_backend_error(f"{type(exc).__name__}: {exc}")

    def _stop_backend(self) -> None:
        self.backend_failed = False
        self.last_backend_error = None
        if self.backend_runner and self.backend_runner.isRunning():
            self.backend_runner.stop()
            self._set_status("Stopping", "idle")
        self._update_buttons()

    def _restart_backend(self) -> None:
        if self.backend_runner and self.backend_runner.isRunning():
            self.restart_pending = True
            self._stop_backend()
            return
        self._start_backend()

    def _on_backend_stopped(self) -> None:
        if self.backend_runner:
            self.backend_runner.deleteLater()
        self.backend_runner = None
        if self.restart_pending:
            self.restart_pending = False
            self._start_backend()
            return
        if not self.backend_failed:
            self._set_status("Stopped", "idle")
        self._update_buttons()

    def _on_backend_error(self, message: str) -> None:
        self.backend_failed = True
        self.last_backend_error = message
        self._set_status(message, "error")

    def _browse_model_dir(self) -> None:
        start_dir = self.model_cache_input.text().strip()
        if not start_dir:
            start_dir = str(config_manager.get_default_model_dir())
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select model cache directory",
            start_dir,
        )
        if path:
            self.model_cache_input.setText(path)
            self._refresh_model_list()

    def _get_model_dir(self) -> Path:
        cache_dir = self.model_cache_input.text().strip()
        if cache_dir:
            return Path(cache_dir)
        return config_manager.get_default_model_dir()

    def _scan_model_dir(self, model_dir: Path) -> Dict[str, Path]:
        if not model_dir.exists():
            return {}
        return {path.stem: path for path in model_dir.glob("*.pt") if path.is_file()}

    def _download_selected_model(self) -> None:
        if self.download_worker and self.download_worker.isRunning():
            self._set_status("Download already running", "error")
            return
        target_path, model_name = self._resolve_model_target()
        if target_path.exists():
            self._set_status("Model already present", "idle")
            return
        if model_name not in WHISPER_MODEL_URLS:
            QtWidgets.QMessageBox.warning(
                self,
                "Model download unavailable",
                "Selected model is custom and has no download URL.",
            )
            return
        self._start_download(target_path, model_name)

    def _refresh_model_list(self, preferred: Optional[str] = None) -> None:
        model_dir = self._get_model_dir()
        available = self._scan_model_dir(model_dir)
        current = preferred or self.model_select_combo.currentData() or "medium"

        self.model_select_combo.blockSignals(True)
        self.model_select_combo.clear()

        for key, label in MODEL_OPTIONS:
            suffix = "" if key in available else " (download)"
            self.model_select_combo.addItem(f"{label}{suffix}", key)

        for name in sorted(available.keys()):
            if name in WHISPER_MODEL_URLS:
                continue
            self.model_select_combo.addItem(f"{name} (custom)", name)

        target_index = next(
            (i for i in range(self.model_select_combo.count()) if self.model_select_combo.itemData(i) == current),
            None,
        )
        if target_index is None:
            self.model_select_combo.addItem(f"{current} (missing)", current)
            target_index = self.model_select_combo.count() - 1
        self.model_select_combo.setCurrentIndex(target_index)
        self.model_select_combo.blockSignals(False)

    def _save_config_to_disk(self) -> None:
        updated = self._collect_config_from_ui()
        self.config = updated
        config_manager.write_config(self.config_path, updated)

    def _save_defaults(self) -> None:
        self._save_config_to_disk()
        self.status_bar.showMessage("Defaults saved", 2000)

    def _reset_defaults(self) -> None:
        defaults = config_manager.load_default_config()
        self.config = defaults
        self._load_config_into_ui(defaults)
        self.status_bar.showMessage("Reset to defaults", 2000)

    def _collect_config_from_ui(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        cfg["server"] = dict(cfg.get("server") or {})
        cfg["stt"] = dict(cfg.get("stt") or {})
        cfg["subtitle"] = dict(cfg.get("subtitle") or {})

        cfg["server"]["host"] = self.host_input.text().strip() or "127.0.0.1"
        cfg["server"]["port"] = int(self.port_input.value())

        cfg["stt"].pop("simulstreaming", None)
        model_dir = self._get_model_dir()
        cfg["stt"]["model_cache_dir"] = str(model_dir)
        cfg["stt"].pop("model_path", None)

        cfg["subtitle"]["max_chars"] = int(self.subtitle_max_chars_input.value())
        cfg["subtitle"]["max_sentences_default"] = int(
            self.subtitle_max_sentences_default_input.value()
        )
        cfg["subtitle"]["max_sentences_cjk"] = int(
            self.subtitle_max_sentences_cjk_input.value()
        )
        cfg["subtitle"]["history_lines"] = int(self.subtitle_history_lines_input.value())
        cfg["subtitle"]["show_partial"] = bool(
            self.subtitle_show_partial_check.isChecked()
        )
        cfg["stt"]["stall_timeout_sec"] = int(self.stall_timeout_input.value())
        cfg["stt"]["stall_check_interval_sec"] = int(
            self.stall_check_interval_input.value()
        )
        return cfg

    def _load_config_into_ui(self, cfg: Dict[str, Any]) -> None:
        server_cfg = cfg.get("server", {})
        self.host_input.setText(str(server_cfg.get("host", "127.0.0.1")))
        try:
            self.port_input.setValue(int(server_cfg.get("port", 8765)))
        except (TypeError, ValueError):
            self.port_input.setValue(8765)

        stt_cfg = cfg.get("stt", {})
        model_path = stt_cfg.get("model_path")
        model_cache_dir = stt_cfg.get("model_cache_dir")
        selected_model = stt_cfg.get("model", "medium")
        if model_path:
            try:
                selected_model = Path(model_path).stem
                if not model_cache_dir:
                    model_cache_dir = str(Path(model_path).parent)
            except OSError:
                pass
        if model_cache_dir:
            self.model_cache_input.setText(str(model_cache_dir))
        else:
            self.model_cache_input.setText(str(config_manager.get_default_model_dir()))
        self._refresh_model_list(selected_model)
        model_index = next(
            (i for i in range(self.model_select_combo.count()) if self.model_select_combo.itemData(i) == selected_model),
            None,
        )
        if model_index is not None:
            self.model_select_combo.setCurrentIndex(model_index)

        subtitle_cfg = cfg.get("subtitle", {})
        try:
            self.subtitle_max_chars_input.setValue(
                int(subtitle_cfg.get("max_chars", SUBTITLE_MAX_CHARS_DEFAULT))
            )
        except (TypeError, ValueError):
            self.subtitle_max_chars_input.setValue(SUBTITLE_MAX_CHARS_DEFAULT)
        try:
            self.subtitle_max_sentences_default_input.setValue(
                int(
                    subtitle_cfg.get(
                        "max_sentences_default", SUBTITLE_MAX_SENTENCES_DEFAULT
                    )
                )
            )
        except (TypeError, ValueError):
            self.subtitle_max_sentences_default_input.setValue(
                SUBTITLE_MAX_SENTENCES_DEFAULT
            )
        try:
            self.subtitle_max_sentences_cjk_input.setValue(
                int(
                    subtitle_cfg.get(
                        "max_sentences_cjk", SUBTITLE_MAX_SENTENCES_CJK
                    )
                )
            )
        except (TypeError, ValueError):
            self.subtitle_max_sentences_cjk_input.setValue(
                SUBTITLE_MAX_SENTENCES_CJK
            )
        try:
            self.subtitle_history_lines_input.setValue(
                int(subtitle_cfg.get("history_lines", SUBTITLE_HISTORY_LINES_DEFAULT))
            )
        except (TypeError, ValueError):
            self.subtitle_history_lines_input.setValue(SUBTITLE_HISTORY_LINES_DEFAULT)
        self.subtitle_show_partial_check.setChecked(
            bool(subtitle_cfg.get("show_partial", SUBTITLE_SHOW_PARTIAL_DEFAULT))
        )

        try:
            self.stall_timeout_input.setValue(
                int(stt_cfg.get("stall_timeout_sec", STALL_TIMEOUT_DEFAULT))
            )
        except (TypeError, ValueError):
            self.stall_timeout_input.setValue(STALL_TIMEOUT_DEFAULT)
        try:
            self.stall_check_interval_input.setValue(
                int(
                    stt_cfg.get(
                        "stall_check_interval_sec", STALL_CHECK_INTERVAL_DEFAULT
                    )
                )
            )
        except (TypeError, ValueError):
            self.stall_check_interval_input.setValue(STALL_CHECK_INTERVAL_DEFAULT)

        self._update_ws_url()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
