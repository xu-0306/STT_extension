from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

APP_DIR_NAME = "STT-TTS"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
CONFIG_ENV_VAR = "STT_CONFIG_PATH"


def _default_app_dir() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        base = Path.home() / ".stt-tts"
    target = base / APP_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def _ensure_model_cache_dir(cfg: Dict[str, Any]) -> None:
    stt_cfg = cfg.get("stt")
    if not isinstance(stt_cfg, dict):
        stt_cfg = {}
        cfg["stt"] = stt_cfg
    if not stt_cfg.get("model_cache_dir"):
        model_dir = _default_app_dir() / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        stt_cfg["model_cache_dir"] = str(model_dir)


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_override = path or os.getenv(CONFIG_ENV_VAR)
    config_path = Path(config_override) if config_override else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    _ensure_model_cache_dir(data)
    return data
