from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

APP_DIR_NAME = "STT-TTS"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "backend" / "config.yaml"


def get_app_dir() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        base = Path(appdata)
    else:
        base = Path.home() / ".stt-tts"
    target = base / APP_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_user_config_path() -> Path:
    return get_app_dir() / "config.yaml"


def get_default_model_dir() -> Path:
    path = get_app_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_default_config() -> Dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Default config root must be a mapping")
    return data


def ensure_user_config() -> Path:
    path = get_user_config_path()
    if not path.exists():
        data = load_default_config()
        write_config(path, data)
    return path


def load_config(path: Path | None = None) -> Dict[str, Any]:
    config_path = path or ensure_user_config()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def write_config(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
