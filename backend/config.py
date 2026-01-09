from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
CONFIG_ENV_VAR = "STT_CONFIG_PATH"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_override = path or os.getenv(CONFIG_ENV_VAR)
    config_path = Path(config_override) if config_override else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data
