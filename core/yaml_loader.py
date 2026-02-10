from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Generic YAML loader (backwards compatible)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_vetro_yaml(path: str) -> Dict[str, Any]:
    return load_yaml(path)


def load_prompts_yaml(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    if "symbols" not in cfg and "prompts" in cfg:
        cfg["symbols"] = cfg["prompts"]
    if "symbols" not in cfg or not isinstance(cfg["symbols"], dict):
        cfg["symbols"] = {}
    return cfg