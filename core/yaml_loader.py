from __future__ import annotations

from typing import Any, Dict
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Generic YAML loader used by the pipeline. Always returns a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    return obj if isinstance(obj, dict) else {}


def load_vetro_yaml(path: str) -> Dict[str, Any]:
    """
    Backwards-compatible alias.
    """
    return load_yaml(path)


def load_prompt_yaml(path: str) -> Dict[str, Any]:
    """
    Backwards-compatible alias for prompts config.
    """
    return load_yaml(path)
