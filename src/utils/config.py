from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration files are malformed."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a nested dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        raise ConfigError(f"Config file {config_path} is empty.")
    if not isinstance(data, Mapping):
        raise ConfigError("Expected configuration root to be a mapping.")
    return copy.deepcopy(dict(data))


def update_dict(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = update_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
