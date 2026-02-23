"""
EnergeX — Config Loader

Loads ProjectConfig from JSON or YAML files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from energex.domain.schemas import ProjectConfig


def load_config(path: str | Path) -> ProjectConfig:
    """
    Load a ProjectConfig from a JSON or YAML file.

    Parameters
    ----------
    path : str | Path
        Path to config file (.json or .yaml/.yml)

    Returns
    -------
    ProjectConfig
        Validated configuration object.

    Raises
    ------
    ValueError
        If the file format is not supported.
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".json":
        with open(p) as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml to load YAML configs: pip install pyyaml") from exc
        with open(p) as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .json or .yaml")

    return ProjectConfig.model_validate(data)


def load_config_dict(data: dict[str, Any]) -> ProjectConfig:
    """Load a ProjectConfig from a dictionary."""
    return ProjectConfig.model_validate(data)
