"""Project configuration.

This module provides a small helper to load a YAML config file and access
its values via an immutable dataclass.

Note: the current pipeline can run without a config file (CLI defaults),
but the loader is kept for extensibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Immutable wrapper around a parsed YAML configuration."""

    raw: dict[str, Any]

    @property
    def env(self) -> str:
        """Environment name (e.g., dev/prod)."""
        return str(self.raw["main"]["env"])

    @property
    def seed(self) -> int:
        """Random seed used across the pipeline."""
        return int(self.raw["main"]["seed"])

    @property
    def log_dir(self) -> str:
        """Directory where logs should be written."""
        return str(self.raw["main"]["log_dir"])

    @property
    def log_name(self) -> str:
        """Base name for log files."""
        return str(self.raw["main"]["log_name"])

    def path(self, key: str) -> str:
        """Get a path from the `paths` section by key."""
        return str(self.raw["paths"][key])

    @property
    def val_block(self) -> int:
        """Validation date_block_num used for the temporal split."""
        return int(self.raw["train"]["val_block"])

    @property
    def feature_columns(self) -> list[str]:
        """Configured feature columns."""
        return list(self.raw["train"]["features"])

    @property
    def clip_min(self) -> float:
        """Lower clipping bound for predictions."""
        return float(self.raw["clip"]["min_pred"])

    @property
    def clip_max(self) -> float:
        """Upper clipping bound for predictions."""
        return float(self.raw["clip"]["max_pred"])


def load_config(path: str) -> AppConfig:
    """Load a YAML config file from disk."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must parse to a dict")

    return AppConfig(raw=raw)
