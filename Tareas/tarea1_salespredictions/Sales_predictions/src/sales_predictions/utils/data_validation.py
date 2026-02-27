"""Data validation utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_file_exists(path: Path, *, hint: str | None = None) -> None:
    """Raise a helpful error if a required file is missing."""
    if not path.exists():
        msg = f"Required file not found: {path}"
        if hint:
            msg = f"{msg}. {hint}"
        raise FileNotFoundError(msg)
