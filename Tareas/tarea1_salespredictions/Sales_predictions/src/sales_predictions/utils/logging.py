"""Logging utilities.

Creates a logger that logs both to console and to a timestamped file under
`artifacts/logs/`.

Security note:
- Never log secrets (passwords, tokens, API keys).
- Prefer project-relative paths.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("artifacts/logs")
DEFAULT_LEVEL = logging.INFO


def get_logger(name: str, *, level: int = DEFAULT_LEVEL) -> logging.Logger:
    """Return a configured logger with file + stream handlers."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(
        LOG_DIR / f"{_safe_name(name)}_{timestamp}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name).strip("_")
