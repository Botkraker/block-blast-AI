"""stdlib logging setup and run directories."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers, force=True)
    return logging.getLogger("blockblast")


def make_run_dir(root: Path, run_name: str) -> Path:
    """Create ``root/run_name`` (sanitized) and return it."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_name)
    path = root / safe
    path.mkdir(parents=True, exist_ok=True)
    return path
