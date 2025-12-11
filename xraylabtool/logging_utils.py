"""Centralized logging utilities for XRayLabTool.

This module provides a single entry point to configure logging across the library,
CLI, and GUI. It keeps setup lightweight (stdlib-only) while still offering:

- Rotating file logs (default location: ~/.cache/xraylabtool/logs/xraylabtool.log)
- Optional console output (stderr) with readable timestamps
- Environment-variable overrides for level, file path, and rotation
- Quieting of noisy third-party libraries (matplotlib, asyncio)

Environment variables
---------------------
- ``XRAYLABTOOL_LOG_LEVEL``: Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.
- ``XRAYLABTOOL_LOG_FILE``: Path to log file. If empty, file logging is disabled.
- ``XRAYLABTOOL_LOG_DIR``: Directory for default log file (overrides cache path).
- ``XRAYLABTOOL_LOG_MAX_BYTES``: Max size per log file before rotation (bytes). Default: 5_000_000.
- ``XRAYLABTOOL_LOG_BACKUPS``: Number of rotated backups to keep. Default: 3.
- ``XRAYLABTOOL_LOG_CONSOLE``: ``1``/``0`` toggle for console logging. Default: on.

Library consumers can opt in by calling ``configure_logging()`` early. The CLI and
GUI call it automatically.
"""

# ruff: noqa: I001

from __future__ import annotations

import logging
import os
import platform
import sys
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path

from xraylabtool import __version__

_STATE: dict[str, object | None] = {"configured": False, "log_file": None}


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def configure_logging(
    *,
    level: str | int | None = None,
    log_file: str | os.PathLike[str] | None = None,
    console: bool | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure package-wide logging once and return the base logger.

    Parameters
    ----------
    level : str | int | None
        Logging level (DEBUG/INFO/...). Defaults to ``XRAYLABTOOL_LOG_LEVEL`` or INFO.
    log_file : str | PathLike | None
        Path to log file. If ``""`` or ``None`` after env resolution, file logging is
        disabled. Defaults to ``XRAYLABTOOL_LOG_FILE`` or
        ``~/.cache/xraylabtool/logs/xraylabtool.log``.
    console : bool | None
        Whether to emit logs to stderr. Defaults to ``XRAYLABTOOL_LOG_CONSOLE`` (on).
    force : bool
        If True, reconfigure even if logging was already set up (useful in tests).
    """

    if _STATE["configured"] and not force:
        return logging.getLogger("xraylabtool")

    level_env = os.getenv("XRAYLABTOOL_LOG_LEVEL", "INFO").upper()
    level_value = level if level is not None else level_env
    resolved_level = (
        logging.getLevelName(level_value)
        if isinstance(level_value, str)
        else level_value
    )
    resolved_level = resolved_level if isinstance(resolved_level, int) else logging.INFO

    console_enabled = (
        _bool_env("XRAYLABTOOL_LOG_CONSOLE", True) if console is None else console
    )

    if log_file is None:
        env_file = os.getenv("XRAYLABTOOL_LOG_FILE")
        if env_file is not None:
            log_file = env_file
        else:
            base_dir = os.getenv(
                "XRAYLABTOOL_LOG_DIR",
                Path.home() / ".cache" / "xraylabtool" / "logs",
            )
            log_dir = Path(base_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "xraylabtool.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] [%(process)d:%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("xraylabtool")
    logger.setLevel(resolved_level)
    logger.propagate = False

    if _STATE["configured"] or force:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    if console_enabled:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _STATE["log_file"] = str(log_file) if log_file not in (None, "") else None

    if log_file not in (None, ""):
        max_bytes = _coerce_int_env("XRAYLABTOOL_LOG_MAX_BYTES", 5_000_000)
        backups = _coerce_int_env("XRAYLABTOOL_LOG_BACKUPS", 3)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backups
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Quiet common noisy libraries without muting real warnings
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    _STATE["configured"] = True
    logger.debug(
        "Logging configured",
        extra={
            "level": resolved_level,
            "console": console_enabled,
            "log_file": str(log_file) if log_file else None,
        },
    )
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the xraylabtool namespace.

    Calling this function ensures logging is configured once.
    """

    configure_logging()
    if name:
        return logging.getLogger(f"xraylabtool.{name}")
    return logging.getLogger("xraylabtool")


def get_log_file_path() -> str | None:
    """Return the configured log file path if file logging is enabled."""

    configure_logging()
    return _STATE.get("log_file")  # type: ignore[return-value]


def log_environment(
    logger: logging.Logger,
    *,
    component: str,
    extra_keys: Iterable[tuple[str, str]] | None = None,
) -> None:
    """Emit a single structured line capturing runtime context."""

    base = {
        "component": component,
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
    }
    if extra_keys:
        base.update(dict(extra_keys))
    logger.info("Runtime environment", extra=base)


def reset_logging() -> None:
    """Reset logging configuration (intended for tests)."""

    logger = logging.getLogger("xraylabtool")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _STATE["configured"] = False
    _STATE["log_file"] = None
