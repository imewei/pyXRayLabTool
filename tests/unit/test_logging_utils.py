from __future__ import annotations

import logging
from pathlib import Path

from xraylabtool.logging_utils import configure_logging, get_logger, reset_logging


def test_configure_logging_writes_file(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XRAYLABTOOL_LOG_FILE", str(tmp_path / "run.log"))
    reset_logging()
    logger = configure_logging(level="DEBUG", console=False, force=True)
    logger.debug("hello from test")
    file_path = tmp_path / "run.log"
    logger.handlers[0].flush()
    assert file_path.exists()
    contents = file_path.read_text()
    assert "hello from test" in contents


def test_get_logger_child_inherits_config(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("XRAYLABTOOL_LOG_FILE", str(tmp_path / "child.log"))
    reset_logging()
    base = configure_logging(level=logging.INFO, console=False, force=True)
    child = get_logger("test.child")
    child.info("child message")
    for handler in base.handlers:
        handler.flush()
    data = (tmp_path / "child.log").read_text()
    assert "child message" in data
    assert "xraylabtool.test.child" in data
