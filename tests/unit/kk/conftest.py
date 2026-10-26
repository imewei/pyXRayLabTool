"""Fixtures shared by the ``xraylabtool.kk`` unit tests."""

from __future__ import annotations

import logging

import jax
import pytest


@pytest.fixture
def kk_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture the ``xraylabtool`` logger (it does not propagate to the root)."""
    logger = logging.getLogger("xraylabtool")
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="xraylabtool")
    yield caplog
    logger.removeHandler(caplog.handler)


@pytest.fixture
def x64_restored():
    """Restore ``jax_enable_x64`` after a test that switches it off."""
    yield
    jax.config.update("jax_enable_x64", True)
