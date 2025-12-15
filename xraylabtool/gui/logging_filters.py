from __future__ import annotations

from contextlib import contextmanager
import io
import os
import sys


@contextmanager
def suppress_qt_noise() -> None:
    """Filter noisy Qt offscreen stderr lines (e.g., propagateSizeHints).

    Keeps other stderr intact so real warnings still surface.
    """

    original = sys.stderr

    class _Filter(io.TextIOBase):
        noisy = ("propagateSizeHints",)

        def write(self, s: str) -> int:  # type: ignore[override]
            if any(token in s for token in self.noisy):
                return len(s)
            return original.write(s)

        def flush(self) -> None:  # type: ignore[override]
            original.flush()

    sys.stderr = _Filter()
    try:
        yield
    finally:
        sys.stderr = original


def enable_offscreen_quiet_env() -> None:
    """Set env vars that quiet common Qt offscreen noise without muting real warnings.

    Prefers the quieter "minimal" platform when available, otherwise falls back to
    "offscreen". Does not override an explicit user choice.
    """

    os.environ.setdefault(
        "QT_LOGGING_RULES", "*.debug=false;*.info=false;qt.qpa.*=false"
    )
    if "QT_QPA_PLATFORM" not in os.environ:
        # Try minimal first; if unavailable Qt will fall back automatically or emit once
        os.environ["QT_QPA_PLATFORM"] = "minimal"
    # Caller can still override by setting QT_QPA_PLATFORM before import


def suggest_quiet_platforms() -> list[str]:
    """Return optional platform plugins that can be quieter offscreen (opt-in)."""

    return ["minimal", "offscreen"]
