from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import io
import sys


@contextmanager
def suppress_qt_noise() -> Generator[None, None, None]:
    """Filter noisy Qt offscreen stderr lines (e.g., propagateSizeHints).

    Keeps other stderr intact so real warnings still surface.
    """

    original = sys.stderr

    class _Filter(io.TextIOBase):
        noisy = ("propagateSizeHints",)

        def write(self, s: str) -> int:
            if any(token in s for token in self.noisy):
                return len(s)
            return original.write(s)

        def flush(self) -> None:
            original.flush()

    sys.stderr = _Filter()
    try:
        yield
    finally:
        sys.stderr = original
