"""Small shared helpers used across CLI command modules."""

from typing import Any

import numpy as np


def create_batch_progress_tracker(**kwargs: Any) -> Any:
    """Stub implementation for the removed monitoring/performance progress module."""
    from contextlib import nullcontext

    class _NoOpProgress:
        # The batch loop calls progress.update(1); yield an object that accepts
        # it. A bare nullcontext() yields None, which crashes that call.
        def update(self, _n: int = 1) -> None:
            pass

    return nullcontext(_NoOpProgress())


def parse_energy_string(energy_str: str) -> np.ndarray:
    """Parse energy string and return numpy array."""
    if "," in energy_str:
        # Comma-separated values
        return np.array([float(x.strip()) for x in energy_str.split(",")])
    elif "-" in energy_str and ":" in energy_str:
        # Range format: start-end:count or start-end:count:spacing
        parts = energy_str.split(":")
        range_part = parts[0]
        count = int(parts[1])
        spacing = parts[2] if len(parts) > 2 else "linear"

        start, end = map(float, range_part.split("-"))

        if spacing.lower() == "log":
            return np.logspace(np.log10(start), np.log10(end), count)
        else:
            return np.linspace(start, end, count)
    else:
        # Single value
        return np.array([float(energy_str)])
