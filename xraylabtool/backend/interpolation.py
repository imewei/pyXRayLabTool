from __future__ import annotations

from typing import Any

from scipy.interpolate import PchipInterpolator

from xraylabtool.backend.array_ops import ops


class InterpolationFactory:
    @staticmethod
    def create_pchip(x: Any, y: Any, extrapolate: bool = False) -> Any:
        if ops.is_jax():
            import interpax  # type: ignore[import-not-found]

            return interpax.PchipInterpolator(x, y)
        else:
            return PchipInterpolator(x, y, extrapolate=extrapolate)
