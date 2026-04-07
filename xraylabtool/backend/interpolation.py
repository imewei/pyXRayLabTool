from __future__ import annotations

from typing import Any

from scipy.interpolate import PchipInterpolator
from xraylabtool.backend.array_ops import JaxBackend, _backend


class InterpolationFactory:
    @staticmethod
    def create_pchip(x: Any, y: Any, extrapolate: bool = False) -> Any:
        if isinstance(_backend, JaxBackend):
            import interpax  # type: ignore[import-not-found]
            return interpax.PchipInterpolator(x, y)
        else:
            return PchipInterpolator(x, y, extrapolate=extrapolate)
