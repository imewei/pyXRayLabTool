from __future__ import annotations

from typing import Any

from scipy.interpolate import PchipInterpolator

from xraylabtool.backend.array_ops import ops


class InterpolationFactory:
    """Builds interpolators matching the active array backend."""

    @staticmethod
    def create_pchip(x: Any, y: Any, extrapolate: bool = False) -> Any:
        """Create a monotone piecewise-cubic Hermite (PCHIP) interpolator.

        Parameters
        ----------
        x : array_like
            Strictly increasing sample points.
        y : array_like
            Values at ``x``.
        extrapolate : bool, default False
            Whether to extrapolate outside ``[x[0], x[-1]]``. Honored by the
            NumPy backend (``scipy.interpolate.PchipInterpolator``); ignored
            by the JAX backend (``interpax.PchipInterpolator``).

        Returns
        -------
        Callable
            Interpolator evaluated by calling it with query points.
        """
        if ops.is_jax():
            import interpax

            return interpax.PchipInterpolator(x, y)
        else:
            return PchipInterpolator(x, y, extrapolate=extrapolate)
