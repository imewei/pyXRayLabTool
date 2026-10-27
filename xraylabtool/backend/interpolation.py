from __future__ import annotations

import functools
from typing import Any

import numpy as np


def _pchip_slopes(x: Any, y: Any) -> Any:
    """Fritsch-Carlson node slopes, following ``scipy.interpolate.PchipInterpolator``.

    Interior slopes are the weighted harmonic mean of the adjacent secants (zero
    where they differ in sign or either vanishes); end slopes use the one-sided
    three-point formula, clipped to preserve shape.
    """
    import jax.numpy as jnp

    h = jnp.diff(x)
    m = jnp.diff(y) / h
    if x.shape[0] == 2:
        return jnp.stack([m[0], m[0]])

    sm = jnp.sign(m)
    flat = (sm[1:] != sm[:-1]) | (m[1:] == 0) | (m[:-1] == 0)
    w1 = 2 * h[1:] + h[:-1]
    w2 = h[1:] + 2 * h[:-1]
    # ``flat`` slots would divide by zero; their value is discarded by the where.
    safe_a = jnp.where(flat, 1.0, m[:-1])
    safe_b = jnp.where(flat, 1.0, m[1:])
    interior = jnp.where(flat, 0.0, (w1 + w2) / (w1 / safe_a + w2 / safe_b))

    def edge(h0: Any, h1: Any, m0: Any, m1: Any) -> Any:
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        d = jnp.where(jnp.sign(d) != jnp.sign(m0), 0.0, d)
        clip = (jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > 3 * jnp.abs(m0))
        return jnp.where(clip, 3 * m0, d)

    return jnp.concatenate(
        [
            edge(h[0], h[1], m[0], m[1])[None],
            interior,
            edge(h[-1], h[-2], m[-1], m[-2])[None],
        ]
    )


@functools.cache
def _jit_pchip_eval() -> Any:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def evaluate(x: Any, y: Any, d: Any, xq: Any) -> Any:
        i = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, x.shape[0] - 2)
        h = x[i + 1] - x[i]
        t = (xq - x[i]) / h
        t2 = t * t
        t3 = t2 * t
        value = (
            (2 * t3 - 3 * t2 + 1) * y[i]
            + (t3 - 2 * t2 + t) * h * d[i]
            + (3 * t2 - 2 * t3) * y[i + 1]
            + (t3 - t2) * h * d[i + 1]
        )
        inside = (xq >= x[0]) & (xq <= x[-1])
        return jnp.where(inside, value, jnp.nan)

    return evaluate


class JaxPchipInterpolator:
    """Monotone piecewise-cubic Hermite (PCHIP) interpolator in ``jax.numpy``.

    Reproduces ``scipy.interpolate.PchipInterpolator(x, y, extrapolate=False)``
    to float64 round-off, so it matches SciPy. Queries outside
    ``[x[0], x[-1]]`` return NaN. Evaluation is jit-compiled and the node
    slopes are computed once at construction.

    Parameters
    ----------
    x : array_like, shape (n,)
        Strictly increasing sample points, ``n >= 2``.
    y : array_like, shape (n,)
        Values at ``x``.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not 1-D of equal length ``>= 2``, contain
        complex or non-finite values, or ``x`` is not strictly increasing.

    Notes
    -----
    Construction and evaluation use local x64 scopes so another caller's
    disabled-x64 setting cannot reduce the interpolation precision.
    """

    def __init__(self, x: Any, y: Any) -> None:
        import jax
        import jax.numpy as jnp

        # Without x64 jnp.asarray would silently truncate the tables to float32.
        jax.config.update("jax_enable_x64", True)
        x_np = np.asarray(x)
        y_np = np.asarray(y)
        if np.iscomplexobj(x_np) or np.iscomplexobj(y_np):
            raise ValueError("x and y must contain real values")
        x_np = np.asarray(x_np, dtype=np.float64)
        y_np = np.asarray(y_np, dtype=np.float64)
        if x_np.ndim != 1 or x_np.shape != y_np.shape or x_np.shape[0] < 2:
            raise ValueError("x and y must be 1-D arrays of equal length >= 2")
        if not (np.isfinite(x_np).all() and np.isfinite(y_np).all()):
            raise ValueError("x and y must be finite")
        if not (np.diff(x_np) > 0).all():
            raise ValueError("x must be strictly increasing")

        # A caller's thread-local x64 setting overrides the global default.
        with jax.enable_x64(True):
            self.x = jnp.asarray(x_np, dtype=jnp.float64)
            self.y = jnp.asarray(y_np, dtype=jnp.float64)
            self.d = _pchip_slopes(self.x, self.y)

    def __call__(self, xq: Any) -> Any:
        import jax
        import jax.numpy as jnp

        # Cached tables may be queried after another caller disables x64.
        with jax.enable_x64(True):
            return _jit_pchip_eval()(
                self.x, self.y, self.d, jnp.asarray(xq, dtype=jnp.float64)
            )
