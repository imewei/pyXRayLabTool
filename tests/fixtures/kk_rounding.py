"""Rounding-error scale of the piecewise-polynomial KK sum, for test tolerances.

The closed-form KK_PP sum adds per-segment terms that can be ~1e12 times
larger than the result (Au at 13 eV: terms of 7.9e12 cancel to 3.2e3), so
two float64 evaluations that differ only in operation order (NumPy vs XLA
fusion, another chunk shape, another CPU's ``log``) differ by a few ulp of
the largest term. Tolerances are therefore ``rtol·|ref| + K·scale(E)``.
"""

from __future__ import annotations

import numpy as np

# Measured worst |difference| / scale over all transform goldens, NumPy and
# JAX, chunk sizes 1, 7, 1000 and default: 9.2 (Si). K = 32 leaves ~3x margin.
ROUNDING_K = 32.0


def kk_rounding_scale(
    eval_ev: np.ndarray, seg_ev: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    """``eps · max_j |term_j(E)| / (πE)`` per eval energy, shape (m,)."""
    e = np.asarray(eval_ev, dtype=np.float64)[:, None]
    x1, x2 = seg_ev[:-1], seg_ev[1:]
    a1, a0, am1, am2, am3 = np.abs(coeffs).T
    log_ratio = np.log(x2 / x1)
    poly = a1 * e**2 + a0 * e + am1 + am2 / e + am3 / e**2
    linear = (
        a1 * e * (x2 - x1)
        + a0 * (x2 - x1)
        + 0.5 * a1 * (x2**2 - x1**2)
        + (am2 / e + am3 / e**2) * log_ratio
        + am3 / e * np.abs(x2**-1 - x1**-1)
    )
    plus = poly * np.abs(np.log((x2 + e) / (x1 + e)))
    adjacent = (x2 == e) | (x1 == e)
    ratio = np.where(adjacent, 1.0, (x2 - e) / np.where(adjacent, 1.0, x1 - e))
    minus = np.where(adjacent, 0.0, poly * np.abs(np.log(np.abs(ratio))))
    largest = np.max(np.maximum(np.maximum(linear, plus), minus), axis=1)
    return np.finfo(np.float64).eps * largest / (np.pi * e[:, 0])


def assert_kk_close(
    actual: np.ndarray,
    expected: np.ndarray,
    eval_ev: np.ndarray,
    seg_ev: np.ndarray,
    coeffs: np.ndarray,
    rtol: float,
) -> None:
    """Assert ``|actual − expected| ≤ rtol·|expected| + K·scale(E)``."""
    assert np.isfinite(actual).all()
    bound = rtol * np.abs(expected) + ROUNDING_K * kk_rounding_scale(
        eval_ev, seg_ev, coeffs
    )
    excess = np.abs(actual - expected) - bound
    worst = int(np.argmax(excess))
    assert excess[worst] <= 0, (
        f"E={eval_ev[worst]!r} eV: |{actual[worst]!r} - {expected[worst]!r}| "
        f"exceeds {bound[worst]:.3e}"
    )
