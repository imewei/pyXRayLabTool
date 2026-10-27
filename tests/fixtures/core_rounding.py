"""Generic ulp-scale comparison for the core calculators' float64 results.

Where XLA fuses or reorders an operation differently from another compile
(CPU vs CUDA, another bucket shape), two correct float64 results can differ by
an ulp or two. Goldens stay at ``rtol=1e-12`` and are never loosened;
:func:`assert_ulp_close` is for comparing two *program* outputs whose only
difference is rounding, with the rationale stated in the calling test.
(``kk_rounding.assert_kk_close`` is specific to the Kramers-Kronig sum.)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def assert_ulp_close(actual: Any, expected: Any, ulps: int = 4) -> None:
    """Assert ``|actual - expected| <= ulps * spacing(max(|actual|, |expected|))``.

    Both arguments are converted to float64 arrays and must have the same shape.
    NaN must sit at the same positions; infinities must be equal (same sign);
    finite entries are compared in ulps: the bound is
    ``ulps * spacing(max(|actual|, |expected|))``. Where ``expected`` is exactly 0
    only 0 or one of the smallest ``ulps`` subnormals (multiples of ``5e-324`` up to
    ``ulps * 5e-324``) passes.

    Parameters
    ----------
    actual, expected : array_like
        Values to compare.
    ulps : int, default 4
        Largest allowed distance in units in the last place.

    Raises
    ------
    AssertionError
        On a shape mismatch, a NaN/inf mismatch, or a finite difference above
        the bound; the message names the worst entry.

    Examples
    --------
    >>> from tests.fixtures.core_rounding import assert_ulp_close
    >>> assert_ulp_close([1.0, float("nan")], [1.0 + 2.2e-16, float("nan")])
    """
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape}"
    assert (np.isnan(a) == np.isnan(e)).all(), "NaN positions differ"
    inf = np.isinf(a) | np.isinf(e)
    assert np.array_equal(a[inf], e[inf]), "infinite values differ"
    finite = np.isfinite(a) & np.isfinite(e)
    a, e = a[finite], e[finite]
    if a.size == 0:
        return
    bound = ulps * np.spacing(np.maximum(np.abs(a), np.abs(e)))
    excess = np.abs(a - e) - bound
    worst = int(np.argmax(excess))
    assert excess[worst] <= 0, (
        f"{a[worst]!r} vs {e[worst]!r} differ by "
        f"{abs(a[worst] - e[worst]) / np.spacing(max(abs(a[worst]), abs(e[worst]))):.1f}"
        f" ulp (allowed {ulps})"
    )
