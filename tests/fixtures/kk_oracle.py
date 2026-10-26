"""Independent longdouble quadrature of Henke's f1, the oracle for the KK kernel.

Evaluates ``f1(E) = Z* + (2/π) PV∫ x f2(x) / (E² − x²) dx`` over a piecewise
Laurent f2 without using the closed-form segment sums of
:func:`xraylabtool.kk.transform.kk_piecewise_poly`.

With ``h(x) = x f2(x)`` the integrand on a segment is split exactly as

    h(x) / (E² − x²) = [h(x) − h(E)] / (E² − x²) + h(E) / (E² − x²).

The first term has a removable singularity at ``x = E``; polynomial division
removes it analytically (``(h(x) − h(E)) / (E − x) = −D(x)`` with ``D``
below), so it is integrated by Gauss-Legendre quadrature in ``ln x`` on
intervals split at every polynomial boundary and at ``E``. The second term's
principal value is two logarithms. All arithmetic is ``np.longdouble``; the
node count is doubled until the result changes by less than 1e-15 relative
(or the longdouble summation noise, 16 eps Σ|terms|, if that is larger: heavy
elements at low E sum terms ~1e7 times the result).

Valid for ``E`` strictly inside the data range and, at an interior knot, only
where f2 is continuous (the principal value is then an ordinary limit). Jump
knots and the two end points are upstream conventions, not integrals; they are
checked against the locked goldens instead.
"""

from __future__ import annotations

from functools import cache, lru_cache

import numpy as np
from numpy.typing import NDArray

LD = np.longdouble
_REL_TOL = 1e-15
_KNOT_CONTINUITY_RTOL = 1e-9
_PI = np.arccos(LD(-1))
_NOISE_K = 16.0


@cache
def _gauss_legendre(n: int) -> tuple[NDArray[np.longdouble], NDArray[np.longdouble]]:
    """Nodes and weights on [-1, 1], Newton-polished in longdouble."""
    x = np.polynomial.legendre.leggauss(n)[0].astype(LD)
    for _ in range(4):
        p0, p1 = np.ones_like(x), x.copy()
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        x = x - p1 / dp
    p0, p1 = np.ones_like(x), x.copy()
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    dp = n * (x * p1 - p0) / (x * x - 1)
    return x, 2 / ((1 - x * x) * dp * dp)


def _h(c: NDArray[np.longdouble], x: NDArray[np.longdouble]) -> NDArray[np.longdouble]:
    """``x f2(x)`` per segment row ``c`` (broadcasts against ``x``)."""
    a1, a0, am1, am2, am3 = c.T
    return a1 * x * x + a0 * x + am1 + am2 / x + am3 / (x * x)


def _quadrature(
    e: np.longdouble,
    lo: NDArray[np.longdouble],
    hi: NDArray[np.longdouble],
    c: NDArray[np.longdouble],
    n: int,
) -> tuple[np.longdouble, np.longdouble]:
    """``∫ −D(x)/(E + x) dx`` over sub-intervals ``[lo, hi]`` and Σ|terms|."""
    t, w = _gauss_legendre(n)
    half = (np.log(hi) - np.log(lo)) / 2
    mid = (np.log(hi) + np.log(lo)) / 2
    x = np.exp(mid[:, None] + half[:, None] * t[None, :])
    a1, a0, _, am2, am3 = (c[:, i, None] for i in range(5))
    d = a1 * (x + e) + a0 - am2 / (x * e) - am3 * (e + x) / (x * x * e * e)
    # dx = x dt, t = mid + half * node
    terms = -d / (e + x) * x * w[None, :] * half[:, None]
    return np.sum(terms), np.sum(np.abs(terms))


def _pv_integral(
    e: np.longdouble,
    seg: NDArray[np.longdouble],
    coeffs: NDArray[np.longdouble],
    n: int,
) -> tuple[np.longdouble, np.longdouble]:
    """``PV∫ x f2 / (E² − x²) dx`` with ``n`` nodes, and the Σ|terms| it sums."""
    edges = seg if (e <= seg[0] or e >= seg[-1]) else np.union1d(seg, [e])
    idx = np.searchsorted(seg, edges[:-1], side="right") - 1
    total, scale = _quadrature(e, edges[:-1], edges[1:], coeffs[idx], n)

    # h_j(E) * PV∫ dx/(E²−x²) per original segment, logs of non-singular args.
    a, b = seg[:-1], seg[1:]
    he = _h(coeffs, e)
    knot = np.flatnonzero(seg == e)
    ln_minus = np.zeros_like(he)
    hit_a, hit_b = a == e, b == e
    ok = ~(hit_a | hit_b)
    ln_minus[ok] = np.log(np.abs((e - a[ok]) / (e - b[ok])))
    ln_minus[hit_a] = -np.log(np.abs(e - b[hit_a]))
    ln_minus[hit_b] = np.log(np.abs(e - a[hit_b]))
    if knot.size:
        # The two divergent ln|x_k − E| terms cancel iff h is continuous there.
        k = int(knot[0])
        if k in (0, seg.size - 1):
            raise ValueError("E is an end point: the integral is not defined")
        left, right = he[k - 1], he[k]
        if abs(left - right) > _KNOT_CONTINUITY_RTOL * max(abs(left), abs(right)):
            raise ValueError(f"f2 jumps at the knot E={e}: not a principal value")
    ln_plus = np.log((e + b) / (e + a))
    logs = he * (ln_minus + ln_plus) / (2 * e)
    return total + np.sum(logs), scale + np.sum(np.abs(logs))


def f1_reference(
    energy_ev: float,
    seg_ev: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    rel_corr: float,
    *,
    n0: int = 16,
    max_nodes: int = 1024,
) -> float:
    """Henke f1 at ``energy_ev`` by longdouble quadrature of the tabulated f2.

    Parameters
    ----------
    energy_ev : float
        Strictly inside ``(seg_ev[0], seg_ev[-1])``; if it equals an interior
        knot, f2 must be continuous there.
    seg_ev : array, shape (n + 1,)
        Strictly increasing segment edges in eV.
    coeffs : array, shape (n, 5)
        ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment.
    rel_corr : float
        Relativistic correction Z*.
    n0, max_nodes : int
        Starting and maximum Gauss-Legendre nodes per interval; the count is
        doubled until the result changes by less than 1e-15 relative.

    Returns
    -------
    float
        f1 in electrons per formula unit.

    Raises
    ------
    ValueError
        At an end point or a jump knot (no principal value).
    RuntimeError
        If the quadrature does not converge within ``max_nodes`` nodes.
    """
    e = LD(energy_ev)
    seg = np.asarray(seg_ev, dtype=LD)
    c = np.asarray(coeffs, dtype=LD)
    prev, _ = _pv_integral(e, seg, c, n0)
    n = n0
    while n < max_nodes:
        n *= 2
        cur, scale = _pv_integral(e, seg, c, n)
        f_cur = LD(rel_corr) + 2 * cur / _PI
        change = 2 * abs(cur - prev) / _PI
        # Heavy elements at low E sum terms ~1e7 times the result, so the
        # longdouble summation noise exceeds 1e-15 relative; allow that floor.
        floor = _NOISE_K * np.finfo(LD).eps * 2 * scale / _PI
        if change <= max(_REL_TOL * max(abs(f_cur), LD(1)), floor):
            return float(f_cur)
        prev = cur
    raise RuntimeError(f"quadrature at E={energy_ev} eV not converged at {n} nodes")
