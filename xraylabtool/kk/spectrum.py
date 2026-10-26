"""Compound imaginary scattering-factor spectrum on a piecewise-polynomial grid.

Altered from the original: a vectorised re-implementation of
``calculate_asf`` and ``coeffs_to_ASF`` from kkcalc/data.py of the
Kramers-Kronig Calculator (kkcalc, https://github.com/benajamin/kkcalc,
commit f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013
Benjamin Watts, Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``). The
method is described in B. Watts, Opt. Express 22(19), 23628-23639 (2014),
doi:10.1364/OE.22.023628.

Energies are in eV throughout this module (the tabulated coefficients are in
eV). This module is host-side NumPy: it only assembles and evaluates the
tabulated f2 polynomials, the Kramers-Kronig kernel lives in
:mod:`xraylabtool.kk.transform`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from xraylabtool.exceptions import FormulaError
from xraylabtool.kk.database import load_asf

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray


def _real_float64(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """float64 array of ``values``; complex input raises instead of losing its imaginary part."""
    arr = np.asarray(values)
    if arr.dtype.kind == "c":
        raise ValueError(f"{name} must be real numbers, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def compound_spectrum(
    symbols: Sequence[str], counts: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Piecewise-polynomial f2 of a formula unit on the union of element grids.

    The segment edges are the sorted union of the elemental edges; on each
    union segment the coefficients are the count-weighted sum of the
    polynomial that each element uses there (``Σ nᵢ f2ᵢ``).

    Parameters
    ----------
    symbols : sequence of str
        Element symbols (case-sensitive, ``"H"`` .. ``"U"``).
    counts : sequence of float
        Atoms of each element per formula unit; finite and positive.

    Returns
    -------
    seg_ev : numpy.ndarray, shape (n + 1,)
        Strictly increasing segment edges in eV (10 eV to 500 keV).
    coeffs : numpy.ndarray, shape (n, 5)
        Columns ``a₁, a₀, a₋₁, a₋₂, a₋₃`` of ``E¹ … E⁻³`` (``E`` in eV) for
        the segment ``[seg_ev[i], seg_ev[i + 1])``.

    Raises
    ------
    FormulaError
        If no elements are given, ``symbols`` and ``counts`` differ in length,
        a symbol is not a supported element (Z = 1..92), or a count is not
        finite and positive.

    Notes
    -----
    Elements are accumulated in the order given, so the sums are bit-identical
    to upstream's. All elements share the 10 eV - 500 keV range.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014): the f2 spectrum is the
    stoichiometric sum of the tabulated piecewise polynomials entering
    Eq. (8). The union-grid summation is upstream-only
    (``kkcalc.data.calculate_asf``), not an equation of the paper.

    Examples
    --------
    >>> from xraylabtool.kk import compound_spectrum
    >>> seg_ev, coeffs = compound_spectrum(["Si", "O"], [1.0, 2.0])
    >>> seg_ev.shape[0] == coeffs.shape[0] + 1
    True
    >>> coeffs.shape[1]
    5
    >>> float(seg_ev[0]), float(seg_ev[-1])
    (10.0, 500000.0)
    """
    if len(symbols) == 0:
        raise FormulaError("No elements given")
    if len(symbols) != len(counts):
        raise FormulaError(
            f"symbols and counts differ in length ({len(symbols)} != {len(counts)})"
        )
    db = load_asf()
    atomic_numbers = []
    for symbol, count in zip(symbols, counts, strict=True):
        if symbol not in db.symbols:
            raise FormulaError(f"Unsupported element {symbol!r} (Z = 1..92 only)")
        if not (math.isfinite(count) and count > 0):
            raise FormulaError(
                f"Element count must be finite and positive, got {count!r} "
                f"for {symbol!r}"
            )
        atomic_numbers.append(db.symbols.index(symbol) + 1)

    seg_ev = np.unique(np.concatenate([db.energy_ev(z) for z in atomic_numbers]))
    coeffs = np.zeros((seg_ev.size - 1, 5))
    for z, count in zip(atomic_numbers, counts, strict=True):
        # Segment of element z that covers each union segment's lower edge.
        idx = np.searchsorted(db.energy_ev(z), seg_ev[:-1], side="right") - 1
        coeffs += count * db.im_coeffs(z)[idx]
    return seg_ev, coeffs


def evaluate_f2(
    energy_ev: ArrayLike, seg_ev: ArrayLike, coeffs: ArrayLike
) -> NDArray[np.float64]:
    """Evaluate the piecewise Laurent polynomial f2 at given energies.

    ``f2(E) = a₁E + a₀ + a₋₁/E + a₋₂/E² + a₋₃/E³`` with the coefficients of
    the segment ``[seg_ev[i], seg_ev[i + 1])`` containing ``E``. Segments are
    right-continuous, so an energy on an interior edge uses the segment that
    starts there; the final edge uses the last polynomial, as upstream does.

    Parameters
    ----------
    energy_ev : array_like, shape (m,)
        Photon energies in eV, within ``[seg_ev[0], seg_ev[-1]]``.
    seg_ev : array_like, shape (n + 1,)
        Strictly increasing segment edges in eV.
    coeffs : array_like, shape (n, 5)
        Coefficients ``a₁, a₀, a₋₁, a₋₂, a₋₃`` per segment.

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        f2 in electrons per formula unit.

    Raises
    ------
    ValueError
        If an input is complex or has the wrong shape, ``energy_ev`` is not
        finite or lies outside the segment range, or ``seg_ev`` is not finite
        and strictly increasing.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eq. (8): the piecewise
    Laurent form of f2. The right-continuous edge convention is upstream's
    (``kkcalc.data.coeffs_to_ASF``).

    Examples
    --------
    >>> from xraylabtool.kk import evaluate_f2
    >>> seg = [1.0, 2.0, 4.0]
    >>> c = [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0]]
    >>> evaluate_f2([1.5, 2.0, 4.0], seg, c).tolist()
    [1.0, 1.0, 0.5]
    """
    e = _real_float64("energy_ev", energy_ev)
    seg = _real_float64("seg_ev", seg_ev)
    c = _real_float64("coeffs", coeffs)
    if e.ndim != 1:
        raise ValueError(f"energy_ev must be 1-D, got shape {e.shape}")
    if seg.ndim != 1 or seg.size < 2:
        raise ValueError(
            f"seg_ev must be 1-D with at least 2 edges, got shape {seg.shape}"
        )
    if c.shape != (seg.size - 1, 5):
        raise ValueError(f"coeffs must have shape ({seg.size - 1}, 5), got {c.shape}")
    if not np.isfinite(e).all():
        raise ValueError("energy_ev must be finite")
    if not np.isfinite(seg).all():
        raise ValueError("seg_ev must be finite")
    if (np.diff(seg) <= 0).any():
        raise ValueError("seg_ev must be strictly increasing")
    if e.size and (e.min() < seg[0] or e.max() > seg[-1]):
        raise ValueError(
            f"energy_ev must lie within [{seg[0]}, {seg[-1]}] eV, "
            f"got [{e.min()}, {e.max()}]"
        )
    k = c[np.clip(np.searchsorted(seg, e, side="right") - 1, 0, c.shape[0] - 1)]
    return np.asarray(
        k[:, 0] * e + k[:, 1] + k[:, 2] / e + k[:, 3] / e**2 + k[:, 4] / e**3,
        dtype=np.float64,
    )
