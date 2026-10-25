"""Adaptive refinement of a sampled scattering-factor spectrum.

Altered from the original: a re-implementation of ``improve_accuracy`` from
kkcalc/kk.py of the Kramers-Kronig Calculator (kkcalc,
https://github.com/benajamin/kkcalc, commit
f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013 Benjamin Watts,
Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``).

Differences from upstream: the grid need not coincide with the segment edges
of the tabulated spectrum (f2 is evaluated at each midpoint from the segment
that contains it), an interval next to a tabulated range end is left alone
where f2 exceeds the tolerance there (upstream skips its first interval
unconditionally, see :func:`refine_spectrum`), the pass and point budgets are
explicit and reported, and an interval that float spacing cannot halve ends
the loop instead of repeating.

Energies are in eV and the tolerance is absolute, in electrons per formula
unit. This module is host-side NumPy: the number of new points is data
dependent, so only the batched transform of the midpoints runs through the
(possibly jitted) kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

# Imported at runtime so ``typing.get_type_hints`` resolves the annotations.
import numpy as np
from numpy.typing import ArrayLike, NDArray

from xraylabtool.exceptions import ValidationError
from xraylabtool.kk.spectrum import evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum
from xraylabtool.logging_utils import get_logger

logger = get_logger("kk")


@dataclass(frozen=True, eq=False)
class RefineOutcome:
    """Result of :func:`refine_spectrum`.

    The dataclass is frozen and compares by identity (``eq=False``): the
    array fields have no element-wise-safe ``==`` or ``hash``.

    Attributes
    ----------
    energy_ev : numpy.ndarray, shape (m + n_added,)
        Strictly increasing energies in eV: the input grid plus the added
        midpoints.
    f1 : numpy.ndarray, shape (m + n_added,)
        Real part at ``energy_ev``, electrons per formula unit.
    f2 : numpy.ndarray, shape (m + n_added,)
        Imaginary part at ``energy_ev``, electrons per formula unit.
    n_added : int
        Number of midpoints added.
    exhausted : bool
        ``True`` if refinement stopped on a limit (``max_passes``,
        ``max_points`` or float spacing) before every interval met the
        tolerance.

    References
    ----------
    Upstream-only: ``improve_accuracy`` of kkcalc is not part of B. Watts,
    Opt. Express 22(19), 23628 (2014).

    Examples
    --------
    >>> import numpy as np
    >>> out = RefineOutcome(np.array([1.0]), np.array([2.0]), np.array([3.0]), 0, False)
    >>> out.n_added, out.exhausted
    (0, False)
    """

    energy_ev: NDArray[np.float64]
    f1: NDArray[np.float64]
    f2: NDArray[np.float64]
    n_added: int
    exhausted: bool


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValidationError(f"{name} must be a positive integer", name, value)
    return int(value)


def _grid_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        arr = np.array(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be real numbers: {exc}", name) from exc
    if arr.ndim != 1 or arr.size == 0:
        raise ValidationError(f"{name} must be a non-empty 1-D array", name, arr.shape)
    if not np.isfinite(arr).all():
        raise ValidationError(f"{name} must be finite", name)
    return arr


def _checked_inputs(
    energy_ev: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    seg_ev: ArrayLike,
    tolerance: float,
    max_passes: int,
    max_points: int,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Validated ``(energy, f1, f2, seg)`` float64 arrays of :func:`refine_spectrum`."""
    e = _grid_array(energy_ev, "energy_ev")
    f1_arr = _grid_array(f1, "f1")
    f2_arr = _grid_array(f2, "f2")
    if not (e.size == f1_arr.size == f2_arr.size):
        raise ValidationError(
            "energy_ev, f1 and f2 must have the same length",
            "shapes",
            (e.size, f1_arr.size, f2_arr.size),
        )
    if e[0] <= 0 or (np.diff(e) <= 0).any():
        raise ValidationError(
            "energy_ev must be positive, strictly increasing and unique",
            "energy_ev",
        )
    seg = np.asarray(seg_ev, dtype=np.float64)
    if e[0] < seg[0] or e[-1] > seg[-1]:
        raise ValidationError(
            "energy_ev must lie inside the tabulated range",
            "energy_ev",
            (float(e[0]), float(e[-1])),
        )
    if isinstance(tolerance, bool) or not isinstance(
        tolerance, int | float | np.number
    ):
        raise ValidationError("tolerance must be a number", "tolerance", tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValidationError(
            "tolerance must be finite and positive", "tolerance", tolerance
        )
    _positive_int(max_passes, "max_passes")
    _positive_int(max_points, "max_points")

    return e, f1_arr, f2_arr, seg


def refine_spectrum(
    energy_ev: ArrayLike,
    f1: ArrayLike,
    f2: ArrayLike,
    seg_ev: ArrayLike,
    coeffs: ArrayLike,
    rel_corr: float,
    *,
    tolerance: float,
    max_passes: int = 50,
    max_points: int = 100_000,
) -> RefineOutcome:
    """Insert midpoints where linear interpolation of f1 or f2 is inaccurate.

    Each interval between neighbouring energies is bisected. The midpoint's
    f1 (Kramers-Kronig transform) and f2 (tabulated polynomial) are computed
    and compared with the average of the interval's end values; if either
    differs by more than ``tolerance`` the midpoint is kept and both halves
    are checked in the next pass. Intervals that pass are final. Only
    midpoints are tested, so a feature narrower than an interval can still be
    missed. Input points are never dropped or changed.

    Parameters
    ----------
    energy_ev : array_like, shape (m,)
        Strictly increasing, unique, positive energies in eV inside
        ``[seg_ev[0], seg_ev[-1]]``.
    f1 : array_like, shape (m,)
        Real part at ``energy_ev``, electrons per formula unit.
    f2 : array_like, shape (m,)
        Imaginary part at ``energy_ev``, electrons per formula unit.
    seg_ev : array_like, shape (n + 1,)
        Segment edges of the spectrum in eV, as for
        :func:`~xraylabtool.kk.transform.kk_real_spectrum`.
    coeffs : array_like, shape (n, 5)
        Polynomial coefficients per segment.
    rel_corr : float
        Relativistic correction Z* (see
        :func:`~xraylabtool.kk.transform.relativistic_correction`).
    tolerance : float
        Largest accepted interpolation error, absolute, in electrons per
        formula unit; finite and positive.
    max_passes : int, default 50
        Maximum number of bisection passes (an interval can be halved at most
        this many times); at least 1.
    max_points : int, default 100000
        Maximum number of points to add; at least 1. When a pass would
        exceed it, the leftmost intervals of that pass get their midpoints
        and refinement stops.

    Returns
    -------
    RefineOutcome
        The sorted union of the input energies and the added midpoints with
        f1 and f2, the number of points added and whether a limit stopped
        refinement. Points added per pass are logged at DEBUG, the total at
        INFO and an exhausted limit at WARNING (logger ``xraylabtool.kk``).

    Raises
    ------
    ValidationError
        If an array is empty, not 1-D, of unequal length or not finite,
        ``energy_ev`` is not strictly increasing, positive and inside the
        tabulated range, ``tolerance`` is not finite and positive, or a limit
        is not a positive integer.
    ValueError
        If ``seg_ev``, ``coeffs`` or ``rel_corr`` are invalid (raised by the
        kernel).

    Notes
    -----
    An interval whose midpoint equals one of its ends in floating point
    cannot be split further; it is left as is and reported as exhausted.

    f1 diverges logarithmically at a tabulated range end ``seg_ev[0]`` or
    ``seg_ev[-1]`` where f2 is nonzero, so bisecting towards it never
    converges. An input interval that ends there with ``|f2| > tolerance`` is
    therefore not refined. Upstream skips the first interval of its grid for
    this reason; it is the same rule on the grids upstream uses (the lower
    end has f2 well above the usual tolerance, the upper end far below).

    References
    ----------
    Upstream-only: this is kkcalc's ``improve_accuracy`` (not in B. Watts,
    Opt. Express 22(19), 23628 (2014)).

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.kk import compound_spectrum, evaluate_f2
    >>> from xraylabtool.kk import kk_real_spectrum, relativistic_correction
    >>> seg, coeffs = compound_spectrum(["Si"], [1.0])
    >>> rel = relativistic_correction([14], [1.0])
    >>> e = np.array([100.0, 1000.0, 10000.0])
    >>> out = refine_spectrum(
    ...     e,
    ...     kk_real_spectrum(e, seg, coeffs, rel),
    ...     evaluate_f2(e, seg, coeffs),
    ...     seg,
    ...     coeffs,
    ...     rel,
    ...     tolerance=0.1,
    ... )
    >>> bool(out.n_added > 0 and out.energy_ev.size == 3 + out.n_added)
    True
    """
    e, f1_arr, f2_arr, seg = _checked_inputs(
        energy_ev, f1, f2, seg_ev, tolerance, max_passes, max_points
    )
    # Active intervals as rows lo, hi, f1_lo, f1_hi, f2_lo, f2_hi; replaced every pass.
    iv = np.vstack([e[:-1], e[1:], f1_arr[:-1], f1_arr[1:], f2_arr[:-1], f2_arr[1:]])
    max_passes, max_points = int(max_passes), int(max_points)
    pole = ((iv[0] == seg[0]) & (np.abs(iv[4]) > tolerance)) | (
        (iv[1] == seg[-1]) & (np.abs(iv[5]) > tolerance)
    )
    iv = iv[:, ~pole]
    new_pts: list[NDArray[np.float64]] = []  # (3, k) blocks: energy, f1, f2
    added = 0
    passes = 0
    reason = ""
    while iv.shape[1] and passes < max_passes:
        passes += 1
        mid = 0.5 * (iv[0] + iv[1])
        splittable = (mid > iv[0]) & (mid < iv[1])
        if not splittable.all():
            reason = reason or "float spacing"
            iv, mid = iv[:, splittable], mid[splittable]
        if not mid.size:
            break
        f1_mid = kk_real_spectrum(mid, seg, coeffs, rel_corr)
        f2_mid = evaluate_f2(mid, seg, coeffs)
        bad = (np.abs(0.5 * (iv[2] + iv[3]) - f1_mid) > tolerance) | (
            np.abs(0.5 * (iv[4] + iv[5]) - f2_mid) > tolerance
        )
        if added + int(bad.sum()) > max_points:
            keep = np.flatnonzero(bad)[: max_points - added]
            bad = np.zeros_like(bad)
            bad[keep] = True
            reason = f"max_points={max_points}"
        n_bad = int(bad.sum())
        logger.debug(
            "refine_spectrum: pass %d added %d points (of %d intervals checked)",
            passes,
            n_bad,
            mid.size,
        )
        added += n_bad
        new_pts.append(np.vstack([mid, f1_mid, f2_mid])[:, bad])
        if reason.startswith("max_points"):
            break
        left = np.vstack([iv[0], mid, iv[2], f1_mid, iv[4], f2_mid])
        right = np.vstack([mid, iv[1], f1_mid, iv[3], f2_mid, iv[5]])
        iv = np.hstack([left[:, bad], right[:, bad]])
    if not reason and iv.shape[1] and passes >= max_passes:
        reason = f"max_passes={max_passes}"

    pts = np.hstack([np.vstack([e, f1_arr, f2_arr]), *new_pts])
    pts = pts[:, np.argsort(pts[0], kind="stable")]
    exhausted = bool(reason)
    logger.info(
        "refine_spectrum: added %d points in %d passes (tolerance %g)",
        added,
        passes,
        tolerance,
    )
    if exhausted:
        logger.warning(
            "refine_spectrum exhausted (%s) with %d points added; "
            "the tolerance %g is not met everywhere",
            reason,
            added,
            tolerance,
        )
    return RefineOutcome(
        energy_ev=pts[0],
        f1=pts[1],
        f2=pts[2],
        n_added=added,
        exhausted=exhausted,
    )
