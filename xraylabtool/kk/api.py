"""High-level Kramers-Kronig calculation of f1 from the tabulated f2.

Altered from the original: a re-implementation of ``kk_calculate_real`` from
kkcalc/kk.py of the Kramers-Kronig Calculator (kkcalc,
https://github.com/benajamin/kkcalc, commit
f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013 Benjamin Watts,
Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``). The
method is described in B. Watts, Opt. Express 22(19), 23628-23639 (2014),
doi:10.1364/OE.22.023628.

The public API takes energies in keV; the transform runs in eV (the
tabulated coefficients are in eV) after a single conversion at the boundary.
"""

from __future__ import annotations

import numpy as np

# Imported at runtime so ``typing.get_type_hints`` resolves the annotations.
from numpy.typing import ArrayLike, NDArray

from xraylabtool.backend.array_ops import ops
from xraylabtool.constants import KEV_TO_EV, TABULATED_MAX_KEV, TABULATED_MIN_KEV
from xraylabtool.exceptions import (
    EnergyError,
    FormulaError,
    NearEdgeDataError,
    ValidationError,
)
from xraylabtool.kk.database import load_asf
from xraylabtool.kk.nearedge import InputType, NearEdgeSpectrum, _merge, to_asf_scale
from xraylabtool.kk.refine import _positive_int, refine_spectrum
from xraylabtool.kk.result import KKResult
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction
from xraylabtool.logging_utils import get_logger
from xraylabtool.utils import parse_formula

logger = get_logger("kk")

# Default output window (keV): the package's tabulated range, 0.03-30 keV.
_DEFAULT_MIN_KEV = TABULATED_MIN_KEV
_DEFAULT_MAX_KEV = TABULATED_MAX_KEV


def _validated_energies_ev(
    energies_kev: ArrayLike, lo_ev: float, hi_ev: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(energies_kev, energies_ev)`` after validation (the one unit conversion)."""
    try:
        raw = np.asarray(energies_kev)
    except (TypeError, ValueError) as exc:
        raise EnergyError(f"Energies must be real numbers: {exc}") from exc
    # np.array(..., dtype=float64) would drop a complex part and accept
    # numeric strings and bools; only integer and float input is an energy.
    if raw.dtype.kind not in "iuf":
        raise EnergyError(f"Energies must be real numbers, got dtype {raw.dtype}")
    e_kev = np.atleast_1d(np.array(raw, dtype=np.float64))
    if e_kev.ndim != 1:
        raise EnergyError(f"Energies must be a 1-D array, got shape {e_kev.shape}")
    if e_kev.size == 0:
        raise EnergyError("Energies must not be empty")
    if not np.isfinite(e_kev).all():
        raise EnergyError("Energies must be finite (no NaN or inf)")
    e_ev = e_kev * KEV_TO_EV
    out = (e_ev < lo_ev) | (e_ev > hi_ev)
    if out.any():
        raise EnergyError(
            "Energy outside the tabulated range",
            energy=float(e_kev[out][0]),
            valid_range=f"{lo_ev / KEV_TO_EV:g}-{hi_ev / KEV_TO_EV:g} keV",
        )
    return e_kev, e_ev


def _merge_points_ev(
    merge_points_kev: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """The merge points converted keV -> eV (the one unit conversion)."""
    if merge_points_kev is None:
        return None
    try:
        low, high = merge_points_kev
        return float(low) * KEV_TO_EV, float(high) * KEV_TO_EV
    except (TypeError, ValueError) as exc:
        raise NearEdgeDataError(
            f"merge_points_kev must be two energies in keV: {exc}"
        ) from exc


def _reported_merge_kev(
    given_kev: tuple[float, float] | None,
    given_ev: tuple[float, float] | None,
    used_ev: tuple[float, float],
) -> tuple[float, float]:
    """Merge points in keV as used: the caller's value unless snapping moved it.

    Keeping the caller's own value avoids ``(k / 1000) * 1000 != k`` noise.
    """
    low, high = (ev / KEV_TO_EV for ev in used_ev)
    if given_kev is not None and given_ev is not None:
        if given_ev[0] == used_ev[0]:
            low = float(given_kev[0])
        if given_ev[1] == used_ev[1]:
            high = float(given_kev[1])
    return low, high


def _check_refine_arguments(
    tolerance: float | None, max_passes: int, max_points: int
) -> None:
    """Raise :class:`ValidationError` for unusable refinement arguments."""
    if tolerance is not None and (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, int | float | np.number)
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValidationError(
            "refine_tolerance must be a finite positive number",
            "refine_tolerance",
            tolerance,
        )
    _positive_int(max_passes, "refine_max_passes")
    _positive_int(max_points, "refine_max_points")


def kk_calculate_real(
    formula: str,
    *,
    energies_kev: ArrayLike | None = None,
    chunk_size: int | None = None,
    nearedge: NearEdgeSpectrum | None = None,
    input_type: InputType | None = None,
    merge_points_kev: tuple[float, float] | None = None,
    refine_tolerance: float | None = None,
    refine_max_passes: int = 50,
    refine_max_points: int = 100_000,
) -> KKResult:
    """Real part f1 of the scattering factor from the tabulated f2.

    Evaluates the Kramers-Kronig transform of the tabulated imaginary
    scattering factor of ``formula`` (Henke tables, 10 eV - 500 keV) and the
    f2 spectrum itself.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``"SiO2"`` or ``"Ca5(PO4)3OH"``.
    energies_kev : array_like, shape (m,), optional
        Photon energies in keV, 0.01 to 500, finite, any order, repeats
        allowed. Default: the knots of the tabulated spectrum (union of the
        elemental grids) inside 0.03-30 keV; the number of points clipped
        away is logged at INFO.
    chunk_size : int, optional
        Energies per block of the transform, bounding memory; the result does
        not depend on it beyond rounding. Also used for the refinement
        midpoints. Default: about 32 MiB per float64 temporary.
    nearedge : NearEdgeSpectrum, optional
        Measured near-edge absorption data (energies in eV, see
        :func:`~xraylabtool.kk.nearedge.load_nearedge`). It is converted to
        the f2 scale, normalised to the tabulated f2 at the merge points and
        replaces the tabulated spectrum between them.
    input_type : {"nexafs", "photoabsorption", "xanes", "beta", "asf"}, optional
        Kind of ``nearedge`` values; required with ``nearedge`` (see
        :func:`~xraylabtool.kk.nearedge.to_asf_scale`).
    merge_points_kev : tuple of float, optional
        ``(low, high)`` merge energies in keV, inside the near-edge data and
        the tabulated range. Default: the whole overlap of the two ranges.
        Only with ``nearedge``.
    refine_tolerance : float, optional
        Absolute interpolation error (electrons per formula unit), finite and
        positive. If given, midpoints are inserted wherever linear
        interpolation of f1 or f2 between neighbouring energies is off by
        more than this (see :func:`~xraylabtool.kk.refine.refine_spectrum`),
        after any near-edge merge. The output is then sorted and
        de-duplicated: the unique requested energies (keV values that are
        equal in eV count once) plus the added midpoints. Without it, the
        energies keep the caller's order.
    refine_max_passes : int, default 50
        Maximum bisection passes of the refinement; at least 1.
    refine_max_points : int, default 100000
        Maximum number of points the refinement may add; at least 1.

    Returns
    -------
    KKResult
        ``f1`` and ``f2`` in electrons per formula unit at ``energy_kev``.
        With near-edge data, ``merge_points_kev`` holds the merge energies
        used and ``nearedge_scaled_kev_f2`` the normalised near-edge data
        ``(energy_kev, f2)`` on its own grid. The default output grid is then
        the merged spectrum's knots. ``n_refined_points`` counts the points
        added by refinement (0 without ``refine_tolerance``).

    Raises
    ------
    FormulaError
        If the formula is not a string, is empty or malformed, contains an
        element outside Z = 1..92, or has a non-positive composition.
    EnergyError
        If ``energies_kev`` is empty, not 1-D, not finite, or outside
        0.01-500 keV (nothing is extrapolated); or, without ``energies_kev``,
        if no knot of the (merged) spectrum lies inside 0.03-30 keV, so the
        default grid would be empty.
    ValidationError
        If ``chunk_size``, ``refine_max_passes`` or ``refine_max_points`` is
        not a positive integer, ``refine_tolerance`` is not a finite positive
        number, or ``input_type`` is unknown.
    TypeError
        If ``nearedge`` is given without ``input_type``, is not a
        :class:`~xraylabtool.kk.nearedge.NearEdgeSpectrum`, or
        ``input_type``/``merge_points_kev`` is given without ``nearedge``.
    NearEdgeDataError
        If the near-edge data cannot be merged (no overlap with the
        tabulated range, bad merge points, flat end points, non-finite
        normalised values); see :func:`~xraylabtool.kk.nearedge.merge_spectra`.

    Notes
    -----
    Core energies may lie anywhere in 10 eV - 500 keV; only the conversion to
    an :class:`~xraylabtool.calculators.XRayResult` is restricted to
    0.03-30 keV. The transform is numerically exact for any energy but
    f1 below ~20 eV for heavy elements is good to about 1e-4 absolute (see
    :func:`~xraylabtool.kk.transform.kk_piecewise_poly`). Within a few eV of
    10 eV or 500 keV, f1 is dominated by the truncation of the integral at the
    data range, not by the data.

    Near-edge values that are negative after normalisation are kept and
    logged at WARNING, never clipped.

    Refinement never drops a requested energy; it logs the points added per
    pass (DEBUG) and in total (INFO), and a WARNING if it stops on
    ``refine_max_passes`` or ``refine_max_points`` before the tolerance is
    met everywhere.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7)-(10): f1 from the
    piecewise-polynomial f2. The keV interface, validation and default
    window are this package's additions; the call sequence follows upstream
    ``kkcalc.kk.kk_calculate_real`` (near-edge merge: see
    :func:`~xraylabtool.kk.nearedge.merge_spectra`).

    Examples
    --------
    >>> from xraylabtool.kk import NearEdgeSpectrum, kk_calculate_real
    >>> result = kk_calculate_real("Si", energies_kev=[1.0, 2.0])
    >>> result.energy_kev.tolist()
    [1.0, 2.0]
    >>> bool((result.f1 > 0).all() and (result.f2 > 0).all())
    True
    >>> result.backend in ("numpy", "jax")
    True
    >>> edge = NearEdgeSpectrum(
    ...     [280.0, 290.0, 300.0, 310.0, 320.0], [0.05, 0.1, 0.3, 0.45, 0.5]
    ... )
    >>> merged = kk_calculate_real(
    ...     "C", nearedge=edge, input_type="nexafs", merge_points_kev=(0.28, 0.32)
    ... )
    >>> merged.merge_points_kev
    (0.28, 0.32)
    >>> 0.29 in merged.energy_kev
    True
    >>> refined = kk_calculate_real(
    ...     "Si", energies_kev=[5.0, 0.1, 5.0], refine_tolerance=0.05
    ... )
    >>> refined.n_refined_points > 0 and bool((refined.energy_kev[1:] > refined.energy_kev[:-1]).all())
    True
    """
    if not isinstance(formula, str):
        raise FormulaError(f"Formula must be a string, got {type(formula).__name__}")
    if chunk_size is not None:
        _positive_int(chunk_size, "chunk_size")

    _check_refine_arguments(refine_tolerance, refine_max_passes, refine_max_points)

    if nearedge is None:
        if input_type is not None or merge_points_kev is not None:
            raise TypeError("input_type and merge_points_kev require nearedge")
    elif not isinstance(nearedge, NearEdgeSpectrum):
        raise TypeError(
            f"nearedge must be a NearEdgeSpectrum, got {type(nearedge).__name__}"
        )
    elif input_type is None:
        raise TypeError(
            "input_type is required with nearedge (one of nexafs, photoabsorption, "
            "xanes, beta, asf)"
        )

    symbols, counts = parse_formula(formula)
    seg_ev, coeffs = compound_spectrum(symbols, counts)
    used_merge_kev: tuple[float, float] | None = None
    scaled_kev_f2: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    # input_type is never None here with a nearedge (checked above); the test narrows it.
    if nearedge is not None and input_type is not None:
        merge_points_ev = _merge_points_ev(merge_points_kev)
        merged = _merge(
            to_asf_scale(nearedge, input_type), seg_ev, coeffs, merge_points_ev
        )
        seg_ev, coeffs = merged.seg_ev, merged.coeffs
        used_merge_kev = _reported_merge_kev(
            merge_points_kev, merge_points_ev, merged.merge_points_ev
        )
        scaled_kev_f2 = (nearedge.energy_ev / KEV_TO_EV, merged.scaled)
    db = load_asf()
    rel_corr = relativistic_correction(
        [db.symbols.index(s) + 1 for s in symbols], counts
    )

    if energies_kev is None:
        keep = (seg_ev >= _DEFAULT_MIN_KEV * KEV_TO_EV) & (
            seg_ev <= _DEFAULT_MAX_KEV * KEV_TO_EV
        )
        energy_ev = seg_ev[keep]
        energy_kev = energy_ev / KEV_TO_EV
        logger.info(
            "kk_calculate_real(%s): default grid clipped to %g-%g keV, "
            "kept %d of %d tabulated points (%d below, %d above)",
            formula,
            _DEFAULT_MIN_KEV,
            _DEFAULT_MAX_KEV,
            energy_ev.size,
            seg_ev.size,
            int((seg_ev < _DEFAULT_MIN_KEV * KEV_TO_EV).sum()),
            int((seg_ev > _DEFAULT_MAX_KEV * KEV_TO_EV).sum()),
        )
        if energy_ev.size == 0:
            raise EnergyError(
                f"No merged or tabulated knots inside {_DEFAULT_MIN_KEV:g}-"
                f"{_DEFAULT_MAX_KEV:g} keV for {formula}, so the default output "
                "grid is empty; pass explicit energies (energies_kev, or "
                "--energies on the command line)",
                valid_range=f"{_DEFAULT_MIN_KEV:g}-{_DEFAULT_MAX_KEV:g} keV",
            )
    else:
        energy_kev, energy_ev = _validated_energies_ev(
            energies_kev, float(seg_ev[0]), float(seg_ev[-1])
        )

    n_refined = 0
    if refine_tolerance is not None:
        # refine_spectrum requires a strictly increasing grid of unique eV
        # values; keV values equal in eV count once. Keeping the eV array (not
        # recomputing it from keV) leaves default-grid knots bit-exact, and
        # evaluate_f2 is right-continuous at edges.
        energy_ev, first = np.unique(energy_ev, return_index=True)
        energy_kev = energy_kev[first]
    f1 = kk_real_spectrum(energy_ev, seg_ev, coeffs, rel_corr, chunk_size)
    f2 = evaluate_f2(energy_ev, seg_ev, coeffs)
    if refine_tolerance is not None:
        refined = refine_spectrum(
            energy_ev,
            f1,
            f2,
            seg_ev,
            coeffs,
            rel_corr,
            tolerance=float(refine_tolerance),
            max_passes=refine_max_passes,
            max_points=refine_max_points,
            chunk_size=chunk_size,
        )
        # Requested energies keep their exact keV value; only midpoints convert.
        original = np.searchsorted(refined.energy_ev, energy_ev)
        energy_ev, f1, f2 = refined.energy_ev, refined.f1, refined.f2
        n_refined = refined.n_added
        energy_kev_out = energy_ev / KEV_TO_EV
        energy_kev_out[original] = energy_kev
        energy_kev = energy_kev_out
    return KKResult(
        formula=formula,
        energy_kev=energy_kev,
        f1=f1,
        f2=f2,
        relativistic_correction=rel_corr,
        merge_points_kev=used_merge_kev,
        n_refined_points=n_refined,
        backend="jax" if ops.is_jax() else "numpy",
        nearedge_scaled_kev_f2=scaled_kev_f2,
    )
