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

from typing import TYPE_CHECKING

import numpy as np

from xraylabtool.backend.array_ops import ops
from xraylabtool.exceptions import EnergyError, ValidationError
from xraylabtool.kk.database import load_asf
from xraylabtool.kk.result import KKResult
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction
from xraylabtool.logging_utils import get_logger
from xraylabtool.utils import parse_formula

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

logger = get_logger("kk")

# Default output window (keV): the package's tabulated range, 0.03-30 keV.
_DEFAULT_MIN_KEV = 0.03
_DEFAULT_MAX_KEV = 30.0
_EV_PER_KEV = 1000.0


def _validated_energies_ev(
    energies_kev: ArrayLike, lo_ev: float, hi_ev: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(energies_kev, energies_ev)`` after validation (the one unit conversion)."""
    try:
        e_kev = np.array(energies_kev, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EnergyError(f"Energies must be real numbers: {exc}") from exc
    e_kev = np.atleast_1d(e_kev)
    if e_kev.ndim != 1:
        raise EnergyError(f"Energies must be a 1-D array, got shape {e_kev.shape}")
    if e_kev.size == 0:
        raise EnergyError("Energies must not be empty")
    if not np.isfinite(e_kev).all():
        raise EnergyError("Energies must be finite (no NaN or inf)")
    e_ev = e_kev * _EV_PER_KEV
    out = (e_ev < lo_ev) | (e_ev > hi_ev)
    if out.any():
        raise EnergyError(
            "Energy outside the tabulated range",
            energy=float(e_kev[out][0]),
            valid_range=f"{lo_ev / _EV_PER_KEV:g}-{hi_ev / _EV_PER_KEV:g} keV",
        )
    return e_kev, e_ev


def kk_calculate_real(
    formula: str,
    *,
    energies_kev: ArrayLike | None = None,
    chunk_size: int | None = None,
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
        not depend on it beyond rounding. Default: about 32 MiB per float64
        temporary.

    Returns
    -------
    KKResult
        ``f1`` and ``f2`` in electrons per formula unit at ``energy_kev``.

    Raises
    ------
    FormulaError
        If the formula is empty or malformed, contains an element outside
        Z = 1..92, or has a non-positive composition.
    EnergyError
        If ``energies_kev`` is empty, not 1-D, not finite, or outside
        0.01-500 keV (nothing is extrapolated).
    ValidationError
        If ``chunk_size`` is not a positive integer.

    Notes
    -----
    Core energies may lie anywhere in 10 eV - 500 keV; only the conversion to
    an :class:`~xraylabtool.calculators.XRayResult` is restricted to
    0.03-30 keV. The transform is numerically exact for any energy but
    f1 below ~20 eV for heavy elements is good to about 1e-4 absolute (see
    :func:`~xraylabtool.kk.transform.kk_piecewise_poly`).

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014), Eqs. (7)-(10): f1 from the
    piecewise-polynomial f2. The keV interface, validation and default
    window are this package's additions; the call sequence follows upstream
    ``kkcalc.kk.kk_calculate_real`` without near-edge data.

    Examples
    --------
    >>> result = kk_calculate_real("Si", energies_kev=[1.0, 2.0])
    >>> result.energy_kev.tolist()
    [1.0, 2.0]
    >>> bool((result.f1 > 0).all() and (result.f2 > 0).all())
    True
    >>> result.backend in ("numpy", "jax")
    True
    """
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int | np.integer)
        or chunk_size < 1
    ):
        raise ValidationError(
            "chunk_size must be a positive integer", "chunk_size", chunk_size
        )

    symbols, counts = parse_formula(formula)
    seg_ev, coeffs = compound_spectrum(symbols, counts)
    db = load_asf()
    rel_corr = relativistic_correction(
        [db.symbols.index(s) + 1 for s in symbols], counts
    )

    if energies_kev is None:
        keep = (seg_ev >= _DEFAULT_MIN_KEV * _EV_PER_KEV) & (
            seg_ev <= _DEFAULT_MAX_KEV * _EV_PER_KEV
        )
        energy_ev = seg_ev[keep]
        energy_kev = energy_ev / _EV_PER_KEV
        logger.info(
            "kk_calculate_real(%s): default grid clipped to %g-%g keV, "
            "kept %d of %d tabulated points (%d below, %d above)",
            formula,
            _DEFAULT_MIN_KEV,
            _DEFAULT_MAX_KEV,
            energy_ev.size,
            seg_ev.size,
            int((seg_ev < _DEFAULT_MIN_KEV * _EV_PER_KEV).sum()),
            int((seg_ev > _DEFAULT_MAX_KEV * _EV_PER_KEV).sum()),
        )
    else:
        energy_kev, energy_ev = _validated_energies_ev(
            energies_kev, float(seg_ev[0]), float(seg_ev[-1])
        )

    f1 = kk_real_spectrum(energy_ev, seg_ev, coeffs, rel_corr, chunk_size)
    f2 = evaluate_f2(energy_ev, seg_ev, coeffs)
    return KKResult(
        formula=formula,
        energy_kev=energy_kev,
        f1=f1,
        f2=f2,
        relativistic_correction=rel_corr,
        merge_points_kev=None,
        n_refined_points=0,
        backend="jax" if ops.is_jax() else "numpy",
    )
