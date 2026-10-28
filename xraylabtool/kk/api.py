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

The numerical part is a device pipeline (:mod:`xraylabtool.kk._pipeline`,
:mod:`xraylabtool.kk.refine`): one upload of all inputs, one status fetch
after assembly and merge, one per refinement pass, one final download.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, cast

import jax
import numpy as np

# Imported at runtime so ``typing.get_type_hints`` resolves the annotations.
from numpy.typing import ArrayLike, NDArray

from xraylabtool.constants import KEV_TO_EV, TABULATED_MAX_KEV, TABULATED_MIN_KEV
from xraylabtool.exceptions import (
    EnergyError,
    FormulaError,
    NearEdgeDataError,
    ValidationError,
)
from xraylabtool.kk import _device
from xraylabtool.kk._pipeline import (
    AssembleStatus,
    NearEdgeInputs,
    PipelineInputs,
    _assemble,
    _evaluate,
)
from xraylabtool.kk._shapes import bucket, bucket_pow2, energy_blocks
from xraylabtool.kk.database import AsfDatabase, load_asf
from xraylabtool.kk.nearedge import (
    InputType,
    NearEdgeSpectrum,
    _asf_scale_arguments,
    _rescaled,
    _resolve_merge_points,
    raise_or_log,
    value_exponent,
)
from xraylabtool.kk.refine import (
    _INT_MAX,
    _Knobs,
    _outcome,
    _positive_int,
    _refine_device,
)
from xraylabtool.kk.result import KKResult
from xraylabtool.kk.spectrum import _atomic_numbers, _element_arrays, _real_float64
from xraylabtool.kk.transform import relativistic_correction
from xraylabtool.logging_utils import get_logger
from xraylabtool.utils import parse_formula

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger("kk")

# Default output window (keV): the package's tabulated range, 0.03-30 keV.
_DEFAULT_MIN_KEV = TABULATED_MIN_KEV
_DEFAULT_MAX_KEV = TABULATED_MAX_KEV
_WINDOW_EV = (_DEFAULT_MIN_KEV * KEV_TO_EV, _DEFAULT_MAX_KEV * KEV_TO_EV)

_runtime_logged = threading.Event()
_runtime_lock = threading.Lock()


def _log_runtime_once() -> None:
    """Log platform, device, jax version and x64 once per process (INFO)."""
    with _runtime_lock:
        if _runtime_logged.is_set():
            return
        _runtime_logged.set()
    device = jax.devices()[0]
    logger.info(
        "kk runtime: jax %s, platform %s, device %s, x64 %s",
        jax.__version__,
        device.platform,
        device,
        jax.config.jax_enable_x64,
    )


def _validated_energies_ev(
    energies_kev: ArrayLike, lo_ev: float, hi_ev: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(energies_kev, energies_ev)`` after validation (the one unit conversion)."""
    if isinstance(energies_kev, jax.Array):
        energies_kev = _device.fetch(energies_kev)  # counted input-preparation fetch
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
        low, high = _real_float64("merge_points_kev", merge_points_kev)
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
        or not isinstance(tolerance, int | float | np.integer | np.floating)
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


@dataclass(frozen=True)
class _NearEdgeCall:
    """Resolved near-edge arguments of one call (host side)."""

    spectrum: NearEdgeSpectrum
    kind: str
    density: float | None
    formula_mass: float | None
    low_ev: float
    high_ev: float
    given_ev: tuple[float, float] | None  # the merge points as the caller gave them


@dataclass(frozen=True)
class _Plan:
    """Everything decided on the host before anything is uploaded."""

    z_list: list[int]
    counts: Sequence[float]
    rel_corr: float
    n_segments: int  # of the compound spectrum, exact (from the host tables)
    capacity: int  # compound spectrum
    final_capacity: int  # after the near-edge merge
    near: _NearEdgeCall | None
    explicit: bool
    energy_kev: NDArray[np.float64]  # explicit energies (first kept under refinement)
    energy_ev: NDArray[np.float64]
    deferred: EnergyError | None  # bad energies, raised once the merge has been checked


def _near_edge_call(
    nearedge: NearEdgeSpectrum,
    input_type: InputType,
    merge_points_kev: tuple[float, float] | None,
    tab_range: tuple[float, float],
) -> _NearEdgeCall:
    """Validate the near-edge arguments (merge points, then input type, then overlap)."""
    given_ev = _merge_points_ev(merge_points_kev)
    kind, density, formula_mass = _asf_scale_arguments(input_type, None, None)
    low, high = _resolve_merge_points(nearedge, *tab_range, given_ev)
    return _NearEdgeCall(nearedge, kind, density, formula_mass, low, high, given_ev)


def _explicit_energies(
    energies_kev: ArrayLike,
    tab_range: tuple[float, float],
    near: _NearEdgeCall | None,
    refine: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], EnergyError | None]:
    """Validated ``(energy_kev, energy_ev, deferred_error)`` of explicit energies.

    With near-edge data a bad grid does not raise yet: a data error in the
    merge outranks it, so
    the error is returned and a one-point stand-in grid is used. Under
    refinement the grid is made unique in eV, keeping the first keV value of
    each group, because ``refine_spectrum`` needs a strictly increasing grid.
    """
    try:
        energy_kev, energy_ev = _validated_energies_ev(energies_kev, *tab_range)
    except EnergyError as exc:
        if near is None:
            raise
        return np.array([tab_range[0] / KEV_TO_EV]), np.array([tab_range[0]]), exc
    if refine:
        energy_ev, first = np.unique(energy_ev, return_index=True)
        energy_kev = energy_kev[first]
    return energy_kev, energy_ev, None


def _compound_segments(db: AsfDatabase, z_list: list[int]) -> int:
    """Exact segment count of the compound spectrum: distinct elemental knots minus one.

    Counted on the host from the tables, so the capacity below is the
    smallest ladder value that cannot overflow (``compound_capacity`` is the
    looser sum bound, up to twice as large for several elements).
    """
    return int(np.unique(np.concatenate([db.energy_ev(z) for z in z_list])).size) - 1


def _compound_capacity(n_segments: int) -> int:
    """Segment capacity of the compound spectrum for ``n_segments`` true segments."""
    return bucket(n_segments)


def _plan(
    formula: str,
    energies_kev: ArrayLike | None,
    near_args: tuple[
        NearEdgeSpectrum | None, InputType | None, tuple[float, float] | None
    ],
    refine: bool,
) -> _Plan:
    """Host side of a call: formula, near-edge arguments, energies, capacities."""
    symbols, counts = parse_formula(formula)
    z_list = _atomic_numbers(symbols, counts)
    db = load_asf()
    # Every element shares the tabulated range, which the merge keeps.
    tab_range = (
        min(float(db.energy_ev(z)[0]) for z in z_list),
        max(float(db.energy_ev(z)[-1]) for z in z_list),
    )
    n_segments = _compound_segments(db, z_list)
    capacity = _compound_capacity(n_segments)
    nearedge, input_type, merge_points_kev = near_args
    near = (
        None
        if nearedge is None or input_type is None
        else _near_edge_call(nearedge, input_type, merge_points_kev, tab_range)
    )
    explicit = energies_kev is not None
    energy_kev = energy_ev = np.empty(0)
    deferred = None
    if energies_kev is not None:
        energy_kev, energy_ev, deferred = _explicit_energies(
            energies_kev, tab_range, near, refine
        )
    final = (
        capacity
        if near is None
        else bucket(n_segments + near.spectrum.energy_ev.size + 2)
    )
    return _Plan(
        z_list,
        counts,
        relativistic_correction(z_list, counts),
        n_segments,
        capacity,
        final,
        near,
        explicit,
        energy_kev,
        energy_ev,
        deferred,
    )


def _host_inputs(
    plan: _Plan,
    block_shape: tuple[int, int],
    refine: tuple[float, int] | None,
) -> PipelineInputs:
    """NumPy-leaved :class:`PipelineInputs`: one pytree for one upload.

    Explicit energies are padded to ``block * n_blocks`` with their first
    entry (an in-range energy), near-edge arrays to a bucketed length with
    their last. ``refine`` is ``(tolerance, max_points)``.
    """
    z_pad, n_pad = _element_arrays(plan.z_list, plan.counts)
    grid = n_grid = None
    if plan.explicit:
        e = plan.energy_ev
        grid = np.concatenate(
            [e, np.full(block_shape[0] * block_shape[1] - e.size, e[0])]
        )
        n_grid = np.int32(e.size)
    ne = None
    if plan.near is not None:
        near = plan.near
        energy = near.spectrum.energy_ev
        values = _rescaled(near.spectrum.values, value_exponent(near.spectrum.values))
        pad = bucket_pow2(energy.size) - energy.size
        ne = NearEdgeInputs(
            np.concatenate([energy, np.full(pad, energy[-1])]),
            np.concatenate([values, np.full(pad, values[-1])]),
            np.int32(energy.size),
            np.float64(near.low_ev),
            np.float64(near.high_ev),
            None if near.density is None else np.float64(near.density),
            None if near.formula_mass is None else np.float64(near.formula_mass),
        )
    return PipelineInputs(
        z_pad,
        n_pad,
        np.int32(len(plan.z_list)),
        np.float64(plan.rel_corr),
        grid,
        n_grid,
        ne,
        None if refine is None else np.float64(refine[0]),
        None if refine is None else np.int64(min(refine[1], _INT_MAX)),
    )


def _checked_merge(
    plan: _Plan, status: AssembleStatus, merge_points_kev: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Raise what the assembly status reports; return the merge points used (keV).

    Order of the checks: capacity, near-edge conversion overflow, merge data
    errors, then the energy error deferred by :func:`_explicit_energies`.
    """
    _device.check_capacity(int(status.n_compound), plan.capacity)
    _device.check_capacity(int(status.n_segments), plan.final_capacity)
    if status.converted_nonfinite:
        raise NearEdgeDataError("Near-edge energies and values must be finite")
    used = None
    if (
        plan.near is not None
        and status.merge is not None
        and status.used_points_ev is not None
    ):
        raise_or_log(
            status.merge,
            plan.near.low_ev,
            plan.near.high_ev,
            value_exponent(plan.near.spectrum.values),
        )
        used = _reported_merge_kev(
            merge_points_kev,
            plan.near.given_ev,
            (float(status.used_points_ev[0]), float(status.used_points_ev[1])),
        )
    if plan.deferred is not None:
        raise plan.deferred
    return used


def _default_grid_size(formula: str, status: AssembleStatus) -> int:
    """Size of the default grid from the status; logs the clipping, raises if empty."""
    size = int(status.n_grid)
    logger.info(
        "kk_calculate_real(%s): default grid clipped to %g-%g keV, "
        "kept %d of %d tabulated points (%d below, %d above)",
        formula,
        _DEFAULT_MIN_KEV,
        _DEFAULT_MAX_KEV,
        size,
        int(status.n_segments) + 1,
        int(status.n_below),
        int(status.n_above),
    )
    if size == 0:
        raise EnergyError(
            f"No merged or tabulated knots inside {_DEFAULT_MIN_KEV:g}-"
            f"{_DEFAULT_MAX_KEV:g} keV for {formula}, so the default output "
            "grid is empty; pass explicit energies (energies_kev, or "
            "--energies on the command line)",
            valid_range=f"{_DEFAULT_MIN_KEV:g}-{_DEFAULT_MAX_KEV:g} keV",
        )
    return size


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
        away is logged at INFO. A ``jax.Array`` is validated on the host, so
        it costs one extra counted fetch (input preparation) before the
        calculation's own single upload.
    chunk_size : int, optional
        Energies per block of the transform, bounding memory; the result does
        not depend on it beyond rounding. Also used for the refinement
        midpoints. Default: platform dependent, at most about 2 MiB (CPU) or
        32 MiB (accelerator) per float64 temporary.
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
    >>> result.backend
    'jax'
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
    _device.ensure_x64()
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

    plan = _plan(
        formula,
        energies_kev,
        (nearedge, input_type, merge_points_kev),
        refine_tolerance is not None,
    )
    _log_runtime_once()
    refine = (
        None
        if refine_tolerance is None
        else (float(refine_tolerance), refine_max_points)
    )
    block_shape = (
        energy_blocks(chunk_size, plan.final_capacity, plan.energy_ev.size)
        if plan.explicit
        else (0, 0)
    )
    tables = load_asf().device_tables()
    dev = _device.upload(_host_inputs(plan, block_shape, refine))
    assembled = _assemble(
        tables,
        # neither the output grid nor the refinement knobs enter assembly;
        # leaving them in the pytree would recompile _assemble per
        # energy-length bucket and per refined/unrefined mode
        dev._replace(energies_ev=None, tolerance=None, max_points=None),
        capacity=plan.capacity,
        merge_capacity=plan.final_capacity,
        input_type=None if plan.near is None else plan.near.kind,
        window_ev=_WINDOW_EV,
    )
    status = _device.fetch_status(assembled.status)
    used_merge_kev = _checked_merge(plan, status, merge_points_kev)

    m = plan.energy_ev.size
    if not plan.explicit:
        m = _default_grid_size(formula, status)
        block_shape = energy_blocks(chunk_size, assembled.spectrum.coeffs.shape[0], m)
    grid_ev, f1_d, f2_d = _evaluate(
        assembled.spectrum,
        dev.energies_ev,
        assembled.n_grid,
        dev.rel_corr,
        block=block_shape[0],
        n_blocks=block_shape[1],
        window_ev=_WINDOW_EV,
    )

    n_refined = 0
    exhausted = False
    if refine is None:
        grid_h, f1_h, f2_h, scaled = _device.fetch(
            (None if plan.explicit else grid_ev, f1_d, f2_d, assembled.scaled_f2)
        )
        f1, f2 = np.asarray(f1_h)[:m], np.asarray(f2_h)[:m]
        energy_kev = (
            plan.energy_kev if plan.explicit else np.asarray(grid_h)[:m] / KEV_TO_EV
        )
    else:
        arrays, info = _refine_device(
            grid_ev,
            f1_d,
            f2_d,
            assembled.spectrum,
            dev.rel_corr,
            _Knobs(
                cast("jax.Array", dev.tolerance),
                cast("jax.Array", dev.max_points),
                assembled.n_grid,
            ),
            m=m,
            max_passes=refine_max_passes,
            max_points=refine_max_points,
            chunk_size=chunk_size,
        )
        arrays_h, scaled = _device.fetch((arrays, assembled.scaled_f2))
        refined = _outcome(arrays_h, info, refine[0])
        f1, f2, n_refined = refined.f1, refined.f2, refined.n_added
        exhausted = refined.exhausted
        energy_kev = refined.energy_ev / KEV_TO_EV
        if plan.explicit:
            # Requested energies keep their exact keV value; only midpoints convert.
            energy_kev[np.searchsorted(refined.energy_ev, plan.energy_ev)] = (
                plan.energy_kev
            )
    return KKResult(
        formula=formula,
        energy_kev=energy_kev,
        f1=f1,
        f2=f2,
        relativistic_correction=plan.rel_corr,
        merge_points_kev=used_merge_kev,
        n_refined_points=n_refined,
        refine_exhausted=exhausted,
        backend="jax",
        nearedge_scaled_kev_f2=None
        if plan.near is None
        else (
            plan.near.spectrum.energy_ev / KEV_TO_EV,
            np.asarray(scaled, dtype=np.float64)[: plan.near.spectrum.energy_ev.size],
        ),
    )


def warm_up(
    formulas: Sequence[str] = ("SiO2",),
    refine_tolerance: float | None = 0.01,
) -> None:
    """Compile the programs a default call of each formula touches.

    Runs the default-grid calculation of every formula, and the refined one
    when ``refine_tolerance`` is given (the tolerance decides which active
    buckets the refinement visits), so later calls with the same buckets
    compile nothing. Idempotent; returns after the results are on the host,
    i.e. after everything is compiled and executed. The GUI does not call it: it
    warms the programs of its own Compute click on an idle worker
    (``gui.services.warm_up_kk``). The CLI never warms up, so a one-shot call
    never pays for formulas it was not asked for.

    Parameters
    ----------
    formulas : sequence of str, default ("SiO2",)
        Chemical formulas whose default calculations are compiled.
    refine_tolerance : float or None, default 0.01
        Also compile a refined run at this tolerance; ``None`` skips it.

    Raises
    ------
    FormulaError
        If a formula is invalid (as for :func:`kk_calculate_real`).

    References
    ----------
    Design spec §10 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.kk import warm_up
    >>> warm_up(("Si",), refine_tolerance=None)
    """
    if isinstance(formulas, str):
        formulas = (formulas,)
    for formula in formulas:
        kk_calculate_real(formula)
        if refine_tolerance is not None:
            kk_calculate_real(formula, refine_tolerance=refine_tolerance)
