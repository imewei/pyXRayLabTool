"""Near-edge absorption data: loading, conversion and merging.

Altered from the original: a re-implementation of ``convert_data``,
``convert_NEXAFS_to_ASF``, ``convert_Beta_to_ASF`` and ``merge_spectra`` from
kkcalc/data.py of the Kramers-Kronig Calculator (kkcalc,
https://github.com/benajamin/kkcalc, commit
f331ca31a654af946f7d550a22cf4e9c11fa8afc), Copyright (c) 2013 Benjamin Watts,
Daniel J. Lauk, zlib/libpng licence (full text in
``xraylabtool/data/kkcalc/LICENSE-kkcalc.txt``; see also ``NOTICE``). The
method is described in B. Watts, Opt. Express 22(19), 23628-23639 (2014),
doi:10.1364/OE.22.023628.

The conversion constants come from :mod:`xraylabtool.constants`, the
"add background" and "fix distortions" options of upstream are not ported,
and every input is validated (a malformed spectrum, merge points outside the
data, flat end points or non-finite values raise
:class:`~xraylabtool.exceptions.NearEdgeDataError` instead of producing
``inf`` or silently wrong coefficients).

Energies are in eV throughout this module (the tabulated coefficients are in
eV). Loading, validation and merge-point resolution are host-side NumPy; the
numerical part (:func:`to_asf_scale_device`, :func:`merge_device`) is pure
``jax.numpy`` and stays on the device, with :func:`raise_or_log` turning its
status pytree into the data errors and the D6 warning on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from xraylabtool import constants
from xraylabtool.exceptions import DataFileError, NearEdgeDataError, ValidationError
from xraylabtool.kk import _device
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk._shapes import bucket, bucket_pow2
from xraylabtool.kk.spectrum import _real_float64, _union_knots, evaluate_f2_device
from xraylabtool.logging_utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger("kk")

InputType = Literal["nexafs", "photoabsorption", "xanes", "beta", "asf"]

_NEXAFS_TYPES = ("nexafs", "photoabsorption", "xanes")
_INPUT_TYPES = (*_NEXAFS_TYPES, "beta", "asf")
_ENERGY_UNITS = {"eV": 1.0, "keV": constants.KEV_TO_EV}
_COMMENT_PREFIXES = ("#", "%", ";")

# hc in eV m, from the package constants.
_HC_EV_M = constants.SPEED_OF_LIGHT * constants.PLANCK / constants.ELEMENT_CHARGE

# Smallest usable relative step between the two end-point values: the affine
# matching divides by it, so anything near rounding level would amplify noise
# by 1e6 or more. Real pre-edge to post-edge steps are percent to orders of
# magnitude.
_MIN_RELATIVE_STEP = 1e-6

# Data whose largest magnitude is below this are rescaled by a power of two
# before they reach the device: XLA flushes subnormals (inputs and results),
# which would turn the tiny differences of such data into zeros.
_TINY_PEAK = 2.0**-500


def value_exponent(values: NDArray[np.float64]) -> int:
    """Power-of-two exponent that brings tiny near-edge ``values`` to order one.

    ``0`` (no rescaling, bit-identical results) unless the largest finite
    magnitude is below ``2**-500``. Multiplying by ``2**exponent`` is exact,
    and every later step (conversion, the affine normalisation of the merge)
    is homogeneous in the values, so the merged spectrum does not depend on it.
    """
    finite = np.abs(values[np.isfinite(values)])
    peak = float(finite.max()) if finite.size else 0.0
    return -int(np.frexp(peak)[1]) if 0.0 < peak < _TINY_PEAK else 0


def _rescaled(values: NDArray[np.float64], exponent: int) -> NDArray[np.float64]:
    """``values * 2**exponent`` (exact); ``values`` itself when ``exponent`` is 0."""
    return np.ldexp(values, exponent) if exponent else values


# A merge point within this relative distance of a range end is that end: the
# keV API multiplies by 1000, and (k / 1000) * 1000 != k for some integers k.
_SNAP_RTOL = 1e-12


def _fmt_range(lo: float, hi: float) -> str:
    return f"{lo:.12g}-{hi:.12g} eV"


def _snap(value: float, bound: float) -> float:
    """``bound`` if ``value`` is within rounding of it, else ``value``."""
    return (
        bound if math.isclose(value, bound, rel_tol=_SNAP_RTOL, abs_tol=0.0) else value
    )


def _validate_arrays(
    energy_ev: object, values: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validated float64 copies of an energy and value array."""
    arrays = []
    for name, raw in (("energy_ev", energy_ev), ("values", values)):
        try:
            arr = np.asarray(raw)
        except ValueError as exc:  # ragged input
            raise NearEdgeDataError(f"{name} is not a numeric array: {exc}") from exc
        if arr.dtype.kind not in "iuf":
            raise NearEdgeDataError(
                f"{name} must be real numbers, got dtype {arr.dtype}"
            )
        if arr.ndim != 1:
            raise NearEdgeDataError(f"{name} must be 1-D, got shape {arr.shape}")
        arrays.append(arr.astype(np.float64))
    e, v = arrays
    if e.shape != v.shape:
        raise NearEdgeDataError(
            f"energy_ev and values must have the same shape, got {e.shape} and "
            f"{v.shape}"
        )
    if e.size < 2:
        raise NearEdgeDataError(
            f"A near-edge spectrum needs at least 2 points, got {e.size}"
        )
    if not (np.isfinite(e).all() and np.isfinite(v).all()):
        raise NearEdgeDataError("Near-edge energies and values must be finite")
    if e[0] <= 0.0:
        raise NearEdgeDataError(
            f"Near-edge energies must be positive, got {e.min():g} eV"
        )
    steps = np.diff(e)
    if (steps <= 0).any():
        bad = int(np.flatnonzero(steps <= 0)[0])
        raise NearEdgeDataError(
            "Near-edge energies must be strictly increasing, but "
            f"energy[{bad}]={e[bad]:g} >= energy[{bad + 1}]={e[bad + 1]:g}"
        )
    return e, v


@dataclass(frozen=True, eq=False)
class NearEdgeSpectrum:
    """A measured absorption spectrum over a limited energy range.

    Construction validates the data exactly as :func:`load_nearedge` does, so
    a spectrum built directly from arrays cannot bypass the checks. The
    arrays are stored as read-only float64 copies.

    Parameters
    ----------
    energy_ev : array_like, shape (n,)
        Photon energies in eV, finite, positive and strictly increasing,
        ``n >= 2``.
    values : array_like, shape (n,)
        Absorption values in the units of the input type (see
        :func:`to_asf_scale`), finite.

    Raises
    ------
    NearEdgeDataError
        If the arrays are not real, 1-D and of equal length, hold fewer than
        two points, contain NaN or inf, or the energies are not positive and
        strictly increasing.

    References
    ----------
    Package addition: a validated container for the near-edge data that
    upstream kkcalc keeps as a bare two-column array; the merge it feeds is
    described in B. Watts, Opt. Express 22(19), 23628 (2014).

    Examples
    --------
    >>> from xraylabtool.kk import NearEdgeSpectrum
    >>> spec = NearEdgeSpectrum([280.0, 290.0, 300.0], [0.1, 0.4, 0.3])
    >>> spec.energy_ev.tolist()
    [280.0, 290.0, 300.0]
    >>> NearEdgeSpectrum([2.0, 1.0], [0.0, 0.0])
    Traceback (most recent call last):
        ...
    xraylabtool.exceptions.NearEdgeDataError: Near-edge energies must be strictly increasing, but energy[0]=2 >= energy[1]=1
    """

    energy_ev: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        e, v = _validate_arrays(self.energy_ev, self.values)
        e.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "energy_ev", e)
        object.__setattr__(self, "values", v)


def load_nearedge(
    path: str | Path, *, energy_unit: Literal["eV", "keV"] = "eV"
) -> NearEdgeSpectrum:
    """Read a near-edge spectrum from a whitespace-separated text file.

    The first column is the photon energy and the last column the absorption
    value; columns in between are ignored. Non-numeric lines before the first
    data row (headers) and lines starting with ``#``, ``%`` or ``;`` are
    skipped; any other non-numeric line after the data has started is an
    error, never silently dropped.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with at least two numeric columns.
    energy_unit : {"eV", "keV"}, optional
        Unit of the first column. Default ``"eV"``, the beamline convention;
        keV values are converted to eV once here.

    Returns
    -------
    NearEdgeSpectrum
        Energies in eV, shape ``(n,)``, and the raw absorption values.

    Raises
    ------
    DataFileError
        If the file cannot be read as text.
    ValidationError
        If ``energy_unit`` is not ``"eV"`` or ``"keV"``.
    NearEdgeDataError
        If the file holds no numeric rows, rows with differing column counts
        or fewer than two columns, a stray non-numeric line inside the data,
        or data that fail the :class:`NearEdgeSpectrum` checks (fewer than
        two points, NaN/inf, non-positive or non-increasing energies).

    References
    ----------
    Package addition. Upstream ``kkcalc.data.load_data`` reads the same
    two columns but silently skips every unparsable line and applies no
    checks.

    Examples
    --------
    >>> from pathlib import Path
    >>> from xraylabtool.kk import load_nearedge
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     name = os.path.join(d, "edge.txt")
    ...     _ = Path(name).write_text("% header\\n280.0 0.1\\n290.0 0.4\\n")
    ...     spec = load_nearedge(name)
    >>> spec.energy_ev.tolist(), spec.values.tolist()
    ([280.0, 290.0], [0.1, 0.4])
    """
    if energy_unit not in _ENERGY_UNITS:
        raise ValidationError(
            "energy_unit must be 'eV' or 'keV'", "energy_unit", energy_unit
        )
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8-sig")  # tolerate a UTF-8 BOM
    except (OSError, UnicodeDecodeError) as exc:
        raise DataFileError(f"Cannot read near-edge file ({exc})", str(file)) from exc

    rows: list[list[float]] = []
    n_header = 0
    for number, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if not tokens or tokens[0].startswith(_COMMENT_PREFIXES):
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            if rows:
                raise NearEdgeDataError(
                    f"Non-numeric data line {number} in {file}: {line.strip()!r}"
                ) from None
            n_header += 1
    if not rows:
        raise NearEdgeDataError(
            f"No numeric rows found in {file} (columns must be whitespace-separated)"
        )
    if n_header:
        logger.info(
            "load_nearedge: skipped %d header line(s) before the first data row in %s",
            n_header,
            file,
        )
    n_cols = {len(r) for r in rows}
    if len(n_cols) != 1 or min(n_cols) < 2:
        raise NearEdgeDataError(
            f"{file} must have the same number (at least 2) of columns in every "
            f"numeric row, found column counts {sorted(n_cols)}"
        )
    table = np.array(rows, dtype=np.float64)
    return NearEdgeSpectrum(table[:, 0] * _ENERGY_UNITS[energy_unit], table[:, -1])


def _check_positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float | np.number)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValidationError(f"{name} must be positive and finite", name, value)
    return float(value)


def _asf_scale_arguments(
    input_type: InputType, density: float | None, formula_mass: float | None
) -> tuple[str, float | None, float | None]:
    """Validated ``(kind, density, formula_mass)`` of :func:`to_asf_scale`.

    ``beta`` input falls back to a density of 1 g/cm3 and a formula mass of
    100 g/mol (each logged at INFO); the other kinds ignore both.
    """
    kind = input_type.lower() if isinstance(input_type, str) else input_type
    if kind not in _INPUT_TYPES:
        raise ValidationError(
            f"input_type must be one of {', '.join(_INPUT_TYPES)}",
            "input_type",
            input_type,
        )
    density = _check_positive("density", density)
    formula_mass = _check_positive("formula_mass", formula_mass)

    if kind == "beta":
        if density is None:
            logger.info("beta input: assuming a mass density of 1 g/cm3")
            density = 1.0
        if formula_mass is None:
            logger.info("beta input: assuming a formula mass of 100 g/mol")
            formula_mass = 100.0
    return kind, density, formula_mass


def to_asf_scale(
    spec: NearEdgeSpectrum,
    input_type: InputType,
    *,
    density: float | None = None,
    formula_mass: float | None = None,
) -> NearEdgeSpectrum:
    """Convert near-edge absorption data to the scattering-factor f2 scale.

    Parameters
    ----------
    spec : NearEdgeSpectrum
        Energies in eV and absorption values in the units of ``input_type``.
    input_type : {"nexafs", "photoabsorption", "xanes", "beta", "asf"}
        ``"nexafs"``, ``"photoabsorption"`` and ``"xanes"`` are the same
        linear absorption coefficient (mass or cross-section scale), which
        is multiplied by ``E / (2 r_e hc)``. ``"beta"`` is the imaginary part
        of the refractive index, converted with the number density
        ``density * N_A / formula_mass``. ``"asf"`` is already f2 and is
        returned unchanged.
    density : float, optional
        Mass density in g/cm^3, ``"beta"`` only. Default 1 (logged at INFO).
    formula_mass : float, optional
        Formula mass in g/mol, ``"beta"`` only. Default 100 (logged at INFO).

    Returns
    -------
    NearEdgeSpectrum
        Same energies (eV), values on the f2 scale (up to the scale factor
        that the affine normalisation of :func:`merge_spectra` removes).

    Raises
    ------
    ValidationError
        If ``input_type`` is unknown, or ``density``/``formula_mass`` is not
        positive and finite.
    NearEdgeDataError
        If the converted values overflow to a non-finite number.

    Notes
    -----
    :func:`merge_spectra` rescales the data to the tabulated f2 at the merge
    points, so a constant factor (the absolute scale of ``nexafs`` data, the
    density and formula mass of ``beta`` data) drops out of the merged
    spectrum; only the shape of the measured curve matters. Data whose
    largest magnitude is below ``2**-500`` are scaled by an exact power of two
    for the device calculation (XLA flushes subnormals) and the result is
    scaled back, so the outcome does not depend on the absolute scale.

    References
    ----------
    Upstream-only conversions (``convert_NEXAFS_to_ASF`` and
    ``convert_Beta_to_ASF`` of kkcalc), using the package constants; the
    paper (Opt. Express 22(19), 23628) describes the normalisation to the
    tabulated scattering factors but not these unit factors.

    Examples
    --------
    >>> from xraylabtool.kk import NearEdgeSpectrum, to_asf_scale
    >>> spec = NearEdgeSpectrum([280.0, 290.0], [0.1, 0.2])
    >>> to_asf_scale(spec, "asf").values.tolist()
    [0.1, 0.2]
    >>> out = to_asf_scale(spec, "nexafs")
    >>> bool(out.values[1] / out.values[0] > 2.0)  # E * mu: ratio 2 * 290/280
    True
    """
    kind, density, formula_mass = _asf_scale_arguments(
        input_type, density, formula_mass
    )

    energy, values = spec.energy_ev, spec.values
    exponent = value_exponent(values)
    values = _rescaled(values, exponent)
    pad = bucket_pow2(energy.size) - energy.size
    _device.ensure_x64()
    # The beta scalars travel in the same single upload (a Python float passed
    # to the jitted function would be an implicit transfer).
    e_d, v_d, density_d, mass_d = _device.upload(
        (
            np.concatenate([energy, np.ones(pad)]),
            np.concatenate([values, np.ones(pad)]),
            None if density is None else np.float64(density),
            None if formula_mass is None else np.float64(formula_mass),
        )
    )
    converted = _device.fetch(
        to_asf_scale_device(e_d, v_d, kind, density=density_d, formula_mass=mass_d)
    )
    out = np.asarray(converted[: energy.size])
    return NearEdgeSpectrum(energy, _rescaled(out, -exponent))


@partial(jax.jit, static_argnames="input_type")
def to_asf_scale_device(
    energy_ev: jax.Array,
    values: jax.Array,
    input_type: str,
    *,
    density: float | jax.Array | None = None,
    formula_mass: float | jax.Array | None = None,
) -> jax.Array:
    """Convert near-edge absorption values to the f2 scale on the device.

    Device counterpart of :func:`to_asf_scale`: the same formulas, element by
    element, so padded entries do not affect real ones. The host wrapper owns
    the argument validation and the INFO log of the ``beta`` defaults; invalid
    arithmetic (overflow) yields ``inf`` and is reported by the host wrapper.

    Parameters
    ----------
    energy_ev : jax.Array
        Photon energies in eV (float64), any 1-D length.
    values : jax.Array
        Absorption values in the units of ``input_type``, same shape.
    input_type : {"nexafs", "photoabsorption", "xanes", "beta", "asf"}
        Static (a different value compiles a separate program); see
        :func:`to_asf_scale`.
    density : float or jax.Array, optional
        Mass density in g/cm^3, ``"beta"`` only. Default 1. Pass a device
        scalar inside a transfer-guarded region (a Python float is an
        implicit transfer).
    formula_mass : float or jax.Array, optional
        Formula mass in g/mol, ``"beta"`` only. Default 100; same note.

    Returns
    -------
    jax.Array
        Values on the f2 scale, same shape as ``values``.

    Raises
    ------
    ValueError
        If ``input_type`` is not one of the supported kinds (at trace time).

    References
    ----------
    Upstream ``kkcalc.data.convert_NEXAFS_to_ASF`` and ``convert_Beta_to_ASF``
    with the package constants; design spec §6 (``2026-10-25-kk-jax-first-design.md``).
    """
    if input_type in _NEXAFS_TYPES:
        return energy_ev * values / (2.0 * constants.THOMPSON * _HC_EV_M)
    if input_type == "beta":
        density = 1.0 if density is None else density
        formula_mass = 100.0 if formula_mass is None else formula_mass
        number_density = density * constants.AVOGADRO / formula_mass
        return (
            1.0e-6
            * values
            * 2.0
            * jnp.pi
            * energy_ev**2
            / (number_density * constants.THOMPSON * _HC_EV_M**2)
        )
    if input_type == "asf":
        return values
    raise ValueError(f"input_type must be one of {', '.join(_INPUT_TYPES)}")


class MergeStatus(NamedTuple):
    """Scalars the host needs to raise or log after a device merge.

    All fields are device scalars (NumPy scalars after
    :func:`~xraylabtool.kk._device.fetch_status`); padded near-edge entries
    take no part in any of them.

    Attributes
    ----------
    n_segments : int32
        True segment count of the merged spectrum.
    v_low, v_high : float
        Measured near-edge values at the two merge points.
    a_low, a_high : float
        Tabulated f2 at the two merge points.
    measured_step, tabulated_step : float
        ``v_high - v_low`` and ``a_high - a_low``.
    nonfinite : bool
        Any normalised near-edge value is not finite.
    n_negative : int32
        Normalised values below zero strictly inside the merge window.
    f2_min : float
        Their minimum over the window (``inf`` when the window holds none).
    n_inside : int32
        Near-edge points strictly inside the merge window.

    References
    ----------
    Design spec §6 (``2026-10-25-kk-jax-first-design.md``).
    """

    n_segments: jax.Array
    v_low: jax.Array
    v_high: jax.Array
    a_low: jax.Array
    a_high: jax.Array
    measured_step: jax.Array
    tabulated_step: jax.Array
    nonfinite: jax.Array
    n_negative: jax.Array
    f2_min: jax.Array
    n_inside: jax.Array


class MergeResult(NamedTuple):
    """Output of :func:`merge_device`.

    Attributes
    ----------
    spectrum : Spectrum
        Merged table at the requested capacity.
    scaled_f2 : jax.Array, shape (m,)
        The full normalised near-edge curve, including the points outside the
        replaced window; entries past the true count are ``0``.
    used_points_ev : jax.Array, shape (2,)
        The ``(low, high)`` merge points the splice used.
    status : MergeStatus
        Everything :func:`raise_or_log` needs.
    """

    spectrum: Spectrum
    scaled_f2: jax.Array
    used_points_ev: jax.Array
    status: MergeStatus


def _resolve_merge_points(
    spec: NearEdgeSpectrum,
    tab_lo: float,
    tab_hi: float,
    merge_points_ev: Sequence[float] | None,
) -> tuple[float, float]:
    """Validated ``(low, high)`` merge points in eV inside the tabulated range."""
    data_lo, data_hi = float(spec.energy_ev[0]), float(spec.energy_ev[-1])
    if data_hi <= tab_lo or data_lo >= tab_hi:
        raise NearEdgeDataError(
            f"Near-edge data (parsed as {_fmt_range(data_lo, data_hi)}) do not "
            f"overlap the tabulated range {_fmt_range(tab_lo, tab_hi)}; the "
            "merge points must lie inside both. If the file holds keV values, "
            "load it with energy_unit='keV'."
        )
    if merge_points_ev is None:
        return max(data_lo, tab_lo), min(data_hi, tab_hi)
    try:
        points = _real_float64("merge points", merge_points_ev)
    except (TypeError, ValueError) as exc:
        raise NearEdgeDataError(f"Merge points must be two numbers: {exc}") from exc
    if points.shape != (2,):
        raise NearEdgeDataError(
            f"Merge points must be two energies (low, high), got shape {points.shape}"
        )
    low, high = float(points[0]), float(points[1])
    if not (math.isfinite(low) and math.isfinite(high)):
        raise NearEdgeDataError(f"Merge points must be finite, got {low}, {high}")
    if not low < high:
        raise NearEdgeDataError(
            f"Merge points must be increasing (low < high), got {low:.12g}, {high:.12g} eV"
        )
    low, high = _snap(low, data_lo), _snap(high, data_hi)
    if low < data_lo or high > data_hi:
        raise NearEdgeDataError(
            f"Merge points {low:.12g}, {high:.12g} eV must lie inside the near-edge "
            f"data range {_fmt_range(data_lo, data_hi)}"
        )
    low, high = _snap(low, tab_lo), _snap(high, tab_hi)
    if low < tab_lo or high > tab_hi:
        raise NearEdgeDataError(
            f"Merge points {low:.12g}, {high:.12g} eV must lie inside the tabulated "
            f"range {_fmt_range(tab_lo, tab_hi)}"
        )
    return low, high


def _interp(x: jax.Array, xp: jax.Array, fp: jax.Array, n: jax.Array) -> jax.Array:
    """``numpy.interp`` over the first ``n`` entries of ``xp`` (padding must be ``+inf``).

    Same arithmetic as NumPy (``slope * (x - xp[j]) + fp[j]``, the product
    pinned so XLA cannot fuse it), constant outside the data, and ``fp[j]``
    exactly on a knot (an overflowing slope times a zero offset would be NaN).
    NumPy's NaN fallback (retry from the right knot) is not reproduced: off a
    knot an overflowing slope gives ``±inf``, never NaN, and that is reported
    by the status.
    """
    j = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, jnp.maximum(n - 2, 0))
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    inner = jnp.where(
        x == xp[j], fp[j], jax.lax.optimization_barrier(slope * (x - xp[j])) + fp[j]
    )
    last = jnp.maximum(n - 1, 0)
    return jnp.where(x >= xp[last], fp[last], jnp.where(x < xp[0], fp[0], inner))


def _spliced_coeffs(
    spec: Spectrum,
    seg_out: jax.Array,
    n_out: jax.Array,
    window: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    knot_energy: jax.Array,
    knot_f2: jax.Array,
) -> jax.Array:
    """Coefficients of the merged table for the segments of ``seg_out``.

    The source of each segment is chosen by its lower boundary (right
    continuous, clamped to the last real segment): inside ``[low, high)`` the
    piecewise-linear near-edge piece between its two knots, elsewhere the
    tabulated polynomial that starts at or below the boundary. A midpoint
    would pick the wrong polynomial for adjacent representable floats.
    """
    low, high, a_low, a_high = window
    lower, upper = seg_out[:-1], seg_out[1:]
    tab = spec.coeffs[
        jnp.clip(
            jnp.searchsorted(spec.seg_ev, lower, side="right") - 1,
            0,
            spec.n_segments - 1,
        )
    ]

    def height(x: jax.Array) -> jax.Array:
        k = jnp.clip(jnp.searchsorted(knot_energy, x), 0, knot_energy.shape[0] - 1)
        return jnp.where(x == low, a_low, jnp.where(x == high, a_high, knot_f2[k]))

    h_lower, h_upper = height(lower), height(upper)
    slope = (h_upper - h_lower) / (upper - lower)
    # The barrier keeps slope * E from fusing with the subtraction (FMA).
    offset = h_lower - jax.lax.optimization_barrier(slope * lower)
    linear = jnp.zeros((lower.shape[0], 5)).at[:, 0].set(slope).at[:, 1].set(offset)
    coeffs = jnp.where(((lower >= low) & (lower < high))[:, None], linear, tab)
    return jnp.where((jnp.arange(lower.shape[0]) < n_out)[:, None], coeffs, 0.0)


@partial(jax.jit, static_argnames="capacity")
def merge_device(
    spec: Spectrum,
    ne_energy_ev: jax.Array,
    ne_f2: jax.Array,
    n_ne: jax.Array,
    low_ev: float | jax.Array,
    high_ev: float | jax.Array,
    *,
    capacity: int,
) -> MergeResult:
    """Normalise near-edge data to the tabulated f2 and splice it in, on the device.

    Device counterpart of :func:`merge_spectra` (the merge points are
    already resolved and snapped by the host). Between ``low_ev`` and
    ``high_ev`` the tabulated segments are replaced by the piecewise-linear
    curve through the normalised near-edge points; outside they are copied
    unchanged. No data check is raised here: the flatness and finiteness
    conditions are returned in :class:`MergeStatus` for :func:`raise_or_log`,
    and computed without trapping on invalid arithmetic.

    Parameters
    ----------
    spec : Spectrum
        Tabulated table (``n`` true segments).
    ne_energy_ev : jax.Array, shape (m,)
        Near-edge energies in eV, strictly increasing over the first ``n_ne``
        entries; the rest is padding and is masked.
    ne_f2 : jax.Array, shape (m,)
        Near-edge values on the f2 scale (:func:`to_asf_scale_device`).
    n_ne : jax.Array
        Number of real near-edge points (int scalar, at least 2).
    low_ev, high_ev : float or jax.Array
        Merge points, ``low_ev < high_ev``. Inside a
        ``jax.transfer_guard("disallow_explicit")`` region they must already be
        device scalars: converting a Python float is an implicit transfer.
    capacity : int
        Static output segment capacity; use
        ``bucket(n_segments_upper_bound + n_near_edge + 2)``, a smaller value
        silently truncates the knot union.

    Returns
    -------
    MergeResult
        Merged :class:`~xraylabtool.kk.Spectrum` at ``capacity``, the full
        normalised near-edge curve, the merge points used and the status.

    Notes
    -----
    Candidate knots are the real tabulated knots outside ``[low, high]``,
    ``low``, the near-edge energies strictly inside the window and ``high``;
    every excluded or padded entry is sent to ``+inf`` before the union, so
    the result does not depend on ``capacity`` or on the padding. The products
    that feed sums are pinned with ``jax.lax.optimization_barrier`` so the
    rounding is that of the NumPy reference rather than an FMA's.

    References
    ----------
    B. Watts, Opt. Express 22(19), 23628 (2014): normalisation of the
    near-edge data to the tabulated scattering factors; design spec §6
    (``2026-10-25-kk-jax-first-design.md``).
    """
    low = jnp.asarray(low_ev, dtype=jnp.float64)
    high = jnp.asarray(high_ev, dtype=jnp.float64)
    valid = jnp.arange(ne_energy_ev.shape[0]) < n_ne
    energy = jnp.where(valid, ne_energy_ev, jnp.inf)
    values = jnp.where(valid, ne_f2, 0.0)

    points = jnp.stack([low, high])
    v_low, v_high = _interp(points, energy, values, n_ne)
    a_low, a_high = evaluate_f2_device(points, spec)
    step = v_high - v_low
    tab_step = a_high - a_low
    scaled = jax.lax.optimization_barrier((values - v_low) * (tab_step / step)) + a_low
    scaled = jnp.where(valid, scaled, 0.0)

    inside = valid & (energy > low) & (energy < high)
    kept = (jnp.arange(spec.seg_ev.shape[0]) <= spec.n_segments) & (
        (spec.seg_ev < low) | (spec.seg_ev > high)
    )
    candidates = jnp.concatenate(
        [
            jnp.where(kept, spec.seg_ev, jnp.inf),
            jnp.where(inside, energy, jnp.inf),
            points,
        ]
    )
    union, n_knots = _union_knots(candidates, capacity + 1)
    pos = jnp.arange(capacity + 1)
    last = union[jnp.maximum(n_knots - 1, 0)]
    seg_out = jnp.where(pos < n_knots, union, last * (pos - n_knots + 2))
    n_out = n_knots - 1

    coeffs = _spliced_coeffs(
        spec, seg_out, n_out, (low, high, a_low, a_high), energy, scaled
    )
    negative = inside & (scaled < 0.0)
    status = MergeStatus(
        n_segments=n_out,
        v_low=v_low,
        v_high=v_high,
        a_low=a_low,
        a_high=a_high,
        measured_step=step,
        tabulated_step=tab_step,
        nonfinite=~jnp.all(jnp.isfinite(scaled) | ~valid),
        n_negative=jnp.sum(negative, dtype=jnp.int32),
        f2_min=jnp.min(jnp.where(inside, scaled, jnp.inf)),
        n_inside=jnp.sum(inside, dtype=jnp.int32),
    )
    return MergeResult(Spectrum(seg_out, coeffs, n_out), scaled, points, status)


def raise_or_log(
    status: MergeStatus, low: float, high: float, exponent: int = 0
) -> None:
    """Raise or log what a device merge reported (host side).

    Parameters
    ----------
    status : MergeStatus
        The fetched status (NumPy scalars, see
        :func:`~xraylabtool.kk._device.fetch_status`).
    low, high : float
        The merge points in eV, for the messages.
    exponent : int, default 0
        The :func:`value_exponent` the values were rescaled by before the
        merge; the reported end-point values are shown unscaled.

    Raises
    ------
    NearEdgeDataError
        If the near-edge values at the merge points are (nearly) equal, the
        tabulated f2 is equal there, or a normalised value is not finite.

    Notes
    -----
    Normalised values below zero inside the window are kept and logged at
    WARNING (count, window size and minimum), never clipped.

    References
    ----------
    Design spec §1 and §6 (``2026-10-25-kk-jax-first-design.md``).
    """
    v_low, v_high = float(status.v_low), float(status.v_high)
    if not abs(float(status.measured_step)) > _MIN_RELATIVE_STEP * max(
        abs(v_low), abs(v_high)
    ):
        v_low, v_high = math.ldexp(v_low, -exponent), math.ldexp(v_high, -exponent)
        raise NearEdgeDataError(
            f"Near-edge data have flat endpoints: the values at the merge points {low:g} "
            f"and {high:g} eV ({v_low:.6g}, {v_high:.6g}) differ by less than "
            f"{_MIN_RELATIVE_STEP:g} of their size, so the data cannot be "
            "normalised to the tabulated f2. Choose merge points on the "
            "pre-edge and post-edge."
        )
    a_low, a_high = float(status.a_low), float(status.a_high)
    if not abs(float(status.tabulated_step)) > 1e-12 * max(abs(a_low), abs(a_high)):
        raise NearEdgeDataError(
            f"The tabulated f2 is equal at the merge points {low:g} and "
            f"{high:g} eV ({a_low:.6g}, {a_high:.6g}); the near-edge data "
            "cannot be normalised. Choose merge points on either side of the edge."
        )
    if bool(status.nonfinite):
        raise NearEdgeDataError(
            "Near-edge data are not finite after normalisation to the "
            "tabulated f2 (values too large for the merge-point step)"
        )
    if int(status.n_negative):
        logger.warning(
            "%d of %d near-edge points inside the merge window are negative "
            "after normalisation to the tabulated f2 (minimum %.6g); kept "
            "unchanged, not clipped",
            int(status.n_negative),
            int(status.n_inside),
            float(status.f2_min),
        )


def _padded_table(
    seg: NDArray[np.float64], coef: NDArray[np.float64], capacity: int
) -> Spectrum:
    """NumPy-leaved padded table without the positivity and finiteness checks.

    The merge accepts non-positive leading edges; the
    last edge is always positive because a merge point lies inside the range,
    so the usual pad edges ``x_n (1 + k)`` are increasing, and finite for any
    physical edge (they overflow to ``inf`` only if the last edge exceeds about
    ``1.8e308 / (capacity + 1)``, far outside the 10 eV-500 keV tables).
    """
    k = np.arange(1, capacity - coef.shape[0] + 1)
    last = seg[-1]
    pad_ev = last * (1.0 + k)
    return Spectrum(
        np.concatenate([seg, pad_ev]),  # type: ignore[arg-type]
        np.concatenate([coef, np.zeros((k.size, 5))]),  # type: ignore[arg-type]
        np.int32(coef.shape[0]),  # type: ignore[arg-type]
    )


def merge_spectra(
    spec: NearEdgeSpectrum,
    seg_ev: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    *,
    merge_points_ev: tuple[float, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Normalise near-edge data to the tabulated f2 and splice it in.

    The near-edge values are rescaled and shifted (affine endpoint matching)
    so that they equal the tabulated f2 at the two merge points; between them
    the tabulated segments are replaced by the piecewise-linear near-edge
    curve. Outside the merge points the tabulated segments are unchanged.

    Parameters
    ----------
    spec : NearEdgeSpectrum
        Near-edge data on the f2 scale (see :func:`to_asf_scale`), energies
        in eV.
    seg_ev : numpy.ndarray, shape (n+1,)
        Segment edges of the tabulated spectrum in eV, strictly increasing.
    coeffs : numpy.ndarray, shape (n, 5)
        Polynomial coefficients (``E¹, E⁰, E⁻¹, E⁻², E⁻³``) of each segment.
    merge_points_ev : tuple of float or None
        ``(low, high)`` in eV with ``low < high``, both inside the near-edge
        data and the tabulated range. ``None`` merges over the whole overlap
        of the two ranges.

    Returns
    -------
    seg_ev : numpy.ndarray, shape (m+1,)
        Segment edges of the merged spectrum in eV, strictly increasing; the
        merge points and the near-edge energies between them are edges.
    coeffs : numpy.ndarray, shape (m, 5)
        Coefficients of the merged spectrum, ready for
        :func:`~xraylabtool.kk.transform.kk_real_spectrum`.

    Raises
    ------
    NearEdgeDataError
        If the near-edge data do not overlap the tabulated range (the message
        states both ranges), the merge points are not two finite increasing
        energies inside both ranges, the near-edge values at the merge points
        are (nearly) equal so the matching would divide by ~0, the tabulated
        f2 is equal at the merge points, or the normalised values are not
        finite.
    ValidationError
        If ``seg_ev`` and ``coeffs`` do not have shapes ``(n+1,)`` and
        ``(n, 5)``, or are complex.
    ValueError
        If ``seg_ev`` is not finite or not strictly increasing.

    Notes
    -----
    Normalised values below zero are kept and logged at WARNING (count and
    minimum), never clipped: a noisy pre-edge can legitimately dip below the
    tabulated level. At a tabulated edge that coincides with a merge point the
    segment that starts there is used, as in upstream.

    References
    ----------
    Upstream ``kkcalc.data.merge_spectra`` without the "add background" and
    "fix distortions" options; the validation and the conditioning guard are
    package additions. The normalisation to the tabulated scattering factors
    is described in B. Watts, Opt. Express 22(19), 23628 (2014).

    Examples
    --------
    >>> from xraylabtool.kk import NearEdgeSpectrum, merge_spectra
    >>> from xraylabtool.kk.spectrum import compound_spectrum
    >>> seg, coeffs = compound_spectrum(["C"], [1.0])
    >>> edge = NearEdgeSpectrum([280.0, 290.0, 300.0, 310.0], [0.1, 0.2, 0.4, 0.5])
    >>> merged, merged_coeffs = merge_spectra(
    ...     edge, seg, coeffs, merge_points_ev=(280.0, 310.0)
    ... )
    >>> bool(merged.size == merged_coeffs.shape[0] + 1)
    True
    >>> bool({290.0, 300.0} <= set(merged.tolist()))
    True
    """
    try:
        seg = _real_float64("seg_ev", seg_ev)
        coef = _real_float64("coeffs", coeffs)
    except ValueError as exc:
        raise ValidationError(str(exc), "seg_ev/coeffs") from exc
    if seg.ndim != 1 or seg.size < 2 or coef.shape != (seg.size - 1, 5):
        raise ValidationError(
            "seg_ev must have shape (n+1,) and coeffs shape (n, 5), got "
            f"{seg.shape} and {coef.shape}"
        )
    low, high = _resolve_merge_points(
        spec, float(seg[0]), float(seg[-1]), merge_points_ev
    )
    # Edges only need to be finite and increasing (not positive); the
    # coefficients are not checked here.
    if not np.isfinite(seg).all():
        raise ValueError("seg_ev must be finite")
    if (np.diff(seg) <= 0).any():
        raise ValueError("seg_ev must be strictly increasing")
    table = _padded_table(seg, coef, bucket(coef.shape[0]))

    energy, values = spec.energy_ev, spec.values
    exponent = value_exponent(values)
    values = _rescaled(values, exponent)
    n_ne = energy.size
    pad = bucket_pow2(n_ne) - n_ne
    _device.ensure_x64()
    table_d, e_d, v_d, n_d, low_d, high_d = _device.upload(
        (
            table,
            np.concatenate([energy, np.full(pad, energy[-1])]),
            np.concatenate([values, np.full(pad, values[-1])]),
            np.int32(n_ne),
            np.float64(low),
            np.float64(high),
        )
    )
    result = merge_device(
        table_d,
        e_d,
        v_d,
        n_d,
        low_d,
        high_d,
        capacity=bucket(coef.shape[0] + n_ne + 2),
    )
    raise_or_log(_device.fetch_status(result.status), low, high, exponent)
    out = _device.fetch(result)
    n_out = int(out.spectrum.n_segments)
    _device.check_capacity(n_out, out.spectrum.coeffs.shape[0])
    return (
        np.asarray(out.spectrum.seg_ev[: n_out + 1], dtype=np.float64),
        np.asarray(out.spectrum.coeffs[:n_out], dtype=np.float64),
    )
