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
eV). This module is host-side NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from xraylabtool import constants
from xraylabtool.exceptions import DataFileError, NearEdgeDataError, ValidationError
from xraylabtool.kk.spectrum import evaluate_f2
from xraylabtool.logging_utils import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger("kk")

InputType = Literal["nexafs", "photoabsorption", "xanes", "beta", "asf"]

_NEXAFS_TYPES = ("nexafs", "photoabsorption", "xanes")
_INPUT_TYPES = (*_NEXAFS_TYPES, "beta", "asf")
_ENERGY_UNITS = {"eV": 1.0, "keV": 1000.0}
_COMMENT_PREFIXES = ("#", "%", ";")

# hc in eV m, from the package constants.
_HC_EV_M = constants.SPEED_OF_LIGHT * constants.PLANCK / constants.ELEMENT_CHARGE

# Smallest usable relative step between the two end-point values: the affine
# matching divides by it, so anything near rounding level would amplify noise
# by 1e6 or more. Real pre-edge to post-edge steps are percent to orders of
# magnitude.
_MIN_RELATIVE_STEP = 1e-6


def _fmt_range(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} eV"


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
        text = file.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise DataFileError(f"Cannot read near-edge file ({exc})", str(file)) from exc

    rows: list[list[float]] = []
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
    if not rows:
        raise NearEdgeDataError(f"No numeric rows found in {file}")
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
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be positive and finite", name, value)
    return float(value)


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
    spectrum; only the shape of the measured curve matters.

    References
    ----------
    Upstream-only conversions (``convert_NEXAFS_to_ASF`` and
    ``convert_Beta_to_ASF`` of kkcalc), using the package constants; the
    paper (Opt. Express 22(19), 23628) describes the normalisation to the
    tabulated scattering factors but not these unit factors.

    Examples
    --------
    >>> spec = NearEdgeSpectrum([280.0, 290.0], [0.1, 0.2])
    >>> to_asf_scale(spec, "asf").values.tolist()
    [0.1, 0.2]
    >>> out = to_asf_scale(spec, "nexafs")
    >>> bool(out.values[1] / out.values[0] > 2.0)  # E * mu: ratio 2 * 290/280
    True
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

    energy, values = spec.energy_ev, spec.values
    with np.errstate(all="ignore"):
        if kind in _NEXAFS_TYPES:
            converted = energy * values / (2.0 * constants.THOMPSON * _HC_EV_M)
        elif kind == "beta":
            if density is None:
                logger.info("beta input: assuming a mass density of 1 g/cm3")
                density = 1.0
            if formula_mass is None:
                logger.info("beta input: assuming a formula mass of 100 g/mol")
                formula_mass = 100.0
            number_density = density * constants.AVOGADRO / formula_mass
            converted = (
                1.0e-6
                * values
                * 2.0
                * np.pi
                * energy**2
                / (number_density * constants.THOMPSON * _HC_EV_M**2)
            )
        else:
            converted = values.copy()
    return NearEdgeSpectrum(energy, converted)


class _Merged(NamedTuple):
    seg_ev: NDArray[np.float64]
    coeffs: NDArray[np.float64]
    scaled: NDArray[np.float64]
    merge_points_ev: tuple[float, float]


def _resolve_merge_points(
    spec: NearEdgeSpectrum,
    seg_ev: NDArray[np.float64],
    merge_points_ev: Sequence[float] | None,
) -> tuple[float, float]:
    """Validated ``(low, high)`` merge points in eV."""
    data_lo, data_hi = float(spec.energy_ev[0]), float(spec.energy_ev[-1])
    tab_lo, tab_hi = float(seg_ev[0]), float(seg_ev[-1])
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
        points = np.asarray(merge_points_ev, dtype=np.float64)
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
            f"Merge points must be increasing (low < high), got {low:g}, {high:g} eV"
        )
    if low < data_lo or high > data_hi:
        raise NearEdgeDataError(
            f"Merge points {low:g}, {high:g} eV must lie inside the near-edge "
            f"data range {_fmt_range(data_lo, data_hi)}"
        )
    if low < tab_lo or high > tab_hi:
        raise NearEdgeDataError(
            f"Merge points {low:g}, {high:g} eV must lie inside the tabulated "
            f"range {_fmt_range(tab_lo, tab_hi)}"
        )
    return low, high


def _merge(
    spec: NearEdgeSpectrum,
    seg_ev: object,
    coeffs: object,
    merge_points_ev: Sequence[float] | None,
) -> _Merged:
    """Implementation of :func:`merge_spectra`, also returning the scaled data."""
    seg = np.asarray(seg_ev, dtype=np.float64)
    coef = np.asarray(coeffs, dtype=np.float64)
    if seg.ndim != 1 or seg.size < 2 or coef.shape != (seg.size - 1, 5):
        raise ValidationError(
            "seg_ev must have shape (n+1,) and coeffs shape (n, 5), got "
            f"{seg.shape} and {coef.shape}"
        )
    low, high = _resolve_merge_points(spec, seg, merge_points_ev)
    energy, values = spec.energy_ev, spec.values

    points = np.array([low, high])
    v_low, v_high = (float(x) for x in np.interp(points, energy, values))
    a_low, a_high = (float(x) for x in evaluate_f2(points, seg, coef))
    step = v_high - v_low
    if not abs(step) > _MIN_RELATIVE_STEP * max(abs(v_low), abs(v_high)):
        raise NearEdgeDataError(
            f"Near-edge data have flat endpoints: the values at the merge points {low:g} "
            f"and {high:g} eV ({v_low:.6g}, {v_high:.6g}) differ by less than "
            f"{_MIN_RELATIVE_STEP:g} of their size, so the data cannot be "
            "normalised to the tabulated f2. Choose merge points on the "
            "pre-edge and post-edge."
        )
    tab_step = a_high - a_low
    if not abs(tab_step) > 1e-12 * max(abs(a_low), abs(a_high)):
        raise NearEdgeDataError(
            f"The tabulated f2 is equal at the merge points {low:g} and "
            f"{high:g} eV ({a_low:.6g}, {a_high:.6g}); the near-edge data "
            "cannot be normalised. Choose merge points on either side of the edge."
        )

    with np.errstate(all="ignore"):
        scaled = (values - v_low) * (tab_step / step) + a_low
    if not np.isfinite(scaled).all():
        raise NearEdgeDataError(
            "Near-edge data are not finite after normalisation to the "
            "tabulated f2 (values too large for the merge-point step)"
        )

    inside = (energy > low) & (energy < high)
    negative = scaled[inside] < 0.0
    if negative.any():
        logger.warning(
            "%d of %d near-edge points inside the merge window are negative "
            "after normalisation to the tabulated f2 (minimum %.6g); kept "
            "unchanged, not clipped",
            int(negative.sum()),
            int(inside.sum()),
            float(scaled[inside].min()),
        )

    # Piecewise-linear near-edge segments between the merge points.
    knots = np.concatenate(([low], energy[inside], [high]))
    heights = np.concatenate(([a_low], scaled[inside], [a_high]))
    slope = np.diff(heights) / np.diff(knots)
    linear = np.zeros((knots.size - 1, 5))
    linear[:, 0] = slope
    linear[:, 1] = heights[:-1] - slope * knots[:-1]

    n_left = int(np.searchsorted(seg, low, side="left"))  # knots below `low`
    n_right = int(np.searchsorted(seg, high, side="right"))  # knots <= `high`
    full_ev = np.concatenate((seg[:n_left], knots, seg[n_right:]))
    if n_right < seg.size:
        # Tabulated polynomial that continues above the last merge point.
        tail = coef[n_right - 1 : n_right]
        full_coeffs = np.vstack((coef[:n_left], linear, tail, coef[n_right:]))
    else:  # high is the last tabulated edge: no segment above it
        full_coeffs = np.vstack((coef[:n_left], linear))
    return _Merged(full_ev, full_coeffs, scaled, (low, high))


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
        ``(n, 5)``.

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
    merged = _merge(spec, seg_ev, coeffs, merge_points_ev)
    return merged.seg_ev, merged.coeffs
