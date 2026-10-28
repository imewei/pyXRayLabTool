"""Pure JAX optics: PCHIP lookup of f1/f2 and the derived X-ray quantities.

The functions here are pure and ``jit``/``vmap``/``grad``-compatible with x64 as
a precondition (callers run :func:`xraylabtool.backend.runtime.ensure_x64` and the
programs inside ``jax.enable_x64(True)``). The scattering tables are a program
*argument* (see :mod:`xraylabtool.calculators.tables`), never a closed-over
constant. :func:`encode_formula` is the host-side half: it turns a formula
into the integer/float arrays the device programs consume.

This module imports ``jax`` at module level, so callers import it inside
functions; importing it has no side effect (no ``ensure_x64()``).

References
----------
Design: ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md`` §4-§5.
Fritsch, F. N. and Carlson, R. E. (1980), SIAM J. Numer. Anal. 17, 238 (PCHIP).
"""

# ruff: noqa: RUF002

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xraylabtool.calculators.tables import ScatteringTables, element_index, load_tables
from xraylabtool.constants import (
    AVOGADRO,
    ENERGY_TO_WAVELENGTH_FACTOR,
    METER_TO_ANGSTROM,
    PI,
    RADIANS_TO_DEGREES,
    SCATTERING_FACTOR,
)
from xraylabtool.data_handling.atomic_cache import get_bulk_atomic_data_fast
from xraylabtool.utils import parse_formula

__all__ = [
    "OUT_FIELDS",
    "OpticalProps",
    "ScatteringTables",
    "attenuation_length_from_beta",
    "critical_angle_from_delta",
    "derive",
    "encode_formula",
    "interpolate_f",
    "load_tables",
    "optical_properties",
    "transmission",
]

# Order of the per-energy rows of the ``out`` array of the programs.
OUT_FIELDS = (
    "wavelength",
    "dispersion",
    "absorption",
    "f1",
    "f2",
    "critical_angle",
    "attenuation_length",
    "re_sld",
    "im_sld",
)


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[*OUT_FIELDS, "electron_density"],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True, eq=False)
class OpticalProps:
    """Optical properties of one material at ``M`` energies.

    Attributes
    ----------
    wavelength : Array, shape ``[M]``
        Wavelength in angstrom.
    dispersion, absorption : Array, shape ``[M]``
        The refractive-index decrement δ and absorption index β.
    f1, f2 : Array, shape ``[M]``
        Count-weighted sums of the atomic scattering factors.
    critical_angle : Array, shape ``[M]``
        Critical angle in degrees (0 where δ < 0).
    attenuation_length : Array, shape ``[M]``
        1/e attenuation length in cm (β clamped at ``1e-30``).
    re_sld, im_sld : Array, shape ``[M]``
        Real and imaginary scattering-length density in Å⁻².
    electron_density : Array, shape ``()``
        Electrons per Å³.
    """

    wavelength: Any
    dispersion: Any
    absorption: Any
    f1: Any
    f2: Any
    critical_angle: Any
    attenuation_length: Any
    re_sld: Any
    im_sld: Any
    electron_density: Any


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _pchip_pair(x: Any, coef: Any, n: Any, q: Any) -> Any:
    """Evaluate f1 and f2 of one padded table row at ``q`` with one ``searchsorted``.

    Parameters
    ----------
    x : Array, shape ``[N]``
        Knots (eV), strictly increasing; entries past ``n`` are padding.
    coef : Array, shape ``[N-1, 9]``
        Row of :attr:`ScatteringTables.coef`: ``x[i]`` and the power-basis
        cubics of f1 and f2 per interval, fetched with one gather.
    n : Array, int scalar
        Number of real knots.
    q : Array, shape ``[M]`` or scalar
        Query energies (eV).

    Returns
    -------
    Array, shape ``[2, M]`` (``[2]`` for a scalar ``q``)
        Cubic Hermite values, NaN outside ``[x[0], x[n-1]]``.
    """
    # "scan_unrolled" returns the same indices as the default "scan" (integers, so
    # bit-identical results) but is ~2.6x faster on XLA-CPU for the batch program.
    i = jnp.clip(
        jnp.searchsorted(x, q, side="right", method="scan_unrolled") - 1, 0, n - 2
    )
    k = coef[i]  # (..., 9): one gather instead of ten; q may be a scalar
    t = q - k[..., 0]
    f1 = ((k[..., 4] * t + k[..., 3]) * t + k[..., 2]) * t + k[..., 1]
    f2 = ((k[..., 8] * t + k[..., 7]) * t + k[..., 6]) * t + k[..., 5]
    inside = (q >= x[0]) & (q <= x[n - 1])
    return jnp.where(inside, jnp.stack([f1, f2]), jnp.nan)


def _row_pair(tables: ScatteringTables, zi: Any, q: Any) -> Any:
    """f1 and f2 of table row ``zi`` at ``q`` (eV), shape ``[2, M]``."""
    return _pchip_pair(tables.x[zi], tables.coef[zi], tables.n[zi], q)


def interpolate_f(tables: ScatteringTables, z: Any, counts: Any, e_ev: Any) -> Any:
    """Count-weighted f1 and f2 sums of a material at ``e_ev``.

    Parameters
    ----------
    tables : ScatteringTables
        Resident tables from :func:`load_tables`.
    z : Array, int, shape ``[K]``
        Table rows (atomic number minus one); padding rows use 0.
    counts : Array, float64, shape ``[K]``
        Atoms per formula unit; padding rows use 0 and contribute nothing, even
        where their own row would be NaN.
    e_ev : Array, float64, shape ``[M]``
        Energies in eV.

    Returns
    -------
    Array, shape ``[2, M]``
        ``Σ_k counts[k]·f1_k`` (row 0) and ``Σ_k counts[k]·f2_k`` (row 1); NaN
        where any real element is outside its table.
    """

    pairs = jax.vmap(lambda zi: _row_pair(tables, zi, e_ev))(z)  # (K, 2, M)
    c = counts[:, None, None]
    weighted = jnp.where(c == 0, 0.0, pairs * c)
    # A left fold over K: a reduce's association tree depends on K, and the padded
    # zero rows must not change a single bit of the sum of the real rows.
    total, _ = jax.lax.scan(
        lambda acc, w: (acc + w, None), jnp.zeros_like(weighted[0]), weighted
    )
    return total


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def _derived(
    wavelength_m: Any,
    dispersion: Any,
    absorption: Any,
    rho: Any,
    mw: Any,
    n_electrons: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Electron density, critical angle, attenuation length and SLDs.

    Reproduces the former NumPy derived-quantities kernel and the electron-density
    expression of ``derived_quantities.calculate_derived_quantities`` operation for
    operation, including the ``max(2δ, 0)`` critical-angle rule and the ``1e-30`` β clamp.
    """
    electron_density = jnp.asarray(1e6 * rho / mw * AVOGADRO * n_electrons / 1e30)
    # Double ``where``: sqrt of the clamped 0 has an infinite slope that the clamp's
    # zero slope turns into a NaN gradient (every energy with delta < 0).
    two_delta = 2.0 * dispersion
    positive = two_delta > 0.0
    critical_angle = jnp.where(
        positive, jnp.sqrt(jnp.where(positive, two_delta, 1.0)), 0.0
    ) * (180.0 / PI)
    absorption_safe = jnp.maximum(absorption, 1e-30)
    attenuation_length = wavelength_m / absorption_safe / (4 * PI) * 1e2
    wavelength_sq = jnp.square(wavelength_m)
    sld_factor = 2 * PI / 1e20
    # One shared factor: ``δ·c/λ²`` is otherwise reassociated differently for one
    # material than for several, which moves the last bit.
    sld_scale = sld_factor / wavelength_sq
    re_sld = dispersion * sld_scale
    im_sld = absorption * sld_scale
    return electron_density, critical_angle, attenuation_length, re_sld, im_sld


def derive(
    energy_kev: Any, f1_sum: Any, f2_sum: Any, rho: Any, mw: Any, n_electrons: Any
) -> OpticalProps:
    """Optical properties from count-weighted f1/f2 totals.

    Reproduces, operation for operation, the former NumPy
    ``core._properties_from_totals`` and derived-quantity kernel:
    δ = ((λ²·S·ρ)·f1)/M, β likewise with f2, with ``S = SCATTERING_FACTOR``. The
    legacy ``λ²·(S·ρ/M)·f`` is reassociated so that the division by ``M`` comes
    last: for the physical range, neither a huge ``M`` (``S·ρ/M`` or ``f/M``
    subnormal, flushed to zero by XLA) nor a tiny ``f`` underflows an
    intermediate. At the opposite extreme the product ``(λ²·S·ρ)·f`` can overflow
    (a factor sum of about 1e300 or more with a large ``ρ``); the status check
    then reports the non-finite failure instead of returning a zero.

    Parameters
    ----------
    energy_kev : Array, shape ``[M]``
        Energies in keV.
    f1_sum, f2_sum : Array, shape ``[M]``
        ``Σ counts·f1`` and ``Σ counts·f2``.
    rho, mw, n_electrons : scalar
        Density (g/cm³), molecular weight (g/mol) and electrons per formula unit.

    Returns
    -------
    OpticalProps
    """
    # The barrier stops XLA from folding ``(K/E)/x`` into ``K/(E*x)`` when the batch
    # is one material, which would change the last bit of the SLD and attenuation.
    wavelength = jax.lax.optimization_barrier(ENERGY_TO_WAVELENGTH_FACTOR / energy_kev)
    wave_factor = jnp.square(wavelength) * (SCATTERING_FACTOR * rho)
    dispersion = (wave_factor * f1_sum) / mw
    absorption = (wave_factor * f2_sum) / mw
    ed, critical, attenuation, re_sld, im_sld = _derived(
        wavelength, dispersion, absorption, rho, mw, n_electrons
    )
    return OpticalProps(
        wavelength=wavelength * METER_TO_ANGSTROM,
        dispersion=dispersion,
        absorption=absorption,
        f1=f1_sum,
        f2=f2_sum,
        critical_angle=critical,
        attenuation_length=attenuation,
        re_sld=re_sld,
        im_sld=im_sld,
        electron_density=ed,
    )


def _status(props: OpticalProps, rho: Any, mw: Any) -> Any:
    """True where δ and β are finite at every energy.

    The legacy ``S·ρ/M`` overflowed to ``inf`` (hence non-finite δ and β) for a tiny
    ``M``; δ is now formed without that product, so its finiteness is tested too.
    """
    return (
        jnp.isfinite(props.dispersion).all()
        & jnp.isfinite(props.absorption).all()
        & jnp.isfinite(SCATTERING_FACTOR * rho / mw)
    )


def optical_properties(
    tables: ScatteringTables,
    z: Any,
    counts: Any,
    rho: Any,
    mw: Any,
    n_electrons: Any,
    energy_kev: Any,
) -> tuple[OpticalProps, Any]:
    """Optical properties of one material; ``vmap`` over its arguments for a batch.

    Pure and ``jit``/``vmap``/``grad``-compatible with x64 as a precondition.
    With the other arguments fixed, ``δ`` is linear in ``rho``, so
    ``grad(rho -> optical_properties(...)[0].dispersion.sum())`` equals
    ``dispersion.sum() / rho`` (the NaN-outside-range branch is not
    differentiated; gradients w.r.t. energy are not part of the contract).

    Parameters
    ----------
    tables : ScatteringTables
        From :func:`load_tables`.
    z : Array, int, shape ``[K]``
    counts : Array, float64, shape ``[K]``
    rho, mw, n_electrons : scalar
    energy_kev : Array, float64, shape ``[M]``

    Returns
    -------
    props : OpticalProps
    ok : Array, bool scalar
        ``all(isfinite(δ)) & all(isfinite(β))`` and ``S·ρ/M`` finite.

    References
    ----------
    Design spec §4-§5.
    """
    # The barrier keeps XLA from contracting ``keV*1000 - knot`` into an FMA in
    # some bucket shapes only, which would break bit-exact padding invariance.
    e_ev = jax.lax.optimization_barrier(energy_kev * 1000.0)
    f = interpolate_f(tables, z, counts, e_ev)
    props = derive(energy_kev, f[0], f[1], rho, mw, n_electrons)
    return props, _status(props, rho, mw)


def _pack(props: OpticalProps) -> Any:
    return jnp.stack([getattr(props, name) for name in OUT_FIELDS])


@functools.partial(jax.jit, static_argnums=(2, 3))
def _optical_program(tables: ScatteringTables, packed: Any, f: int, k: int) -> Any:
    """Optical properties of ``F`` materials at ``M`` shared energies.

    Run it inside ``jax.enable_x64(True)``.

    Parameters
    ----------
    tables : ScatteringTables
    packed : Array, float64, shape ``[F·(2K+3) + M]``
        One upload: ``F`` rows ``z[K] | counts[K] | mw | rho | n_electrons``, then
        the ``M`` energies (keV).
    f, k : int
        Static material and element counts (padded).

    Returns
    -------
    dict
        ``{"out": (F, 9, M) rows in OUT_FIELDS order, "electron_density": (F,),
        "ok": (F,)}``.
    """
    width = 2 * k + 3
    rows = packed[: f * width].reshape(f, width)
    energies = packed[f * width :]
    z = rows[:, :k].astype(jnp.int32)
    counts = rows[:, k : 2 * k]
    mw, rho, n_electrons = rows[:, 2 * k], rows[:, 2 * k + 1], rows[:, 2 * k + 2]

    def one(zi: Any, ci: Any, mwi: Any, rhoi: Any, nei: Any) -> Any:
        props, ok = optical_properties(tables, zi, ci, rhoi, mwi, nei, energies)
        return _pack(props), props.electron_density, ok

    out, electron_density, ok = jax.vmap(one)(z, counts, mw, rho, n_electrons)
    return {"out": out, "electron_density": electron_density, "ok": ok}


@jax.jit
def _tail_program(batch: Any, f1: Any, f2: Any, energies: Any) -> Any:
    """Optical properties from supplied f1/f2 (``calculate_properties_from_factors``).

    Run it inside ``jax.enable_x64(True)``.

    Parameters
    ----------
    batch : tuple
        ``(mw[F], rho[F], n_electrons[F])``.
    f1, f2 : Array, shape ``[F, M]``
        Supplied scattering-factor totals.
    energies : Array, shape ``[M]``
        keV.

    Returns
    -------
    dict
        As :func:`_optical_program`.
    """
    mw, rho, n_electrons = batch

    def one(f1i: Any, f2i: Any, mwi: Any, rhoi: Any, nei: Any) -> Any:
        props = derive(energies, f1i, f2i, rhoi, mwi, nei)
        return _pack(props), props.electron_density, _status(props, rhoi, mwi)

    out, electron_density, ok = jax.vmap(one)(f1, f2, mw, rho, n_electrons)
    return {"out": out, "electron_density": electron_density, "ok": ok}


@jax.jit
def _row_program(tables: ScatteringTables, z: Any, q: Any) -> Any:
    """f1 (row 0) and f2 (row 1) of table row ``z`` at ``q`` eV, NaN outside the table.

    Run it inside ``jax.enable_x64(True)``. ``z`` is a traced scalar, so every
    element shares one compiled program per query length.
    """
    return _row_pair(tables, z, q)


@jax.jit
def _field_program(tables: ScatteringTables, z: Any, field: Any, q: Any) -> Any:
    """f1 (``field=0``) or f2 (``field=1``) of table row ``z`` at ``q`` eV.

    Run it inside ``jax.enable_x64(True)``; ``z`` and ``field`` are traced scalars.
    """
    return _row_pair(tables, z, q)[field]


@jax.jit
def _scattering_program(
    tables: ScatteringTables,
    batch: Any,
    e_ev: Any,
    wavelength: Any,
    rho: Any,
    mw: Any,
) -> Any:
    """δ, β, f1 and f2 totals of one material at the caller's ``e_ev`` and ``wavelength``.

    Backs ``calculate_scattering_factors``, which takes energy and wavelength as
    separate arguments. Run it inside ``jax.enable_x64(True)``.

    Parameters
    ----------
    batch : tuple
        ``(z[K], counts[K])``; padding rows use ``z=0, counts=0``.
    e_ev, wavelength : Array, shape ``[M]``
        Energies in eV and wavelengths in metres.
    rho, mw : scalar
        Density (g/cm³) and molecular weight (g/mol); applied as in :func:`derive`.

    Returns
    -------
    Array, shape ``[4, M]``
        Rows δ, β, Σ counts·f1, Σ counts·f2.
    """
    z, counts = batch
    f = interpolate_f(tables, z, counts, e_ev)
    wave_factor = jnp.square(wavelength) * (SCATTERING_FACTOR * rho)
    return jnp.stack([(wave_factor * f[0]) / mw, (wave_factor * f[1]) / mw, f[0], f[1]])


@jax.jit
def _derived_program(args: Any) -> Any:
    """Clamped derived quantities for ``calculate_derived_quantities``.

    Run it inside ``jax.enable_x64(True)``.

    Parameters
    ----------
    args : tuple
        ``(wavelength_m[M], dispersion[M], absorption[M], rho, mw, n_electrons)``.

    Returns
    -------
    dict
        ``{"electron_density": scalar, "out": (4, M)}`` with rows critical angle,
        attenuation length, real and imaginary SLD.
    """
    wavelength, dispersion, absorption, rho, mw, n_electrons = args
    ed, critical, attenuation, re_sld, im_sld = _derived(
        wavelength, dispersion, absorption, rho, mw, n_electrons
    )
    return {
        "electron_density": ed,
        "out": jnp.stack([critical, attenuation, re_sld, im_sld]),
    }


# ---------------------------------------------------------------------------
# Standalone primitives (unclamped; they differ from the in-program rules)
# ---------------------------------------------------------------------------


def critical_angle_from_delta(delta: Any) -> Any:
    """Critical angle in degrees, ``sqrt(2δ)·180/π``; NaN for δ < 0."""
    return jnp.sqrt(2.0 * delta) * RADIANS_TO_DEGREES


def attenuation_length_from_beta(wavelength_angstrom: Any, beta: Any) -> Any:
    """1/e attenuation length in cm, ``λ/(4πβ)``; ``inf`` for β = 0."""
    wavelength_m = wavelength_angstrom * 1e-10
    length_m = wavelength_m / (4 * PI * beta)
    return length_m * 100


def transmission(thickness_cm: Any, attenuation_length_cm: Any) -> Any:
    """Transmission ``exp(-t/L)`` through ``thickness_cm`` of material."""
    return jnp.exp(-thickness_cm / attenuation_length_cm)


# ---------------------------------------------------------------------------
# Host encoding
# ---------------------------------------------------------------------------


def encode_formula(formula: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Encode a formula for the device programs.

    The molecular weight and electron count are accumulated as Python floats in
    the order of ``core._calculate_molecular_properties``, so they are
    bit-identical to the values the NumPy-era path produced.

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``"Ca5(PO4)3OH"``.

    Returns
    -------
    z_idx : ndarray, int32, shape ``[K]``
        Table rows (atomic number minus one), one per distinct element.
    counts : ndarray, float64, shape ``[K]``
    molecular_weight : float
        g/mol.
    number_of_electrons : float

    Raises
    ------
    FormulaError
        For an empty or malformed formula.
    UnknownElementError
        For a symbol without tabulated data (the legacy message, where the
        element appears twice, comes from the atomic-data lookup).

    Examples
    --------
    >>> from xraylabtool.calculators.optics import encode_formula
    >>> z, counts, mw, ne = encode_formula("SiO2")
    >>> z.tolist(), counts.tolist(), ne
    ([13, 7], [1.0, 2.0], 30.0)
    """
    symbols, counts = parse_formula(formula)
    bulk = get_bulk_atomic_data_fast(tuple(symbols))
    molecular_weight = 0.0
    number_of_electrons = 0.0
    for symbol, count in zip(symbols, counts, strict=False):
        data = bulk[symbol]
        molecular_weight += count * data["atomic_weight"]
        number_of_electrons += data["atomic_number"] * count
    z_idx = [element_index(s) for s in symbols]
    return (
        np.asarray(z_idx, dtype=np.int32),
        np.asarray(counts, dtype=np.float64),
        molecular_weight,
        number_of_electrons,
    )
