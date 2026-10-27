"""Pure ``jax.numpy`` physics behind the :mod:`xraylabtool.utils` wrappers.

The wrappers in :mod:`xraylabtool.utils` validate on the host, run these
functions through :func:`xraylabtool.calculators.host.run_map` or
:func:`xraylabtool.calculators.host.execute` and return Python floats or NumPy
arrays. Every function is jit/vmap/grad-compatible and assumes float64 (the
caller ran :func:`xraylabtool.backend.runtime.ensure_x64`); ``units``,
``method`` and ``window_size`` are static (bound with ``functools.partial``).

Unit conversion, Bragg and lattice formulas are elementwise. The signal helpers
take a series padded to a shape bucket plus its true length ``n``; indices and
reductions are masked, so padding only changes a result through the order in
which XLA sums the masked terms (about 1e-16 relative, 1e-11 at points sitting on
the mean for ``standard``). A given length always maps to one bucket, so results
are deterministic.

This module imports ``jax`` at the top, so ``utils.py`` imports it inside each
wrapper. Host by design (spec section 9): ``find_peaks``, ``progress_bar``,
``save_processed_data``, ``parse_formula``.

References
----------
Design: ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md`` §10.
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from xraylabtool.constants import ELEMENT_CHARGE, PLANCK, SPEED_OF_LIGHT

__all__ = [
    "angle_from_q",
    "bragg_angle",
    "d_spacing_cubic",
    "d_spacing_orthorhombic",
    "d_spacing_tetragonal",
    "energy_to_wavelength",
    "linear_background",
    "linear_background_columns",
    "normalize_intensity",
    "polynomial_background",
    "q_from_angle",
    "smooth_data",
    "wavelength_to_energy",
]

_PLANCK = float(PLANCK)
_LIGHT = float(SPEED_OF_LIGHT)
_CHARGE = float(ELEMENT_CHARGE)
_TO_METRES = {"angstrom": 1e-10, "nm": 1e-9, "m": None}


def wavelength_to_energy(wavelength: jax.Array, *, units: str) -> jax.Array:
    """Energy in keV of a wavelength in ``units`` (``E = hc/λ``)."""
    scale = _TO_METRES[units]
    wavelength_m = wavelength if scale is None else wavelength * scale
    return ((_PLANCK * _LIGHT) / wavelength_m) / (_CHARGE * 1000)


def energy_to_wavelength(energy: jax.Array, *, units: str) -> jax.Array:
    """Wavelength in ``units`` of an energy in keV (``λ = hc/E``)."""
    scale = _TO_METRES[units]
    wavelength_m = (_PLANCK * _LIGHT) / (energy * _CHARGE * 1000)
    return wavelength_m if scale is None else wavelength_m / scale


def bragg_angle(
    d_spacing: jax.Array, wavelength: jax.Array, order: jax.Array
) -> jax.Array:
    """Bragg angle in degrees, ``θ = asin(nλ / 2d)``."""
    return jnp.degrees(jnp.arcsin((order * wavelength) / (2 * d_spacing)))


def d_spacing_cubic(
    h: jax.Array, k: jax.Array, miller_l: jax.Array, a: jax.Array
) -> jax.Array:
    """Cubic d-spacing in Å."""
    return a / jnp.sqrt(h**2 + k**2 + miller_l**2)


def d_spacing_tetragonal(
    h: jax.Array, k: jax.Array, miller_l: jax.Array, a: jax.Array, c: jax.Array
) -> jax.Array:
    """Tetragonal d-spacing in Å."""
    return 1 / jnp.sqrt((h**2 + k**2) / a**2 + miller_l**2 / c**2)


def d_spacing_orthorhombic(
    h: jax.Array,
    k: jax.Array,
    miller_l: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
) -> jax.Array:
    """Orthorhombic d-spacing in Å."""
    return 1 / jnp.sqrt(h**2 / a**2 + k**2 / b**2 + miller_l**2 / c**2)


def q_from_angle(two_theta: jax.Array, wavelength: jax.Array) -> jax.Array:
    """Momentum transfer ``q = 4π sin θ / λ`` in Å⁻¹ from 2θ in degrees."""
    return (4 * jnp.pi * jnp.sin(jnp.radians(two_theta / 2))) / wavelength


def angle_from_q(q: jax.Array, wavelength: jax.Array) -> jax.Array:
    """Scattering angle 2θ in degrees from ``q`` in Å⁻¹."""
    return 2 * jnp.degrees(jnp.arcsin((q * wavelength) / (4 * jnp.pi)))


def _origin(y: jax.Array) -> jax.Array:
    """The first sample, or 0 when it is not finite (so NaN/inf stay local)."""
    return jnp.where(jnp.isfinite(y[0]), y[0], 0.0)


def smooth_data(y: jax.Array, n: jax.Array, *, window_size: int) -> jax.Array:
    """Moving average with edge replication over the first ``n`` samples of ``y``.

    A window of at least ``n`` samples returns the mean everywhere. Entries of
    the result at or beyond ``n`` are meaningless.
    """
    size = y.shape[0]
    half = window_size // 2
    index = jnp.clip(jnp.arange(size + 2 * half) - half, 0, n - 1)
    kernel = jnp.ones(window_size, dtype=y.dtype) / window_size
    # Windowed sums are local, so they stay on the raw values (a shift would round
    # small values next to a large first sample). The global mean is the one sum
    # that cancels: it uses offsets from the first sample, which keeps a constant
    # series exactly constant.
    moving = jnp.convolve(y[index], kernel, mode="valid")[:size]
    first = _origin(y)
    mean = first + jnp.sum(jnp.where(jnp.arange(size) < n, y - first, 0.0)) / n
    return jnp.where(window_size >= n, mean, moving)


def linear_background(y: jax.Array, n: jax.Array) -> jax.Array:
    """``y`` minus the straight line through its first and last (``n``-th) samples."""
    position = jnp.arange(y.shape[0])
    last = n - 1
    step = jnp.where(n > 1, (y[last] - y[0]) / jnp.maximum(last, 1), 0.0)
    # The barrier keeps XLA from fusing the multiply and add into one rounding.
    ramp = lax.optimization_barrier(position * step) + y[0]
    return y - jnp.where(position == last, y[last], ramp)


# One series per column: ``y`` is ``[width, columns]``, ``n`` the true row count.
linear_background_columns = jax.vmap(linear_background, in_axes=(1, None), out_axes=1)


def polynomial_background(
    x: jax.Array, y: jax.Array, n: jax.Array, coefficients: jax.Array
) -> jax.Array:
    """``y`` minus a polynomial (highest power first) evaluated by Horner's rule.

    ``n`` is accepted so the signature matches the other series programs. The
    coefficients come from the host (3-point fit, see ``utils.background_subtraction``).
    """
    del n
    background = jnp.zeros_like(x)
    for coefficient in coefficients:
        background = lax.optimization_barrier(background * x) + coefficient
    return y - background


def normalize_intensity(y: jax.Array, n: jax.Array, *, method: str) -> jax.Array:
    """``y`` divided by its maximum, trapezoid area, or standardised (first ``n`` samples)."""
    live = jnp.arange(y.shape[0]) < n
    if method == "max":
        return y / jnp.max(jnp.where(live, y, -jnp.inf))
    if method == "area":
        pair = jnp.where(live[1:], (y[1:] + y[:-1]) / 2.0, 0.0)
        return y / jnp.sum(pair)
    # Deviations from the first sample: a constant series has exactly zero
    # deviation (0/0 = nan, as NumPy) and a large offset does not cancel the sums.
    shifted = y - _origin(y)
    mean = jnp.sum(jnp.where(live, shifted, 0.0)) / n
    deviation = shifted - mean
    std = jnp.sqrt(jnp.sum(jnp.where(live, deviation**2, 0.0)) / n)
    return deviation / std
