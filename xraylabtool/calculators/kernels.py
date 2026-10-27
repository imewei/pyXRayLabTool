"""Legacy math kernels of the NumPy/JAX backend twin; the calculators no longer call them.

The public ``calculate_scattering_factors`` and ``calculate_derived_quantities``
now live in :mod:`xraylabtool.calculators.derived_quantities` and run on the
device program.
"""

from __future__ import annotations

from typing import Any

from xraylabtool.backend import ops


def _scattering_math_kernel(
    f1_matrix: Any,
    f2_matrix: Any,
    counts: Any,
    wave_sq: Any,
    common_factor: float,
) -> tuple[Any, Any, Any, Any]:
    """JIT-compiled pure math kernel for scattering factor computation.

    This function contains no Python side effects, interpolation calls, or
    data-dependent branching — safe for @jax.jit compilation.
    """
    f1_weighted = f1_matrix * counts.reshape(-1, 1)
    f2_weighted = f2_matrix * counts.reshape(-1, 1)

    f1_total = ops.sum(f1_weighted, axis=0)
    f2_total = ops.sum(f2_weighted, axis=0)

    wave_factor = wave_sq * common_factor
    dispersion = wave_factor * f1_total
    absorption = wave_factor * f2_total

    return dispersion, absorption, f1_total, f2_total


def _derived_quantities_kernel(
    wavelength: Any,
    dispersion: Any,
    absorption: Any,
    pi: float,
) -> tuple[Any, Any, Any, Any]:
    """JIT-compiled pure math kernel for derived X-ray quantities.

    Computes critical angle, attenuation length, and SLD from dispersion
    and absorption arrays. No validation or branching — pure array math.
    """
    critical_angle = ops.sqrt(ops.maximum(2.0 * dispersion, 0.0)) * (180.0 / pi)

    absorption_safe = ops.maximum(absorption, 1e-30)
    attenuation_length = wavelength / absorption_safe / (4 * pi) * 1e2

    wavelength_sq = ops.square(wavelength)
    sld_factor = 2 * pi / 1e20

    re_sld = dispersion * sld_factor / wavelength_sq
    im_sld = absorption * sld_factor / wavelength_sq

    return critical_angle, attenuation_length, re_sld, im_sld


# Keep raw functions and lazily apply JIT when JAX backend is active.
# This ensures `set_backend('jax')` at runtime also gets JIT-compiled kernels.
_scattering_math_kernel_raw = _scattering_math_kernel
_derived_quantities_kernel_raw = _derived_quantities_kernel
_jit_cache: dict[str, Any] = {}


def _get_scattering_kernel() -> Any:
    """Return JIT-compiled kernel when JAX is active, raw function otherwise."""
    from xraylabtool.backend.array_ops import JaxBackend, _backend

    if isinstance(_backend, JaxBackend):
        if "scatter" not in _jit_cache:
            import jax

            # common_factor (arg 4) must stay a traced argument: marking it static
            # would recompile the kernel for every distinct density/material.
            _jit_cache["scatter"] = jax.jit(_scattering_math_kernel_raw)
        return _jit_cache["scatter"]
    return _scattering_math_kernel_raw


def _get_derived_kernel() -> Any:
    """Return JIT-compiled kernel when JAX is active, raw function otherwise."""
    from xraylabtool.backend.array_ops import JaxBackend, _backend

    if isinstance(_backend, JaxBackend):
        if "derived" not in _jit_cache:
            import jax

            _jit_cache["derived"] = jax.jit(
                _derived_quantities_kernel_raw, static_argnums=(3,)
            )
        return _jit_cache["derived"]
    return _derived_quantities_kernel_raw
