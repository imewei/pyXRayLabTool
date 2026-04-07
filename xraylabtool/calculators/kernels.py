"""JIT-compiled math kernels and scattering factor calculation functions."""

# ruff: noqa: RUF002

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xraylabtool.backend import ops

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        EnergyArray,
        FloatLike,
        InterpolatorProtocol,
        OpticalConstantArray,
        WavelengthArray,
    )


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
            import jax  # type: ignore[import-not-found]

            _jit_cache["scatter"] = jax.jit(
                _scattering_math_kernel_raw, static_argnums=(4,)
            )
        return _jit_cache["scatter"]
    return _scattering_math_kernel_raw


def _get_derived_kernel() -> Any:
    """Return JIT-compiled kernel when JAX is active, raw function otherwise."""
    from xraylabtool.backend.array_ops import JaxBackend, _backend

    if isinstance(_backend, JaxBackend):
        if "derived" not in _jit_cache:
            import jax  # type: ignore[import-not-found]

            _jit_cache["derived"] = jax.jit(
                _derived_quantities_kernel_raw, static_argnums=(3,)
            )
        return _jit_cache["derived"]
    return _derived_quantities_kernel_raw


def calculate_scattering_factors(
    energy_ev: EnergyArray,
    wavelength: WavelengthArray,
    mass_density: FloatLike,
    molecular_weight: FloatLike,
    element_data: list[tuple[float, InterpolatorProtocol, InterpolatorProtocol]],
) -> tuple[
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
]:
    """
    Optimized vectorized calculation of X-ray scattering factors and properties.

    This function performs the core calculation of dispersion, absorption, and total
    scattering factors for a material based on its elemental composition.
    Interpolation runs in Python; the linear algebra is JIT-compiled when JAX is active.

    Args:
        energy_ev: X-ray energies in eV (numpy array)
        wavelength: Corresponding wavelengths in meters (numpy array)
        mass_density: Material density in g/cm³
        molecular_weight: Molecular weight in g/mol
        element_data: List of tuples (count, f1_interp, f2_interp) for each element

    Returns:
        Tuple of (dispersion, absorption, f1_total, f2_total) arrays

    Mathematical Background:
    The dispersion and absorption coefficients are calculated using:
    - δ = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f1ᵢ) / M  # noqa: RUF002
    - β = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f2ᵢ) / M  # noqa: RUF002

    Where:
    - λ: X-ray wavelength
    - rₑ: Thomson scattering length
    - ρ: Mass density
    - Nₐ: Avogadro's number
    - nᵢ: Number of atoms of element i
    - f1ᵢ, f2ᵢ: Atomic scattering factors for element i
    - M: Molecular weight
    """
    from xraylabtool.constants import SCATTERING_FACTOR

    n_energies = len(energy_ev)
    n_elements = len(element_data)

    # Handle empty element data case
    if n_elements == 0:
        z = ops.zeros(n_energies, dtype=ops.float64)
        return z, z, z, z

    common_factor = SCATTERING_FACTOR * mass_density / molecular_weight
    wave_sq = ops.square(wavelength)

    # Evaluate interpolators (Python-side, outside JIT boundary)
    f1_rows = []
    f2_rows = []
    count_list = []
    for count, f1_interp, f2_interp in element_data:
        f1_rows.append(ops.asarray(f1_interp(energy_ev), dtype=ops.float64))
        f2_rows.append(ops.asarray(f2_interp(energy_ev), dtype=ops.float64))
        count_list.append(float(count))

    f1_matrix = ops.asarray(f1_rows, dtype=ops.float64)
    f2_matrix = ops.asarray(f2_rows, dtype=ops.float64)
    counts = ops.asarray(count_list, dtype=ops.float64)

    # Dispatch to JIT-compiled math kernel
    kernel = _get_scattering_kernel()
    return kernel(f1_matrix, f2_matrix, counts, wave_sq, common_factor)


def calculate_derived_quantities(
    wavelength: WavelengthArray,
    dispersion: OpticalConstantArray,
    absorption: OpticalConstantArray,
    mass_density: FloatLike,
    molecular_weight: FloatLike,
    number_of_electrons: FloatLike,
) -> tuple[
    float,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
]:
    """
    Calculate derived X-ray optical quantities from dispersion and absorption.

    Args:
        wavelength: X-ray wavelengths in meters (numpy array)
        dispersion: Dispersion coefficients δ (numpy array)
        absorption: Absorption coefficients β (numpy array)
        mass_density: Material density in g/cm³
        molecular_weight: Molecular weight in g/mol
        number_of_electrons: Total electrons per molecule

    Returns:
        Tuple of (electron_density, critical_angle, attenuation_length, re_sld, im_sld)
        - electron_density: Electron density in electrons/Å³ (scalar)
        - critical_angle: Critical angle in degrees (numpy array)
        - attenuation_length: Attenuation length in cm (numpy array)
        - re_sld: Real part of SLD in Å⁻² (numpy array)
        - im_sld: Imaginary part of SLD in Å⁻² (numpy array)
    """
    from xraylabtool.constants import AVOGADRO, PI

    # Numerical stability checks must run outside any JIT region because they
    # branch on concrete array values (ops.any returns a Python bool here).
    # These guards are intentionally kept as host-side checks.
    if not ops.all(ops.isfinite(dispersion)) or not ops.all(ops.isfinite(absorption)):
        raise ValueError(
            "Non-finite values (NaN or Inf) detected in dispersion or absorption coefficients"
        )

    if ops.any(ops.asarray(dispersion) < 0):
        raise ValueError("Negative dispersion values detected (physically unrealistic)")

    # Calculate electron density (electrons per unit volume).
    electron_density = float(
        1e6
        * float(mass_density)
        / float(molecular_weight)
        * AVOGADRO
        * float(number_of_electrons)
        / 1e30
    )

    # Dispatch to JIT-compiled math kernel for array computations
    kernel = _get_derived_kernel()
    critical_angle, attenuation_length, re_sld, im_sld = kernel(
        wavelength, dispersion, absorption, PI
    )

    return electron_density, critical_angle, attenuation_length, re_sld, im_sld
