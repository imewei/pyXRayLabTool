"""
Derived quantities calculator for X-ray optical properties.

This module contains functions for calculating derived X-ray properties such as
critical angles, attenuation lengths, and other useful quantities. Each takes
host arrays and returns NumPy arrays; the arithmetic runs in one JAX device
program call (one upload, one fetch).
"""

# ruff: noqa: PLC0415, RUF002

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from xraylabtool.constants import PI, SCATTERING_FACTOR

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        EnergyArray,
        FloatLike,
        InterpolatorProtocol,
        OpticalConstantArray,
        WavelengthArray,
    )


def _map(fn: Any, *arrays: Any) -> Any:
    """Apply the elementwise device function ``fn`` to broadcast host arrays."""
    from xraylabtool.calculators import host

    out = host.run_map(fn, *arrays)
    # ``[()]`` turns a 0-d result into a NumPy scalar and leaves arrays alone.
    return out[()] if isinstance(out, np.ndarray) else out


def calculate_critical_angle(dispersion_delta: np.ndarray) -> np.ndarray:
    """
    Calculate critical angle for total external reflection.

    Args:
        dispersion_delta: Dispersion coefficient δ

    Returns:
        Critical angle in degrees, ``sqrt(2δ)·180/π``; NaN where δ < 0
    """
    from xraylabtool.calculators.optics import critical_angle_from_delta

    return _map(critical_angle_from_delta, dispersion_delta)  # type: ignore[no-any-return]


def calculate_attenuation_length(
    wavelength_angstrom: np.ndarray, absorption_beta: np.ndarray
) -> np.ndarray:
    """
    Calculate X-ray attenuation length (1/e length).

    Args:
        wavelength_angstrom: X-ray wavelength in Angstroms
        absorption_beta: Absorption coefficient β

    Returns:
        Attenuation length in centimeters; ``inf`` where β = 0
    """
    from xraylabtool.calculators.optics import attenuation_length_from_beta

    return _map(attenuation_length_from_beta, wavelength_angstrom, absorption_beta)  # type: ignore[no-any-return]


def calculate_transmission(
    thickness_cm: float, attenuation_length_cm: np.ndarray
) -> np.ndarray:
    """
    Calculate transmission through a material thickness.

    Args:
        thickness_cm: Material thickness in centimeters
        attenuation_length_cm: Attenuation length in centimeters

    Returns:
        Transmission coefficient (0-1)
    """
    from xraylabtool.calculators.optics import transmission

    return _map(transmission, thickness_cm, attenuation_length_cm)  # type: ignore[no-any-return]


def _scattering_length_density(
    dispersion_delta: Any, absorption_beta: Any, wavelength_angstrom: Any
) -> tuple[Any, Any]:
    return (
        2.0 * PI * dispersion_delta / wavelength_angstrom**2,
        2.0 * PI * absorption_beta / wavelength_angstrom**2,
    )


def calculate_scattering_length_density(
    dispersion_delta: np.ndarray,
    absorption_beta: np.ndarray,
    wavelength_angstrom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate real and imaginary scattering length densities.

    Args:
        dispersion_delta: Dispersion coefficient δ
        absorption_beta: Absorption coefficient β
        wavelength_angstrom: X-ray wavelength in Angstroms

    Returns:
        Tuple of (real_sld, imaginary_sld) in units of Å⁻²
    """
    # SLD = 2π·(δ or β)/λ²; with λ in Å this yields Å⁻² directly. The real part
    # is positive for δ > 0.
    return _map(  # type: ignore[no-any-return]
        _scattering_length_density,
        dispersion_delta,
        absorption_beta,
        wavelength_angstrom,
    )


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
    Calculate X-ray scattering factors and properties of one material.

    This function performs the core calculation of dispersion, absorption, and total
    scattering factors for a material based on its elemental composition. The PCHIP
    lookup and the arithmetic run in one device program call.

    Args:
        energy_ev: X-ray energies in eV (numpy array)
        wavelength: Corresponding wavelengths in meters (numpy array)
        mass_density: Material density in g/cm³
        molecular_weight: Molecular weight in g/mol
        element_data: List of tuples (count, f1_interp, f2_interp) for each
            element; the interpolators must come from
            ``create_scattering_factor_interpolators``

    Returns:
        Tuple of (dispersion, absorption, f1_total, f2_total) NumPy arrays;
        NaN where an energy lies outside an element's table.

    Raises:
        TypeError: If an interpolator is not a ``TableInterpolator`` (an arbitrary
            callable is rejected before any entry is evaluated)
        ValueError: If an f1/f2 pair does not belong to one element

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
    from xraylabtool.calculators.cache import TableInterpolator

    n_energies = len(energy_ev)

    # Handle empty element data case
    if len(element_data) == 0:
        return (
            np.zeros(n_energies),
            np.zeros(n_energies),
            np.zeros(n_energies),
            np.zeros(n_energies),
        )

    for _count, f1_interp, f2_interp in element_data:
        if not (
            isinstance(f1_interp, TableInterpolator)
            and isinstance(f2_interp, TableInterpolator)
        ):
            raise TypeError(
                "element_data interpolators must come from "
                "create_scattering_factor_interpolators (TableInterpolator), got "
                f"{type(f1_interp).__name__} and {type(f2_interp).__name__}"
            )
        if f1_interp.field != 0 or f2_interp.field != 1 or f1_interp.z != f2_interp.z:
            raise ValueError(
                "each entry needs the f1 and f2 interpolators of one element"
            )

    from xraylabtool.calculators import host

    factor = SCATTERING_FACTOR * mass_density / molecular_weight
    got = host.run_scattering(
        [entry[1].z for entry in element_data],  # type: ignore[attr-defined]
        [float(entry[0]) for entry in element_data],
        energy_ev,
        wavelength,
        float(factor),
    )
    dispersion, absorption, f1_total, f2_total = (row.copy() for row in got)
    return dispersion, absorption, f1_total, f2_total


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

    Negative dispersion is physical near absorption edges (anomalous dispersion):
    the critical angle is 0 there, and β is clamped at ``1e-30`` in the
    attenuation length. This differs from the standalone helpers
    :func:`calculate_critical_angle` and :func:`calculate_attenuation_length`,
    which return NaN and ``inf`` at those points.

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

    Raises:
        ValueError: If dispersion or absorption holds NaN or Inf
    """
    from xraylabtool.calculators import host

    wl, delta, beta = np.broadcast_arrays(
        *(np.asarray(a, dtype=np.float64) for a in (wavelength, dispersion, absorption))
    )
    # The finiteness guard branches on concrete values, so it stays on the host.
    if not np.isfinite(delta).all() or not np.isfinite(beta).all():
        raise ValueError(
            "Non-finite values (NaN or Inf) detected in dispersion or absorption coefficients"
        )
    electron_density, out = host.run_derived(
        wl.ravel(),
        delta.ravel(),
        beta.ravel(),
        float(mass_density),
        float(molecular_weight),
        float(number_of_electrons),
    )
    critical_angle, attenuation_length, re_sld, im_sld = (
        row.reshape(wl.shape).copy() for row in out
    )
    return electron_density, critical_angle, attenuation_length, re_sld, im_sld
