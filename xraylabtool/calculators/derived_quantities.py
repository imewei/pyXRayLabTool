"""
Derived quantities calculator for X-ray optical properties.

This module contains functions for calculating derived X-ray properties such as
critical angles, attenuation lengths, and other useful quantities.
"""

import numpy as np

from xraylabtool.constants import PI, RADIANS_TO_DEGREES


def calculate_critical_angle(dispersion_delta: np.ndarray) -> np.ndarray:
    """
    Calculate critical angle for total external reflection.

    Args:
        dispersion_delta: Dispersion coefficient δ

    Returns:
        Critical angle in degrees
    """
    theta_c_rad = np.sqrt(2.0 * dispersion_delta)
    return theta_c_rad * RADIANS_TO_DEGREES


def calculate_attenuation_length(
    wavelength_angstrom: np.ndarray, absorption_beta: np.ndarray
) -> np.ndarray:
    """
    Calculate X-ray attenuation length (1/e length).

    Args:
        wavelength_angstrom: X-ray wavelength in Angstroms
        absorption_beta: Absorption coefficient β

    Returns:
        Attenuation length in centimeters
    """
    wavelength_m = wavelength_angstrom * 1e-10  # Convert to meters
    length_m = wavelength_m / (4 * PI * absorption_beta)
    return length_m * 100  # Convert to centimeters


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
    return np.exp(-thickness_cm / attenuation_length_cm)


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
    wavelength_ang = wavelength_angstrom
    real_sld = -dispersion_delta / (PI * wavelength_ang**2) * 1e20  # Convert to Å⁻²
    imaginary_sld = absorption_beta / (PI * wavelength_ang**2) * 1e20  # Convert to Å⁻²

    return real_sld, imaginary_sld
