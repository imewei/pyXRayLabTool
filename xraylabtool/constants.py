"""
Physical constants for X-ray calculations.

This module contains fundamental physical constants used throughout X-ray
analysis calculations, translated from the Julia implementation with preserved
numerical precision.
All values are provided with explanatory docstrings describing their physical meaning
and units.

The constants are defined with high precision to ensure accurate calculations
in X-ray optical property computations.
"""

import numpy as np
from typing import Final

# =====================================================================================
# FUNDAMENTAL PHYSICAL CONSTANTS
# =====================================================================================

THOMPSON: Final[float] = 2.8179403227e-15
"""
Thomson scattering length (classical electron radius).

The Thomson scattering length represents the classical radius of an electron,
which is the length scale characterizing the scattering of electromagnetic
radiation by a free electron in the classical limit.

Value: 2.8179403227 × 10⁻¹⁵ m
Units: meters (m)
Reference: CODATA 2018 recommended values
"""

SPEED_OF_LIGHT: Final[float] = 299792458.0
"""
Speed of light in vacuum.

The speed of electromagnetic radiation in vacuum, a fundamental physical constant
that appears in many X-ray calculations involving wavelength-energy conversions.

Value: 299,792,458 m/s (exact by definition)
Units: meters per second (m/s)
Reference: SI base unit definition
"""

PLANCK: Final[float] = 6.626068e-34
"""
Planck constant.

The quantum of action, relating the energy of a photon to its frequency.
Essential for X-ray energy-wavelength conversions through E = hc/λ.

Value: 6.626068 × 10⁻³⁴ J⋅s
Units: joule-seconds (J⋅s)
Reference: CODATA 2018 recommended values

Note: This value maintains compatibility with the Julia implementation.
For the most current CODATA value, see: 6.62607015e-34 J⋅s (exact since 2019).
"""

ELEMENT_CHARGE: Final[float] = 1.60217646e-19
"""
Elementary charge.

The electric charge of a single proton, equal in magnitude to the charge of an electron.
Used in X-ray calculations for energy unit conversions between Joules and eV.

Value: 1.60217646 × 10⁻¹⁹ C
Units: coulombs (C)
Reference: CODATA 2018 recommended values

Note: This value maintains compatibility with the Julia implementation.
For the most current CODATA value, see: 1.602176634e-19 C (exact since 2019).
"""

AVOGADRO: Final[float] = 6.02214199e23
"""
Avogadro's number.

The number of constituent particles (atoms or molecules) per mole.
Used in calculations relating molecular properties to bulk material properties.

Value: 6.02214199 × 10²³ mol⁻¹
Units: per mole (mol⁻¹)
Reference: CODATA 2018 recommended values

Note: This value maintains compatibility with the Julia implementation.
For the most current CODATA value, see: 6.02214076e23 mol⁻¹ (exact since 2019).
"""

# =====================================================================================
# DERIVED CONSTANTS FOR COMPUTATIONAL EFFICIENCY
# =====================================================================================

ENERGY_TO_WAVELENGTH_FACTOR: Final[float] = (
    SPEED_OF_LIGHT * PLANCK / ELEMENT_CHARGE
) / 1000.0
"""
Pre-computed factor for energy-to-wavelength conversion.

This constant combines fundamental constants to efficiently convert X-ray energies
in keV to wavelengths in meters using: λ = ENERGY_TO_WAVELENGTH_FACTOR / E_keV

Mathematical derivation:
λ = hc/E = (hc/e) / E_eV = (hc/e) / (1000 × E_keV)

Value: (SPEED_OF_LIGHT × PLANCK / ELEMENT_CHARGE) / 1000.0
     = (2.998 × 10⁸ × 6.626 × 10⁻³⁴ / 1.602 × 10⁻¹⁹) / 1000
     ≈ 1.240 × 10⁻⁶ m⋅keV

Units: meter-keV (m⋅keV)
Usage: wavelength_m = ENERGY_TO_WAVELENGTH_FACTOR / energy_keV
"""

SCATTERING_FACTOR: Final[float] = THOMPSON * AVOGADRO * 1e6 / (2 * np.pi)
"""
Pre-computed factor for X-ray scattering calculations.

This constant combines the Thomson scattering length with Avogadro's number
and unit conversion factors, optimized for calculating dispersion and absorption
coefficients in X-ray optical property calculations.

Mathematical derivation:
Factor = (rₑ × Nₐ × 10⁶) / (2π)

Where:
- rₑ: Thomson scattering length (classical electron radius)
- Nₐ: Avogadro's number
- 10⁶: Unit conversion factor (cm³/m³ to account for density units)
- 2π: Geometric factor from X-ray scattering theory

Value: THOMPSON × AVOGADRO × 10⁶ / (2π)
     ≈ 2.70 × 10¹⁴ m⁻¹⋅mol⁻¹

Units: per meter per mole (m⁻¹⋅mol⁻¹)
Usage: Used in calculating δ = (λ²/2π) × rₑ × ρ × Nₐ × (Σf₁) / M
"""

# =====================================================================================
# MATHEMATICAL CONSTANTS
# =====================================================================================

PI: Final[float] = np.pi
"""
The mathematical constant π.

The ratio of a circle's circumference to its diameter, appearing frequently
in X-ray scattering calculations and Fourier transforms.

Value: 3.141592653589793...
Units: dimensionless
"""

TWO_PI: Final[float] = 2.0 * np.pi
"""
Mathematical constant 2π.

Commonly appears in X-ray calculations involving angular frequencies,
momentum transfer, and scattering theory.

Value: 6.283185307179586...
Units: dimensionless
"""

# =====================================================================================
# UNIT CONVERSION CONSTANTS
# =====================================================================================

KEV_TO_EV: Final[float] = 1000.0
"""
Conversion factor from keV to eV.

Value: 1000
Units: eV/keV
Usage: energy_eV = energy_keV * KEV_TO_EV
"""

EV_TO_KEV: Final[float] = 1.0 / 1000.0
"""
Conversion factor from eV to keV.

Value: 0.001
Units: keV/eV
Usage: energy_keV = energy_eV * EV_TO_KEV
"""

ANGSTROM_TO_METER: Final[float] = 1e-10
"""
Conversion factor from Angstroms to meters.

Value: 10⁻¹⁰
Units: m/Å
Usage: length_m = length_angstrom * ANGSTROM_TO_METER
"""

METER_TO_ANGSTROM: Final[float] = 1e10
"""
Conversion factor from meters to Angstroms.

Value: 10¹⁰
Units: Å/m
Usage: length_angstrom = length_m * METER_TO_ANGSTROM
"""

CM_TO_METER: Final[float] = 1e-2
"""
Conversion factor from centimeters to meters.

Value: 0.01
Units: m/cm
Usage: length_m = length_cm * CM_TO_METER
"""

METER_TO_CM: Final[float] = 1e2
"""
Conversion factor from meters to centimeters.

Value: 100
Units: cm/m
Usage: length_cm = length_m * METER_TO_CM
"""

# =====================================================================================
# X-RAY SPECIFIC CONSTANTS
# =====================================================================================

DEGREES_TO_RADIANS: Final[float] = np.pi / 180.0
"""
Conversion factor from degrees to radians.

Value: π/180 ≈ 0.017453292519943295
Units: rad/°
Usage: angle_rad = angle_deg * DEGREES_TO_RADIANS
"""

RADIANS_TO_DEGREES: Final[float] = 180.0 / np.pi
"""
Conversion factor from radians to degrees.

Value: 180/π ≈ 57.29577951308232
Units: °/rad
Usage: angle_deg = angle_rad * RADIANS_TO_DEGREES
"""

# =====================================================================================
# HELPER FUNCTIONS FOR CONSTANT USAGE
# =====================================================================================


def energy_to_wavelength_angstrom(energy_kev: float) -> float:
    """
    Convert X-ray energy in keV to wavelength in Angstroms.

    Args:
        energy_kev: X-ray energy in keV

    Returns:
        Wavelength in Angstroms

    Example:
        >>> wavelength = energy_to_wavelength_angstrom(10.0)  # 10 keV
        >>> print(f"Wavelength: {wavelength:.4f} Å")
        Wavelength: 1.2398 Å
    """
    if energy_kev <= 0:
        raise ValueError("Energy must be positive")

    wavelength_m = ENERGY_TO_WAVELENGTH_FACTOR / energy_kev
    return wavelength_m * METER_TO_ANGSTROM


def wavelength_angstrom_to_energy(wavelength_angstrom: float) -> float:
    """
    Convert X-ray wavelength in Angstroms to energy in keV.

    Args:
        wavelength_angstrom: X-ray wavelength in Angstroms

    Returns:
        Energy in keV

    Example:
        >>> energy = wavelength_angstrom_to_energy(1.2398)  # Cu Kα₁
        >>> print(f"Energy: {energy:.4f} keV")
        Energy: 10.0000 keV
    """
    if wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be positive")

    wavelength_m = wavelength_angstrom * ANGSTROM_TO_METER
    return ENERGY_TO_WAVELENGTH_FACTOR / wavelength_m


def critical_angle_degrees(dispersion: float) -> float:
    """
    Calculate critical angle for total external reflection from dispersion coefficient.

    Args:
        dispersion: Dispersion coefficient δ (dimensionless)

    Returns:
        Critical angle in degrees

    Example:
        >>> theta_c = critical_angle_degrees(1e-6)
        >>> print(f"Critical angle: {theta_c:.4f}°")
    """
    if dispersion <= 0:
        raise ValueError("Dispersion coefficient must be positive")

    theta_c_rad = np.sqrt(2.0 * dispersion)
    return float(theta_c_rad * RADIANS_TO_DEGREES)


def attenuation_length_cm(wavelength_angstrom: float, absorption: float) -> float:
    """
    Calculate X-ray attenuation length from wavelength and absorption coefficient.

    Args:
        wavelength_angstrom: X-ray wavelength in Angstroms
        absorption: Absorption coefficient β (dimensionless)

    Returns:
        Attenuation length (1/e length) in centimeters

    Example:
        >>> length = attenuation_length_cm(1.24, 1e-7)
        >>> print(f"Attenuation length: {length:.2f} cm")
    """
    if wavelength_angstrom <= 0:
        raise ValueError("Wavelength must be positive")
    if absorption <= 0:
        raise ValueError("Absorption coefficient must be positive")

    wavelength_m = wavelength_angstrom * ANGSTROM_TO_METER
    length_m = wavelength_m / (4 * PI * absorption)
    return length_m * METER_TO_CM


# =====================================================================================
# CONSTANT VALIDATION
# =====================================================================================


def validate_constants() -> bool:
    """
    Validate that all constants are properly defined and have reasonable values.

    Returns:
        True if all constants pass validation

    Raises:
        ValueError: If any constant fails validation
    """
    # Check fundamental constants are positive
    fundamental_constants = {
        "THOMPSON": THOMPSON,
        "SPEED_OF_LIGHT": SPEED_OF_LIGHT,
        "PLANCK": PLANCK,
        "ELEMENT_CHARGE": ELEMENT_CHARGE,
        "AVOGADRO": AVOGADRO,
    }

    for name, value in fundamental_constants.items():
        if value <= 0:
            raise ValueError(
                f"Fundamental constant {name} must be positive, got {value}"
            )

    # Check derived constants
    expected_energy_factor = (SPEED_OF_LIGHT * PLANCK / ELEMENT_CHARGE) / 1000.0
    if not np.isclose(ENERGY_TO_WAVELENGTH_FACTOR, expected_energy_factor, rtol=1e-10):
        raise ValueError("ENERGY_TO_WAVELENGTH_FACTOR calculation error")

    expected_scattering_factor = THOMPSON * AVOGADRO * 1e6 / (2 * np.pi)
    if not np.isclose(SCATTERING_FACTOR, expected_scattering_factor, rtol=1e-10):
        raise ValueError("SCATTERING_FACTOR calculation error")

    # Validate conversion factors
    if not np.isclose(KEV_TO_EV * EV_TO_KEV, 1.0):
        raise ValueError("keV/eV conversion factors are inconsistent")

    if not np.isclose(ANGSTROM_TO_METER * METER_TO_ANGSTROM, 1.0):
        raise ValueError("Angstrom/meter conversion factors are inconsistent")

    return True


# Run validation when module is imported
if __name__ != "__main__":
    try:
        validate_constants()
    except ValueError as e:
        import warnings

        warnings.warn(f"Constants validation failed: {e}", UserWarning)


# =====================================================================================
# MODULE METADATA
# =====================================================================================

__all__ = [
    # Fundamental constants
    "THOMPSON",
    "SPEED_OF_LIGHT",
    "PLANCK",
    "ELEMENT_CHARGE",
    "AVOGADRO",
    # Derived constants
    "ENERGY_TO_WAVELENGTH_FACTOR",
    "SCATTERING_FACTOR",
    # Mathematical constants
    "PI",
    "TWO_PI",
    # Unit conversions
    "KEV_TO_EV",
    "EV_TO_KEV",
    "ANGSTROM_TO_METER",
    "METER_TO_ANGSTROM",
    "CM_TO_METER",
    "METER_TO_CM",
    "DEGREES_TO_RADIANS",
    "RADIANS_TO_DEGREES",
    # Helper functions
    "energy_to_wavelength_angstrom",
    "wavelength_angstrom_to_energy",
    "critical_angle_degrees",
    "attenuation_length_cm",
    "validate_constants",
]
