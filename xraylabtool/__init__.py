"""
XRayLabTool - A Python package for X-ray laboratory analysis tools.

This package provides tools and utilities for X-ray crystallography
and related laboratory analysis tasks.
"""

__version__ = "0.1.9"
__author__ = "Wei Chen"
__email__ = "wchen@anl.gov"

# Import main modules for easy access
from . import constants, core, utils

# Import useful constants
from .constants import (
    AVOGADRO,
    ELEMENT_CHARGE,
    PLANCK,
    SPEED_OF_LIGHT,
    THOMPSON,
    attenuation_length_cm,
    critical_angle_degrees,
    energy_to_wavelength_angstrom,
    wavelength_angstrom_to_energy,
)

# Import key classes and functions for easy access
from .core import (
    XRayResult,
    calculate_derived_quantities,
    calculate_multiple_xray_properties,
    calculate_scattering_factors,
    calculate_single_material_properties,
    calculate_xray_properties,
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
    get_cached_elements,
    is_element_cached,
    load_scattering_factor_data,
)

# Import useful utility functions
from .utils import (
    bragg_angle,
    energy_to_wavelength,
    get_atomic_number,
    get_atomic_weight,
    parse_formula,
    wavelength_to_energy,
)

# Performance optimization modules (imported on demand to avoid unused
# import warnings)
_PERFORMANCE_MODULES_AVAILABLE = True

__all__ = [
    # Main modules
    "constants",
    "core",
    "utils",
    # Core functionality - Main API
    "XRayResult",
    "calculate_single_material_properties",
    "calculate_xray_properties",
    # Core functionality - Advanced/Internal
    "load_scattering_factor_data",
    "get_cached_elements",
    "clear_scattering_factor_cache",
    "is_element_cached",
    "create_scattering_factor_interpolators",
    "calculate_scattering_factors",
    "calculate_derived_quantities",
    "calculate_xray_properties",
    "calculate_multiple_xray_properties",
    # Utility functions
    "wavelength_to_energy",
    "energy_to_wavelength",
    "bragg_angle",
    "parse_formula",
    "get_atomic_number",
    "get_atomic_weight",
    # Physical constants
    "THOMPSON",
    "SPEED_OF_LIGHT",
    "PLANCK",
    "ELEMENT_CHARGE",
    "AVOGADRO",
    # Convenient conversion functions
    "energy_to_wavelength_angstrom",
    "wavelength_angstrom_to_energy",
    "critical_angle_degrees",
    "attenuation_length_cm",
]
