"""
XRayLabTool Calculators Module.

This module contains the core calculation engines for X-ray optical properties.
"""

from xraylabtool.calculators.core import (
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
from xraylabtool.calculators.derived_quantities import (
    calculate_attenuation_length,
    calculate_critical_angle,
    calculate_scattering_length_density,
    calculate_transmission,
)

__all__ = [
    # Core calculation functions
    "XRayResult",
    "calculate_single_material_properties",
    "calculate_xray_properties",
    "calculate_multiple_xray_properties",
    "calculate_scattering_factors",
    "calculate_derived_quantities",
    # Cache management
    "clear_scattering_factor_cache",
    "get_cached_elements",
    "is_element_cached",
    # Data loading
    "load_scattering_factor_data",
    "create_scattering_factor_interpolators",
    # Derived quantities
    "calculate_attenuation_length",
    "calculate_critical_angle",
    "calculate_scattering_length_density",
    "calculate_transmission",
]
