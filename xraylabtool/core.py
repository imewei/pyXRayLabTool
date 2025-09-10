"""
Core functionality for XRayLabTool.

This module imports and re-exports core functionality from the calculators package
to maintain backward compatibility. New code should import directly from
xraylabtool.calculators.core for better organization.

All core classes and functions are imported from xraylabtool.calculators.core.
"""

import warnings

# Re-export everything from calculators.core for backward compatibility
from xraylabtool.calculators.core import (
    XRayResult,
    calculate_derived_quantities,
    calculate_scattering_factors,
    calculate_single_material_properties,
    calculate_xray_properties,
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
    get_bulk_atomic_data,
    get_cached_elements,
    is_element_cached,
    load_scattering_factor_data,
)

# Emit a deprecation warning when this module is imported
warnings.warn(
    "Importing from xraylabtool.core is deprecated. "
    "Please import from xraylabtool.calculators.core instead. "
    "Support for xraylabtool.core imports will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "XRayResult",
    "calculate_single_material_properties",
    "calculate_xray_properties",
    "calculate_scattering_factors",
    "calculate_derived_quantities",
    "load_scattering_factor_data",
    "get_cached_elements",
    "get_bulk_atomic_data",
    "clear_scattering_factor_cache",
    "is_element_cached",
    "create_scattering_factor_interpolators",
]
