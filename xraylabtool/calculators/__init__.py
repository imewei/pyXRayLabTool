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
    warm_up,
)
from xraylabtool.calculators.derived_quantities import (
    calculate_attenuation_length,
    calculate_critical_angle,
    calculate_scattering_length_density,
    calculate_transmission,
)

_OPTICS_EXPORTS = frozenset(
    {
        "OpticalProps",
        "ScatteringTables",
        "encode_formula",
        "load_tables",
        "optical_properties",
    }
)


def __getattr__(name: str) -> object:
    """Resolve the pure device API of :mod:`.optics` on first access (it imports JAX)."""
    if name in _OPTICS_EXPORTS:
        from xraylabtool.calculators import optics

        return getattr(optics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core calculation functions
    "OpticalProps",
    "ScatteringTables",
    "XRayResult",
    # Derived quantities
    "calculate_attenuation_length",
    "calculate_critical_angle",
    "calculate_derived_quantities",
    "calculate_multiple_xray_properties",
    "calculate_scattering_factors",
    "calculate_scattering_length_density",
    "calculate_single_material_properties",
    "calculate_transmission",
    "calculate_xray_properties",
    # Cache management
    "clear_scattering_factor_cache",
    "create_scattering_factor_interpolators",
    "encode_formula",
    "get_cached_elements",
    "is_element_cached",
    # Data loading
    "load_scattering_factor_data",
    "load_tables",
    "optical_properties",
    "warm_up",
]
