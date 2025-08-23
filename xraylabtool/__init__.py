"""
XRayLabTool - A Python package for X-ray laboratory analysis tools.

This package provides tools and utilities for X-ray crystallography
and related laboratory analysis tasks.
"""

__version__ = "0.1.6"
__author__ = "Wei Chen"
__email__ = "wchen@anl.gov"

# Import main modules for easy access
from . import constants
from . import core
from . import utils

# Import key classes and functions for easy access
from .core import (
    XRayResult,
    load_scattering_factor_data,
    get_cached_elements,
    clear_scattering_factor_cache,
    is_element_cached,
    create_scattering_factor_interpolators,
    calculate_scattering_factors,
    calculate_derived_quantities,
    calculate_multiple_xray_properties,
    calculate_single_material_properties,
    calculate_xray_properties,
)

# Import useful utility functions
from .utils import (
    wavelength_to_energy,
    energy_to_wavelength,
    bragg_angle,
    parse_formula,
    get_atomic_number,
    get_atomic_weight,
)

# Import useful constants
from .constants import (
    THOMPSON,
    SPEED_OF_LIGHT,
    PLANCK,
    ELEMENT_CHARGE,
    AVOGADRO,
    energy_to_wavelength_angstrom,
    wavelength_angstrom_to_energy,
    critical_angle_degrees,
    attenuation_length_cm,
)

# Import performance optimization modules
try:
    from .atomic_data_cache import (
        get_atomic_data_fast,
        get_bulk_atomic_data_fast,
        warm_up_cache,
        get_cache_stats,
        is_element_preloaded,
    )

    from .batch_processor import (
        calculate_batch_properties,
        save_batch_results,
        load_batch_input,
        BatchConfig,
        MemoryMonitor,
    )

    _PERFORMANCE_MODULES_AVAILABLE = True
except ImportError as e:
    # Gracefully handle missing optional dependencies
    _PERFORMANCE_MODULES_AVAILABLE = False
    import warnings

    warnings.warn(f"Performance optimization modules not available: {e}")

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
