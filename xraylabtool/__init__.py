"""
XRayLabTool: High-Performance X-ray Optical Properties Calculator

A comprehensive Python package for calculating X-ray optical properties of materials
with ultra-fast performance, comprehensive CLI tools, and scientific accuracy.

**Key Features:**
- **Ultra-fast calculations**: 150,000+ calculations/second with 350x speed improvement
- **CXRO/NIST databases**: Authoritative atomic scattering factor data
- **Complete Python API**: Full programmatic access with descriptive field names
- **Powerful CLI**: 8 specialized commands for batch processing and analysis
- **High performance caching**: Preloaded data for 92 elements (H-U)
- **Cross-platform**: Windows, macOS, Linux with shell completion support

**Quick Start:**
    >>> import xraylabtool as xlt
    >>> result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
    >>> print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
    Critical angle: 0.174°

**Main Functions:**
- :func:`calculate_single_material_properties`: Single material calculations
- :func:`calculate_xray_properties`: Multiple materials (parallel processing)
- :func:`parse_formula`: Chemical formula parsing
- :func:`energy_to_wavelength`, :func:`wavelength_to_energy`: Unit conversions

**Data Structures:**
- :class:`XRayResult`: Complete X-ray properties dataclass

**Physical Constants:**
- :data:`PLANCK`, :data:`SPEED_OF_LIGHT`, :data:`AVOGADRO`: Fundamental constants
- :func:`critical_angle_degrees`, :func:`attenuation_length_cm`: Conversion functions

**Command-Line Interface:**
Access via ``xraylabtool`` command with specialized subcommands:
- ``calc``: Single material calculations
- ``batch``: Multi-material processing
- ``convert``: Energy/wavelength conversions
- ``formula``: Chemical formula analysis
- And more... (use ``xraylabtool --help``)

**Scientific Applications:**
- Synchrotron beamline design and commissioning
- X-ray reflectometry (XRR) and diffraction (XRD)
- Materials characterization and thin film analysis
- Small-angle X-ray scattering (SAXS) contrast calculations
- Medical imaging and industrial radiography optimization

For complete documentation, visit: https://pyxraylabtool.readthedocs.io
"""

__version__ = "0.2.2"
__author__ = "Wei Chen"
__email__ = "wchen@anl.gov"

# Import main modules for easy access
from xraylabtool import (
    calculators,
    constants,
    data_handling,
    interfaces,
    io,
    utils,
    validation,
)

# Import key classes and functions for easy access
from xraylabtool.calculators import (
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

# Import useful constants
from xraylabtool.constants import (
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

# Import CLI main function
from xraylabtool.interfaces import main

# Import I/O functions
from xraylabtool.io import (
    export_to_csv,
    export_to_json,
    format_xray_result,
    load_data_file,
)

# Import useful utility functions
from xraylabtool.utils import (
    bragg_angle,
    energy_to_wavelength,
    get_atomic_number,
    get_atomic_weight,
    parse_formula,
    wavelength_to_energy,
)

# Import exceptions and validation functions for external use
from xraylabtool.validation import (
    AtomicDataError,
    BatchProcessingError,
    CalculationError,
    ConfigurationError,
    DataFileError,
    EnergyError,
    FormulaError,
    UnknownElementError,
    ValidationError,
    XRayLabToolError,
    validate_chemical_formula,
    validate_density,
    validate_energy_range,
)

# Performance optimization modules (imported on demand to avoid unused
# import warnings)
_PERFORMANCE_MODULES_AVAILABLE = True

__all__ = [
    # Main modules
    "constants",
    "utils",
    "calculators",
    "data_handling",
    "interfaces",
    "io",
    "validation",
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
    "energy_to_wavelength_angstrom",
    "wavelength_angstrom_to_energy",
    "critical_angle_degrees",
    "attenuation_length_cm",
    # I/O functions
    "format_xray_result",
    "load_data_file",
    "export_to_csv",
    "export_to_json",
    # CLI interface
    "main",
    # Domain-specific exceptions
    "XRayLabToolError",
    "CalculationError",
    "FormulaError",
    "EnergyError",
    "DataFileError",
    "ValidationError",
    "AtomicDataError",
    "UnknownElementError",
    "BatchProcessingError",
    "ConfigurationError",
    # Validation functions
    "validate_energy_range",
    "validate_chemical_formula",
    "validate_density",
]
