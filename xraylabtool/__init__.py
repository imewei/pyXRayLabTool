"""
XRayLabTool: High-Performance X-ray Optical Properties Calculator.

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

import sys

if sys.version_info < (3, 12):  # noqa: UP036
    raise ImportError(
        "XRayLabTool requires Python 3.12+; please upgrade your interpreter."
    )

__version__ = "0.3.0"
__author__ = "Wei Chen"
__email__ = "wchen@anl.gov"

# Logging helpers (lightweight, stdlib-only)
from xraylabtool.logging_utils import configure_logging, get_logger, log_environment

# All modules are now imported lazily via __getattr__ for ultra-fast startup


# Lazy import heavy modules and functions to improve startup time
def __getattr__(name):
    # Lazy module imports
    if name == "calculators":
        from xraylabtool import calculators

        globals()["calculators"] = calculators
        return calculators
    elif name == "constants":
        from xraylabtool import constants

        globals()["constants"] = constants
        return constants
    elif name == "data_handling":
        from xraylabtool import data_handling

        globals()["data_handling"] = data_handling
        return data_handling
    elif name == "io":
        from xraylabtool import io

        globals()["io"] = io
        return io
    elif name == "utils":
        from xraylabtool import utils

        globals()["utils"] = utils
        return utils
    elif name == "validation":
        from xraylabtool import validation

        globals()["validation"] = validation
        return validation
    elif name == "analysis":
        from xraylabtool import analysis

        globals()["analysis"] = analysis
        return analysis
    elif name == "cleanup":
        from xraylabtool import cleanup

        globals()["cleanup"] = cleanup
        return cleanup
    elif name == "export":
        from xraylabtool import export

        globals()["export"] = export
        return export

    # Lazy function imports from calculators
    elif name in [
        "XRayResult",
        "calculate_derived_quantities",
        "calculate_scattering_factors",
        "calculate_single_material_properties",
        "calculate_xray_properties",
        "clear_scattering_factor_cache",
        "create_scattering_factor_interpolators",
        "get_cached_elements",
        "is_element_cached",
        "load_scattering_factor_data",
    ]:
        from xraylabtool.calculators import (
            XRayResult,
            calculate_derived_quantities,
            calculate_scattering_factors,
            calculate_single_material_properties,
            calculate_xray_properties,
            clear_scattering_factor_cache,
            create_scattering_factor_interpolators,
            get_cached_elements,
            is_element_cached,
            load_scattering_factor_data,
        )

        globals().update(
            {
                "XRayResult": XRayResult,
                "calculate_derived_quantities": calculate_derived_quantities,
                "calculate_scattering_factors": calculate_scattering_factors,
                "calculate_single_material_properties": (
                    calculate_single_material_properties
                ),
                "calculate_xray_properties": calculate_xray_properties,
                "clear_scattering_factor_cache": clear_scattering_factor_cache,
                "create_scattering_factor_interpolators": (
                    create_scattering_factor_interpolators
                ),
                "get_cached_elements": get_cached_elements,
                "is_element_cached": is_element_cached,
                "load_scattering_factor_data": load_scattering_factor_data,
            }
        )
        return globals()[name]

    # Lazy function imports from constants
    elif name in [
        "AVOGADRO",
        "ELEMENT_CHARGE",
        "ELECTRON_CHARGE",
        "PLANCK",
        "SPEED_OF_LIGHT",
        "THOMPSON",
        "attenuation_length_cm",
        "critical_angle_degrees",
        "energy_to_wavelength_angstrom",
        "wavelength_angstrom_to_energy",
    ]:
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

        globals().update(
            {
                "AVOGADRO": AVOGADRO,
                "ELEMENT_CHARGE": ELEMENT_CHARGE,
                "ELECTRON_CHARGE": ELEMENT_CHARGE,  # Alias for backward compatibility
                "PLANCK": PLANCK,
                "SPEED_OF_LIGHT": SPEED_OF_LIGHT,
                "THOMPSON": THOMPSON,
                "attenuation_length_cm": attenuation_length_cm,
                "critical_angle_degrees": critical_angle_degrees,
                "energy_to_wavelength_angstrom": energy_to_wavelength_angstrom,
                "wavelength_angstrom_to_energy": wavelength_angstrom_to_energy,
            }
        )
        return globals()[name]

    # Lazy function imports from export
    elif name in ["export_to_csv", "export_to_json"]:
        from xraylabtool.export import export_to_csv, export_to_json

        globals().update(
            {
                "export_to_csv": export_to_csv,
                "export_to_json": export_to_json,
            }
        )
        return globals()[name]

    # Lazy function imports from io
    elif name in ["format_xray_result", "load_data_file"]:
        from xraylabtool.io import format_xray_result, load_data_file

        globals().update(
            {
                "format_xray_result": format_xray_result,
                "load_data_file": load_data_file,
            }
        )
        return globals()[name]

    # Lazy function imports from utils
    elif name in [
        "bragg_angle",
        "energy_to_wavelength",
        "get_atomic_number",
        "get_atomic_weight",
        "parse_formula",
        "wavelength_to_energy",
    ]:
        from xraylabtool.utils import (
            bragg_angle,
            energy_to_wavelength,
            get_atomic_number,
            get_atomic_weight,
            parse_formula,
            wavelength_to_energy,
        )

        globals().update(
            {
                "bragg_angle": bragg_angle,
                "energy_to_wavelength": energy_to_wavelength,
                "get_atomic_number": get_atomic_number,
                "get_atomic_weight": get_atomic_weight,
                "parse_formula": parse_formula,
                "wavelength_to_energy": wavelength_to_energy,
            }
        )
        return globals()[name]

    # Lazy function imports from validation (exceptions and functions)
    elif name in [
        "AtomicDataError",
        "BatchProcessingError",
        "CalculationError",
        "ConfigurationError",
        "DataFileError",
        "EnergyError",
        "FormulaError",
        "UnknownElementError",
        "ValidationError",
        "XRayLabToolError",
        "validate_chemical_formula",
        "validate_density",
        "validate_energy_range",
    ]:
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

        globals().update(
            {
                "AtomicDataError": AtomicDataError,
                "BatchProcessingError": BatchProcessingError,
                "CalculationError": CalculationError,
                "ConfigurationError": ConfigurationError,
                "DataFileError": DataFileError,
                "EnergyError": EnergyError,
                "FormulaError": FormulaError,
                "UnknownElementError": UnknownElementError,
                "ValidationError": ValidationError,
                "XRayLabToolError": XRayLabToolError,
                "validate_chemical_formula": validate_chemical_formula,
                "validate_density": validate_density,
                "validate_energy_range": validate_energy_range,
            }
        )
        return globals()[name]

    # Lazy import for completion_installer module
    elif name == "completion_installer":
        from xraylabtool.interfaces import completion as completion_installer

        globals().update({"completion_installer": completion_installer})
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# All functions and constants are now imported lazily via __getattr__

# CLI main function available via lazy import (use: from xraylabtool.interfaces import main)
# Removed direct import to avoid startup performance penalty


# Performance optimization modules (imported on demand to avoid unused
# import warnings)
_PERFORMANCE_MODULES_AVAILABLE = True

__all__ = [
    "AVOGADRO",
    "ELEMENT_CHARGE",
    "PLANCK",
    "SPEED_OF_LIGHT",
    # Physical constants
    "THOMPSON",
    "AtomicDataError",
    "BatchProcessingError",
    "CalculationError",
    "ConfigurationError",
    "DataFileError",
    "EnergyError",
    "FormulaError",
    "UnknownElementError",
    "ValidationError",
    # Domain-specific exceptions
    "XRayLabToolError",
    # Core functionality - Main API
    "XRayResult",
    "attenuation_length_cm",
    "bragg_angle",
    "calculate_derived_quantities",
    "calculate_scattering_factors",
    "calculate_single_material_properties",
    "calculate_xray_properties",
    "calculators",
    "clear_scattering_factor_cache",
    # Main modules
    "constants",
    "create_scattering_factor_interpolators",
    "critical_angle_degrees",
    "data_handling",
    "energy_to_wavelength",
    "energy_to_wavelength_angstrom",
    "export_to_csv",
    "export_to_json",
    # I/O functions
    "format_xray_result",
    "get_atomic_number",
    "get_atomic_weight",
    "get_cached_elements",
    "io",
    "is_element_cached",
    "load_data_file",
    # Core functionality - Advanced/Internal
    "load_scattering_factor_data",
    "parse_formula",
    "utils",
    "validate_chemical_formula",
    "validate_density",
    # Validation functions
    "validate_energy_range",
    "validation",
    "wavelength_angstrom_to_energy",
    # Utility functions
    "wavelength_to_energy",
]
