"""
XRayLabTool Validation Module.

This module contains data validation, error handling, and exception classes.
"""

from xraylabtool.exceptions import (
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
)
from xraylabtool.validation.validators import (
    validate_calculation_parameters,
    validate_chemical_formula,
    validate_density,
    validate_energy_range,
)

__all__ = [
    # Exception classes
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
    "validate_calculation_parameters",
]
