"""
Basic validation system for XRayLabTool.

Simplified validation functionality for scientific input validation.
"""

# Import exceptions from main exceptions module for compatibility
from ..exceptions import (
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
from .validators import (
    validate_chemical_formula,
    validate_density,
    validate_energy_range,
)

__all__ = [
    # Exceptions
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
    # Validation functions
    "validate_chemical_formula",
    "validate_density",
    "validate_energy_range",
]
