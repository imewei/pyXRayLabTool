"""
Validation exceptions module.

This module re-exports exception classes from the main exceptions module
to maintain compatibility with documentation and import patterns.
"""

# Re-export all exception classes from the main exceptions module
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

__all__ = [
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
]
