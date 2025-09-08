"""
Custom exceptions for XRayLabTool.

This module defines domain-specific exception classes that provide
better error handling and more informative error messages for
X-ray analysis operations.
"""

from typing import Any

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


class XRayLabToolError(Exception):
    """Base exception for all XRayLabTool-related errors."""

    pass


class CalculationError(XRayLabToolError):
    """Exception raised when X-ray property calculations fail."""

    def __init__(
        self,
        message: str,
        formula: str | None = None,
        energy: float | None = None,
    ):
        self.formula = formula
        self.energy = energy

        if formula and energy:
            message = f"{message} (formula: {formula}, energy: {energy} keV)"
        elif formula:
            message = f"{message} (formula: {formula})"
        elif energy:
            message = f"{message} (energy: {energy} keV)"

        super().__init__(message)


class FormulaError(XRayLabToolError):
    """Exception raised for invalid chemical formulas."""

    def __init__(self, message: str, formula: str | None = None):
        self.formula = formula

        if formula:
            message = f"{message}: '{formula}'"

        super().__init__(message)


class EnergyError(XRayLabToolError):
    """Exception raised for invalid energy values or ranges."""

    def __init__(
        self,
        message: str,
        energy: float | None = None,
        valid_range: str | None = None,
    ):
        self.energy = energy
        self.valid_range = valid_range

        if energy and valid_range:
            message = f"{message}: {energy} keV (valid range: {valid_range})"
        elif energy:
            message = f"{message}: {energy} keV"
        elif valid_range:
            message = f"{message} (valid range: {valid_range})"

        super().__init__(message)


class DataFileError(XRayLabToolError):
    """Exception raised when data files cannot be loaded or processed."""

    def __init__(self, message: str, filename: str | None = None):
        self.filename = filename

        if filename:
            message = f"{message}: {filename}"

        super().__init__(message)


class ValidationError(XRayLabToolError):
    """Exception raised when input validation fails."""

    def __init__(
        self, message: str, parameter: str | None = None, value: Any | None = None
    ):
        self.parameter = parameter
        self.value = value

        if parameter and value is not None:
            message = f"{message}: {parameter}={value}"
        elif parameter:
            message = f"{message}: {parameter}"

        super().__init__(message)


class AtomicDataError(XRayLabToolError):
    """Exception raised when atomic data cannot be retrieved or processed."""

    def __init__(self, message: str, element: str | None = None):
        self.element = element

        if element:
            message = f"{message} (element: {element})"

        super().__init__(message)


class UnknownElementError(AtomicDataError):
    """Exception raised when an unknown element symbol is provided."""

    def __init__(self, element: str):
        self.element = element
        message = f"Unknown element symbol: '{element}'"
        super().__init__(message, element)


class BatchProcessingError(XRayLabToolError):
    """Exception raised during batch processing operations."""

    def __init__(
        self,
        message: str,
        failed_items: list[Any] | None = None,
        total_items: int | None = None,
    ):
        self.failed_items = failed_items or []
        self.total_items = total_items

        if failed_items and total_items:
            message = f"{message} ({len(failed_items)}/{total_items} items failed)"
        elif failed_items:
            message = f"{message} ({len(failed_items)} items failed)"

        super().__init__(message)


class ConfigurationError(XRayLabToolError):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str, config_key: str | None = None):
        self.config_key = config_key

        if config_key:
            message = f"{message} (config: {config_key})"

        super().__init__(message)
