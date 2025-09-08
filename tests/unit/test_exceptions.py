"""
Tests for custom exception classes.

This module tests the behavior of domain-specific exceptions
to ensure they provide appropriate error information.
"""

import pytest

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


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_base_exception(self):
        """Test XRayLabToolError is the base exception."""
        error = XRayLabToolError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from XRayLabToolError."""
        exceptions_to_test = [
            CalculationError("test"),
            FormulaError("test"),
            EnergyError("test"),
            DataFileError("test"),
            ValidationError("test"),
            AtomicDataError("test"),
            UnknownElementError("H"),
            BatchProcessingError("test"),
            ConfigurationError("test"),
        ]

        for error in exceptions_to_test:
            assert isinstance(error, XRayLabToolError)
            assert isinstance(error, Exception)

    def test_unknown_element_inherits_from_atomic_data(self):
        """Test UnknownElementError inherits from AtomicDataError."""
        error = UnknownElementError("Xx")
        assert isinstance(error, AtomicDataError)
        assert isinstance(error, XRayLabToolError)


class TestCalculationError:
    """Test CalculationError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = CalculationError("Calculation failed")
        assert str(error) == "Calculation failed"

    def test_with_formula(self):
        """Test error with formula context."""
        error = CalculationError("Invalid calculation", formula="SiO2")
        assert "SiO2" in str(error)
        assert error.formula == "SiO2"

    def test_with_energy(self):
        """Test error with energy context."""
        error = CalculationError("Invalid energy", energy=10.0)
        assert "10.0 keV" in str(error)
        assert error.energy == 10.0

    def test_with_formula_and_energy(self):
        """Test error with both formula and energy context."""
        error = CalculationError("Calculation failed", formula="Al2O3", energy=8.0)
        assert "Al2O3" in str(error)
        assert "8.0 keV" in str(error)
        assert error.formula == "Al2O3"
        assert error.energy == 8.0


class TestFormulaError:
    """Test FormulaError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = FormulaError("Invalid formula")
        assert str(error) == "Invalid formula"

    def test_with_formula(self):
        """Test error with formula context."""
        error = FormulaError("Invalid syntax", formula="Si$O2")
        assert "Si$O2" in str(error)
        assert error.formula == "Si$O2"


class TestEnergyError:
    """Test EnergyError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = EnergyError("Invalid energy")
        assert str(error) == "Invalid energy"

    def test_with_energy(self):
        """Test error with energy value."""
        error = EnergyError("Out of range", energy=-5.0)
        assert "-5.0 keV" in str(error)
        assert error.energy == -5.0

    def test_with_valid_range(self):
        """Test error with valid range information."""
        error = EnergyError("Invalid range", valid_range="0.03-30 keV")
        assert "0.03-30 keV" in str(error)
        assert error.valid_range == "0.03-30 keV"

    def test_with_energy_and_range(self):
        """Test error with both energy and range."""
        error = EnergyError("Out of range", energy=50.0, valid_range="0.03-30 keV")
        assert "50.0 keV" in str(error)
        assert "0.03-30 keV" in str(error)


class TestDataFileError:
    """Test DataFileError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = DataFileError("File not found")
        assert str(error) == "File not found"

    def test_with_filename(self):
        """Test error with filename context."""
        error = DataFileError("Cannot read file", filename="data.csv")
        assert "data.csv" in str(error)
        assert error.filename == "data.csv"


class TestValidationError:
    """Test ValidationError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"

    def test_with_parameter(self):
        """Test error with parameter name."""
        error = ValidationError("Invalid value", parameter="density")
        assert "density" in str(error)
        assert error.parameter == "density"

    def test_with_parameter_and_value(self):
        """Test error with parameter and value."""
        error = ValidationError("Must be positive", parameter="energy", value=-1.0)
        assert "energy=-1.0" in str(error)
        assert error.parameter == "energy"
        assert error.value == -1.0


class TestAtomicDataError:
    """Test AtomicDataError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = AtomicDataError("Data unavailable")
        assert str(error) == "Data unavailable"

    def test_with_element(self):
        """Test error with element context."""
        error = AtomicDataError("Missing data", element="Tc")
        assert "Tc" in str(error)
        assert error.element == "Tc"


class TestUnknownElementError:
    """Test UnknownElementError class."""

    def test_message_format(self):
        """Test automatic message formatting."""
        error = UnknownElementError("Xx")
        assert "Unknown element symbol: 'Xx'" in str(error)
        assert error.element == "Xx"

    def test_inheritance(self):
        """Test proper inheritance from AtomicDataError."""
        error = UnknownElementError("Zz")
        assert isinstance(error, AtomicDataError)
        assert error.element == "Zz"


class TestBatchProcessingError:
    """Test BatchProcessingError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = BatchProcessingError("Batch failed")
        assert str(error) == "Batch failed"

    def test_with_failed_items(self):
        """Test error with failed items list."""
        failed = ["item1", "item2"]
        error = BatchProcessingError("Processing failed", failed_items=failed)
        assert "2" in str(error)
        assert error.failed_items == failed

    def test_with_total_items(self):
        """Test error with failed and total items."""
        failed = ["item1"]
        error = BatchProcessingError("Some failed", failed_items=failed, total_items=5)
        assert "1/5" in str(error)
        assert error.failed_items == failed
        assert error.total_items == 5


class TestConfigurationError:
    """Test ConfigurationError class."""

    def test_basic_message(self):
        """Test basic error message."""
        error = ConfigurationError("Config error")
        assert str(error) == "Config error"

    def test_with_config_key(self):
        """Test error with configuration key."""
        error = ConfigurationError("Invalid setting", config_key="max_workers")
        assert "max_workers" in str(error)
        assert error.config_key == "max_workers"


class TestExceptionUsage:
    """Test practical usage patterns of exceptions."""

    def test_catching_base_exception(self):
        """Test catching all custom exceptions with base class."""
        errors = [
            CalculationError("calc failed"),
            FormulaError("formula invalid"),
            EnergyError("energy bad"),
        ]

        for error in errors:
            try:
                raise error
            except XRayLabToolError as e:
                assert isinstance(e, XRayLabToolError)

    def test_specific_exception_catching(self):
        """Test catching specific exception types."""
        with pytest.raises(UnknownElementError) as exc_info:
            raise UnknownElementError("Xx")

        assert exc_info.value.element == "Xx"
        assert isinstance(exc_info.value, AtomicDataError)

    def test_exception_chaining(self):
        """Test exception chaining with from clause."""
        original_error = ValueError("original problem")

        try:
            try:
                raise original_error
            except ValueError as e:
                raise CalculationError("calculation failed") from e
        except CalculationError as calc_error:
            assert calc_error.__cause__ is original_error
