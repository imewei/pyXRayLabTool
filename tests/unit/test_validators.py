"""Tests for validation/validators.py module.

This module tests validation functions for X-ray calculations,
including energy ranges, chemical formulas, density, and parameter validation.
"""

import math

import pytest

from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError
from xraylabtool.validation.validators import (
    validate_calculation_parameters,
    validate_chemical_formula,
    validate_density,
    validate_energy_range,
)


class TestValidateEnergyRange:
    """Tests for validate_energy_range function."""

    def test_single_valid_energy(self):
        """Test validation of single valid energy value."""
        result = validate_energy_range(5.0)
        assert float(result) == 5.0

    def test_array_valid_energies(self):
        """Test validation of array of valid energy values."""
        energies = [1.0, 5.0, 10.0]
        result = validate_energy_range(energies)
        assert len(result) == 3
        assert float(result[0]) == 1.0
        assert float(result[1]) == 5.0
        assert float(result[2]) == 10.0

    def test_custom_range(self):
        """Test validation with custom min/max range."""
        result = validate_energy_range(50.0, min_energy=10.0, max_energy=100.0)
        assert float(result) == 50.0

    def test_energy_below_minimum(self):
        """Test that energy below minimum raises EnergyError."""
        with pytest.raises(EnergyError, match="below minimum"):
            validate_energy_range(0.05)

    def test_energy_above_maximum(self):
        """Test that energy above maximum raises EnergyError."""
        with pytest.raises(EnergyError, match="above maximum"):
            validate_energy_range(150.0)

    def test_negative_energy(self):
        """Test that negative energy raises EnergyError."""
        with pytest.raises(EnergyError, match="positive"):
            validate_energy_range(-5.0)

    def test_zero_energy(self):
        """Test that zero energy raises EnergyError."""
        with pytest.raises(EnergyError, match="positive"):
            validate_energy_range(0.0)

    def test_nan_energy(self):
        """Test that NaN energy raises EnergyError."""
        with pytest.raises(EnergyError, match="finite"):
            validate_energy_range(float("nan"))

    def test_infinite_energy(self):
        """Test that infinite energy raises EnergyError."""
        with pytest.raises(EnergyError, match="finite"):
            validate_energy_range(float("inf"))


class TestValidateChemicalFormula:
    """Tests for validate_chemical_formula function."""

    def test_simple_formula(self):
        """Test validation of simple chemical formula."""
        result = validate_chemical_formula("H2O")
        assert "H" in result
        assert "O" in result

    def test_complex_formula(self):
        """Test validation of complex chemical formula."""
        result = validate_chemical_formula("Ca0.5Sr0.5TiO3")
        assert "Ca" in result
        assert "Sr" in result
        assert "Ti" in result
        assert "O" in result

    def test_single_element(self):
        """Test validation of single element."""
        result = validate_chemical_formula("Fe")
        assert "Fe" in result

    def test_oxide_formula(self):
        """Test validation of common oxide formula."""
        result = validate_chemical_formula("SiO2")
        assert "Si" in result
        assert "O" in result

    def test_empty_string(self):
        """Test that empty string raises FormulaError."""
        with pytest.raises(FormulaError, match="empty"):
            validate_chemical_formula("")

    def test_whitespace_only(self):
        """Test that whitespace-only string raises FormulaError."""
        with pytest.raises(FormulaError, match="empty"):
            validate_chemical_formula("   ")

    def test_none_input(self):
        """Test that None input raises FormulaError."""
        with pytest.raises(FormulaError, match="non-empty"):
            validate_chemical_formula(None)  # type: ignore

    def test_invalid_characters(self):
        """Test that invalid characters raise FormulaError."""
        with pytest.raises(FormulaError, match="invalid characters"):
            validate_chemical_formula("H@2O!")

    def test_unknown_element(self):
        """Test that unknown element raises FormulaError."""
        with pytest.raises(FormulaError, match="Unknown element"):
            validate_chemical_formula("Xx3")

    def test_formula_with_spaces(self):
        """Test that formula with spaces is handled."""
        # Spaces should be stripped and if valid pattern remains, should work
        result = validate_chemical_formula("  H2O  ")
        assert "H" in result
        assert "O" in result


class TestValidateDensity:
    """Tests for validate_density function."""

    def test_valid_density(self):
        """Test validation of valid density value."""
        result = validate_density(2.5)
        assert result == 2.5

    def test_custom_range(self):
        """Test validation with custom min/max range."""
        result = validate_density(15.0, min_density=10.0, max_density=20.0)
        assert result == 15.0

    def test_density_at_minimum(self):
        """Test density at minimum boundary."""
        result = validate_density(0.001)
        assert result == 0.001

    def test_density_at_maximum(self):
        """Test density at maximum boundary."""
        result = validate_density(30.0)
        assert result == 30.0

    def test_density_below_minimum(self):
        """Test that density below minimum raises ValidationError."""
        with pytest.raises(ValidationError, match="below minimum"):
            validate_density(0.0001)

    def test_density_above_maximum(self):
        """Test that density above maximum raises ValidationError."""
        with pytest.raises(ValidationError, match="above maximum"):
            validate_density(50.0)

    def test_negative_density(self):
        """Test that negative density raises ValidationError."""
        with pytest.raises(ValidationError, match="positive"):
            validate_density(-1.0)

    def test_zero_density(self):
        """Test that zero density raises ValidationError."""
        with pytest.raises(ValidationError, match="positive"):
            validate_density(0.0)

    def test_nan_density(self):
        """Test that NaN density raises ValidationError."""
        with pytest.raises(ValidationError, match="finite"):
            validate_density(float("nan"))

    def test_infinite_density(self):
        """Test that infinite density raises ValidationError."""
        with pytest.raises(ValidationError, match="finite"):
            validate_density(float("inf"))

    def test_non_numeric_density(self):
        """Test that non-numeric density raises ValidationError."""
        with pytest.raises(ValidationError, match="numeric"):
            validate_density("2.5")  # type: ignore


class TestValidateCalculationParameters:
    """Tests for validate_calculation_parameters function."""

    def test_valid_parameters(self):
        """Test validation of valid calculation parameters."""
        formula, energies, density = validate_calculation_parameters("SiO2", 5.0, 2.2)
        assert formula == "SiO2"
        # Single energy value returns 0-d array
        assert float(energies) == 5.0
        assert density == 2.2

    def test_array_energies(self):
        """Test validation with array of energies."""
        formula, energies, density = validate_calculation_parameters(
            "H2O", [1.0, 5.0, 10.0], 1.0
        )
        assert formula == "H2O"
        assert len(energies) == 3
        assert density == 1.0

    def test_invalid_formula(self):
        """Test that invalid formula raises error."""
        with pytest.raises(FormulaError):
            validate_calculation_parameters("Xx", 5.0, 2.0)

    def test_invalid_energy(self):
        """Test that invalid energy raises error."""
        with pytest.raises(EnergyError):
            validate_calculation_parameters("H2O", 500.0, 2.0)

    def test_invalid_density(self):
        """Test that invalid density raises error."""
        with pytest.raises(ValidationError):
            validate_calculation_parameters("H2O", 5.0, 50.0)

    def test_multiple_invalid_parameters(self):
        """Test that first invalid parameter is caught."""
        # Will fail on formula first
        with pytest.raises(FormulaError):
            validate_calculation_parameters("", 5.0, 2.0)
