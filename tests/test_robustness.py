"""
Tests for robustness improvements in core.py and utils.py.

These tests verify that the improvements made to handle edge cases,
type conversions, and error conditions work correctly.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import xraylabtool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xraylabtool.core import calculate_single_material_properties
from xraylabtool.utils import get_atomic_number, get_atomic_weight


class TestComplexNumberHandling:
    """Test complex number handling in energy conversions."""

    def test_complex_energy_handling(self):
        """Test that complex numbers are handled properly in energy conversion."""
        # Test with complex energy - should use the real part
        complex_energy = 8.0 + 2.0j
        try:
            result = calculate_single_material_properties("Si", complex_energy, 2.33)  # type: ignore
            # Should succeed and use real part (8.0)
            assert result.Energy[0] == 8.0
        except ValueError as e:
            # Should not raise ValueError for complex numbers
            pytest.fail(f"Complex number handling failed: {e}")

    def test_scalar_energy_types(self):
        """Test various scalar energy types."""
        # Test with different numeric types
        energies_to_test = [
            8,  # int
            8.0,  # float
            np.int32(8),  # numpy int
            np.float64(8.0),  # numpy float
        ]

        for energy in energies_to_test:
            result = calculate_single_material_properties("Si", energy, 2.33)
            assert len(result.Energy) == 1
            assert np.isclose(result.Energy[0], 8.0)


class TestAtomicDataRobustness:
    """Test robustness of atomic data functions."""

    def test_atomic_number_types(self):
        """Test that atomic number function handles various return types."""
        # Test basic functionality
        atomic_num = get_atomic_number("C")
        assert isinstance(atomic_num, int)
        assert atomic_num == 6

        # Test that it works with different element symbols
        test_elements = ["H", "He", "Li", "C", "N", "O", "Si", "Fe"]
        expected_numbers = [1, 2, 3, 6, 7, 8, 14, 26]

        for element, expected in zip(test_elements, expected_numbers):
            result = get_atomic_number(element)
            assert isinstance(result, int)
            assert result == expected

    def test_atomic_weight_types(self):
        """Test that atomic weight function handles various return types."""
        # Test basic functionality
        atomic_weight = get_atomic_weight("C")
        assert isinstance(atomic_weight, float)
        assert 12.0 < atomic_weight < 12.1  # Carbon atomic weight ~12.011

        # Test that it works with different element symbols
        test_elements = ["H", "C", "O", "Si"]

        for element in test_elements:
            result = get_atomic_weight(element)
            assert isinstance(result, float)
            assert result > 0  # All atomic weights should be positive

    def test_atomic_data_error_handling(self):
        """Test error handling for invalid element symbols."""
        from xraylabtool.utils import UnknownElementError, AtomicDataError

        # Test with invalid element symbols
        invalid_elements = ["Xx", "InvalidElement", "ABC"]

        for invalid_element in invalid_elements:
            with pytest.raises((UnknownElementError, AtomicDataError)):
                get_atomic_number(invalid_element)

            with pytest.raises((UnknownElementError, AtomicDataError)):
                get_atomic_weight(invalid_element)


class TestArrayTypeHandling:
    """Test array type handling improvements."""

    def test_numpy_array_conversions(self):
        """Test that numpy array conversions work correctly."""
        # Test with different array-like inputs for energy
        energy_arrays = [
            [8.0, 10.0, 12.0],  # list
            np.array([8.0, 10.0, 12.0]),  # numpy array
            (8.0, 10.0, 12.0),  # tuple
        ]

        for energies in energy_arrays:
            result = calculate_single_material_properties("Si", energies, 2.33)
            assert len(result.Energy) == 3
            assert isinstance(result.Energy, np.ndarray)
            assert np.allclose(result.Energy, [8.0, 10.0, 12.0])

    def test_dataclass_field_types(self):
        """Test that XRayResult dataclass fields have correct types."""
        result = calculate_single_material_properties("SiO2", [8.0, 10.0], 2.2)

        # Check string fields
        assert isinstance(result.Formula, str)
        assert result.Formula == "SiO2"

        # Check float fields
        assert isinstance(result.MW, float)
        assert isinstance(result.Number_Of_Electrons, float)
        assert isinstance(result.Density, float)
        assert isinstance(result.Electron_Density, float)

        # Check array fields
        array_fields = [
            "Energy",
            "Wavelength",
            "Dispersion",
            "Absorption",
            "f1",
            "f2",
            "Critical_Angle",
            "Attenuation_Length",
            "reSLD",
            "imSLD",
        ]

        for field_name in array_fields:
            field_value = getattr(result, field_name)
            assert isinstance(field_value, np.ndarray)
            assert len(field_value) == 2  # Two energy points


class TestErrorMessages:
    """Test that error messages are informative."""

    def test_invalid_formula_error(self):
        """Test error message for invalid formula."""
        with pytest.raises(ValueError) as exc_info:
            calculate_single_material_properties("", [8.0], 2.2)

        assert "Formula must be a non-empty string" in str(exc_info.value)

    def test_negative_density_error(self):
        """Test error message for negative density."""
        with pytest.raises(ValueError) as exc_info:
            calculate_single_material_properties("Si", [8.0], -2.33)

        assert "Mass density must be positive" in str(exc_info.value)

    def test_invalid_energy_range_error(self):
        """Test error message for out-of-range energies."""
        # Test energy too low
        with pytest.raises(ValueError) as exc_info:
            calculate_single_material_properties("Si", [0.01], 2.33)

        assert "Energy is out of range" in str(exc_info.value)

        # Test energy too high
        with pytest.raises(ValueError) as exc_info:
            calculate_single_material_properties("Si", [35.0], 2.33)

        assert "Energy is out of range" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])
