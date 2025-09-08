"""
Tests for numerical stability improvements.

This test module validates the numerical stability enhancements added to
prevent calculation failures, handle edge cases, and ensure robust behavior
under extreme conditions.
"""

import numpy as np
import pytest

import xraylabtool as xlt
from xraylabtool.calculators.core import calculate_derived_quantities


class TestNumericalStabilityChecks:
    """Test numerical stability checks in calculate_derived_quantities."""

    def test_nan_detection_in_dispersion(self):
        """Test NaN detection in dispersion coefficients."""
        wavelength = np.array([1.24e-10, 1.55e-10])  # Valid wavelengths
        dispersion = np.array([1e-6, np.nan])  # One NaN value
        absorption = np.array([1e-8, 1e-8])  # Valid absorption

        with pytest.raises(ValueError, match="NaN values detected in dispersion"):
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )

    def test_nan_detection_in_absorption(self):
        """Test NaN detection in absorption coefficients."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, 1e-6])
        absorption = np.array([1e-8, np.nan])  # One NaN value

        with pytest.raises(ValueError, match="NaN values detected.*absorption"):
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )

    def test_infinity_detection_in_dispersion(self):
        """Test infinity detection in dispersion coefficients."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, np.inf])  # One infinite value
        absorption = np.array([1e-8, 1e-8])

        with pytest.raises(ValueError, match="Infinite values detected in dispersion"):
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )

    def test_infinity_detection_in_absorption(self):
        """Test infinity detection in absorption coefficients."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, 1e-6])
        absorption = np.array([1e-8, np.inf])  # One infinite value

        with pytest.raises(ValueError, match="Infinite values detected.*absorption"):
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )

    def test_negative_dispersion_detection(self):
        """Test detection of negative dispersion values (physically unrealistic)."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, -1e-6])  # One negative value
        absorption = np.array([1e-8, 1e-8])

        with pytest.raises(ValueError, match="Negative dispersion values detected"):
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )

    def test_safe_critical_angle_calculation(self):
        """Test safe critical angle calculation with np.maximum."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, 0.0])  # One zero value
        absorption = np.array([1e-8, 1e-8])

        # Should not raise error and handle zero dispersion gracefully
        electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )
        )

        # Critical angle should be finite and non-negative
        assert np.all(np.isfinite(critical_angle))
        assert np.all(critical_angle >= 0)
        assert (
            critical_angle[1] == 0.0
        )  # Zero dispersion should give zero critical angle

    def test_safe_attenuation_length_calculation(self):
        """Test safe attenuation length calculation with epsilon handling."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, 1e-6])
        absorption = np.array([1e-8, 1e-50])  # One extremely small value

        # Should not raise error and handle small absorption gracefully
        electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )
        )

        # Attenuation length should be finite
        assert np.all(np.isfinite(attenuation_length))
        assert np.all(attenuation_length > 0)  # Should be positive

        # Very small absorption should give very large attenuation length
        assert attenuation_length[1] > attenuation_length[0]

    def test_zero_absorption_handling(self):
        """Test handling of zero absorption values."""
        wavelength = np.array([1.24e-10, 1.55e-10])
        dispersion = np.array([1e-6, 1e-6])
        absorption = np.array([1e-8, 0.0])  # One zero value

        electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density=2.2,
                molecular_weight=60.08,
                number_of_electrons=30.0,
            )
        )

        # Should handle zero absorption with minimum epsilon
        assert np.all(np.isfinite(attenuation_length))
        assert np.all(attenuation_length > 0)


class TestExtremeConditions:
    """Test calculations under extreme conditions."""

    def test_very_low_energy_calculation(self):
        """Test calculation at very low energies (near the lower bound)."""
        # Test at 0.03 keV (lower bound)
        result = xlt.calculate_single_material_properties("SiO2", 0.03, 2.2)

        # Results should be finite and physically reasonable
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert np.all(result.critical_angle_degrees > 0)
        assert np.all(result.attenuation_length_cm > 0)

    def test_very_high_energy_calculation(self):
        """Test calculation at very high energies (near the upper bound)."""
        # Test at 30 keV (upper bound)
        result = xlt.calculate_single_material_properties("SiO2", 30.0, 2.2)

        # Results should be finite and physically reasonable
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert np.all(result.critical_angle_degrees > 0)
        assert np.all(result.attenuation_length_cm > 0)

    def test_very_low_density_material(self):
        """Test calculation with very low density materials."""
        # Hydrogen gas at very low density
        result = xlt.calculate_single_material_properties("H", 10.0, 0.00009)

        # Should handle low density gracefully
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert result.critical_angle_degrees[0] > 0

    def test_very_high_density_material(self):
        """Test calculation with very high density materials."""
        # Gold at high density
        result = xlt.calculate_single_material_properties("Au", 10.0, 19.3)

        # Should handle high density gracefully
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert result.critical_angle_degrees[0] > 0

    def test_large_energy_array(self):
        """Test calculation with large energy arrays."""
        # Large energy array (1000 points)
        energies = np.linspace(1.0, 25.0, 1000)
        result = xlt.calculate_single_material_properties("Si", energies, 2.33)

        # All results should be finite
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert np.all(np.isfinite(result.dispersion_delta))
        assert np.all(np.isfinite(result.absorption_beta))
        assert len(result.energy_kev) == 1000

    def test_logarithmic_energy_spacing(self):
        """Test calculation with logarithmic energy spacing."""
        # Logarithmic spacing from 0.1 to 30 keV
        energies = np.logspace(np.log10(0.1), np.log10(30), 100)

        # Filter to valid range (0.03-30 keV)
        valid_energies = energies[(energies >= 0.03) & (energies <= 30.0)]

        result = xlt.calculate_single_material_properties("SiO2", valid_energies, 2.2)

        # All results should be finite across the entire energy range
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert len(result.energy_kev) == len(valid_energies)


class TestBoundaryConditions:
    """Test calculations at boundary conditions."""

    def test_energy_boundary_validation(self):
        """Test energy boundary validation."""
        # Test below lower bound
        with pytest.raises(ValueError, match="Energy.*range"):
            xlt.calculate_single_material_properties("SiO2", 0.02, 2.2)  # Too low

        # Test above upper bound
        with pytest.raises(ValueError, match="Energy.*range"):
            xlt.calculate_single_material_properties("SiO2", 31.0, 2.2)  # Too high

        # Test exactly at bounds (should work)
        result_low = xlt.calculate_single_material_properties("SiO2", 0.03, 2.2)
        result_high = xlt.calculate_single_material_properties("SiO2", 30.0, 2.2)

        assert np.isfinite(result_low.critical_angle_degrees[0])
        assert np.isfinite(result_high.critical_angle_degrees[0])

    def test_density_boundary_validation(self):
        """Test density boundary validation."""
        # Test zero density (should fail)
        with pytest.raises(ValueError, match="Mass density must be positive"):
            xlt.calculate_single_material_properties("SiO2", 10.0, 0.0)

        # Test negative density (should fail)
        with pytest.raises(ValueError, match="Mass density must be positive"):
            xlt.calculate_single_material_properties("SiO2", 10.0, -1.0)

        # Test very small positive density (should work)
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 1e-6)
        assert np.isfinite(result.critical_angle_degrees[0])

    def test_numerical_precision_at_boundaries(self):
        """Test numerical precision at energy boundaries."""
        # Test calculation very close to boundaries
        near_low = 0.030001  # Just above lower bound
        near_high = 29.9999  # Just below upper bound

        result_low = xlt.calculate_single_material_properties("SiO2", near_low, 2.2)
        result_high = xlt.calculate_single_material_properties("SiO2", near_high, 2.2)

        # Results should be finite and reasonable
        assert np.isfinite(result_low.critical_angle_degrees[0])
        assert np.isfinite(result_high.critical_angle_degrees[0])
        assert (
            result_low.critical_angle_degrees[0] > result_high.critical_angle_degrees[0]
        )  # Physical expectation


class TestPhysicalRealism:
    """Test that results maintain physical realism."""

    def test_critical_angle_monotonicity(self):
        """Test that critical angle decreases with increasing energy."""
        energies = np.array([5.0, 10.0, 15.0, 20.0])
        result = xlt.calculate_single_material_properties("SiO2", energies, 2.2)

        # Critical angle should generally decrease with increasing energy
        angles = result.critical_angle_degrees
        for i in range(len(angles) - 1):
            assert angles[i] >= angles[i + 1], (
                "Critical angle should decrease with energy"
            )

    def test_attenuation_length_positive(self):
        """Test that attenuation length is always positive."""
        energies = np.linspace(1.0, 25.0, 50)
        materials = [("SiO2", 2.2), ("Al2O3", 3.95), ("Fe2O3", 5.24)]

        for formula, density in materials:
            result = xlt.calculate_single_material_properties(
                formula, energies, density
            )
            assert np.all(result.attenuation_length_cm > 0), (
                f"Attenuation length should be positive for {formula}"
            )

    def test_electron_density_consistency(self):
        """Test that electron density is consistent with formula and density."""
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        # SiO2: Si (14 electrons) + 2*O (8 electrons each) = 30 electrons
        expected_electrons = 14 + 2 * 8
        assert abs(result.total_electrons - expected_electrons) < 1e-6

        # Electron density should be positive and reasonable
        assert result.electron_density_per_ang3 > 0
        assert result.electron_density_per_ang3 < 1.0  # Reasonable upper bound

    def test_scattering_factors_consistency(self):
        """Test that scattering factors are physically reasonable."""
        energies = np.array([8.0, 10.0, 12.0])
        result = xlt.calculate_single_material_properties("Si", energies, 2.33)

        # f1 should be close to atomic number at low energies
        # f2 should be positive (related to absorption)
        assert np.all(result.scattering_factor_f1 > 0)  # Should be positive
        assert np.all(result.scattering_factor_f2 > 0)  # Should be positive

        # For Silicon (Z=14), f1 should be reasonably close to 14
        assert np.all(result.scattering_factor_f1 < 20)  # Upper bound check
        assert np.all(result.scattering_factor_f1 > 5)  # Lower bound check
