"""
Tests for core physics calculations implemented in Step 7.

This module tests the vectorized calculate_scattering_factors function
and derived quantity calculations that were ported from Julia.
"""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path to import xraylabtool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xraylabtool.core import (
    calculate_scattering_factors,
    calculate_derived_quantities,
    create_scattering_factor_interpolators,
    calculate_single_material_properties,
    calculate_multiple_xray_properties,
    clear_scattering_factor_cache,
    XRayResult,
)
from xraylabtool.constants import (
    ENERGY_TO_WAVELENGTH_FACTOR,
)


class TestCalculateScatteringFactors:
    """Test the core calculate_scattering_factors function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_calculate_scattering_factors_basic(self):
        """Test basic functionality of calculate_scattering_factors."""
        try:
            # Simple test case: SiO2 at 10 keV
            energy_ev = np.array([10000.0])  # 10 keV in eV
            wavelength = np.array(
                [ENERGY_TO_WAVELENGTH_FACTOR / 10.0]
            )  # Corresponding wavelength
            mass_density = 2.2  # g/cm³ for quartz
            molecular_weight = 60.08  # SiO2 molecular weight

            # Create interpolators for Si and O
            si_f1_interp, si_f2_interp = create_scattering_factor_interpolators("Si")
            o_f1_interp, o_f2_interp = create_scattering_factor_interpolators("O")

            # Element data: [(count, f1_interp, f2_interp), ...]
            element_data = [
                (1.0, si_f1_interp, si_f2_interp),  # 1 Si atom
                (2.0, o_f1_interp, o_f2_interp),  # 2 O atoms
            ]

            # Calculate scattering factors
            dispersion, absorption, f1_total, f2_total = calculate_scattering_factors(
                energy_ev, wavelength, mass_density, molecular_weight, element_data
            )

            # Verify output structure
            assert isinstance(dispersion, np.ndarray)
            assert isinstance(absorption, np.ndarray)
            assert isinstance(f1_total, np.ndarray)
            assert isinstance(f2_total, np.ndarray)

            assert len(dispersion) == 1
            assert len(absorption) == 1
            assert len(f1_total) == 1
            assert len(f2_total) == 1

            # All values should be finite and positive for typical X-ray energies
            assert np.all(np.isfinite(dispersion))
            assert np.all(np.isfinite(absorption))
            assert np.all(np.isfinite(f1_total))
            assert np.all(np.isfinite(f2_total))

            # For SiO2, f1_total should be approximately Si_f1 + 2*O_f1
            si_f1_val = si_f1_interp(energy_ev[0])
            o_f1_val = o_f1_interp(energy_ev[0])
            expected_f1_total = si_f1_val + 2 * o_f1_val
            assert abs(f1_total[0] - expected_f1_total) < 1e-10

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_calculate_scattering_factors_multiple_energies(self):
        """Test with multiple energy values (vectorized operation)."""
        try:
            # Multiple energies
            energies_kev = np.array([8.0, 10.0, 12.0])
            energy_ev = energies_kev * 1000.0
            wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energies_kev
            mass_density = 2.2
            molecular_weight = 60.08

            # Create interpolators
            si_f1_interp, si_f2_interp = create_scattering_factor_interpolators("Si")
            o_f1_interp, o_f2_interp = create_scattering_factor_interpolators("O")

            element_data = [
                (1.0, si_f1_interp, si_f2_interp),
                (2.0, o_f1_interp, o_f2_interp),
            ]

            # Calculate scattering factors
            dispersion, absorption, f1_total, f2_total = calculate_scattering_factors(
                energy_ev, wavelength, mass_density, molecular_weight, element_data
            )

            # Verify all arrays have correct length
            assert len(dispersion) == 3
            assert len(absorption) == 3
            assert len(f1_total) == 3
            assert len(f2_total) == 3

            # All values should be finite
            assert np.all(np.isfinite(dispersion))
            assert np.all(np.isfinite(absorption))
            assert np.all(np.isfinite(f1_total))
            assert np.all(np.isfinite(f2_total))

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_calculate_scattering_factors_empty_element_data(self):
        """Test with empty element data."""
        energy_ev = np.array([10000.0])
        wavelength = np.array([ENERGY_TO_WAVELENGTH_FACTOR / 10.0])
        mass_density = 2.2
        molecular_weight = 60.08
        element_data = []  # Empty

        dispersion, absorption, f1_total, f2_total = calculate_scattering_factors(
            energy_ev, wavelength, mass_density, molecular_weight, element_data
        )

        # All should be zero
        assert np.allclose(dispersion, 0.0)
        assert np.allclose(absorption, 0.0)
        assert np.allclose(f1_total, 0.0)
        assert np.allclose(f2_total, 0.0)


class TestCalculateDerivedQuantities:
    """Test the calculate_derived_quantities function."""

    def test_calculate_derived_quantities_basic(self):
        """Test basic derived quantity calculations."""
        # Simple test case
        wavelength = np.array([1.24e-10])  # ~10 keV in meters
        dispersion = np.array([1e-6])  # Typical dispersion value
        absorption = np.array([1e-8])  # Typical absorption value
        mass_density = 2.2
        molecular_weight = 60.08
        number_of_electrons = 30.0  # Si: 14, O: 8*2 = 16, total = 30

        electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density,
                molecular_weight,
                number_of_electrons,
            )
        )

        # Verify types and values
        assert isinstance(electron_density, (float, np.floating))
        assert isinstance(critical_angle, np.ndarray)
        assert isinstance(attenuation_length, np.ndarray)
        assert isinstance(re_sld, np.ndarray)
        assert isinstance(im_sld, np.ndarray)

        assert electron_density > 0
        assert critical_angle[0] > 0
        assert attenuation_length[0] > 0
        assert re_sld[0] != 0  # Can be positive or negative
        assert im_sld[0] != 0  # Can be positive or negative

        # Check physical reasonableness
        assert electron_density < 1  # electrons/Å³ should be reasonable
        assert critical_angle[0] < 90  # Critical angle should be < 90 degrees
        assert attenuation_length[0] > 0.001  # Should be at least microns

    def test_calculate_derived_quantities_multiple_values(self):
        """Test with multiple wavelength/dispersion/absorption values."""
        wavelength = np.array([1.0e-10, 1.24e-10, 1.5e-10])
        dispersion = np.array([1e-6, 2e-6, 1.5e-6])
        absorption = np.array([1e-8, 1.5e-8, 2e-8])
        mass_density = 2.2
        molecular_weight = 60.08
        number_of_electrons = 30.0

        electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
            calculate_derived_quantities(
                wavelength,
                dispersion,
                absorption,
                mass_density,
                molecular_weight,
                number_of_electrons,
            )
        )

        # Verify all arrays have correct length
        assert len(critical_angle) == 3
        assert len(attenuation_length) == 3
        assert len(re_sld) == 3
        assert len(im_sld) == 3

        # All values should be finite
        assert np.all(np.isfinite(critical_angle))
        assert np.all(np.isfinite(attenuation_length))
        assert np.all(np.isfinite(re_sld))
        assert np.all(np.isfinite(im_sld))


class TestCalculateXrayProperties:
    """Test the high-level calculate_xray_properties function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_calculate_xray_properties_single_energy(self):
        """Test calculation for single energy."""
        try:
            result = calculate_single_material_properties("SiO2", 10.0, 2.2)

            # Verify result is XRayResult object
            assert isinstance(result, XRayResult)

            # Verify basic attributes
            assert result.Formula == "SiO2"
            assert isinstance(result.MW, (float, np.floating))
            assert isinstance(result.Number_Of_Electrons, (float, np.floating))
            assert result.Density == 2.2

            # Arrays should have length 1 for single energy
            assert len(result.Energy) == 1
            assert len(result.Wavelength) == 1
            assert len(result.Dispersion) == 1
            assert len(result.Absorption) == 1
            assert len(result.f1) == 1
            assert len(result.f2) == 1
            assert len(result.Critical_Angle) == 1
            assert len(result.Attenuation_Length) == 1
            assert len(result.reSLD) == 1
            assert len(result.imSLD) == 1

            # Check physical reasonableness
            assert result.MW > 0
            assert result.Number_Of_Electrons > 0
            assert result.Electron_Density > 0
            assert result.Energy[0] == 10.0
            assert result.Wavelength[0] > 0

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_calculate_xray_properties_multiple_energies(self):
        """Test calculation for multiple energies."""
        try:
            energies = [8.0, 10.0, 12.0, 15.0]
            result = calculate_single_material_properties("Al2O3", energies, 3.95)

            # Arrays should have correct length
            n_energies = len(energies)
            assert len(result.Energy) == n_energies
            assert len(result.Wavelength) == n_energies
            assert len(result.Dispersion) == n_energies
            assert len(result.Absorption) == n_energies
            assert len(result.f1) == n_energies
            assert len(result.f2) == n_energies
            assert len(result.Critical_Angle) == n_energies
            assert len(result.Attenuation_Length) == n_energies
            assert len(result.reSLD) == n_energies
            assert len(result.imSLD) == n_energies

            # All values should be finite
            assert np.all(np.isfinite(result.Dispersion))
            assert np.all(np.isfinite(result.Absorption))
            assert np.all(np.isfinite(result.f1))
            assert np.all(np.isfinite(result.f2))
            assert np.all(np.isfinite(result.Critical_Angle))
            assert np.all(np.isfinite(result.Attenuation_Length))
            assert np.all(np.isfinite(result.reSLD))
            assert np.all(np.isfinite(result.imSLD))

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_calculate_xray_properties_input_validation(self):
        """Test input validation."""
        # Empty formula
        with pytest.raises(ValueError, match="Formula must be a non-empty string"):
            calculate_single_material_properties("", 10.0, 2.2)

        # Negative density
        with pytest.raises(ValueError, match="Mass density must be positive"):
            calculate_single_material_properties("SiO2", 10.0, -1.0)

        # Zero energy
        with pytest.raises(ValueError, match="All energies must be positive"):
            calculate_single_material_properties("SiO2", 0.0, 2.2)

        # Energy out of range
        with pytest.raises(ValueError, match="Energy is out of range"):
            calculate_single_material_properties("SiO2", 0.01, 2.2)  # Too low

        with pytest.raises(ValueError, match="Energy is out of range"):
            calculate_single_material_properties("SiO2", 50.0, 2.2)  # Too high


class TestCalculateMultipleXrayProperties:
    """Test the calculate_multiple_xray_properties function."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_calculate_multiple_xray_properties_basic(self):
        """Test calculation for multiple formulas."""
        try:
            formulas = ["SiO2", "Al2O3"]
            energy = 10.0
            densities = [2.2, 3.95]

            results = calculate_multiple_xray_properties(formulas, energy, densities)

            # Should have results for both formulas
            assert len(results) >= 1  # At least one should succeed

            for formula in formulas:
                if formula in results:
                    result = results[formula]
                    assert result["formula"] == formula
                    assert isinstance(result["molecular_weight"], (float, np.floating))
                    assert result["molecular_weight"] > 0

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_calculate_multiple_xray_properties_input_validation(self):
        """Test input validation for multiple formulas."""
        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            calculate_multiple_xray_properties(["SiO2", "Al2O3"], 10.0, [2.2])

        # Empty formula list
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_multiple_xray_properties([], 10.0, [])


class TestPhysicsConsistency:
    """Test consistency with known physics and Julia implementation."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_energy_wavelength_consistency(self):
        """Test energy-wavelength conversion consistency."""
        try:
            result = calculate_single_material_properties("SiO2", 10.0, 2.2)

            # Check E = hc/λ relationship
            energy_kev = result.Energy[0]
            wavelength_angstrom = result.Wavelength[0]

            # Convert back to energy
            calculated_energy = ENERGY_TO_WAVELENGTH_FACTOR / (wavelength_angstrom * 1e-10)  # type: ignore

            assert abs(calculated_energy - energy_kev) < 1e-10

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")

    def test_dispersion_absorption_scaling(self):
        """Test that dispersion and absorption scale with density."""
        try:
            # Same material, different densities
            result1 = calculate_single_material_properties("SiO2", 10.0, 2.2)
            result2 = calculate_single_material_properties(
                "SiO2", 10.0, 4.4
            )  # Double density

            # Dispersion and absorption should scale with density
            ratio = result2.Dispersion[0] / result1.Dispersion[0]
            assert abs(ratio - 2.0) < 0.1  # Should be approximately 2

            ratio = result2.Absorption[0] / result1.Absorption[0]
            assert abs(ratio - 2.0) < 0.1  # Should be approximately 2

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
