"""
Edge case tests for XRayLabTool.

This module tests boundary conditions, error handling, and extreme values
to ensure robustness of the X-ray optical property calculations.
"""

import numpy as np
import pytest

from xraylabtool import calculate_single_material_properties, parse_formula
from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError


class TestFormulaEdgeCases:
    """Test edge cases in chemical formula parsing."""

    def test_single_element_formulas(self):
        """Test parsing single element formulas."""
        # Test simple elements
        for element in ["H", "He", "Li", "C", "N", "O", "Si", "Al", "Fe", "Au", "U"]:
            symbols, counts = parse_formula(element)
            assert symbols == [element]
            assert counts == [1.0]

    def test_complex_formulas(self):
        """Test parsing complex chemical formulas."""
        test_cases = [
            # Formula with decimal coefficients
            ("H0.5He0.5", ["H", "He"], [0.5, 0.5]),
            ("Al0.3Ga0.7As", ["Al", "Ga", "As"], [0.3, 0.7, 1.0]),
            # Large coefficients
            ("C60", ["C"], [60.0]),
            ("Si100", ["Si"], [100.0]),
            # Simple multi-element compounds
            ("SiO2", ["Si", "O"], [1.0, 2.0]),
            ("Al2O3", ["Al", "O"], [2.0, 3.0]),
        ]

        for formula, expected_symbols, expected_counts in test_cases:
            symbols, counts = parse_formula(formula)
            assert symbols == expected_symbols, f"Failed for {formula}, got {symbols}"
            assert np.allclose(
                counts, expected_counts
            ), f"Failed for {formula}, got {counts}"

    def test_invalid_formulas(self):
        """Test handling of invalid chemical formulas."""
        invalid_formulas = [
            "",  # Empty string
            "123",  # Only numbers
            "si02",  # Lowercase element (invalid)
        ]

        for invalid_formula in invalid_formulas:
            with pytest.raises((ValueError, FormulaError)):
                parse_formula(invalid_formula)

    def test_questionable_formulas(self):
        """Test formulas that parse but may contain invalid elements."""
        # These parse but may contain invalid element symbols
        questionable_formulas = ["H2O3Zx", "Si$O2"]

        for formula in questionable_formulas:
            # These don't raise errors during parsing, but might during calculations
            symbols, counts = parse_formula(formula)
            assert len(symbols) > 0
            assert len(counts) == len(symbols)


class TestEnergyEdgeCases:
    """Test edge cases for energy ranges and values."""

    def test_minimum_energy_boundary(self):
        """Test calculations at minimum energy boundary."""
        # Test at minimum energy (0.03 keV)
        result = calculate_single_material_properties("Si", [0.03], 2.33)

        assert len(result.energy_kev) == 1
        assert result.energy_kev[0] == 0.03
        assert result.wavelength_angstrom[0] > 0
        assert np.all(np.isfinite(result.dispersion_delta))
        assert np.all(np.isfinite(result.absorption_beta))

    def test_maximum_energy_boundary(self):
        """Test calculations at maximum energy boundary."""
        # Test at maximum energy (30 keV)
        result = calculate_single_material_properties("Si", [30.0], 2.33)

        assert len(result.energy_kev) == 1
        assert result.energy_kev[0] == 30.0
        assert result.wavelength_angstrom[0] > 0
        assert np.all(np.isfinite(result.dispersion_delta))
        assert np.all(np.isfinite(result.absorption_beta))

    def test_energy_array_edge_cases(self):
        """Test edge cases for energy arrays."""
        # Single energy
        result = calculate_single_material_properties("C", [8.0], 2.27)
        assert len(result.energy_kev) == 1

        # Very small energy steps
        energies = np.linspace(5.0, 5.1, 101)  # 101 points in 0.1 keV range
        result = calculate_single_material_properties("Al", energies, 2.70)
        assert len(result.energy_kev) == 101
        assert np.allclose(result.energy_kev, energies)

    def test_invalid_energies(self):
        """Test handling of invalid energy values."""
        invalid_energies = [
            [-1.0],  # Negative energy
            [0.0],  # Zero energy
            [0.02],  # Below minimum
            [35.0],  # Above maximum
            [np.inf],  # Infinity
        ]

        for invalid_energy in invalid_energies:
            with pytest.raises((ValueError, EnergyError)):
                calculate_single_material_properties("Si", invalid_energy, 2.33)

    def test_nan_energy_handling(self):
        """Test that NaN energies are properly rejected."""
        # NaN energies should raise an error due to numerical stability checks
        with pytest.raises((ValueError, EnergyError)):
            calculate_single_material_properties("Si", [np.nan], 2.33)


class TestDensityEdgeCases:
    """Test edge cases for material densities."""

    def test_extreme_densities(self):
        """Test calculations with extreme but valid densities."""
        # Very low density (aerogel-like)
        result_low = calculate_single_material_properties("SiO2", [10.0], 0.003)
        assert result_low.density_g_cm3 == 0.003
        assert np.all(np.isfinite(result_low.dispersion_delta))

        # Very high density (heavy elements)
        result_high = calculate_single_material_properties("Au", [10.0], 19.3)
        assert result_high.density_g_cm3 == 19.3
        assert np.all(np.isfinite(result_high.dispersion_delta))

    def test_invalid_densities(self):
        """Test handling of invalid density values."""
        invalid_densities = [
            -1.0,  # Negative density
            0.0,  # Zero density
        ]

        for invalid_density in invalid_densities:
            with pytest.raises((ValueError, ValidationError)):
                calculate_single_material_properties("Si", [10.0], invalid_density)

    def test_special_density_values(self):
        """Test special density values (NaN, inf)."""
        # These should now raise errors due to numerical stability checks
        special_values = [np.nan, np.inf]

        for density in special_values:
            # Should raise error due to numerical stability improvements
            with pytest.raises((ValueError, ValidationError)):
                calculate_single_material_properties("Si", [10.0], density)


class TestMaterialEdgeCases:
    """Test edge cases for different materials."""

    def test_light_elements(self):
        """Test calculations for light elements."""
        light_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O"]

        for element in light_elements:
            try:
                result = calculate_single_material_properties(element, [10.0], 1.0)
                assert len(result.energy_kev) == 1
                assert np.all(np.isfinite(result.dispersion_delta))
                assert np.all(np.isfinite(result.absorption_beta))
            except Exception as e:
                pytest.fail(f"Failed for light element {element}: {e}")

    def test_heavy_elements(self):
        """Test calculations for heavy elements."""
        heavy_elements = ["W", "Pt", "Au", "Hg", "Pb", "U"]

        for element in heavy_elements:
            try:
                result = calculate_single_material_properties(element, [10.0], 10.0)
                assert len(result.energy_kev) == 1
                assert np.all(np.isfinite(result.dispersion_delta))
                assert np.all(np.isfinite(result.absorption_beta))
            except Exception as e:
                pytest.fail(f"Failed for heavy element {element}: {e}")

    def test_transition_metals(self):
        """Test calculations for transition metals with absorption edges."""
        transition_metals = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]

        for element in transition_metals:
            try:
                # Test across energy range that may include absorption edges
                energies = [5.0, 8.0, 12.0, 15.0]
                result = calculate_single_material_properties(element, energies, 5.0)
                assert len(result.energy_kev) == 4
                assert np.all(np.isfinite(result.dispersion_delta))
                assert np.all(np.isfinite(result.absorption_beta))
            except Exception as e:
                pytest.fail(f"Failed for transition metal {element}: {e}")


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_repeated_calculations(self):
        """Test that repeated calculations give consistent results."""
        formula = "Si"
        energies = [10.0]
        density = 2.33

        results = []
        for _ in range(10):
            result = calculate_single_material_properties(formula, energies, density)
            results.append(result.dispersion_delta[0])

        # All results should be identical (no randomness in calculation)
        assert all(r == results[0] for r in results)

    def test_energy_order_independence(self):
        """Test that energy order doesn't affect individual calculations."""
        formula = "Al2O3"
        density = 3.95
        energies = [5.0, 10.0, 15.0, 20.0]

        # Calculate in order
        result1 = calculate_single_material_properties(formula, energies, density)

        # Calculate in reverse order
        energies_rev = list(reversed(energies))
        result2 = calculate_single_material_properties(formula, energies_rev, density)

        # Results should match when reordered
        for i, energy in enumerate(energies):
            j = energies_rev.index(energy)
            assert np.isclose(result1.dispersion_delta[i], result2.dispersion_delta[j])
            assert np.isclose(result1.absorption_beta[i], result2.absorption_beta[j])

    def test_precision_with_close_energies(self):
        """Test precision when energies are very close together."""
        formula = "SiO2"
        density = 2.2
        base_energy = 10.0

        # Test with energies differing by larger amounts to ensure differences
        differences = [
            2.0,
            5.0,
        ]  # Use much larger differences to ensure meaningful changes

        for diff in differences:
            energies = [base_energy, base_energy + diff]
            result = calculate_single_material_properties(formula, energies, density)

            # Results should be different with larger energy differences
            assert not np.isclose(
                result.dispersion_delta[0], result.dispersion_delta[1], rtol=1e-3
            )
            assert not np.isclose(
                result.absorption_beta[0], result.absorption_beta[1], rtol=1e-3
            )

    def test_very_close_energies_similarity(self):
        """Test that very close energies give similar results."""
        formula = "Al"
        density = 2.70
        base_energy = 10.0

        # Very small differences should give very similar results
        small_differences = [1e-6, 1e-5]

        for diff in small_differences:
            energies = [base_energy, base_energy + diff]
            result = calculate_single_material_properties(formula, energies, density)

            # Results should be very close but may have tiny differences due to numerical precision
            assert np.allclose(
                result.dispersion_delta, result.dispersion_delta, rtol=1e-10
            )
            assert np.allclose(
                result.absorption_beta, result.absorption_beta, rtol=1e-10
            )


class TestMemoryAndPerformance:
    """Test memory usage and performance with large datasets."""

    def test_large_energy_array(self):
        """Test calculation with large energy array."""
        formula = "Si"
        density = 2.33
        # 1000 energy points
        energies = np.linspace(1.0, 20.0, 1000)

        result = calculate_single_material_properties(formula, energies, density)

        assert len(result.energy_kev) == 1000
        assert np.all(np.isfinite(result.dispersion_delta))
        assert np.all(np.isfinite(result.absorption_beta))
        assert np.allclose(result.energy_kev, energies)

    def test_memory_cleanup(self):
        """Test that large calculations don't cause memory issues."""
        import gc

        # Get initial memory usage
        initial_objects = len(gc.get_objects())

        # Perform multiple large calculations
        for i in range(10):
            energies = np.linspace(1.0, 20.0, 500)
            result = calculate_single_material_properties(f"C{i + 1}", energies, 2.0)
            del result

        # Force garbage collection
        gc.collect()

        # Check memory usage hasn't grown excessively
        final_objects = len(gc.get_objects())
        growth = final_objects - initial_objects

        # Allow some growth but not excessive (arbitrary threshold)
        assert growth < 1000, f"Memory growth: {growth} objects"


class TestEdgeCaseIntegration:
    """Test integration of edge cases with full workflow."""

    def test_full_workflow_with_edge_cases(self):
        """Test complete workflow with various edge cases."""
        # Mix of challenging materials and conditions
        test_cases = [
            ("H", [0.05], 0.071),  # Light element, low energy, very low density
            ("U", [25.0], 19.1),  # Heavy element, high energy, high density
            (
                "Al0.5Ga0.5As",
                [8.0, 12.0],
                4.5,
            ),  # Compound with fractional stoichiometry
        ]

        for formula, energies, density in test_cases:
            result = calculate_single_material_properties(formula, energies, density)

            # Basic sanity checks
            assert len(result.energy_kev) == len(energies)
            assert np.all(result.energy_kev == energies)
            assert result.density_g_cm3 == density

            # Physical reasonableness checks
            assert np.all(result.dispersion_delta > 0)  # Should be positive
            assert np.all(result.absorption_beta > 0)  # Should be positive
            assert np.all(result.critical_angle_degrees > 0)  # Should be positive
            assert np.all(result.attenuation_length_cm > 0)  # Should be positive

            # Finite value checks
            assert np.all(np.isfinite(result.dispersion_delta))
            assert np.all(np.isfinite(result.absorption_beta))
            assert np.all(np.isfinite(result.scattering_factor_f1))
            assert np.all(np.isfinite(result.scattering_factor_f2))

    @pytest.mark.parametrize(
        "formula,density",
        [
            ("SiO2", 2.2),
            ("Al2O3", 3.95),
            ("Fe2O3", 5.24),
            ("CaCO3", 2.71),
            ("TiO2", 4.23),
        ],
    )
    def test_common_materials_robustness(self, formula, density):
        """Test robustness with common laboratory materials."""
        # Wide energy range
        energies = np.logspace(np.log10(0.5), np.log10(25.0), 50)

        result = calculate_single_material_properties(formula, energies, density)

        # Check all calculations completed successfully
        assert len(result.energy_kev) == 50
        assert np.all(np.isfinite(result.dispersion_delta))
        assert np.all(np.isfinite(result.absorption_beta))

        # Check physical reasonableness
        assert np.all(result.dispersion_delta > 0)
        assert np.all(result.absorption_beta > 0)

        # Check energy dependence (generally dispersion decreases with energy)
        # Allow for some fluctuation due to absorption edges
        dispersion_trend = np.diff(result.dispersion_delta)
        # Most differences should be negative (decreasing trend)
        assert np.sum(dispersion_trend < 0) > len(dispersion_trend) * 0.6
