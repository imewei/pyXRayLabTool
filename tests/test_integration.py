"""
Integration tests for XRayLabTool Python package.

This test module is a complete translation of the Julia test/runtests.jl test suite,
ensuring identical behavior and validation between the Julia and Python implementations.
Uses numpy.isclose assertions for numerical comparisons as requested.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import xraylabtool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xraylabtool import calculate_xray_properties, calculate_single_material_properties

# Test constants (matching Julia test constants)
DEFAULT_TOL = 1e-6
ENERGIES = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
DENSITIES = [2.2, 1.0]
MATERIALS = ["SiO2", "H2O"]


class TestBasicSetupAndInitialization:
    """Test basic setup and initialization."""

    def test_basic_setup_and_initialization(self):
        """Test that materials are properly initialized."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)

        # Test that materials are properly initialized
        assert "SiO2" in data
        assert "H2O" in data

        # Test data structure integrity
        assert len(data) == 2
        assert all(material in data.keys() for material in MATERIALS)


class TestSiO2Properties:
    """Test SiO2 properties matching Julia test values."""

    def test_sio2_dispersion(self):
        """Test SiO2 dispersion values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        sio2 = data["SiO2"]

        # Expected values for SiO2 dispersion (index-1 based for Python)
        expected_dispersion = [
            (2, 9.451484792575434e-6),  # index 3 in Julia = index 2 in Python
            (4, 5.69919201789506e-06),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_dispersion:
            actual = sio2.Dispersion[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"SiO2 Dispersion[{idx}]: expected {expected_val}, got {actual}"

    def test_sio2_f1_values(self):
        """Test SiO2 f1 values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        sio2 = data["SiO2"]

        # Expected values for SiO2 f1 (index-1 based for Python)
        expected_f1 = [
            (0, 30.641090313037314),  # index 1 in Julia = index 0 in Python
            (2, 30.46419063207884),  # index 3 in Julia = index 2 in Python
            (4, 30.366953850108544),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_f1:
            actual = sio2.f1[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"SiO2 f1[{idx}]: expected {expected_val}, got {actual}"

    def test_sio2_resld_values(self):
        """Test SiO2 reSLD values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        sio2 = data["SiO2"]

        # Expected values for SiO2 reSLD (index-1 based for Python)
        expected_reSLD = [
            (2, 1.8929689855615698e-5),  # index 3 in Julia = index 2 in Python
            (4, 1.886926933936152e-5),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_reSLD:
            actual = sio2.reSLD[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"SiO2 reSLD[{idx}]: expected {expected_val}, got {actual}"


class TestH2OProperties:
    """Test H2O properties matching Julia test values."""

    def test_h2o_dispersion(self):
        """Test H2O dispersion values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        h2o = data["H2O"]

        # Expected values for H2O dispersion (index-1 based for Python)
        expected_dispersion = [
            (2, 4.734311949237782e-6),  # index 3 in Julia = index 2 in Python
            (4, 2.8574954896752405e-6),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_dispersion:
            actual = h2o.Dispersion[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"H2O Dispersion[{idx}]: expected {expected_val}, got {actual}"

    def test_h2o_f1_values(self):
        """Test H2O f1 values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        h2o = data["H2O"]

        # Expected values for H2O f1 (index-1 based for Python)
        expected_f1 = [
            (0, 10.110775776847062),  # index 1 in Julia = index 0 in Python
            (2, 10.065881494924541),  # index 3 in Julia = index 2 in Python
            (4, 10.04313810715771),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_f1:
            actual = h2o.f1[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"H2O f1[{idx}]: expected {expected_val}, got {actual}"

    def test_h2o_resld_values(self):
        """Test H2O reSLD values."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        h2o = data["H2O"]

        # Expected values for H2O reSLD (index-1 based for Python)
        expected_reSLD = [
            (2, 9.482008260671003e-6),  # index 3 in Julia = index 2 in Python
            (4, 9.460584107129207e-6),  # index 5 in Julia = index 4 in Python
        ]

        for idx, expected_val in expected_reSLD:
            actual = h2o.reSLD[idx]
            assert np.isclose(
                actual, expected_val, atol=DEFAULT_TOL
            ), f"H2O reSLD[{idx}]: expected {expected_val}, got {actual}"


class TestSubRefracSiliconProperties:
    """Test SubRefrac Silicon properties matching Julia test values."""

    def test_silicon_properties(self):
        """Test Silicon property values from SubRefrac."""
        si = calculate_single_material_properties("Si", [20.0], 2.33)

        # Expected values for Silicon (index-1 based for Python)
        expected_values = [
            ("Dispersion", 0, 1.20966554922812e-06),
            ("f1", 0, 14.048053047106292),
            ("f2", 0, 0.053331074920700626),
            ("reSLD", 0, 1.9777910804587255e-5),
            ("imSLD", 0, 7.508351793358633e-8),
        ]

        for property_name, idx, expected_val in expected_values:
            actual_val = getattr(si, property_name)[idx]
            assert np.isclose(
                actual_val, expected_val, atol=DEFAULT_TOL
            ), f"Silicon {property_name}[{idx}]: expected {expected_val}, got {actual_val}"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_materials(self):
        """Test with empty materials."""
        with pytest.raises(ValueError, match=r".*empty.*"):
            calculate_xray_properties([], ENERGIES, [])

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        with pytest.raises(
            ValueError, match=r".*Number of formulas.*must match.*number of densities.*"
        ):
            calculate_xray_properties(MATERIALS, ENERGIES, [1.0])  # Wrong density count

    def test_non_existent_material_access(self):
        """Test accessing non-existent material."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)
        assert "NonExistentMaterial" not in data


class TestPropertyConsistency:
    """Test property consistency across materials."""

    def test_property_consistency(self):
        """Test that all materials have the same energy array length."""
        data = calculate_xray_properties(MATERIALS, ENERGIES, DENSITIES)

        for material in MATERIALS:
            material_data = data[material]
            assert len(material_data.f1) == len(ENERGIES)
            assert len(material_data.Dispersion) == len(ENERGIES)
            assert len(material_data.reSLD) == len(ENERGIES)


class TestInputValidationAndErrorHandling:
    """Test comprehensive input validation and error handling."""

    def test_refrac_energy_below_minimum(self):
        """Test energy below 0.03 keV."""
        with pytest.raises(
            ValueError, match=r".*Energy values must be in range 0\.03-30 keV.*"
        ):
            calculate_xray_properties(["SiO2"], [0.02, 5.0], [2.2])

    def test_refrac_energy_above_maximum(self):
        """Test energy above 30 keV."""
        with pytest.raises(
            ValueError, match=r".*Energy values must be in range 0\.03-30 keV.*"
        ):
            calculate_xray_properties(["SiO2"], [5.0, 35.0], [2.2])

    def test_mismatched_list_lengths(self):
        """Test mismatched formulaList & massDensityList lengths."""
        with pytest.raises(
            ValueError, match=r".*Number of formulas.*must match.*number of densities.*"
        ):
            calculate_xray_properties(["SiO2", "Al2O3"], [8.0, 10.0], [2.2])

    def test_non_vector_formula_input(self):
        """Test non-vector formula input."""
        with pytest.raises((TypeError, ValueError)):
            calculate_xray_properties("SiO2", [8.0, 10.0], [2.2])  # type: ignore

    def test_empty_formula_list(self):
        """Test empty formula list."""
        with pytest.raises(ValueError, match=r".*empty.*"):
            calculate_xray_properties([], [8.0, 10.0], [])

    def test_empty_energy_vector(self):
        """Test empty energy vector."""
        with pytest.raises(ValueError, match=r".*empty.*"):
            calculate_xray_properties(["SiO2"], [], [2.2])


class TestSubRefracErrorHandling:
    """Test SubRefrac function error handling."""

    def test_invalid_chemical_formula(self):
        """Test invalid chemical formula."""
        from xraylabtool.utils import UnknownElementError

        with pytest.raises(UnknownElementError, match=r".*Unknown element.*"):
            calculate_single_material_properties("InvalidElement123", [8.0], 2.2)

    def test_empty_formula_string(self):
        """Test empty formula string."""
        with pytest.raises(ValueError, match=r".*Formula must be a non-empty string.*"):
            calculate_single_material_properties("", [8.0], 2.2)


class TestDuplicatedFormulasIndependence:
    """Test that duplicated formulas produce independent results."""

    def test_duplicate_formulas_yield_consistent_results(self):
        """Test duplicate formulas yield consistent results."""
        formulas = ["SiO2", "SiO2", "Al2O3"]
        energies = [8.0, 10.0, 12.0]
        densities = [2.2, 2.2, 3.95]

        results = calculate_xray_properties(formulas, energies, densities)

        # Both SiO2 entries should be present and identical
        assert "SiO2" in results
        assert "Al2O3" in results

        # Results should be consistent for the same material
        sio2_result = results["SiO2"]
        assert sio2_result.Formula == "SiO2"
        assert sio2_result.Density == 2.2
        assert len(sio2_result.Energy) == len(energies)


class TestEdgeCaseInputValues:
    """Test edge case input values."""

    def test_boundary_energy_values(self):
        """Test exactly at boundary energy values."""
        # Test exactly at boundaries (should work)
        result_min = calculate_xray_properties(["SiO2"], [0.03], [2.2])
        assert "SiO2" in result_min

        result_max = calculate_xray_properties(["SiO2"], [30.0], [2.2])
        assert "SiO2" in result_max

    def test_very_small_density(self):
        """Test with very small density."""
        result = calculate_single_material_properties("H2O", [8.0], 0.001)
        assert result.Density == 0.001
        assert result.Formula == "H2O"

    def test_very_large_density(self):
        """Test with very large density."""
        result = calculate_single_material_properties("Au", [8.0], 19.3)  # Gold density
        assert result.density_g_cm3 == 19.3
        assert result.formula == "Au"


# =====================================================================================
# PERFORMANCE BENCHMARK TESTS
# =====================================================================================


class TestPerformanceBenchmarks:
    """Performance benchmark tests using pytest-benchmark."""

    def test_benchmark_single_calculation(self, benchmark):
        """Benchmark single SubRefrac calculation."""

        def single_calculation():
            return calculate_single_material_properties(
                "SiO2", np.arange(1.0, 20.1, 0.1), 2.2
            )

        result = benchmark(single_calculation)
        assert result.Formula == "SiO2"
        assert len(result.Energy) == 191  # 1.0 to 20.0 in steps of 0.1

    def test_benchmark_multi_calculation(self, benchmark):
        """Benchmark multi-material Refrac calculation."""
        formulas = ["SiO2", "Al2O3", "Fe2O3", "CaCO3", "MgO"]
        densities = [2.2, 3.95, 5.24, 2.71, 3.58]
        energies = np.arange(1.0, 20.1, 0.1)

        def multi_calculation():
            return calculate_xray_properties(formulas, energies, densities)

        result = benchmark(multi_calculation)
        assert len(result) == 5
        for formula in formulas:
            assert formula in result
            assert len(result[formula].Energy) == 191

    def test_benchmark_energy_sweep(self, benchmark):
        """Benchmark calculation with large energy sweep."""

        def energy_sweep():
            energies = np.linspace(0.1, 25.0, 1000)
            return calculate_single_material_properties("SiO2", energies, 2.2)

        result = benchmark(energy_sweep)
        assert len(result.Energy) == 1000

    def test_benchmark_formula_complexity(self, benchmark):
        """Benchmark calculation with complex formula."""

        def complex_formula():
            # Use a complex mineral formula if available
            return calculate_single_material_properties(
                "SiO2", [5.0, 8.0, 10.0, 15.0, 20.0], 2.65
            )

        result = benchmark(complex_formula)
        assert len(result.Energy) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
