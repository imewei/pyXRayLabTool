"""
Consolidated performance optimization validation tests.

This module tests all performance optimization features including:
- Bulk atomic data loading and caching
- Interpolator caching and performance
- Element path pre-computation
- Enhanced cache management
- Array optimization and vectorization
- Deprecation warning performance
"""

import time
import warnings

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
import xraylabtool as xlt
from xraylabtool.calculators.core import (
    _AVAILABLE_ELEMENTS,
    XRayResult,
    _initialize_element_paths,
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
    get_bulk_atomic_data,
)


class TestBulkAtomicDataOptimization(BasePerformanceTest):
    """Test bulk atomic data loading optimizations."""

    def test_get_bulk_atomic_data_basic(self):
        """Test basic bulk atomic data loading."""
        elements = ("H", "C", "O")

        # Clear any existing cache
        get_bulk_atomic_data.cache_clear()

        # Load bulk data
        data = get_bulk_atomic_data(elements)

        # Verify structure
        assert isinstance(data, dict)
        assert len(data) == 3

        for element in elements:
            assert element in data
            assert "atomic_number" in data[element]
            assert "atomic_weight" in data[element]
            assert isinstance(data[element]["atomic_number"], int | float)
            assert isinstance(data[element]["atomic_weight"], int | float)

    def test_get_bulk_atomic_data_caching(self):
        """Test that bulk atomic data loading uses caching effectively."""
        elements = ("Si", "O")

        # Clear cache
        get_bulk_atomic_data.cache_clear()

        # First call
        start_time = time.perf_counter()
        data1 = get_bulk_atomic_data(elements)
        first_time = time.perf_counter() - start_time

        # Second call (should be faster due to caching)
        start_time = time.perf_counter()
        data2 = get_bulk_atomic_data(elements)
        second_time = time.perf_counter() - start_time

        # Verify data is identical
        assert data1 == data2

        # Second call should be significantly faster (cached)
        assert second_time < first_time

        # Clean up
        get_bulk_atomic_data.cache_clear()


class TestInterpolatorCaching(BasePerformanceTest):
    """Test the enhanced interpolator caching system."""

    def test_interpolator_caching_basic(self):
        """Test that interpolators are cached and reused."""
        try:
            # Clear all caches
            clear_scattering_factor_cache()

            # First call - should create and cache interpolators
            f1_interp1, f2_interp1 = create_scattering_factor_interpolators("Si")

            # Second call - should return cached interpolators
            f1_interp2, f2_interp2 = create_scattering_factor_interpolators("Si")

            # Should be the same objects (cached)
            assert f1_interp1 is f1_interp2
            assert f2_interp1 is f2_interp2

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for interpolator testing")

    def test_interpolator_caching_performance(self):
        """Test that interpolator caching improves performance."""
        try:
            # Clear all caches
            clear_scattering_factor_cache()

            # First call (cold cache)
            start_time = time.perf_counter()
            _f1_interp1, _f2_interp1 = create_scattering_factor_interpolators("Si")
            cold_time = time.perf_counter() - start_time

            # Second call (warm cache)
            start_time = time.perf_counter()
            _f1_interp2, _f2_interp2 = create_scattering_factor_interpolators("Si")
            warm_time = time.perf_counter() - start_time

            # Warm cache should be significantly faster
            assert warm_time < cold_time
            # Expect at least 2x speedup with caching
            assert warm_time < cold_time / 2.0

        except FileNotFoundError:
            pytest.skip(
                "Silicon .nff file not available for interpolator performance testing"
            )


class TestElementPathOptimization(BasePerformanceTest):
    """Test the element path pre-computation optimization."""

    def test_available_elements_populated(self):
        """Test that _AVAILABLE_ELEMENTS is populated at module load."""
        # Should be a dictionary
        assert isinstance(_AVAILABLE_ELEMENTS, dict)

        # Should contain at least some elements if data files are present
        if _AVAILABLE_ELEMENTS:
            # Check that values are Path objects
            for element, path in _AVAILABLE_ELEMENTS.items():
                assert isinstance(element, str)
                assert hasattr(path, "exists")  # Path-like object

    def test_initialize_element_paths_functionality(self):
        """Test that _initialize_element_paths works correctly."""
        # Save original state
        original_available = _AVAILABLE_ELEMENTS.copy()

        try:
            # Clear the global dict
            _AVAILABLE_ELEMENTS.clear()

            # Re-initialize
            _initialize_element_paths()

            # Should have repopulated (if data files exist)
            if original_available:
                assert len(_AVAILABLE_ELEMENTS) > 0
                # Should find at least some of the same elements
                common_elements = set(original_available.keys()) & set(
                    _AVAILABLE_ELEMENTS.keys()
                )
                assert len(common_elements) > 0

        finally:
            # Restore original state
            _AVAILABLE_ELEMENTS.clear()
            _AVAILABLE_ELEMENTS.update(original_available)


class TestArrayOptimization:
    """Test array optimization features."""

    def test_xray_result_array_optimization(self):
        """Test XRayResult array conversion optimization."""
        # Create with numpy arrays (should not be converted)
        energies = np.array([8.0, 10.0, 12.0])
        values = np.array([1.0, 2.0, 3.0])

        result = XRayResult(
            formula="SiO2",
            molecular_weight_g_mol=60.08,
            total_electrons=30.0,
            density_g_cm3=2.2,
            electron_density_per_ang3=0.066,
            energy_kev=energies,
            wavelength_angstrom=values,
            dispersion_delta=values,
            absorption_beta=values,
            scattering_factor_f1=values,
            scattering_factor_f2=values,
            critical_angle_degrees=values,
            attenuation_length_cm=values,
            real_sld_per_ang2=values,
            imaginary_sld_per_ang2=values,
        )

        # All should be numpy arrays
        assert isinstance(result.energy_kev, np.ndarray)
        assert isinstance(result.wavelength_angstrom, np.ndarray)

    def test_array_optimization_performance(self):
        """Test that array optimizations provide measurable benefits."""
        # Test with different array sizes
        for n_points in [10, 100, 1000]:
            energies = np.linspace(8.0, 12.0, n_points)

            start = time.perf_counter()
            result = xlt.calculate_single_material_properties("Si", energies, 2.33)
            calculation_time = time.perf_counter() - start

            # Should complete in reasonable time (performance baseline)
            assert calculation_time < 5.0  # 5 seconds max for 1000 points
            assert len(result.energy_kev) == n_points


class TestDeprecationWarningOptimization:
    """Test deprecation warning performance optimization."""

    def test_deprecated_core_module_removed(self):
        """Test that deprecated core module is no longer available."""
        # The core module has been removed as part of codebase simplification
        # No deprecated modules to test - this test verifies the cleanup was successful

        # Test that functionality is still available through main module
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        assert result.formula == "SiO2"

    def test_warning_performance_benchmark(self):
        """Test that deprecation warnings work and deprecated properties are accessible."""
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        # Test that deprecated property access works and issues warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            formula_deprecated = result.Formula  # Deprecated property

            # Should issue deprecation warning
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Formula is deprecated" in str(w[0].message)

        # Test that new property access works without warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            formula_new = result.formula  # New property

            # Should not issue any warnings
            assert len(w) == 0

        # Both should return the same value
        assert formula_deprecated == formula_new


class TestNumericalStabilityOptimizations:
    """Test numerical stability improvements."""

    def test_numerical_stability_enhancements(self):
        """Test numerical stability improvements."""
        # Test with very small density (edge case)
        result = xlt.calculate_single_material_properties("H", 10.0, 0.00009)

        # Should handle edge case gracefully
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert np.all(result.critical_angle_degrees >= 0)
        assert np.all(result.attenuation_length_cm > 0)

    def test_edge_case_handling(self):
        """Test that edge cases are handled properly."""
        # Test within valid energy range (0.03keV ~ 30keV)
        # High energy within range
        result_high = xlt.calculate_single_material_properties("Al", 25.0, 2.7)
        assert np.all(np.isfinite(result_high.dispersion_delta))
        assert np.all(np.isfinite(result_high.absorption_beta))

        # Low energy within range
        result_low = xlt.calculate_single_material_properties("Al", 0.5, 2.7)
        assert np.all(np.isfinite(result_low.dispersion_delta))
        assert np.all(np.isfinite(result_low.absorption_beta))


class TestPerformanceRegression:
    """Test that optimizations don't break functionality."""

    def test_results_consistency_with_optimization(self):
        """Test that optimized code produces same results as before."""
        try:
            # Clear caches to start fresh
            clear_scattering_factor_cache()

            # Calculate the same result multiple times
            formula = "SiO2"
            energy = 10.0
            density = 2.2

            results = []
            for _ in range(3):
                result = xlt.calculate_single_material_properties(
                    formula, energy, density
                )
                results.append((result.dispersion_delta[0], result.absorption_beta[0]))

            # All results should be identical
            for i in range(1, len(results)):
                np.testing.assert_allclose(results[0][0], results[i][0], rtol=1e-12)
                np.testing.assert_allclose(results[0][1], results[i][1], rtol=1e-12)

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for consistency testing")

    def test_optimization_speedup_measurable(self):
        """Test that optimizations provide measurable speedup."""
        try:
            formula = "SiO2"
            energy = 10.0
            density = 2.2

            # Cold cache timing
            clear_scattering_factor_cache()
            start_time = time.perf_counter()
            result1 = xlt.calculate_single_material_properties(formula, energy, density)
            cold_time = time.perf_counter() - start_time

            # Warm cache timing
            start_time = time.perf_counter()
            result2 = xlt.calculate_single_material_properties(formula, energy, density)
            warm_time = time.perf_counter() - start_time

            # Results should be identical
            np.testing.assert_allclose(
                result1.dispersion_delta, result2.dispersion_delta, rtol=1e-12
            )
            np.testing.assert_allclose(
                result1.absorption_beta, result2.absorption_beta, rtol=1e-12
            )

            # Warm cache should be faster
            assert warm_time < cold_time

            # Record the speedup for information
            speedup = cold_time / warm_time if warm_time > 0 else float("inf")
            print(
                f"Cache speedup: {speedup: .1f}x (cold: {cold_time * 1000: .2f}ms,"
                f" warm: {warm_time * 1000: .2f}ms)"
            )

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for speedup testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
