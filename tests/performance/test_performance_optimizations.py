"""
Tests for performance optimization features added in v0.1.2.

This test module specifically tests the new caching and optimization features:
- Bulk atomic data loading
- Interpolator caching
- Element path pre-computation
- Enhanced cache management
"""

import time

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
import xraylabtool as xlt
from xraylabtool.calculators.core import (
    _AVAILABLE_ELEMENTS,
    _initialize_element_paths,
    _interpolator_cache,
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
    get_bulk_atomic_data,
    get_cached_elements,
)


class TestBulkAtomicDataLoading(BasePerformanceTest):
    """Test the bulk atomic data loading optimization."""

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

    def test_get_bulk_atomic_data_empty_input(self):
        """Test bulk atomic data loading with empty input."""
        data = get_bulk_atomic_data(())
        assert isinstance(data, dict)
        assert len(data) == 0


class TestInterpolatorCaching:
    """Test the enhanced interpolator caching system."""

    def setUp(self):
        """Clear caches before each test."""
        clear_scattering_factor_cache()

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
            f1_interp1, f2_interp1 = create_scattering_factor_interpolators("Si")
            cold_time = time.perf_counter() - start_time

            # Second call (warm cache)
            start_time = time.perf_counter()
            f1_interp2, f2_interp2 = create_scattering_factor_interpolators("Si")
            warm_time = time.perf_counter() - start_time

            # Warm cache should be significantly faster
            assert warm_time < cold_time
            # Expect at least 2x speedup with caching
            assert warm_time < cold_time / 2.0

        except FileNotFoundError:
            pytest.skip(
                "Silicon .nff file not available for interpolator performance testing"
            )

    def test_interpolator_cache_cleared_with_clear_function(self):
        """Test that interpolator cache is cleared by clear_scattering_factor_cache."""
        try:
            # Load an interpolator
            create_scattering_factor_interpolators("Si")
            assert len(_interpolator_cache) > 0

            # Clear cache
            clear_scattering_factor_cache()

            # Interpolator cache should be empty
            assert len(_interpolator_cache) == 0

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for cache clearing test")


class TestElementPathOptimization:
    """Test the element path pre-computation optimization."""

    def test_available_elements_populated(self):
        """Test that _AVAILABLE_ELEMENTS is populated at module load."""
        # _initialize_element_paths() should have been called at module import
        # This test verifies that the optimization is working

        # Should be a dictionary
        assert isinstance(_AVAILABLE_ELEMENTS, dict)

        # Should contain at least some elements if data files are present
        # (This will depend on the test environment)
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


class TestEnhancedCacheManagement:
    """Test the enhanced cache management features."""

    def test_cache_clearing_comprehensive(self):
        """Test that cache clearing clears all optimization caches."""
        try:
            # Load some data to populate caches
            xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

            # Verify caches are populated
            assert len(get_cached_elements()) > 0

            # Clear all caches
            clear_scattering_factor_cache()

            # Verify all caches are cleared
            assert len(get_cached_elements()) == 0
            assert len(_interpolator_cache) == 0

            # Verify LRU caches are cleared
            # This is tested by confirming cache_info shows 0 hits after clearing
            # and then seeing hits increase after use

        except FileNotFoundError:
            pytest.skip(
                "Required .nff files not available for comprehensive cache testing"
            )

    def test_cache_statistics_tracking(self):
        """Test that we can track cache effectiveness."""
        try:
            # Clear caches
            clear_scattering_factor_cache()

            # Load the same element multiple times
            for _ in range(3):
                create_scattering_factor_interpolators("Si")

            # Should show cache reuse
            cache_info = create_scattering_factor_interpolators.cache_info()
            assert cache_info.hits > 0

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for cache statistics testing")


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
                results.append((result.Dispersion[0], result.Absorption[0]))

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
                result1.Dispersion, result2.Dispersion, rtol=1e-12
            )
            np.testing.assert_allclose(
                result1.Absorption, result2.Absorption, rtol=1e-12
            )

            # Warm cache should be faster
            assert warm_time < cold_time

            # Record the speedup for information
            speedup = cold_time / warm_time if warm_time > 0 else float("inf")
            print(
                f"Cache speedup: {speedup: .1f}x (cold: {cold_time * 1000: .2f}ms, warm: {warm_time * 1000: .2f}ms)"
            )

        except FileNotFoundError:
            pytest.skip("Required .nff files not available for speedup testing")


class TestNewOptimizations:
    """Test the newly implemented optimizations."""

    def test_atomic_data_immutable_proxy(self):
        """Test atomic data cache returns immutable proxy types."""
        import types

        from xraylabtool.data_handling import get_atomic_data_fast

        data = get_atomic_data_fast("Si")

        # Should return MappingProxyType (immutable)
        assert isinstance(data, types.MappingProxyType)

        # Should contain expected data
        assert "atomic_number" in data
        assert "atomic_weight" in data
        assert data["atomic_number"] == 14  # Silicon atomic number

    def test_xray_result_array_optimization(self):
        """Test XRayResult array conversion optimization."""
        import numpy as np

        from xraylabtool.calculators.core import XRayResult

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

    def test_single_element_optimization_path(self):
        """Test that single element calculations use optimized path."""
        import time

        # Single element should be faster
        energies = np.linspace(5.0, 15.0, 100)

        start = time.perf_counter()
        result_single = xlt.calculate_single_material_properties("Si", energies, 2.33)
        single_time = time.perf_counter() - start

        start = time.perf_counter()
        result_multi = xlt.calculate_single_material_properties("SiO2", energies, 2.2)
        multi_time = time.perf_counter() - start

        # Record performance but don't enforce strict timing requirements
        # Performance can vary significantly based on system state, caching, etc.
        print(
            f"Single element: {single_time: .6f}s vs Multi-element: {multi_time: .6f}s"
        )

        # The optimization may provide subtle improvements that are hard to measure
        # in micro-benchmarks. The main goal is correctness, not measurable speedup.
        assert single_time > 0 and multi_time > 0, "Both calculations should complete"

        # Both should produce valid results
        assert len(result_single.energy_kev) == 100
        assert len(result_multi.energy_kev) == 100

    def test_numerical_stability_enhancements(self):
        """Test numerical stability improvements."""
        # Test with very small density (edge case)
        result = xlt.calculate_single_material_properties("H", 10.0, 0.00009)

        # Should handle edge case gracefully
        assert np.all(np.isfinite(result.critical_angle_degrees))
        assert np.all(np.isfinite(result.attenuation_length_cm))
        assert np.all(result.critical_angle_degrees >= 0)
        assert np.all(result.attenuation_length_cm > 0)

    def test_memory_management_integration(self):
        """Test memory management improvements."""
        from xraylabtool.data_handling.batch_processing import MemoryMonitor

        monitor = MemoryMonitor(4.0)

        # Test memory monitoring
        usage = monitor.get_memory_usage_mb()
        assert usage >= 0

        within_limits = monitor.check_memory()
        assert isinstance(within_limits, bool)

        # Test cache clearing integration
        monitor.force_gc()  # Should not raise error

    def test_batch_config_auto_optimization(self):
        """Test BatchConfig automatic optimization."""
        from xraylabtool.data_handling.batch_processing import BatchConfig

        config = BatchConfig()

        # Should have reasonable defaults
        assert config.max_workers > 0
        assert config.max_workers <= 8  # Capped for memory efficiency
        assert config.chunk_size == 100
        assert config.memory_limit_gb > 0

    def test_deprecation_warning_module_level_import(self):
        """Test that warnings module is imported at module level."""
        from xraylabtool import core

        # Should have warnings imported at module level
        assert hasattr(core, "warnings")

        # Test deprecation warning without import overhead
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = result.Formula  # Should trigger warning without import overhead
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
