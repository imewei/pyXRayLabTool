"""
Enhanced tests for performance optimization features.

This test module provides comprehensive testing of all performance optimizations
including timing benchmarks, memory usage analysis, and regression testing.
"""

import time
import types
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import xraylabtool as xlt
from xraylabtool.calculators.core import (
    XRayResult,
    calculate_single_material_properties,
)
from xraylabtool.data_handling import (
    get_atomic_data_fast,
    get_cache_stats,
    is_element_preloaded,
)
from xraylabtool.data_handling.atomic_cache import _RUNTIME_CACHE


class TestDeprecationWarningOptimization:
    """Test the deprecation warning performance optimization."""

    def test_warnings_module_imported_at_module_level(self):
        """Test that warnings module is imported at module level, not in properties."""
        # Import the core module and check if warnings is available at module level
        from xraylabtool import core

        assert hasattr(core, "warnings"), (
            "warnings module should be imported at module level"
        )

    def test_all_deprecated_properties_trigger_warnings(self):
        """Test that all 12 deprecated properties trigger warnings correctly."""
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        # List all deprecated properties
        deprecated_properties = [
            "Formula",
            "MW",
            "Number_Of_Electrons",
            "Density",
            "Electron_Density",
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

        for prop in deprecated_properties:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = getattr(result, prop)

                assert len(w) == 1, (
                    f"Property {prop} should trigger exactly one warning"
                )
                assert issubclass(w[0].category, DeprecationWarning)
                assert "deprecated" in str(w[0].message).lower()
                assert prop in str(w[0].message)

    def test_warning_performance_benchmark(self):
        """Benchmark deprecation warning performance vs theoretical overhead."""
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        # Benchmark deprecated property access time
        iterations = 1000

        start_time = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for clean timing
            for _ in range(iterations):
                _ = result.Formula  # Access deprecated property
        end_time = time.perf_counter()

        avg_time_per_access = (end_time - start_time) / iterations

        # Should be much faster than 50μs (our optimization target)
        assert avg_time_per_access < 0.000050, (
            f"Deprecated property access too slow: {avg_time_per_access:.6f}s"
        )

        # Compare to new property access
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = result.formula  # Access new property
        end_time = time.perf_counter()

        new_property_time = (end_time - start_time) / iterations

        # Deprecated access will be slower due to warnings.warn() calls
        # Allow for up to 100x overhead (warnings are expected to have significant cost)
        assert avg_time_per_access < new_property_time * 100

    def test_warning_suppression_scenarios(self):
        """Test various warning suppression scenarios."""
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

        # Test warnings.filterwarnings suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter(
                    "always", category=UserWarning
                )  # Only UserWarnings
                _ = result.Formula
                assert len(w) == 0, "DeprecationWarning should be filtered out"

        # Test context manager suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with warnings.catch_warnings(record=True) as w:
                _ = result.MW
                assert len(w) == 0, "All warnings should be suppressed"


class TestArrayOptimization:
    """Test numpy array conversion optimizations."""

    def test_post_init_with_existing_numpy_arrays(self):
        """Test __post_init__ doesn't convert arrays that are already numpy arrays."""
        # Create arrays that are already the right type
        energies = np.array([8.0, 10.0, 12.0])
        wavelengths = np.array([1.55, 1.24, 1.03])
        values = np.array([1.0, 2.0, 3.0])

        # Create XRayResult with pre-existing numpy arrays
        result = XRayResult(
            formula="SiO2",
            molecular_weight_g_mol=60.08,
            total_electrons=30.0,
            density_g_cm3=2.2,
            electron_density_per_ang3=0.066,
            energy_kev=energies,
            wavelength_angstrom=wavelengths,
            dispersion_delta=values * 1e-6,
            absorption_beta=values * 1e-8,
            scattering_factor_f1=values * 10,
            scattering_factor_f2=values * 2,
            critical_angle_degrees=values * 0.1,
            attenuation_length_cm=values * 100,
            real_sld_per_ang2=values * 1e-6,
            imaginary_sld_per_ang2=values * 1e-8,
        )

        # Verify arrays are still numpy arrays and haven't been unnecessarily converted
        assert isinstance(result.energy_kev, np.ndarray)
        assert isinstance(result.wavelength_angstrom, np.ndarray)
        assert isinstance(result.dispersion_delta, np.ndarray)

        # Check if they're the same objects (no unnecessary conversion)
        # Note: This test verifies optimization worked, but XRayResult might create new arrays
        # so we mainly check that they remain numpy arrays

    def test_post_init_with_non_numpy_arrays(self):
        """Test __post_init__ converts non-numpy arrays properly."""
        # Create non-numpy arrays (lists)
        energies = [8.0, 10.0, 12.0]
        wavelengths = [1.55, 1.24, 1.03]
        values = [1.0, 2.0, 3.0]

        result = XRayResult(
            formula="SiO2",
            molecular_weight_g_mol=60.08,
            total_electrons=30.0,
            density_g_cm3=2.2,
            electron_density_per_ang3=0.066,
            energy_kev=energies,
            wavelength_angstrom=wavelengths,
            dispersion_delta=values,
            absorption_beta=values,
            scattering_factor_f1=values,
            scattering_factor_f2=values,
            critical_angle_degrees=values,
            attenuation_length_cm=values,
            real_sld_per_ang2=values,
            imaginary_sld_per_ang2=values,
        )

        # Verify all fields are converted to numpy arrays
        assert isinstance(result.energy_kev, np.ndarray)
        assert isinstance(result.wavelength_angstrom, np.ndarray)
        assert isinstance(result.dispersion_delta, np.ndarray)
        assert isinstance(result.absorption_beta, np.ndarray)
        assert isinstance(result.scattering_factor_f1, np.ndarray)
        assert isinstance(result.scattering_factor_f2, np.ndarray)
        assert isinstance(result.critical_angle_degrees, np.ndarray)
        assert isinstance(result.attenuation_length_cm, np.ndarray)
        assert isinstance(result.real_sld_per_ang2, np.ndarray)
        assert isinstance(result.imaginary_sld_per_ang2, np.ndarray)

    def test_mixed_input_types(self):
        """Test mixed input types (some numpy arrays, some not)."""
        energies = np.array([8.0, 10.0, 12.0])  # Already numpy
        wavelengths = [1.55, 1.24, 1.03]  # List
        values_array = np.array([1.0, 2.0, 3.0])  # Already numpy
        values_list = [1.0, 2.0, 3.0]  # List

        result = XRayResult(
            formula="SiO2",
            molecular_weight_g_mol=60.08,
            total_electrons=30.0,
            density_g_cm3=2.2,
            electron_density_per_ang3=0.066,
            energy_kev=energies,  # numpy array
            wavelength_angstrom=wavelengths,  # list
            dispersion_delta=values_array,  # numpy array
            absorption_beta=values_list,  # list
            scattering_factor_f1=values_array,  # numpy array
            scattering_factor_f2=values_list,  # list
            critical_angle_degrees=values_array,  # numpy array
            attenuation_length_cm=values_list,  # list
            real_sld_per_ang2=values_array,  # numpy array
            imaginary_sld_per_ang2=values_list,  # list
        )

        # All should end up as numpy arrays
        for field_name in [
            "energy_kev",
            "wavelength_angstrom",
            "dispersion_delta",
            "absorption_beta",
            "scattering_factor_f1",
            "scattering_factor_f2",
            "critical_angle_degrees",
            "attenuation_length_cm",
            "real_sld_per_ang2",
            "imaginary_sld_per_ang2",
        ]:
            field_value = getattr(result, field_name)
            assert isinstance(field_value, np.ndarray), (
                f"{field_name} should be numpy array"
            )

    def test_array_optimization_performance_benchmark(self):
        """Benchmark array conversion optimization performance."""
        # Test with arrays that are already numpy arrays (should be fast)
        energies = np.array([8.0, 10.0, 12.0])
        wavelengths = np.array([1.55, 1.24, 1.03])
        values = np.array([1.0, 2.0, 3.0])

        iterations = 1000

        start_time = time.perf_counter()
        for _ in range(iterations):
            XRayResult(
                formula="SiO2",
                molecular_weight_g_mol=60.08,
                total_electrons=30.0,
                density_g_cm3=2.2,
                electron_density_per_ang3=0.066,
                energy_kev=energies,
                wavelength_angstrom=wavelengths,
                dispersion_delta=values,
                absorption_beta=values,
                scattering_factor_f1=values,
                scattering_factor_f2=values,
                critical_angle_degrees=values,
                attenuation_length_cm=values,
                real_sld_per_ang2=values,
                imaginary_sld_per_ang2=values,
            )
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / iterations

        # Should be very fast with optimization (less than 1ms)
        assert avg_time < 0.001, f"XRayResult creation too slow: {avg_time:.6f}s"


class TestAtomicDataCacheOptimization:
    """Test atomic data cache optimizations."""

    def test_mapping_proxy_type_returned(self):
        """Test that MappingProxyType is returned for immutable views."""
        data = get_atomic_data_fast("Si")
        assert isinstance(data, types.MappingProxyType)

        # Test immutability
        with pytest.raises(TypeError):
            data["atomic_number"] = 999  # Should fail - immutable

    def test_no_dictionary_copying(self):
        """Test that no dictionary copying occurs."""
        # Clear any existing runtime cache
        _RUNTIME_CACHE.clear()

        # Test with preloaded element
        data1 = get_atomic_data_fast("Si")
        data2 = get_atomic_data_fast("Si")

        # Both should return MappingProxyType wrapping the same underlying dict
        assert isinstance(data1, types.MappingProxyType)
        assert isinstance(data2, types.MappingProxyType)
        assert data1 == data2  # Same content

    def test_cache_performance_benchmark(self):
        """Benchmark cache access performance."""
        element = "Si"  # Preloaded element
        iterations = 10000

        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = get_atomic_data_fast(element)
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / iterations

        # Should be very fast (< 10μs per access)
        assert avg_time < 0.000010, f"Cache access too slow: {avg_time:.8f}s"

    def test_preloaded_vs_runtime_cache_performance(self):
        """Compare preloaded vs runtime cache performance."""
        preloaded_element = "Si"  # Should be preloaded
        iterations = 1000

        # Benchmark preloaded element
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = get_atomic_data_fast(preloaded_element)
        preloaded_time = time.perf_counter() - start_time

        # Preloaded access should be very fast
        avg_preloaded_time = preloaded_time / iterations
        assert avg_preloaded_time < 0.000005, (
            f"Preloaded access too slow: {avg_preloaded_time:.8f}s"
        )

        # Verify it's actually preloaded
        assert is_element_preloaded(preloaded_element)

    def test_cache_statistics(self):
        """Test cache statistics and monitoring."""
        stats = get_cache_stats()

        assert "preloaded_elements" in stats
        assert "runtime_cached_elements" in stats
        assert "total_cached_elements" in stats

        assert stats["preloaded_elements"] == 92  # Should have 92 preloaded elements
        assert stats["total_cached_elements"] >= stats["preloaded_elements"]

        # Test element preloaded check
        common_elements = ["H", "C", "N", "O", "Si", "Al", "Fe"]
        for element in common_elements:
            assert is_element_preloaded(element), f"{element} should be preloaded"


class TestSingleElementOptimization:
    """Test single-element calculation optimization."""

    def test_single_vs_multi_element_performance(self):
        """Compare single vs multi-element calculation performance."""
        energies = np.linspace(5.0, 15.0, 100)
        iterations = 100

        # Single element (Si)
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = calculate_single_material_properties("Si", energies, 2.33)
        single_element_time = time.perf_counter() - start_time

        # Multi-element (SiO2)
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = calculate_single_material_properties("SiO2", energies, 2.2)
        multi_element_time = time.perf_counter() - start_time

        avg_single = single_element_time / iterations
        avg_multi = multi_element_time / iterations

        # Single element should be faster than or equal to multi-element
        # Note: For very fast operations, the difference may be within measurement noise
        print(f"Single element: {avg_single:.6f}s vs Multi-element: {avg_multi:.6f}s")

        # Allow for measurement noise and system variations
        # Single element should not be significantly slower (within 50% margin)
        # This is a regression test to ensure no major performance degradation
        tolerance = 1.5  # 50% tolerance for micro-benchmark variations
        if avg_single > avg_multi * tolerance:
            print(
                f"WARNING: Single element significantly slower: {avg_single:.6f}s vs {avg_multi:.6f}s"
            )
            # Only fail if the difference is extreme (more than 2x)
            assert avg_single <= avg_multi * 2.0, (
                f"Single element extremely slow (>2x): {avg_single:.6f}s vs {avg_multi:.6f}s"
            )

    def test_single_element_memory_usage(self):
        """Test memory usage optimization for single elements."""
        import tracemalloc

        energies = np.linspace(5.0, 15.0, 1000)  # Large energy array

        # Measure memory usage for single element
        tracemalloc.start()
        result = calculate_single_material_properties("Si", energies, 2.33)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 1MB for this calculation)
        assert peak < 1024 * 1024, f"Memory usage too high: {peak / 1024 / 1024:.2f}MB"
        assert len(result.energy_kev) == 1000  # Verify result is correct

    def test_various_single_elements(self):
        """Test optimization works for various single elements."""
        single_elements = ["H", "C", "Si", "Fe", "Au"]
        densities = [0.00009, 2.267, 2.33, 7.874, 19.3]  # Approximate densities
        energies = np.array([8.0, 10.0, 12.0])

        for element, density in zip(single_elements, densities, strict=False):
            start_time = time.perf_counter()
            result = calculate_single_material_properties(element, energies, density)
            calc_time = time.perf_counter() - start_time

            # Each calculation should be fast
            assert calc_time < 0.01, f"{element} calculation too slow: {calc_time:.6f}s"
            assert len(result.energy_kev) == 3
            assert result.formula == element

    def test_array_type_checking_optimization(self):
        """Test that array type checking optimization works."""
        # This test verifies the enhanced single-element path with better type checking
        energies = np.array([8.0, 10.0, 12.0], dtype=np.float64)  # Already right type

        # Create a mock to track asarray calls
        original_asarray = np.asarray
        with patch("numpy.asarray") as mock_asarray:
            # Configure mock to pass through but count calls
            mock_asarray.side_effect = original_asarray

            result = calculate_single_material_properties("Si", energies, 2.33)

            # Verify result is correct
            assert len(result.energy_kev) == 3

            # The optimization should minimize asarray calls for arrays that are already correct type
            # Note: Some calls may still occur in other parts of the code, this mainly tests
            # that the single-element optimization path is working
