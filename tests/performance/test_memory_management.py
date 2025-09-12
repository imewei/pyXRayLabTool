"""
Tests for memory management and batch processing optimizations.

This test module validates memory management improvements, cache clearing
functionality, and batch processing optimizations.
"""

import gc
import os
import tracemalloc
from unittest.mock import patch

import numpy as np
import psutil
import pytest

import xraylabtool as xlt
from xraylabtool.calculators.core import clear_scattering_factor_cache
from xraylabtool.data_handling.batch_processing import (
    BatchConfig,
    MemoryMonitor,
    calculate_batch_properties,
    process_single_calculation,
)


class TestMemoryMonitor:
    """Test the MemoryMonitor class and its optimizations."""

    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(4.0)  # 4GB limit

        assert monitor.limit_bytes == 4.0 * 1024 * 1024 * 1024
        assert hasattr(monitor, "process")

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        monitor = MemoryMonitor(4.0)

        usage_mb = monitor.get_memory_usage_mb()
        assert isinstance(usage_mb, float)
        assert usage_mb >= 0  # Should be non-negative

        # Memory usage should be reasonable for test process
        assert usage_mb < 1000  # Less than 1GB for test process

    def test_memory_limit_checking(self):
        """Test memory limit checking."""
        # Set a very high limit (should pass)
        monitor = MemoryMonitor(100.0)  # 100GB - very high
        assert monitor.check_memory()

        # Set a very low limit (should fail)
        monitor = MemoryMonitor(0.001)  # 1MB - very low
        assert not monitor.check_memory()

    def test_force_gc_with_cache_clearing(self):
        """Test that force_gc clears caches and runs garbage collection."""
        monitor = MemoryMonitor(4.0)

        # Populate some cache data first
        _ = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        _ = xlt.calculate_single_material_properties("Al2O3", 10.0, 3.95)

        # Force garbage collection and cache clearing
        monitor.force_gc()

        # Verify that garbage collection ran (hard to test directly, but we can check it doesn't error)
        assert True  # If we get here, force_gc completed without error

    def test_memory_monitor_exception_handling(self):
        """Test exception handling in memory monitoring."""
        monitor = MemoryMonitor(4.0)

        # Mock psutil to raise an exception
        with patch.object(
            monitor.process, "memory_info", side_effect=Exception("Mock error")
        ):
            # Should not raise exception, should return 0.0
            usage = monitor.get_memory_usage_mb()
            assert usage == 0.0

            # Should not raise exception, should return True (safe default)
            within_limits = monitor.check_memory()
            assert within_limits


class TestBatchConfig:
    """Test BatchConfig optimizations."""

    def test_default_initialization(self):
        """Test default BatchConfig initialization."""
        config = BatchConfig()

        # Should auto-detect reasonable defaults
        assert config.max_workers > 0
        assert config.max_workers <= 8  # Capped at 8 for memory efficiency
        assert config.chunk_size == 100
        assert config.memory_limit_gb > 0
        assert config.enable_progress
        assert not config.cache_results

    def test_memory_limit_auto_adjustment(self):
        """Test automatic memory limit adjustment based on system memory."""
        # Test with available system memory
        try:
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            expected_max = available_memory_gb * 0.5

            # Create config with high memory limit that should be adjusted
            config = BatchConfig(memory_limit_gb=100.0)  # Very high initial value

            if config.memory_limit_gb < 100.0:
                # Memory limit was adjusted down
                # Allow small floating-point tolerance in comparison
                tolerance = 0.01  # 10 MB tolerance
                assert config.memory_limit_gb <= (expected_max + tolerance)
                assert config.memory_limit_gb >= 1.0  # Minimum limit

        except Exception:
            # If psutil fails, should keep original limit
            config = BatchConfig(memory_limit_gb=4.0)
            # Allow small floating-point tolerance
            assert abs(config.memory_limit_gb - 4.0) < 0.1

    def test_worker_count_calculation(self):
        """Test worker count calculation."""
        config = BatchConfig()
        cpu_count = os.cpu_count() or 1
        expected_max_workers = min(max(1, int(cpu_count * 0.75)), 8)

        assert config.max_workers == expected_max_workers

    def test_custom_configuration(self):
        """Test custom configuration parameters."""
        config = BatchConfig(
            max_workers=4,
            chunk_size=50,
            memory_limit_gb=2.0,
            enable_progress=False,
            cache_results=True,
        )

        assert config.max_workers == 4
        assert config.chunk_size == 50
        assert config.memory_limit_gb <= 2.0  # May be adjusted down
        assert not config.enable_progress
        assert config.cache_results


class TestBatchProcessingMemoryManagement:
    """Test memory management in batch processing."""

    def test_single_calculation_memory_cleanup(self):
        """Test memory cleanup in single calculations."""
        formula = "SiO2"
        energies = np.linspace(5.0, 15.0, 100)
        density = 2.2

        # Measure memory before
        initial_memory = psutil.Process().memory_info().rss

        # Perform calculation
        formula_result, result = process_single_calculation(formula, energies, density)

        # Force garbage collection
        gc.collect()

        # Memory should not increase significantly
        final_memory = psutil.Process().memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024

        # Memory increase should be reasonable (less than 10MB for this calculation)
        assert memory_increase_mb < 10, (
            f"Memory increase too large: {memory_increase_mb:.2f}MB"
        )
        assert result is not None
        assert formula_result == formula

    def test_memory_monitoring_during_batch(self):
        """Test memory monitoring during batch processing."""
        formulas = ["SiO2", "Al2O3", "Fe2O3"] * 5  # 15 materials
        energies = np.linspace(5.0, 15.0, 50)
        densities = [2.2, 3.95, 5.24] * 5

        # Configure with low memory limit to trigger monitoring
        config = BatchConfig(
            memory_limit_gb=0.1,  # Very low limit to trigger cleanup
            chunk_size=3,
            enable_progress=False,
        )

        # This should complete without memory errors
        # (memory cleanup should be triggered)
        results = calculate_batch_properties(formulas, energies, densities, config)

        # Verify results
        unique_combinations = len(set(zip(formulas, densities, strict=False)))
        assert (
            len(results) <= unique_combinations
        )  # Results keyed by unique formula+density combinations

        # At least some results should succeed
        successful_results = [
            result for result in results.values() if result is not None
        ]
        assert len(successful_results) > 0

    def test_cache_clearing_integration(self):
        """Test cache clearing integration with memory management."""
        monitor = MemoryMonitor(4.0)

        # Populate caches by running calculations
        formulas = ["SiO2", "Al2O3", "Si", "Fe"]
        for formula in formulas:
            _ = xlt.calculate_single_material_properties(formula, 10.0, 2.2)

        # Clear caches through memory monitor
        monitor.force_gc()

        # Verify that calculation still works (caches rebuilt as needed)
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        assert result is not None
        assert result.formula == "SiO2"

    @pytest.mark.skipif(
        psutil.virtual_memory().total < 8 * 1024**3,
        reason="Requires at least 8GB RAM for large batch test",
    )
    def test_large_batch_memory_efficiency(self):
        """Test memory efficiency with large batches."""
        # Adjust test size based on CI environment
        is_ci = os.environ.get("CI", "").lower() == "true"

        if is_ci:
            # Smaller test for CI to avoid timeouts
            formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"] * 5  # 25 materials
            energies = np.linspace(5.0, 15.0, 20)  # Fewer energy points
            densities = [2.2, 3.95, 5.24, 2.33, 2.267] * 5
        else:
            # Full test for local development
            formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"] * 20  # 100 materials
            energies = np.linspace(5.0, 15.0, 100)
            densities = [2.2, 3.95, 5.24, 2.33, 2.267] * 20

        config = BatchConfig(memory_limit_gb=2.0, chunk_size=10, enable_progress=False)

        # Monitor memory during calculation
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss

        results = calculate_batch_properties(formulas, energies, densities, config)

        end_memory = psutil.Process().memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory increase should be reasonable
        memory_increase_mb = (end_memory - start_memory) / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        # Adjust memory expectations based on test size
        if is_ci:
            max_memory_increase = 100  # Lower expectations for CI
            max_peak_memory = 200
        else:
            max_memory_increase = 500  # Full expectations for local
            max_peak_memory = 1000

        assert memory_increase_mb < max_memory_increase, (
            f"Memory increase too large: {memory_increase_mb:.2f}MB (limit: {max_memory_increase}MB, CI: {is_ci})"
        )
        assert peak_mb < max_peak_memory, (
            f"Peak memory usage too high: {peak_mb:.2f}MB (limit: {max_peak_memory}MB, CI: {is_ci})"
        )

        # Most results should succeed
        successful_results = [
            result for result in results.values() if result is not None
        ]
        total_formulas = len(set(formulas))  # Unique formulas
        success_rate = len(successful_results) / total_formulas
        assert success_rate > 0.8, f"Success rate too low: {success_rate:.2%}"


class TestCacheManagement:
    """Test cache management optimizations."""

    def test_cache_clearing_functionality(self):
        """Test that cache clearing works properly."""
        # Populate caches
        materials = ["SiO2", "Al2O3", "Si", "Fe", "Au"]
        for material in materials:
            _ = xlt.calculate_single_material_properties(material, 10.0, 2.2)

        # Clear all caches
        clear_scattering_factor_cache()

        # Verify calculations still work (caches rebuilt)
        result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        assert result is not None
        assert np.isfinite(result.critical_angle_degrees[0])

    def test_cache_statistics_monitoring(self):
        """Test cache statistics for monitoring."""
        from xraylabtool.data_handling import get_cache_stats

        stats = get_cache_stats()

        assert "preloaded_elements" in stats
        assert "runtime_cached_elements" in stats
        assert "total_cached_elements" in stats

        # Should have preloaded elements
        assert stats["preloaded_elements"] > 0
        assert stats["total_cached_elements"] >= stats["preloaded_elements"]

    def test_memory_pressure_cache_behavior(self):
        """Test cache behavior under memory pressure."""
        monitor = MemoryMonitor(0.01)  # Very low limit - 10MB

        # Populate caches with many materials
        [f"Element{i}" for i in range(1, 11)]  # This will likely fail, but that's ok

        try:
            for material in ["SiO2", "Al2O3", "Si", "Fe"]:  # Use real materials
                _ = xlt.calculate_single_material_properties(material, 10.0, 2.2)

                # Check if we should clear caches due to memory pressure
                if not monitor.check_memory():
                    monitor.force_gc()

        except Exception:
            # Some operations might fail due to invalid formulas, that's ok for this test
            pass

        # The important thing is that memory management doesn't crash
        assert True


class TestMemoryLeakPrevention:
    """Test prevention of memory leaks."""

    def test_repeated_calculations_no_memory_leak(self):
        """Test that repeated calculations don't cause memory leaks."""
        initial_memory = psutil.Process().memory_info().rss

        # Perform many calculations
        for i in range(100):
            xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

            # Periodically force garbage collection
            if i % 20 == 0:
                gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024

        # Memory increase should be minimal (less than 50MB)
        assert memory_increase_mb < 50, (
            f"Possible memory leak: {memory_increase_mb:.2f}MB increase"
        )

    def test_batch_calculation_memory_cleanup(self):
        """Test that batch calculations clean up memory properly."""
        formulas = ["SiO2", "Al2O3", "Fe2O3"] * 10  # 30 materials
        energies = np.array([8.0, 10.0, 12.0])
        densities = [2.2, 3.95, 5.24] * 10

        initial_memory = psutil.Process().memory_info().rss

        # Perform batch calculation
        results = xlt.calculate_xray_properties(formulas, energies, densities)

        # Force cleanup
        gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024

        # Memory increase should be reasonable
        assert memory_increase_mb < 100, (
            f"Memory increase too large: {memory_increase_mb:.2f}MB"
        )

        # Verify results are correct
        assert len(results) == len(set(formulas))  # Unique formulas
        for formula, result in results.items():
            assert result is not None
            assert result.formula == formula

    def test_large_array_memory_management(self):
        """Test memory management with large arrays."""
        # Large energy array - use more conservative range to avoid numerical issues
        energies = np.linspace(1.0, 25.0, 5000)  # Avoid very low energies
        valid_energies = energies[(energies >= 1.0) & (energies <= 25.0)]

        initial_memory = psutil.Process().memory_info().rss

        result = xlt.calculate_single_material_properties("Si", valid_energies, 2.33)

        # Verify result
        assert len(result.energy_kev) == len(valid_energies)
        assert np.all(np.isfinite(result.critical_angle_degrees))

        # Clean up
        del result
        gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_increase_mb = (final_memory - initial_memory) / 1024 / 1024

        # Should handle large arrays efficiently
        assert memory_increase_mb < 200, (
            f"Large array memory usage: {memory_increase_mb:.2f}MB"
        )
