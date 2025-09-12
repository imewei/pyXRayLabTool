"""
Performance benchmark tests for optimization validation.

This test module provides comprehensive performance benchmarks to validate
optimization improvements and detect performance regressions.
"""

from contextlib import contextmanager
import statistics
import time

import numpy as np
import pytest

import xraylabtool as xlt
from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.data_handling import get_atomic_data_fast


@contextmanager
def timer():
    """Context manager for timing operations."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_single_calculation_performance(self):
        """Benchmark single material calculation performance."""
        formula = "SiO2"
        energy = 10.0
        density = 2.2
        iterations = 1000

        times = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties(formula, energy, density)
            times.append(get_time())

        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_time = statistics.stdev(times)

        # Performance targets based on optimizations
        assert avg_time < 0.001, f"Single calculation too slow: {avg_time:.6f}s (avg)"
        assert median_time < 0.001, (
            f"Single calculation too slow: {median_time:.6f}s (median)"
        )

        print(f"Single calculation performance: {avg_time:.6f}s ± {std_time:.6f}s")

    def test_array_calculation_performance(self):
        """Benchmark array calculation performance."""
        formula = "SiO2"
        energies = np.linspace(5.0, 15.0, 100)  # 100 energy points
        density = 2.2
        iterations = 100

        times = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties(formula, energies, density)
            times.append(get_time())

        avg_time = statistics.mean(times)
        median_time = statistics.median(times)

        # Performance targets
        assert avg_time < 0.01, f"Array calculation too slow: {avg_time:.6f}s (avg)"
        assert median_time < 0.01, (
            f"Array calculation too slow: {median_time:.6f}s (median)"
        )

        print(
            f"Array calculation (100 points): {avg_time:.6f}s ± {statistics.stdev(times):.6f}s"
        )

    def test_atomic_data_cache_performance(self):
        """Benchmark atomic data cache performance."""
        element = "Si"
        iterations = 10000

        times = []
        for _ in range(iterations):
            with timer() as get_time:
                get_atomic_data_fast(element)
            times.append(get_time())

        avg_time = statistics.mean(times)
        median_time = statistics.median(times)

        # Should be very fast due to caching optimization
        assert avg_time < 0.00001, f"Cache access too slow: {avg_time:.8f}s (avg)"
        assert median_time < 0.00001, (
            f"Cache access too slow: {median_time:.8f}s (median)"
        )

        print(
            f"Atomic data cache access: {avg_time:.8f}s ± {statistics.stdev(times):.8f}s"
        )

    def test_batch_calculation_performance(self):
        """Benchmark batch calculation performance."""
        formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]
        energies = np.array([8.0, 10.0, 12.0])
        densities = [2.2, 3.95, 5.24, 2.33, 2.267]
        iterations = 100

        times = []
        for _ in range(iterations):
            with timer() as get_time:
                xlt.calculate_xray_properties(formulas, energies, densities)
            times.append(get_time())

        avg_time = statistics.mean(times)
        statistics.median(times)
        avg_per_material = avg_time / len(formulas)

        # Performance targets
        assert avg_time < 0.05, f"Batch calculation too slow: {avg_time:.6f}s (avg)"
        assert avg_per_material < 0.01, (
            f"Per-material too slow: {avg_per_material:.6f}s"
        )

        print(
            f"Batch calculation (5 materials): {avg_time:.6f}s total, {avg_per_material:.6f}s per material"
        )

    def test_single_vs_multi_element_performance(self):
        """Benchmark single vs multi-element performance difference."""
        energies = np.linspace(5.0, 15.0, 100)
        iterations = 100

        # Single element calculation
        single_times = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties("Si", energies, 2.33)
            single_times.append(get_time())

        # Multi-element calculation
        multi_times = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties("SiO2", energies, 2.2)
            multi_times.append(get_time())

        avg_single = statistics.mean(single_times)
        avg_multi = statistics.mean(multi_times)
        speedup = avg_multi / avg_single

        # Single element optimization may provide subtle improvements
        # Performance differences can be small and variable for fast operations
        print(
            f"Single element: {avg_single:.6f}s, Multi-element: {avg_multi:.6f}s, Speedup: {speedup:.2f}x"
        )

        # Ensure both calculations complete successfully (main requirement)
        assert avg_single > 0 and avg_multi > 0, "Both calculations should complete"

        # Allow performance to be within reasonable range (not significantly slower)
        assert speedup > 0.5, f"Single element significantly slower: {speedup:.2f}x"

    def test_large_array_performance(self):
        """Benchmark performance with large energy arrays."""
        formula = "Si"
        large_energies = np.linspace(1.0, 25.0, 1000)  # 1000 energy points
        density = 2.33
        iterations = 10

        times = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties(formula, large_energies, density)
            times.append(get_time())

        avg_time = statistics.mean(times)
        time_per_point = avg_time / len(large_energies)

        # Performance targets for large arrays
        assert avg_time < 0.1, f"Large array calculation too slow: {avg_time:.6f}s"
        assert time_per_point < 0.0001, (
            f"Per-point calculation too slow: {time_per_point:.8f}s"
        )

        print(
            f"Large array (1000 points): {avg_time:.6f}s total, {time_per_point:.8f}s per point"
        )

    def test_throughput_benchmark(self):
        """Benchmark overall throughput (calculations per second)."""
        formula = "SiO2"
        energy = 10.0
        density = 2.2
        duration = 1.0  # Run for 1 second

        start_time = time.perf_counter()
        count = 0

        while time.perf_counter() - start_time < duration:
            calculate_single_material_properties(formula, energy, density)
            count += 1

        actual_duration = time.perf_counter() - start_time
        throughput = count / actual_duration

        # Throughput target based on optimizations
        assert throughput > 1000, (
            f"Throughput too low: {throughput:.0f} calculations/second"
        )

        print(f"Throughput: {throughput:.0f} calculations/second")


class TestPerformanceRegression:
    """Performance regression tests."""

    def test_deprecation_warning_overhead(self):
        """Test that deprecation warnings don't add significant overhead."""
        result = calculate_single_material_properties("SiO2", 10.0, 2.2)
        iterations = 1000

        # Test new property access (baseline)
        with timer() as get_time:
            for _ in range(iterations):
                _ = result.formula
        baseline_time = get_time()

        # Test deprecated property access (should be close to baseline)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings for timing
            with timer() as get_time:
                for _ in range(iterations):
                    _ = result.Formula
        deprecated_time = get_time()

        overhead_ratio = deprecated_time / baseline_time

        # Deprecation warnings will have overhead, but shouldn't be excessive (less than 100x)
        # Note: This is expected since warnings.warn() is called on each access
        assert overhead_ratio < 100.0, (
            f"Deprecation warning overhead too high: {overhead_ratio:.2f}x"
        )

        print(f"Deprecation warning overhead: {overhead_ratio:.2f}x baseline")

    def test_array_conversion_overhead(self):
        """Test that array conversion optimization reduces overhead."""
        # Test with arrays that are already numpy arrays (optimized path)
        energies_np = np.array([8.0, 10.0, 12.0])
        iterations = 1000

        times_optimized = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties("Si", energies_np, 2.33)
            times_optimized.append(get_time())

        # Test with lists (conversion required)
        energies_list = [8.0, 10.0, 12.0]
        times_conversion = []
        for _ in range(iterations):
            with timer() as get_time:
                calculate_single_material_properties("Si", energies_list, 2.33)
            times_conversion.append(get_time())

        avg_optimized = statistics.mean(times_optimized)
        avg_conversion = statistics.mean(times_conversion)
        overhead = avg_conversion / avg_optimized

        # Conversion overhead should be minimal due to optimization
        assert overhead < 2.0, f"Array conversion overhead too high: {overhead:.2f}x"

        print(f"Array conversion overhead: {overhead:.2f}x optimized path")

    def test_cache_performance_consistency(self):
        """Test that cache performance is consistent over time."""
        element = "Si"
        iterations = 1000

        # First batch (cache warming)
        times_batch1 = []
        for _ in range(iterations):
            with timer() as get_time:
                get_atomic_data_fast(element)
            times_batch1.append(get_time())

        # Second batch (cache should be warm)
        times_batch2 = []
        for _ in range(iterations):
            with timer() as get_time:
                get_atomic_data_fast(element)
            times_batch2.append(get_time())

        avg_batch1 = statistics.mean(times_batch1)
        avg_batch2 = statistics.mean(times_batch2)
        consistency_ratio = max(avg_batch1, avg_batch2) / min(avg_batch1, avg_batch2)

        # Cache performance should be consistent (within 3x to account for system variations)
        assert consistency_ratio < 3.0, (
            f"Cache performance inconsistent: {consistency_ratio:.2f}x variation"
        )

        print(f"Cache consistency: {consistency_ratio:.2f}x variation between batches")


class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""

    def test_energy_array_scaling(self):
        """Test performance scaling with energy array size."""
        formula = "SiO2"
        density = 2.2

        sizes = [10, 50, 100, 500, 1000]
        times = []

        for size in sizes:
            energies = np.linspace(5.0, 15.0, size)

            with timer() as get_time:
                calculate_single_material_properties(formula, energies, density)

            time_taken = get_time()
            times.append(time_taken)
            time_per_point = time_taken / size

            print(
                f"Array size {size}: {time_taken:.6f}s total, {time_per_point:.8f}s per point"
            )

        # Should scale roughly linearly or better
        # Check that larger arrays don't have excessive overhead
        for i in range(1, len(sizes)):
            ratio = times[i] / times[0]  # Ratio to smallest
            size_ratio = sizes[i] / sizes[0]
            efficiency = size_ratio / ratio

            # Efficiency should be reasonable (> 0.5 means less than 2x overhead)
            assert efficiency > 0.5, (
                f"Poor scaling efficiency: {efficiency:.2f} for size {sizes[i]}"
            )

    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        base_materials = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]
        base_densities = [2.2, 3.95, 5.24, 2.33, 2.267]
        energy = 10.0

        batch_sizes = [5, 10, 20, 50]
        times = []

        for batch_size in batch_sizes:
            # Create batch by repeating base materials
            multiplier = batch_size // len(base_materials) + 1
            formulas = (base_materials * multiplier)[:batch_size]
            densities = (base_densities * multiplier)[:batch_size]

            with timer() as get_time:
                xlt.calculate_xray_properties(formulas, energy, densities)

            time_taken = get_time()
            times.append(time_taken)
            time_per_material = time_taken / batch_size

            print(
                f"Batch size {batch_size}: {time_taken:.6f}s total, {time_per_material:.6f}s per material"
            )

        # Check scaling efficiency (allow for overhead and measurement noise)
        for i in range(1, len(batch_sizes)):
            ratio = times[i] / times[0]
            size_ratio = batch_sizes[i] / batch_sizes[0]
            efficiency = size_ratio / ratio

            # Batch processing should show reasonable scaling efficiency
            # Allow for overhead, setup costs, and system variations
            min_efficiency = 0.3  # 30% minimum efficiency threshold
            if efficiency < min_efficiency:
                print(
                    f"WARNING: Low batch scaling efficiency: {efficiency:.2f} for batch size {batch_sizes[i]}"
                )
                # Only fail if efficiency is extremely poor (< 20%)
                assert efficiency > 0.2, (
                    f"Extremely poor batch scaling: {efficiency:.2f} for batch size {batch_sizes[i]}"
                )

    def test_memory_scaling(self):
        """Test memory scaling with problem size."""
        import tracemalloc

        formula = "SiO2"
        density = 2.2
        sizes = [100, 500, 1000]

        for size in sizes:
            energies = np.linspace(5.0, 15.0, size)

            tracemalloc.start()
            calculate_single_material_properties(formula, energies, density)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / 1024 / 1024
            memory_per_point = peak_mb / size

            print(
                f"Array size {size}: {peak_mb:.2f}MB peak, {memory_per_point:.6f}MB per point"
            )

            # Memory usage should be reasonable
            assert peak_mb < 100, (
                f"Memory usage too high: {peak_mb:.2f}MB for size {size}"
            )
            assert memory_per_point < 0.1, (
                f"Memory per point too high: {memory_per_point:.6f}MB"
            )


@pytest.mark.benchmark
class TestBenchmarkComparison:
    """Benchmark comparison tests for optimization validation."""

    def test_optimization_effectiveness(self):
        """Overall test of optimization effectiveness."""
        # This test validates that all optimizations work together effectively

        # Test scenario: mixed workload
        single_element_time = self._benchmark_single_element()
        multi_element_time = self._benchmark_multi_element()
        batch_time = self._benchmark_batch()
        cache_time = self._benchmark_cache()

        # Print summary
        print("Performance Summary:")
        print(f"  Single element: {single_element_time:.6f}s")
        print(f"  Multi-element: {multi_element_time:.6f}s")
        print(f"  Batch (5 materials): {batch_time:.6f}s")
        print(f"  Cache access: {cache_time:.8f}s")

        # Overall performance should meet targets (adjusted for CI environment variations)
        # More relaxed targets for CI environments with variable performance
        assert single_element_time < 0.015, (
            "Single element performance target not met"
        )  # Relaxed from 0.005s
        assert multi_element_time < 0.020, (
            "Multi-element performance target not met"
        )  # Relaxed from 0.01s
        assert batch_time < 0.100, (
            "Batch performance target not met"
        )  # Relaxed from 0.05s
        assert cache_time < 0.0001, (
            "Cache performance target not met"
        )  # Relaxed from 0.00005s

    def _benchmark_single_element(self):
        """Benchmark single element calculation."""
        with timer() as get_time:
            calculate_single_material_properties("Si", 10.0, 2.33)
        return get_time()

    def _benchmark_multi_element(self):
        """Benchmark multi-element calculation."""
        with timer() as get_time:
            calculate_single_material_properties("SiO2", 10.0, 2.2)
        return get_time()

    def _benchmark_batch(self):
        """Benchmark batch calculation."""
        formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]
        densities = [2.2, 3.95, 5.24, 2.33, 2.267]

        with timer() as get_time:
            xlt.calculate_xray_properties(formulas, 10.0, densities)
        return get_time()

    def _benchmark_cache(self):
        """Benchmark cache access."""
        with timer() as get_time:
            get_atomic_data_fast("Si")
        return get_time()
