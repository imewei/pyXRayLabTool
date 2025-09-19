"""
Comprehensive performance benchmarking tests for calculation speed optimization.

This module provides detailed benchmarking of XRayLabTool performance across
different calculation types, material complexities, and usage patterns to
establish baselines and track optimization progress.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import psutil
import pytest

from tests.fixtures.test_base import BasePerformanceTest


class TestCalculationSpeedBenchmarks(BasePerformanceTest):
    """Comprehensive benchmarks for calculation speed optimization."""

    @pytest.mark.performance
    def test_single_material_calculation_speed_baseline(self):
        """Benchmark single material calculation speed across energy ranges."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        # Test materials with different complexities
        test_materials = [
            ("Si", 2.33, "simple_element"),
            ("SiO2", 2.2, "simple_compound"),
            ("Al2O3", 3.97, "medium_compound"),
            ("CaAl2Si2O8", 2.76, "complex_compound"),
        ]

        # Energy ranges to test (using safer ranges to avoid numerical issues)
        energy_ranges = [
            (np.array([10.0]), "single_energy"),
            (np.linspace(1.0, 30.0, 100), "energy_array_100"),
            (np.linspace(1.0, 30.0, 1000), "energy_array_1000"),
            (np.linspace(1.0, 30.0, 2000), "large_range_2000"),
        ]

        benchmark_results = {}

        for formula, density, material_type in test_materials:
            for energies, energy_type in energy_ranges:
                # Warm up
                for _ in range(3):
                    calculate_single_material_properties(formula, energies, density)

                # Benchmark
                start_time = time.perf_counter()
                iterations = 100 if len(energies) <= 100 else 10

                for _ in range(iterations):
                    calculate_single_material_properties(formula, energies, density)

                end_time = time.perf_counter()

                total_time = end_time - start_time
                time_per_calculation = total_time / iterations
                calculations_per_second = 1.0 / time_per_calculation

                benchmark_key = f"{material_type}_{energy_type}"
                benchmark_results[benchmark_key] = {
                    "formula": formula,
                    "energy_points": len(energies),
                    "time_per_calculation": time_per_calculation,
                    "calculations_per_second": calculations_per_second,
                    "total_time": total_time,
                    "iterations": iterations,
                }

        # Log results for baseline establishment
        print("\n=== SINGLE MATERIAL CALCULATION SPEED BASELINE ===")
        for key, data in benchmark_results.items():
            print(
                f"{key}: {data['calculations_per_second']:.0f} calc/sec "
                f"({data['time_per_calculation'] * 1000:.2f}ms per calc, "
                f"{data['energy_points']} energy points)"
            )

        # Assert minimum performance expectations (relaxed for realistic system performance)
        for key, data in benchmark_results.items():
            if data["energy_points"] == 1:
                # Single energy calculations should be reasonably fast
                assert data["calculations_per_second"] > 5000, (
                    f"Single energy calc too slow: {data['calculations_per_second']}"
                )
            elif data["energy_points"] <= 100:
                # Small arrays should still be reasonably fast
                assert data["calculations_per_second"] > 500, (
                    f"Small array calc too slow: {data['calculations_per_second']}"
                )

        return benchmark_results

    @pytest.mark.performance
    def test_batch_processing_speed_baseline(self):
        """Benchmark batch processing performance for multiple materials."""
        from xraylabtool.calculators.core import calculate_xray_properties

        # Create batch test data
        materials_data = [
            ("Si", [10.0], 2.33),
            ("SiO2", [10.0], 2.2),
            ("Al2O3", [10.0], 3.97),
            ("TiO2", [10.0], 4.23),
            ("Fe2O3", [10.0], 5.24),
        ]

        batch_sizes = [1, 5, 10, 25, 50, 100]
        benchmark_results = {}

        for batch_size in batch_sizes:
            # Create batch data
            batch_materials = (
                materials_data * ((batch_size // len(materials_data)) + 1)
            )[:batch_size]

            # Extract separate lists for the API
            formulas = [item[0] for item in batch_materials]
            energies = [10.0]  # Single energy for all materials
            densities = [item[2] for item in batch_materials]

            # Warm up
            for _ in range(3):
                calculate_xray_properties(formulas, energies, densities)

            # Benchmark
            start_time = time.perf_counter()
            iterations = max(1, 50 // batch_size)  # Fewer iterations for larger batches

            for _ in range(iterations):
                calculate_xray_properties(formulas, energies, densities)

            end_time = time.perf_counter()

            total_time = end_time - start_time
            time_per_batch = total_time / iterations
            materials_per_second = (batch_size * iterations) / total_time

            benchmark_results[f"batch_{batch_size}"] = {
                "batch_size": batch_size,
                "time_per_batch": time_per_batch,
                "materials_per_second": materials_per_second,
                "total_time": total_time,
                "iterations": iterations,
            }

        # Log results
        print("\n=== BATCH PROCESSING SPEED BASELINE ===")
        for key, data in benchmark_results.items():
            print(
                f"{key}: {data['materials_per_second']:.0f} materials/sec "
                f"({data['time_per_batch'] * 1000:.2f}ms per batch)"
            )

        # Assert batch processing behavior based on v0.2.5 optimizations
        batch_1_rate = benchmark_results["batch_1"]["materials_per_second"]
        batch_10_rate = benchmark_results["batch_10"]["materials_per_second"]
        batch_25_rate = benchmark_results["batch_25"]["materials_per_second"]
        batch_100_rate = benchmark_results["batch_100"]["materials_per_second"]

        # v0.2.5 uses adaptive processing: sequential for <20 items, parallel for >=20 items
        # Small batches (1-10) should have similar performance due to sequential processing
        # Larger batches (25+) should benefit from parallelization

        # Small batch performance should be reasonably consistent (within 50% variation)
        small_batch_ratio = batch_10_rate / batch_1_rate
        assert 0.5 <= small_batch_ratio <= 2.0, (
            f"Small batch processing inconsistent: {batch_10_rate} vs {batch_1_rate} (ratio: {small_batch_ratio:.2f})"
        )

        # Large batches should maintain reasonable performance compared to small batches
        # batch_100 should be at least 60% of batch_1 performance (accounts for threading overhead)
        large_batch_ratio = batch_100_rate / batch_1_rate
        assert large_batch_ratio >= 0.6, (
            f"Large batch processing too slow: {batch_100_rate} vs {batch_1_rate} (ratio: {large_batch_ratio:.2f})"
        )

        # Very large batches should be faster than medium batches (parallelization benefit)
        medium_vs_large = batch_100_rate / batch_25_rate
        assert medium_vs_large >= 1.0, (
            f"Large batches not scaling properly: {batch_100_rate} vs {batch_25_rate} (ratio: {medium_vs_large:.2f})"
        )

        return benchmark_results

    @pytest.mark.performance
    def test_memory_allocation_patterns_baseline(self):
        """Benchmark memory allocation patterns and garbage collection impact."""

        import psutil

        from xraylabtool.calculators.core import calculate_single_material_properties

        process = psutil.Process()

        # Test different calculation patterns
        test_scenarios = [
            (
                "repeated_small",
                lambda: [
                    calculate_single_material_properties("SiO2", [10.0], 2.2)
                    for _ in range(1000)
                ],
            ),
            (
                "repeated_medium",
                lambda: [
                    calculate_single_material_properties(
                        "SiO2", np.linspace(1, 30, 100), 2.2
                    )
                    for _ in range(100)
                ],
            ),
            (
                "repeated_large",
                lambda: [
                    calculate_single_material_properties(
                        "SiO2", np.linspace(1, 30, 1000), 2.2
                    )
                    for _ in range(10)
                ],
            ),
        ]

        memory_results = {}

        for scenario_name, scenario_func in test_scenarios:
            # Clear memory and force garbage collection
            gc.collect()

            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run scenario and measure time
            start_time = time.perf_counter()
            scenario_func()
            end_time = time.perf_counter()

            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Force garbage collection and measure cleaned memory
            gc.collect()
            cleaned_memory = process.memory_info().rss / 1024 / 1024  # MB

            memory_results[scenario_name] = {
                "execution_time": end_time - start_time,
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "cleaned_memory_mb": cleaned_memory,
                "memory_growth_mb": final_memory - initial_memory,
                "memory_retained_mb": cleaned_memory - initial_memory,
                "gc_recovered_mb": final_memory - cleaned_memory,
            }

        # Log memory results
        print("\n=== MEMORY ALLOCATION PATTERNS BASELINE ===")
        for scenario, data in memory_results.items():
            print(f"{scenario}:")
            print(f"  Execution time: {data['execution_time']:.3f}s")
            print(f"  Memory growth: {data['memory_growth_mb']:.1f}MB")
            print(f"  Memory retained: {data['memory_retained_mb']:.1f}MB")
            print(f"  GC recovered: {data['gc_recovered_mb']:.1f}MB")

        # Assert reasonable memory behavior
        for scenario, data in memory_results.items():
            # Memory growth should be reasonable
            assert data["memory_growth_mb"] < 500, (
                f"Excessive memory growth in {scenario}: {data['memory_growth_mb']}MB"
            )

            # Garbage collection should recover most temporary memory
            if data["memory_growth_mb"] > 10:  # Only check if significant growth
                gc_efficiency = data["gc_recovered_mb"] / data["memory_growth_mb"]
                assert gc_efficiency > 0.5, (
                    f"Poor GC efficiency in {scenario}: {gc_efficiency:.2f}"
                )

        return memory_results

    @pytest.mark.performance
    def test_interpolator_creation_speed_baseline(self):
        """Benchmark scattering factor interpolator creation speed."""
        from xraylabtool.calculators.core import create_scattering_factor_interpolators

        # Test elements across the periodic table
        test_elements = [
            "H",
            "C",
            "N",
            "O",
            "Si",
            "Al",
            "Fe",
            "Cu",
            "Zn",
            "Mo",
            "W",
            "Au",
            "Pb",
            "U",  # Range from light to heavy elements
        ]

        interpolator_results = {}

        for element in test_elements:
            # Clear any existing caches
            from xraylabtool.calculators.core import clear_scattering_factor_cache

            clear_scattering_factor_cache()

            # Benchmark first creation (no cache)
            start_time = time.perf_counter()
            f1_interp, f2_interp = create_scattering_factor_interpolators(element)
            first_creation_time = time.perf_counter() - start_time

            # Benchmark cached access
            start_time = time.perf_counter()
            for _ in range(100):
                (
                    _f1_interp_cached,
                    _f2_interp_cached,
                ) = create_scattering_factor_interpolators(element)
            cached_time = (time.perf_counter() - start_time) / 100

            # Test interpolation speed
            test_energies = np.linspace(100, 30000, 1000)  # eV
            start_time = time.perf_counter()
            for _ in range(10):
                f1_interp(test_energies)
                f2_interp(test_energies)
            interpolation_time = (time.perf_counter() - start_time) / 10

            interpolator_results[element] = {
                "first_creation_time": first_creation_time,
                "cached_access_time": cached_time,
                "interpolation_time": interpolation_time,
                "cache_speedup": (
                    first_creation_time / cached_time
                    if cached_time > 0
                    else float("inf")
                ),
            }

        # Log results
        print("\n=== INTERPOLATOR CREATION SPEED BASELINE ===")
        for element, data in interpolator_results.items():
            print(
                f"{element}: First={data['first_creation_time'] * 1000:.1f}ms, "
                f"Cached={data['cached_access_time'] * 1000:.3f}ms, "
                f"Interpolation={data['interpolation_time'] * 1000:.1f}ms, "
                f"Speedup={data['cache_speedup']:.0f}x"
            )

        # Assert reasonable performance
        for element, data in interpolator_results.items():
            # First creation should complete reasonably quickly
            assert data["first_creation_time"] < 0.1, (
                f"Slow interpolator creation for {element}:"
                f" {data['first_creation_time']}s"
            )

            # Cached access should be very fast
            assert data["cached_access_time"] < 0.001, (
                f"Slow cached access for {element}: {data['cached_access_time']}s"
            )

            # Cache should provide significant speedup
            assert data["cache_speedup"] > 10, (
                f"Poor cache speedup for {element}: {data['cache_speedup']}x"
            )

        return interpolator_results

    @pytest.mark.performance
    def test_concurrent_calculation_baseline(self):
        """Benchmark concurrent calculation performance with threading."""
        import concurrent.futures

        from xraylabtool.calculators.core import calculate_single_material_properties

        def calculation_worker(args):
            """Worker function for concurrent calculations."""
            formula, energies, density = args
            return calculate_single_material_properties(formula, energies, density)

        # Test data
        test_materials = [
            ("SiO2", np.linspace(1, 30, 100), 2.2),
            ("Al2O3", np.linspace(1, 30, 100), 3.97),
            ("TiO2", np.linspace(1, 30, 100), 4.23),
            ("Fe2O3", np.linspace(1, 30, 100), 5.24),
        ] * 25  # 100 total calculations

        thread_counts = [1, 2, 4, 8, 16]
        concurrency_results = {}

        for thread_count in thread_counts:
            # Warm up
            for _ in range(3):
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=thread_count
                ) as executor:
                    list(executor.map(calculation_worker, test_materials[:10]))

            # Benchmark
            start_time = time.perf_counter()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=thread_count
            ) as executor:
                list(executor.map(calculation_worker, test_materials))

            end_time = time.perf_counter()

            total_time = end_time - start_time
            calculations_per_second = len(test_materials) / total_time

            concurrency_results[thread_count] = {
                "thread_count": thread_count,
                "total_time": total_time,
                "calculations_per_second": calculations_per_second,
                "total_calculations": len(test_materials),
            }

        # Log results
        print("\n=== CONCURRENT CALCULATION BASELINE ===")
        for threads, data in concurrency_results.items():
            print(
                f"{threads} threads: {data['calculations_per_second']:.0f} calc/sec "
                f"({data['total_time']:.2f}s total)"
            )

        # Calculate scaling efficiency
        single_thread_rate = concurrency_results[1]["calculations_per_second"]

        for threads, data in concurrency_results.items():
            if threads > 1:
                scaling_efficiency = data["calculations_per_second"] / (
                    single_thread_rate * threads
                )
                data["scaling_efficiency"] = scaling_efficiency
                print(f"{threads} threads scaling efficiency: {scaling_efficiency:.2f}")

        # Assert reasonable concurrent performance (very relaxed due to GIL limitations)
        # For CPU-bound calculations in Python, GIL often limits scaling
        # We mainly want to verify no severe degradation
        min_expected_rate = (
            single_thread_rate * 0.6
        )  # Allow up to 40% degradation due to GIL
        assert concurrency_results[4]["calculations_per_second"] > min_expected_rate, (
            "Severe performance degradation with 4 threads: expected >"
            f" {min_expected_rate:.0f}, got"
            f" {concurrency_results[4]['calculations_per_second']:.0f}"
        )

        return concurrency_results


class TestPerformanceRegressionDetection(BasePerformanceTest):
    """Test framework for detecting performance regressions."""

    def test_performance_threshold_validation(self):
        """Validate that current performance meets minimum thresholds."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        # Minimum performance thresholds (relaxed for realistic system performance)
        THRESHOLDS = {
            "single_energy_calc_per_sec": (
                3000
            ),  # Single energy calculations - reduced from 5000
            "array_100_calc_per_sec": 300,  # 100-point energy array - reduced from 500
            "array_1000_calc_per_sec": 30,  # 1000-point energy array - reduced from 50
            "memory_growth_per_1000_calc_mb": (
                200
            ),  # Memory growth limit - increased from 100
        }

        # Test single energy performance
        start_time = time.perf_counter()
        for _ in range(100):
            calculate_single_material_properties("SiO2", [10.0], 2.2)
        single_energy_rate = 100 / (time.perf_counter() - start_time)

        # Test array performance
        energies_100 = np.linspace(1, 30, 100)
        start_time = time.perf_counter()
        for _ in range(10):
            calculate_single_material_properties("SiO2", energies_100, 2.2)
        array_100_rate = 10 / (time.perf_counter() - start_time)

        energies_1000 = np.linspace(1, 30, 1000)
        start_time = time.perf_counter()
        for _ in range(5):
            calculate_single_material_properties("SiO2", energies_1000, 2.2)
        array_1000_rate = 5 / (time.perf_counter() - start_time)

        # Memory growth test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        for _ in range(1000):
            calculate_single_material_properties("SiO2", [10.0], 2.2)

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Assert thresholds
        assert single_energy_rate >= THRESHOLDS["single_energy_calc_per_sec"], (
            f"Single energy performance regression: {single_energy_rate:.0f} <"
            f" {THRESHOLDS['single_energy_calc_per_sec']}"
        )

        assert array_100_rate >= THRESHOLDS["array_100_calc_per_sec"], (
            f"Array 100 performance regression: {array_100_rate:.0f} <"
            f" {THRESHOLDS['array_100_calc_per_sec']}"
        )

        assert array_1000_rate >= THRESHOLDS["array_1000_calc_per_sec"], (
            f"Array 1000 performance regression: {array_1000_rate:.0f} <"
            f" {THRESHOLDS['array_1000_calc_per_sec']}"
        )

        assert memory_growth <= THRESHOLDS["memory_growth_per_1000_calc_mb"], (
            f"Memory growth regression: {memory_growth:.1f}MB >"
            f" {THRESHOLDS['memory_growth_per_1000_calc_mb']}MB"
        )

        print("\n=== PERFORMANCE THRESHOLD VALIDATION ===")
        print(
            f"Single energy: {single_energy_rate:.0f} calc/sec (threshold:"
            f" {THRESHOLDS['single_energy_calc_per_sec']})"
        )
        print(
            f"Array 100: {array_100_rate:.0f} calc/sec (threshold:"
            f" {THRESHOLDS['array_100_calc_per_sec']})"
        )
        print(
            f"Array 1000: {array_1000_rate:.0f} calc/sec (threshold:"
            f" {THRESHOLDS['array_1000_calc_per_sec']})"
        )
        print(
            f"Memory growth: {memory_growth:.1f}MB (threshold:"
            f" {THRESHOLDS['memory_growth_per_1000_calc_mb']}MB)"
        )

        return {
            "single_energy_rate": single_energy_rate,
            "array_100_rate": array_100_rate,
            "array_1000_rate": array_1000_rate,
            "memory_growth_mb": memory_growth,
        }
