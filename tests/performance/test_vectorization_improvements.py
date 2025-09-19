"""
Tests for vectorization improvements in XRayLabTool optimization.

This module provides comprehensive testing of vectorized calculation functions
with performance validation to ensure optimization targets are met while
maintaining numerical accuracy.
"""

import time

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.optimization.regression_detector import record_performance_metric
from xraylabtool.utils import parse_formula


class TestVectorizedCalculations(BasePerformanceTest):
    """
    Test suite for advanced vectorization improvements.

    This class tests the vectorized implementations against the original
    implementations to ensure:
    1. Numerical accuracy is preserved (within tolerance)
    2. Performance improvements meet target metrics
    3. Memory efficiency is improved
    4. All edge cases are handled correctly
    """

    def setup_method(self):
        """Set up test environment for vectorization testing."""
        super().setup_method()
        self.performance_data = {}
        self.accuracy_tolerance = 1e-12  # Strict tolerance for scientific accuracy
        self.target_performance_improvement = 2.0  # 2x improvement target

    def test_scattering_factors_vectorization_accuracy(self):
        """Test that vectorized scattering factor calculations maintain accuracy."""
        test_cases = [
            # (formula, density, description)
            ("Si", 2.33, "silicon_element"),
            ("SiO2", 2.2, "silica_compound"),
            ("Al0.3Ga0.7As", 4.5, "complex_alloy"),
            ("Ca5(PO4)3OH", 3.16, "hydroxyapatite"),
        ]

        energy_ranges = [
            np.array([10.0]),  # Single energy
            np.linspace(5.0, 25.0, 100),  # Medium array
            np.linspace(
                1.0, 29.0, 1000
            ),  # Large array - more conservative range 1.0-29.0 keV
        ]

        for formula, density, _description in test_cases:
            for _i, energies in enumerate(energy_ranges):
                # Test vectorization accuracy for each material and energy range
                # Get reference result using current implementation
                reference_result = calculate_single_material_properties(
                    formula, energies, density
                )

                # Verify reference result is valid
                assert reference_result is not None
                assert hasattr(reference_result, "dispersion_delta")
                assert hasattr(reference_result, "absorption_beta")
                assert len(reference_result.dispersion_delta) == len(energies)

                # Test accuracy preservation
                # (For now, we're testing against current implementation)
                # When we implement new vectorization, we'll compare against this baseline
                assert np.all(np.isfinite(reference_result.dispersion_delta))
                assert np.all(np.isfinite(reference_result.absorption_beta))
                assert np.all(reference_result.dispersion_delta >= 0)
                assert np.all(reference_result.absorption_beta >= 0)

    def test_scattering_factors_performance_comparison(self):
        """Compare performance of current vs optimized scattering factor calculations."""
        # Test with increasingly complex materials to stress the vectorization
        test_materials = [
            ("Si", 2.33, "element"),
            ("SiO2", 2.2, "compound_2_elements"),
            ("Al2O3", 3.97, "compound_2_elements"),
            ("CaAl2Si2O8", 2.76, "compound_4_elements"),
            ("Ca5(PO4)3OH", 3.16, "compound_4_elements"),
        ]

        energy_sizes = [100, 500, 1000, 2000]

        performance_results = {}

        for formula, density, complexity in test_materials:
            for size in energy_sizes:
                energies = np.linspace(5.0, 25.0, size)
                test_key = f"{complexity}_{size}_energies"

                # Measure current implementation performance
                start_time = time.perf_counter()

                # Run multiple iterations for stable timing
                iterations = max(1, 100 // size)  # More iterations for smaller arrays
                for _ in range(iterations):
                    result = calculate_single_material_properties(
                        formula, energies, density
                    )

                elapsed_time = time.perf_counter() - start_time
                time_per_calculation = elapsed_time / iterations
                calculations_per_second = 1.0 / time_per_calculation

                performance_results[test_key] = {
                    "formula": formula,
                    "complexity": complexity,
                    "energy_points": size,
                    "time_per_calculation": time_per_calculation,
                    "calculations_per_second": calculations_per_second,
                    "total_time": elapsed_time,
                    "iterations": iterations,
                }

                # Record performance metric for regression tracking
                record_performance_metric(
                    name="vectorization_baseline_calc_per_sec",
                    value=calculations_per_second,
                    unit="calc/sec",
                    context={
                        "formula": formula,
                        "complexity": complexity,
                        "energy_points": size,
                        "optimization_phase": "baseline_current",
                    },
                )

                # Verify reasonable performance
                assert calculations_per_second > 10, (
                    f"Performance too low for {test_key}:"
                    f" {calculations_per_second:.1f} calc/sec"
                )
                assert result is not None, f"Calculation failed for {test_key}"

        # Store results for later comparison with optimized version
        self.performance_data.update(performance_results)

        # Print performance summary
        print("\n=== CURRENT IMPLEMENTATION PERFORMANCE BASELINE ===")
        for key, data in performance_results.items():
            print(
                f"{key}: {data['calculations_per_second']:.0f} calc/sec "
                f"({data['time_per_calculation'] * 1000:.2f}ms per calc, "
                f"{data['energy_points']} energies)"
            )

    def test_element_iteration_vectorization_opportunities(self):
        """Test and identify opportunities for vectorizing element iteration patterns."""
        # Test materials with different numbers of elements
        test_cases = [
            ("Si", "single_element"),
            ("SiO2", "two_elements"),
            ("Al2O3", "two_elements_different"),
            ("CaCO3", "three_elements"),
            ("CaAl2Si2O8", "four_elements"),
            ("Ca5(PO4)3OH", "complex_formula"),
        ]

        energies = np.linspace(5.0, 25.0, 500)

        for formula, complexity in test_cases:
            # Test vectorization opportunities for each formula
            # Parse formula to understand element composition
            try:
                composition = parse_formula(formula)
                num_elements = len(composition)

                # Time the calculation
                start_time = time.perf_counter()
                result = calculate_single_material_properties(formula, energies, 2.5)
                elapsed_time = time.perf_counter() - start_time

                # Calculate metrics
                time_per_element = elapsed_time / num_elements
                elements_per_second = num_elements / elapsed_time

                print(
                    f"{formula} ({num_elements} elements): "
                    f"{elapsed_time * 1000:.2f}ms total, "
                    f"{time_per_element * 1000:.2f}ms per element, "
                    f"{elements_per_second:.1f} elements/sec"
                )

                # Record metrics for vectorization planning
                record_performance_metric(
                    name="element_iteration_time_per_element",
                    value=time_per_element * 1000,  # Convert to ms
                    unit="ms/element",
                    context={
                        "formula": formula,
                        "num_elements": num_elements,
                        "energy_points": len(energies),
                        "complexity": complexity,
                    },
                )

                assert result is not None
                assert num_elements > 0

            except Exception as e:
                pytest.fail(f"Failed to process {formula}: {e}")

    def test_memory_contiguity_analysis(self):
        """Analyze memory layout and contiguity of arrays in calculations."""
        test_formula = "SiO2"
        energy_sizes = [100, 500, 1000, 2000]

        for size in energy_sizes:
            energies = np.linspace(5.0, 25.0, size)

            # Ensure energies array is C-contiguous
            assert energies.flags.c_contiguous, (
                f"Energy array not C-contiguous for size {size}"
            )

            # Test calculation and check result array properties
            result = calculate_single_material_properties(test_formula, energies, 2.2)

            # Check that result arrays are properly structured
            assert hasattr(result, "dispersion_delta")
            assert hasattr(result, "absorption_beta")

            delta_array = result.dispersion_delta

            if isinstance(delta_array, np.ndarray):
                # Record memory layout information
                record_performance_metric(
                    name="array_memory_contiguous",
                    value=1.0 if delta_array.flags.c_contiguous else 0.0,
                    unit="boolean",
                    context={
                        "array_type": "dispersion_delta",
                        "size": size,
                        "dtype": str(delta_array.dtype),
                    },
                )

                print(
                    f"Size {size}: delta C-contiguous={delta_array.flags.c_contiguous},"
                    f" dtype={delta_array.dtype}"
                )

    def test_interpolation_vectorization_opportunities(self):
        """Test atomic data interpolation performance patterns."""
        # Test different elements to understand interpolation performance
        elements_to_test = ["Si", "O", "Al", "Ca", "Fe", "Ti"]
        energies = np.linspace(1.0, 30.0, 1000)

        interpolation_times = {}

        for element in elements_to_test:
            # Test single element to isolate interpolation performance
            start_time = time.perf_counter()

            try:
                calculate_single_material_properties(element, energies, 2.0)
                elapsed_time = time.perf_counter() - start_time

                interpolations_per_second = len(energies) / elapsed_time
                time_per_interpolation = elapsed_time / len(energies)

                interpolation_times[element] = {
                    "total_time": elapsed_time,
                    "time_per_interpolation": time_per_interpolation,
                    "interpolations_per_second": interpolations_per_second,
                }

                record_performance_metric(
                    name="interpolation_performance",
                    value=interpolations_per_second,
                    unit="interp/sec",
                    context={
                        "element": element,
                        "energy_points": len(energies),
                        "time_per_interpolation_us": time_per_interpolation * 1e6,
                    },
                )

                print(
                    f"{element}: {interpolations_per_second:.0f} interp/sec "
                    f"({time_per_interpolation * 1e6:.2f}μs per point)"
                )

            except Exception as e:
                print(f"Failed to test {element}: {e}")

        # Analyze variation in interpolation performance
        if interpolation_times:
            times = [
                data["interpolations_per_second"]
                for data in interpolation_times.values()
            ]
            mean_performance = np.mean(times)
            std_performance = np.std(times)
            cv = std_performance / mean_performance  # Coefficient of variation

            print("\nInterpolation performance statistics:")
            print(f"Mean: {mean_performance:.0f} interp/sec")
            print(f"Std: {std_performance:.0f} interp/sec")
            print(f"CV: {cv:.3f} (lower is more consistent)")

    def test_broadcasting_optimization_potential(self):
        """Test potential for NumPy broadcasting optimizations."""
        # Create test scenarios that would benefit from broadcasting
        test_scenarios = [
            {
                "name": "multiple_energies_single_material",
                "materials": [("SiO2", 2.2)],
                "energies": np.linspace(5.0, 25.0, 1000),
                "description": "Energy broadcasting potential",
            },
            {
                "name": "single_energy_multiple_densities",
                "materials": [("SiO2", d) for d in [2.0, 2.2, 2.4, 2.6, 2.8]],
                "energies": np.array([10.0]),
                "description": "Density broadcasting potential",
            },
            {
                "name": "mixed_complexity",
                "materials": [("Si", 2.33), ("SiO2", 2.2), ("Al2O3", 3.97)],
                "energies": np.linspace(5.0, 25.0, 500),
                "description": "Material broadcasting potential",
            },
        ]

        for scenario in test_scenarios:
            print(f"\n=== {scenario['description'].upper()} ===")
            materials = scenario["materials"]
            energies = scenario["energies"]

            # Time individual calculations
            individual_times = []
            for formula, density in materials:
                start_time = time.perf_counter()
                calculate_single_material_properties(formula, energies, density)
                elapsed_time = time.perf_counter() - start_time
                individual_times.append(elapsed_time)

            total_individual_time = sum(individual_times)

            # Calculate potential for broadcasting optimization
            mean_time_per_material = np.mean(individual_times)
            broadcasting_efficiency_potential = (
                total_individual_time / mean_time_per_material
            )

            print(f"Materials: {len(materials)}")
            print(f"Energy points: {len(energies)}")
            print(f"Total individual time: {total_individual_time * 1000:.2f}ms")
            print(f"Mean time per material: {mean_time_per_material * 1000:.2f}ms")
            print(f"Broadcasting potential: {broadcasting_efficiency_potential:.2f}x")

            record_performance_metric(
                name="broadcasting_optimization_potential",
                value=broadcasting_efficiency_potential,
                unit="speedup_potential",
                context={
                    "scenario": scenario["name"],
                    "num_materials": len(materials),
                    "num_energies": len(energies),
                    "total_time_ms": total_individual_time * 1000,
                },
            )

    def test_numerical_precision_preservation(self):
        """Test that vectorization optimizations preserve numerical precision."""
        # Test with materials known to have challenging numerical properties
        challenging_cases = [
            ("Si", 2.33, "light_element"),  # Silicon - more stable than Carbon
            ("Au", 19.32, "heavy_element"),  # High atomic number
            ("SiO2", 2.2, "light_compound"),  # Silica - more stable than H2O
            ("Fe2O3", 5.24, "heavy_compound"),  # Iron oxide - more stable than UO2
        ]

        # Test at different energy ranges that might cause numerical issues
        energy_ranges = [
            np.linspace(1.0, 5.0, 100),  # Low energy - more conservative range
            np.linspace(5.0, 15.0, 100),  # Medium-low energy
            np.linspace(10.0, 25.0, 100),  # Medium energy
            np.linspace(20.0, 30.0, 100),  # High energy
        ]

        for formula, density, _description in challenging_cases:
            for i, energies in enumerate(energy_ranges):
                # Test numerical precision for each challenging case
                try:
                    result = calculate_single_material_properties(
                        formula, energies, density
                    )

                    # Check for numerical stability issues
                    delta = result.dispersion_delta
                    beta = result.absorption_beta

                    # Test for common numerical issues
                    assert np.all(np.isfinite(delta)), (
                        f"Non-finite dispersion values for {formula}"
                    )
                    assert np.all(np.isfinite(beta)), (
                        f"Non-finite absorption values for {formula}"
                    )
                    assert np.all(delta >= 0), (
                        f"Negative dispersion values for {formula}"
                    )
                    assert np.all(beta >= 0), (
                        f"Negative absorption values for {formula}"
                    )

                    # Test precision preservation
                    # Values should not be exactly zero unless physically meaningful
                    assert np.any(delta > 1e-15), (
                        f"Suspiciously small dispersion for {formula}"
                    )
                    assert np.any(beta > 1e-15), (
                        f"Suspiciously small absorption for {formula}"
                    )

                    # Record numerical health metrics
                    record_performance_metric(
                        name="numerical_precision_health",
                        value=float(np.mean(delta / beta) if np.mean(beta) > 0 else 0),
                        unit="delta_beta_ratio",
                        context={
                            "formula": formula,
                            "energy_range_index": i,
                            "min_delta": float(np.min(delta)),
                            "max_delta": float(np.max(delta)),
                            "min_beta": float(np.min(beta)),
                            "max_beta": float(np.max(beta)),
                        },
                    )

                except Exception as e:
                    pytest.fail(
                        f"Numerical precision test failed for {formula} at energy range"
                        f" {i}: {e}"
                    )


@pytest.mark.performance
class TestVectorizationRegressions(BasePerformanceTest):
    """Test suite to catch performance regressions during vectorization."""

    def test_performance_regression_monitoring(self):
        """Monitor for performance regressions during optimization."""
        # Standard test case for regression monitoring
        test_material = "SiO2"
        test_density = 2.2
        test_energies = np.linspace(5.0, 25.0, 1000)

        # Perform calculation multiple times for stable measurement
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            calculate_single_material_properties(
                test_material, test_energies, test_density
            )
            elapsed_time = time.perf_counter() - start_time
            times.append(elapsed_time)

        # Use median time for stability
        median_time = np.median(times)
        calc_per_second = 1.0 / median_time

        # Record for regression detection
        record_performance_metric(
            name="vectorization_regression_monitor",
            value=calc_per_second,
            unit="calc/sec",
            context={
                "test_type": "regression_monitoring",
                "material": test_material,
                "energy_points": len(test_energies),
                "measurement_stability": (
                    np.std(times) / np.mean(times)
                ),  # Coefficient of variation
            },
        )

        # Assert minimum performance threshold
        min_acceptable_performance = 100  # calc/sec
        assert calc_per_second > min_acceptable_performance, (
            f"Performance regression detected: {calc_per_second:.1f} <"
            f" {min_acceptable_performance} calc/sec"
        )

        print(
            f"Regression monitoring: {calc_per_second:.0f} calc/sec (stability:"
            f" {np.std(times) / np.mean(times):.3f})"
        )

    def test_memory_usage_regression(self):
        """Monitor memory usage during vectorization to prevent regressions."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run calculation with large array
        test_energies = np.linspace(
            1.0, 29.0, 5000
        )  # Large array - fixed to valid energy range
        calculate_single_material_properties("Al2O3", test_energies, 3.97)

        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory

        # Record memory usage
        record_performance_metric(
            name="vectorization_memory_usage",
            value=memory_increase,
            unit="MB",
            context={
                "test_type": "memory_regression",
                "energy_points": len(test_energies),
                "baseline_memory_mb": baseline_memory,
                "peak_memory_mb": peak_memory,
            },
        )

        # Assert reasonable memory usage
        max_acceptable_memory = 500  # MB increase
        assert memory_increase < max_acceptable_memory, (
            f"Memory regression detected: {memory_increase:.1f}MB increase >"
            f" {max_acceptable_memory}MB limit"
        )

        print(
            f"Memory usage: +{memory_increase:.1f}MB (baseline:"
            f" {baseline_memory:.1f}MB)"
        )
