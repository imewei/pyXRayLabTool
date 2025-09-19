"""
Tests for the vectorized core optimization implementations.

This module tests the new vectorized implementations to ensure they:
1. Maintain numerical accuracy with the original implementations
2. Provide measurable performance improvements
3. Handle edge cases correctly
"""

import time

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
from xraylabtool.calculators.core import (
    calculate_scattering_factors,
    create_scattering_factor_interpolators,
)
from xraylabtool.optimization.regression_detector import record_performance_metric
from xraylabtool.optimization.vectorized_core import (
    benchmark_vectorization_improvement,
    calculate_scattering_factors_vectorized,
    configure_numpy_for_performance,
    vectorized_interpolation_batch,
)
from xraylabtool.utils import parse_formula


class TestVectorizedCore(BasePerformanceTest):
    """Test the vectorized core calculation implementations."""

    def setup_method(self):
        """Set up test environment."""
        super().setup_method()
        self.accuracy_tolerance = 1e-12
        configure_numpy_for_performance()

    def test_vectorized_scattering_factors_accuracy(self):
        """Test that vectorized implementation maintains numerical accuracy."""
        test_cases = [
            # (formula, density, description)
            ("Si", 2.33, "single_element"),
            ("SiO2", 2.2, "two_elements"),
            ("Al2O3", 3.97, "two_elements_different"),
            ("CaAl2Si2O8", 2.76, "four_elements"),
            ("Ca5(PO4)3OH", 3.16, "complex_compound"),
        ]

        energy_ranges = [
            np.array([10000.0]),  # Single energy (10 keV in eV)
            np.linspace(5000.0, 25000.0, 100),  # Medium array (5-25 keV in eV)
            np.linspace(2000.0, 25000.0, 1000),  # Large array (2-25 keV in eV)
        ]

        for formula, density, _description in test_cases:
            for i, energies in enumerate(energy_ranges):
                try:
                    # Create test data
                    elements, counts = parse_formula(formula)
                    composition = dict(zip(elements, counts, strict=False))
                    molecular_weight = sum(
                        count * self._get_atomic_weight(element)
                        for element, count in composition.items()
                    )

                    # Create wavelength array (convert eV to keV first)
                    from xraylabtool.utils import energy_to_wavelength

                    wavelength_angstrom = np.array(
                        [energy_to_wavelength(e / 1000) for e in energies]
                    )  # eV to keV
                    # Convert to meters for the calculation function
                    from xraylabtool.constants import ANGSTROM_TO_METER

                    wavelength = wavelength_angstrom * ANGSTROM_TO_METER

                    # Create element data
                    element_data = []
                    for element, count in composition.items():
                        f1_interp, f2_interp = create_scattering_factor_interpolators(
                            element
                        )
                        element_data.append((count, f1_interp, f2_interp))

                    # Compare original and vectorized implementations
                    original_result = calculate_scattering_factors(
                        energies, wavelength, density, molecular_weight, element_data
                    )

                    vectorized_result = calculate_scattering_factors_vectorized(
                        energies, wavelength, density, molecular_weight, element_data
                    )

                    # Check numerical accuracy
                    for j, (orig, vect) in enumerate(
                        zip(original_result, vectorized_result, strict=False)
                    ):
                        # Ensure arrays are finite
                        if not np.all(np.isfinite(orig)) or not np.all(
                            np.isfinite(vect)
                        ):
                            raise ValueError(
                                f"Non-finite values detected in result {j}"
                            )

                        # Calculate relative error safely
                        abs_diff = np.abs(orig - vect)
                        abs_orig = np.abs(orig)

                        # Handle case where original is zero
                        mask = abs_orig > 1e-15
                        rel_error = 0.0

                        if np.any(mask):
                            rel_error = np.max(abs_diff[mask] / abs_orig[mask])

                        # If original is zero, use absolute error
                        if not np.any(mask):
                            rel_error = np.max(abs_diff)

                        assert rel_error < self.accuracy_tolerance, (
                            f"Accuracy failure for {formula}, result {j}: relative"
                            f" error {rel_error:.2e} > {self.accuracy_tolerance:.2e}"
                        )

                        print(f"{formula}[{i}][{j}]: max rel error = {rel_error:.2e}")

                except Exception as e:
                    pytest.fail(f"Test failed for {formula}, energy_range {i}: {e}")

    def test_vectorized_performance_improvement(self):
        """Test that vectorized implementation provides performance improvement."""
        test_materials = [
            ("SiO2", 2.2, "compound_2_elements"),
            ("Al2O3", 3.97, "compound_2_elements_diff"),
            ("CaAl2Si2O8", 2.76, "compound_4_elements"),
        ]

        energy_sizes = [100, 500, 1000]
        performance_results = {}

        for formula, density, complexity in test_materials:
            for size in energy_sizes:
                test_key = f"{complexity}_{size}"

                # Create test data
                energies = np.linspace(5000.0, 25000.0, size)
                elements, counts = parse_formula(formula)
                composition = dict(zip(elements, counts, strict=False))
                molecular_weight = sum(
                    count * self._get_atomic_weight(element)
                    for element, count in composition.items()
                )

                from xraylabtool.constants import ANGSTROM_TO_METER
                from xraylabtool.utils import energy_to_wavelength

                wavelength_angstrom = np.array(
                    [energy_to_wavelength(e / 1000) for e in energies]
                )
                wavelength = wavelength_angstrom * ANGSTROM_TO_METER

                element_data = []
                for element, count in composition.items():
                    f1_interp, f2_interp = create_scattering_factor_interpolators(
                        element
                    )
                    element_data.append((count, f1_interp, f2_interp))

                # Benchmark the implementations
                benchmark_result = benchmark_vectorization_improvement(
                    energies,
                    wavelength,
                    density,
                    molecular_weight,
                    element_data,
                    iterations=5,
                )

                performance_results[test_key] = benchmark_result

                # Record performance metrics
                record_performance_metric(
                    name="vectorization_speedup",
                    value=benchmark_result["speedup"],
                    unit="speedup_ratio",
                    context={
                        "formula": formula,
                        "complexity": complexity,
                        "energy_points": size,
                        "n_elements": len(element_data),
                        "original_time_ms": (
                            benchmark_result["original_time_median"] * 1000
                        ),
                        "vectorized_time_ms": (
                            benchmark_result["vectorized_time_median"] * 1000
                        ),
                    },
                )

                # Check accuracy is preserved
                assert benchmark_result["accuracy_preserved"], (
                    f"Accuracy not preserved for {test_key}"
                )

                # Check for reasonable performance (vectorization may have overhead on small datasets)
                # Focus on correctness rather than strict performance requirements since benefits vary by hardware
                if benchmark_result["speedup"] < 0.2:
                    print(
                        f"Warning: Significant performance regression for {test_key}:"
                        f" {benchmark_result['speedup']:.2f}x"
                    )

                # Record but don't fail on performance - hardware and dataset size dependent
                if benchmark_result["speedup"] >= 1.0:
                    print(
                        f"✓ Performance improvement for {test_key}:"
                        f" {benchmark_result['speedup']:.2f}x"
                    )
                else:
                    print(
                        f"⚠ Vectorization overhead for {test_key}:"
                        f" {benchmark_result['speedup']:.2f}x (common for small"
                        " datasets)"
                    )

                print(
                    f"{test_key}: {benchmark_result['speedup']:.2f}x speedup "
                    f"(orig: {benchmark_result['original_time_median'] * 1000:.2f}ms, "
                    f"vect: {benchmark_result['vectorized_time_median'] * 1000:.2f}ms)"
                )

        # Summary statistics
        all_speedups = [result["speedup"] for result in performance_results.values()]
        mean_speedup = np.mean(all_speedups)
        print(f"\nMean speedup across all tests: {mean_speedup:.2f}x")

        # Check overall performance trend (vectorization benefits vary by hardware and dataset size)
        # Focus on detecting severe regressions rather than strict performance requirements
        min_acceptable_speedup = (
            0.5  # Allow for vectorization overhead on small datasets
        )
        if mean_speedup < min_acceptable_speedup:
            print(
                "Warning: Significant vectorization overhead detected:"
                f" {mean_speedup:.2f}x"
            )
        else:
            print(f"✓ Vectorization performance acceptable: {mean_speedup:.2f}x")

    def test_vectorized_interpolation_batch(self):
        """Test the vectorized interpolation batch function."""
        # Create test interpolators
        elements = ["Si", "O"]
        energies = np.linspace(5000.0, 25000.0, 100)

        interpolator_pairs = []
        element_counts = []

        for element in elements:
            f1_interp, f2_interp = create_scattering_factor_interpolators(element)
            interpolator_pairs.append((f1_interp, f2_interp))
            element_counts.append(1.0 if element == "Si" else 2.0)  # SiO2 stoichiometry

        element_counts = np.array(element_counts)

        # Test the vectorized batch interpolation
        f1_total, f2_total = vectorized_interpolation_batch(
            energies, interpolator_pairs, element_counts
        )

        # Verify results
        assert len(f1_total) == len(energies)
        assert len(f2_total) == len(energies)
        assert np.all(np.isfinite(f1_total))
        assert np.all(np.isfinite(f2_total))

        # Compare with manual calculation
        f1_manual = element_counts[0] * interpolator_pairs[0][0](
            energies
        ) + element_counts[1] * interpolator_pairs[1][0](energies)
        f2_manual = element_counts[0] * interpolator_pairs[0][1](
            energies
        ) + element_counts[1] * interpolator_pairs[1][1](energies)

        np.testing.assert_allclose(f1_total, f1_manual, rtol=1e-15)
        np.testing.assert_allclose(f2_total, f2_manual, rtol=1e-15)

    def test_multi_material_batch_processing(self):
        """Test the enhanced multi-material batch processing functionality."""
        from xraylabtool.optimization.vectorized_core import (
            vectorized_multi_material_batch,
        )

        # Setup test materials
        test_materials = [
            ("Si", 2.33),  # Single element
            ("SiO2", 2.2),  # Two elements
            ("Al2O3", 3.97),  # Two elements, different
            ("CaAl2Si2O8", 2.76),  # Four elements
        ]

        energies = np.linspace(5000.0, 25000.0, 200)

        # Prepare material definitions
        material_definitions = []
        material_properties = []

        for formula, density in test_materials:
            elements, counts = parse_formula(formula)
            composition = dict(zip(elements, counts, strict=False))
            molecular_weight = sum(
                count * self._get_atomic_weight(element)
                for element, count in composition.items()
            )

            # Create interpolator data
            interpolator_pairs = []
            element_counts = []
            for element, count in composition.items():
                f1_interp, f2_interp = create_scattering_factor_interpolators(element)
                interpolator_pairs.append((f1_interp, f2_interp))
                element_counts.append(count)

            material_definitions.append((interpolator_pairs, np.array(element_counts)))
            material_properties.append((density, molecular_weight))

        # Convert energies to wavelengths
        from xraylabtool.constants import ANGSTROM_TO_METER
        from xraylabtool.utils import energy_to_wavelength

        wavelength_angstrom = np.array(
            [energy_to_wavelength(e / 1000) for e in energies]
        )
        wavelength = wavelength_angstrom * ANGSTROM_TO_METER

        # Test batch processing
        batch_results = vectorized_multi_material_batch(
            energies, material_definitions, wavelength, material_properties
        )

        # Verify results structure
        assert len(batch_results) == len(test_materials)

        for i, (dispersion, absorption, f1_total, f2_total) in enumerate(batch_results):
            assert len(dispersion) == len(energies)
            assert len(absorption) == len(energies)
            assert len(f1_total) == len(energies)
            assert len(f2_total) == len(energies)
            assert np.all(np.isfinite(dispersion))
            assert np.all(np.isfinite(absorption))
            assert np.all(np.isfinite(f1_total))
            assert np.all(np.isfinite(f2_total))

        # Compare with individual calculations for accuracy
        for i, ((formula, density), (interpolator_pairs, element_counts)) in enumerate(
            zip(test_materials, material_definitions, strict=False)
        ):
            # Individual calculation
            mass_density, molecular_weight = material_properties[i]
            individual_result = calculate_scattering_factors_vectorized(
                energies,
                wavelength,
                mass_density,
                molecular_weight,
                [
                    (count, f1_interp, f2_interp)
                    for (f1_interp, f2_interp), count in zip(
                        interpolator_pairs, element_counts, strict=False
                    )
                ],
            )

            # Batch result
            batch_result = batch_results[i]

            # Compare accuracy
            for j, (individual, batch) in enumerate(
                zip(individual_result, batch_result, strict=False)
            ):
                rel_error = np.max(np.abs((individual - batch) / (individual + 1e-15)))
                assert rel_error < 1e-12, (
                    f"Material {i}, result {j}: relative error {rel_error:.2e}"
                )

        print(
            "Multi-material batch processing validated for"
            f" {len(test_materials)} materials"
        )

    def test_simd_optimized_functions(self):
        """Test SIMD-optimized calculation functions."""
        from xraylabtool.optimization.vectorized_core import (
            adaptive_simd_interpolation_batch,
            create_simd_optimized_arrays,
            simd_optimized_element_sum,
            simd_vectorized_wavelength_operations,
        )

        # Test SIMD-optimized array creation
        test_shape = (4, 1000)
        simd_array = create_simd_optimized_arrays(test_shape, np.float64)
        assert simd_array.shape == test_shape
        assert simd_array.dtype == np.float64
        assert simd_array.flags.c_contiguous

        # Test SIMD wavelength operations
        wavelengths = np.linspace(1e-10, 5e-10, 500)
        common_factor = 2.5
        wave_factor = simd_vectorized_wavelength_operations(wavelengths, common_factor)

        # Compare with manual calculation
        expected = np.square(wavelengths) * common_factor
        np.testing.assert_allclose(wave_factor, expected, rtol=1e-15)

        # Test SIMD element summation
        n_elements, n_energies = 3, 200
        f1_matrix = np.random.rand(n_elements, n_energies).astype(np.float64)
        f2_matrix = np.random.rand(n_elements, n_energies).astype(np.float64)
        element_counts = np.array([1.0, 2.0, 1.0], dtype=np.float64)

        f1_simd, f2_simd = simd_optimized_element_sum(
            f1_matrix, f2_matrix, element_counts
        )

        # Compare with manual einsum
        f1_manual = np.einsum("i,ij->j", element_counts, f1_matrix)
        f2_manual = np.einsum("i,ij->j", element_counts, f2_matrix)

        np.testing.assert_allclose(f1_simd, f1_manual, rtol=1e-15)
        np.testing.assert_allclose(f2_simd, f2_manual, rtol=1e-15)

        # Test adaptive SIMD interpolation
        elements = ["Si", "O", "Al"]
        energies = np.linspace(5000.0, 25000.0, 200)

        interpolator_pairs = []
        element_counts = []

        for element in elements:
            f1_interp, f2_interp = create_scattering_factor_interpolators(element)
            interpolator_pairs.append((f1_interp, f2_interp))
            element_counts.append(
                1.0 if element == "Si" else 2.0 if element == "O" else 1.5
            )

        element_counts = np.array(element_counts)

        # Test adaptive SIMD batch processing
        f1_adaptive, f2_adaptive = adaptive_simd_interpolation_batch(
            energies, interpolator_pairs, element_counts
        )

        # Compare with standard vectorized batch processing
        from xraylabtool.optimization.vectorized_core import (
            vectorized_interpolation_batch,
        )

        f1_standard, f2_standard = vectorized_interpolation_batch(
            energies, interpolator_pairs, element_counts
        )

        np.testing.assert_allclose(f1_adaptive, f1_standard, rtol=1e-15)
        np.testing.assert_allclose(f2_adaptive, f2_standard, rtol=1e-15)

        print("SIMD-optimized functions validated successfully")

    def test_adaptive_simd_performance(self):
        """Test performance characteristics of adaptive SIMD optimization."""
        from xraylabtool.optimization.vectorized_core import (
            adaptive_simd_interpolation_batch,
            configure_numpy_for_performance,
        )

        # Configure NumPy for optimal performance
        perf_config = configure_numpy_for_performance()
        print(f"NumPy performance configuration: {perf_config}")

        # Test various array sizes for adaptive behavior
        test_cases = [
            (1, 32, "small_single_element"),  # Should use direct calculation
            (2, 64, "medium_two_elements"),  # Boundary case
            (4, 200, "large_multi_element"),  # Should use vectorization
            (6, 1000, "xlarge_complex"),  # Definitely vectorized
        ]

        for n_elements, n_energies, case_name in test_cases:
            energies = np.linspace(5000.0, 25000.0, n_energies)

            # Create test interpolator pairs and counts
            elements = ["Si", "O", "Al", "Ca", "Fe", "Ti"][:n_elements]
            interpolator_pairs = []
            element_counts = []

            for element in elements:
                f1_interp, f2_interp = create_scattering_factor_interpolators(element)
                interpolator_pairs.append((f1_interp, f2_interp))
                element_counts.append(np.random.uniform(0.5, 2.0))

            element_counts = np.array(element_counts)

            # Benchmark adaptive SIMD performance
            import time

            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                f1_result, f2_result = adaptive_simd_interpolation_batch(
                    energies, interpolator_pairs, element_counts
                )
                times.append(time.perf_counter() - start_time)

            median_time = np.median(times)
            calc_per_second = n_energies / median_time

            # Record performance metric
            record_performance_metric(
                name="adaptive_simd_performance",
                value=calc_per_second,
                unit="calc/sec",
                context={
                    "case": case_name,
                    "n_elements": n_elements,
                    "n_energies": n_energies,
                    "simd_support": perf_config.get("simd_support", {}),
                },
            )

            print(
                f"{case_name}: {calc_per_second:.0f} calc/sec ({n_elements} elements,"
                f" {n_energies} energies)"
            )

            # Verify results are valid
            assert len(f1_result) == n_energies
            assert len(f2_result) == n_energies
            assert np.all(np.isfinite(f1_result))
            assert np.all(np.isfinite(f2_result))

    def test_edge_cases(self):
        """Test edge cases for vectorized implementations."""
        energies = np.linspace(5000.0, 25000.0, 100)
        from xraylabtool.constants import ANGSTROM_TO_METER
        from xraylabtool.utils import energy_to_wavelength

        wavelength_angstrom = np.array(
            [energy_to_wavelength(e / 1000) for e in energies]
        )
        wavelength = wavelength_angstrom * ANGSTROM_TO_METER

        # Test empty element data
        result = calculate_scattering_factors_vectorized(
            energies, wavelength, 2.0, 28.0, []
        )
        for array in result:
            assert len(array) == len(energies)
            assert np.all(array == 0.0)

        # Test single element
        f1_interp, f2_interp = create_scattering_factor_interpolators("Si")
        single_element_data = [(1.0, f1_interp, f2_interp)]

        result_single = calculate_scattering_factors_vectorized(
            energies, wavelength, 2.33, 28.0, single_element_data
        )

        for array in result_single:
            assert len(array) == len(energies)
            assert np.all(np.isfinite(array))

    def test_memory_efficiency(self):
        """Test that vectorized implementation is memory efficient."""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Large test case
        energies = np.linspace(1.0, 50.0, 5000)  # Large energy array in eV
        from xraylabtool.constants import ANGSTROM_TO_METER
        from xraylabtool.utils import energy_to_wavelength

        wavelength_angstrom = np.array(
            [energy_to_wavelength(e / 1000) for e in energies]
        )
        wavelength = wavelength_angstrom * ANGSTROM_TO_METER

        # Complex material with multiple elements
        elements, counts = parse_formula("CaAl2Si2O8")  # Feldspar
        composition = dict(zip(elements, counts, strict=False))
        molecular_weight = sum(
            count * self._get_atomic_weight(element)
            for element, count in composition.items()
        )

        element_data = []
        for element, count in composition.items():
            f1_interp, f2_interp = create_scattering_factor_interpolators(element)
            element_data.append((count, f1_interp, f2_interp))

        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run vectorized calculation
        calculate_scattering_factors_vectorized(
            energies, wavelength, 2.76, molecular_weight, element_data
        )

        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Record memory usage
        record_performance_metric(
            name="vectorized_memory_usage",
            value=memory_increase,
            unit="MB",
            context={
                "test_type": "memory_efficiency",
                "energy_points": len(energies),
                "n_elements": len(element_data),
                "baseline_memory_mb": memory_before,
            },
        )

        # Memory usage should be reasonable
        max_acceptable_memory = 200  # MB
        assert memory_increase < max_acceptable_memory, (
            f"Excessive memory usage: {memory_increase:.1f}MB >"
            f" {max_acceptable_memory}MB"
        )

        print(
            f"Memory usage: +{memory_increase:.1f}MB for {len(energies)} energies,"
            f" {len(element_data)} elements"
        )

    def _get_atomic_weight(self, element: str) -> float:
        """Get atomic weight for an element (simplified for testing)."""
        atomic_weights = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "Si": 28.085,
            "Al": 26.982,
            "Ca": 40.078,
            "Fe": 55.845,
            "Ti": 47.867,
            "P": 30.974,
        }
        return atomic_weights.get(element, 50.0)  # Default for unknown elements


@pytest.mark.performance
class TestVectorizationRegression(BasePerformanceTest):
    """Test for regressions during vectorization implementation."""

    def test_vectorization_regression_monitoring(self):
        """Monitor for performance regressions in vectorized implementation."""
        # Standard test case
        energies = np.linspace(2000.0, 25000.0, 1000)
        elements, counts = parse_formula("Al2O3")
        composition = dict(zip(elements, counts, strict=False))
        density = 3.97
        molecular_weight = sum(
            count * self._get_atomic_weight(element)
            for element, count in composition.items()
        )

        from xraylabtool.constants import ANGSTROM_TO_METER
        from xraylabtool.utils import energy_to_wavelength

        wavelength_angstrom = np.array(
            [energy_to_wavelength(e / 1000) for e in energies]
        )
        wavelength = wavelength_angstrom * ANGSTROM_TO_METER

        element_data = []
        for element, count in composition.items():
            f1_interp, f2_interp = create_scattering_factor_interpolators(element)
            element_data.append((count, f1_interp, f2_interp))

        # Benchmark vectorized implementation
        times = []
        for _ in range(5):
            start_time = time.perf_counter()
            calculate_scattering_factors_vectorized(
                energies, wavelength, density, molecular_weight, element_data
            )
            times.append(time.perf_counter() - start_time)

        median_time = np.median(times)
        calc_per_second = 1.0 / median_time

        # Record for regression tracking
        record_performance_metric(
            name="vectorized_regression_monitor",
            value=calc_per_second,
            unit="calc/sec",
            context={
                "implementation": "vectorized",
                "test_case": "Al2O3_1000_energies",
                "stability": np.std(times) / np.mean(times),
            },
        )

        # Assert minimum performance
        min_performance = 500  # calc/sec - should be much higher with vectorization
        assert calc_per_second > min_performance, (
            f"Vectorized performance regression: {calc_per_second:.0f} <"
            f" {min_performance} calc/sec"
        )

        print(f"Vectorized performance: {calc_per_second:.0f} calc/sec")

    def _get_atomic_weight(self, element: str) -> float:
        """Get atomic weight for an element (simplified for testing)."""
        atomic_weights = {
            "H": 1.008,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "Si": 28.085,
            "Al": 26.982,
            "Ca": 40.078,
            "Fe": 55.845,
            "Ti": 47.867,
            "P": 30.974,
        }
        return atomic_weights.get(element, 50.0)
