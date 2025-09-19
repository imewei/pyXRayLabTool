"""
Performance baseline establishment tests for XRayLabTool optimization.

This module establishes comprehensive performance baselines across different
material types, complexities, and energy ranges to support optimization efforts.
"""

from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
from xraylabtool.calculators import calculate_single_material_properties
from xraylabtool.optimization.regression_detector import (
    PerformanceRegressionDetector,
    record_performance_metric,
)


class TestPerformanceBaselines(BasePerformanceTest):
    """
    Establish performance baselines across material types and complexities.

    This test suite measures and records baseline performance metrics for:
    - Different material types (elements, compounds, alloys)
    - Various energy ranges (single, arrays, wide ranges)
    - Different calculation complexities
    - Memory usage patterns
    """

    def setup_method(self):
        """Set up baseline testing environment."""
        super().setup_method()
        self.baseline_data = {}
        self.detector = PerformanceRegressionDetector(
            data_file=Path("baseline_performance.json")
        )

    def test_single_element_baselines(self):
        """Establish baselines for single element calculations."""
        test_elements = [
            ("Si", 2.33, "Silicon - semiconductor"),
            ("Fe", 7.87, "Iron - transition metal"),
            ("Au", 19.32, "Gold - heavy metal"),
            ("C", 2.26, "Carbon - light element"),
            ("Pb", 11.34, "Lead - heavy element"),
            ("Al", 2.70, "Aluminum - light metal"),
            ("Cu", 8.96, "Copper - conductive metal"),
        ]

        energy_ranges = [
            (np.array([10.0]), "single_10keV"),
            (np.array([1.0, 5.0, 10.0, 20.0]), "discrete_energies"),
            (np.linspace(1.0, 30.0, 100), "linear_100_points"),
            (np.linspace(0.5, 30.0, 500), "wide_range_500_points"),
        ]

        for formula, density, description in test_elements:
            for energies, energy_desc in energy_ranges:
                test_name = f"element_{formula}_{energy_desc}"

                # Warm-up calculation
                calculate_single_material_properties(formula, energies, density)

                # Timed calculation
                start_time = time.perf_counter()
                result = calculate_single_material_properties(
                    formula, energies, density
                )
                elapsed_time = time.perf_counter() - start_time

                # Calculate throughput
                num_calculations = len(energies)
                calc_per_second = (
                    num_calculations / elapsed_time if elapsed_time > 0 else 0
                )

                # Record baseline metric
                record_performance_metric(
                    name="baseline_element_calc_per_sec",
                    value=calc_per_second,
                    unit="calc/sec",
                    context={
                        "material_type": "element",
                        "formula": formula,
                        "density": density,
                        "energy_points": num_calculations,
                        "energy_range": energy_desc,
                        "description": description,
                    },
                )

                # Store for summary
                self.baseline_data[test_name] = {
                    "calc_per_second": calc_per_second,
                    "elapsed_time": elapsed_time,
                    "num_calculations": num_calculations,
                    "material_info": description,
                }

                # Verify reasonable performance
                assert calc_per_second > 1000, (
                    f"Performance too low for {test_name}:"
                    f" {calc_per_second:.1f} calc/sec"
                )
                assert result is not None, f"Calculation failed for {test_name}"

    def test_compound_baselines(self):
        """Establish baselines for compound calculations."""
        test_compounds = [
            ("SiO2", 2.2, "Silica - simple oxide"),
            ("Al2O3", 3.95, "Alumina - ceramic"),
            ("CaCO3", 2.71, "Calcium carbonate - limestone"),
            ("NaCl", 2.16, "Sodium chloride - salt"),
            ("TiO2", 4.23, "Titanium dioxide - pigment"),
            ("Fe2O3", 5.24, "Iron oxide - rust"),
            ("MgO", 3.58, "Magnesium oxide - refractory"),
        ]

        energy_ranges = [
            (np.array([8.0]), "single_8keV"),
            (np.linspace(5.0, 25.0, 200), "medium_range_200_points"),
            (np.linspace(1.0, 30.0, 1000), "wide_range_1000_points"),
        ]

        for formula, density, description in test_compounds:
            for energies, energy_desc in energy_ranges:
                test_name = f"compound_{formula}_{energy_desc}"

                # Warm-up
                calculate_single_material_properties(formula, energies, density)

                # Timed calculation
                start_time = time.perf_counter()
                result = calculate_single_material_properties(
                    formula, energies, density
                )
                elapsed_time = time.perf_counter() - start_time

                num_calculations = len(energies)
                calc_per_second = (
                    num_calculations / elapsed_time if elapsed_time > 0 else 0
                )

                record_performance_metric(
                    name="baseline_compound_calc_per_sec",
                    value=calc_per_second,
                    unit="calc/sec",
                    context={
                        "material_type": "compound",
                        "formula": formula,
                        "density": density,
                        "energy_points": num_calculations,
                        "energy_range": energy_desc,
                        "description": description,
                    },
                )

                self.baseline_data[test_name] = {
                    "calc_per_second": calc_per_second,
                    "elapsed_time": elapsed_time,
                    "num_calculations": num_calculations,
                    "material_info": description,
                }

                # Compounds should be slower than elements but still performant
                assert calc_per_second > 500, (
                    f"Performance too low for {test_name}:"
                    f" {calc_per_second:.1f} calc/sec"
                )
                assert result is not None, f"Calculation failed for {test_name}"

    def test_complex_material_baselines(self):
        """Establish baselines for complex multi-element materials."""
        test_materials = [
            ("Si0.7Ge0.3", 3.5, "Silicon-Germanium alloy"),
            ("Al0.3Ga0.7As", 4.5, "Aluminum Gallium Arsenide"),
            ("Ti0.5Al0.5N", 4.2, "Titanium Aluminum Nitride coating"),
            ("Cr0.2Fe0.7Ni0.1", 7.9, "Stainless steel approximation"),
            ("Ca5(PO4)3OH", 3.16, "Hydroxyapatite - bone mineral"),
            ("BaTiO3", 6.02, "Barium titanate - ferroelectric"),
            ("YBa2Cu3O7", 6.38, "YBCO superconductor"),
        ]

        energy_ranges = [
            (np.linspace(8.0, 12.0, 50), "narrow_range_50_points"),
            (np.linspace(5.0, 30.0, 250), "medium_range_250_points"),
            (np.linspace(1.0, 30.0, 500), "wide_range_500_points"),
        ]

        for formula, density, description in test_materials:
            for energies, energy_desc in energy_ranges:
                test_name = f"complex_{formula.replace('.', '_')}_{energy_desc}"

                # Warm-up
                calculate_single_material_properties(formula, energies, density)

                # Timed calculation
                start_time = time.perf_counter()
                result = calculate_single_material_properties(
                    formula, energies, density
                )
                elapsed_time = time.perf_counter() - start_time

                num_calculations = len(energies)
                calc_per_second = (
                    num_calculations / elapsed_time if elapsed_time > 0 else 0
                )

                record_performance_metric(
                    name="baseline_complex_calc_per_sec",
                    value=calc_per_second,
                    unit="calc/sec",
                    context={
                        "material_type": "complex",
                        "formula": formula,
                        "density": density,
                        "energy_points": num_calculations,
                        "energy_range": energy_desc,
                        "description": description,
                    },
                )

                self.baseline_data[test_name] = {
                    "calc_per_second": calc_per_second,
                    "elapsed_time": elapsed_time,
                    "num_calculations": num_calculations,
                    "material_info": description,
                }

                # Complex materials should still meet minimum performance
                assert calc_per_second > 200, (
                    f"Performance too low for {test_name}:"
                    f" {calc_per_second:.1f} calc/sec"
                )
                assert result is not None, f"Calculation failed for {test_name}"

    def test_energy_scaling_baselines(self):
        """Test performance scaling with different energy array sizes."""
        test_material = ("Si", 2.33, "Silicon reference")
        formula, density, _description = test_material

        energy_sizes = [1, 10, 50, 100, 250, 500, 1000, 2000]

        for size in energy_sizes:
            energies = np.linspace(5.0, 25.0, size)
            test_name = f"energy_scaling_{size}_points"

            # Warm-up
            calculate_single_material_properties(formula, energies, density)

            # Timed calculation
            start_time = time.perf_counter()
            result = calculate_single_material_properties(formula, energies, density)
            elapsed_time = time.perf_counter() - start_time

            calc_per_second = size / elapsed_time if elapsed_time > 0 else 0
            time_per_calc = elapsed_time / size if size > 0 else 0

            record_performance_metric(
                name="baseline_energy_scaling_calc_per_sec",
                value=calc_per_second,
                unit="calc/sec",
                context={
                    "material_type": "element",
                    "formula": formula,
                    "energy_array_size": size,
                    "time_per_calculation": time_per_calc,
                },
            )

            record_performance_metric(
                name="baseline_energy_scaling_time_per_calc",
                value=time_per_calc * 1000,  # Convert to milliseconds
                unit="ms/calc",
                context={
                    "material_type": "element",
                    "formula": formula,
                    "energy_array_size": size,
                    "calc_per_second": calc_per_second,
                },
            )

            self.baseline_data[test_name] = {
                "calc_per_second": calc_per_second,
                "time_per_calc_ms": time_per_calc * 1000,
                "energy_array_size": size,
                "total_time": elapsed_time,
            }

            # Ensure scaling efficiency
            assert calc_per_second > 100, (
                f"Poor scaling performance for {size} points:"
                f" {calc_per_second:.1f} calc/sec"
            )
            assert result is not None, f"Calculation failed for {test_name}"

    def test_memory_usage_baselines(self):
        """Establish memory usage baselines for different calculation sizes."""
        from xraylabtool.optimization.memory_profiler import MemoryProfiler

        profiler = MemoryProfiler()
        test_cases = [
            ("Si", 2.33, np.linspace(1.0, 30.0, 100), "small_array"),
            ("SiO2", 2.2, np.linspace(1.0, 30.0, 500), "medium_array"),
            ("Al2O3", 3.95, np.linspace(1.0, 30.0, 1000), "large_array"),
            ("Ti0.5Al0.5N", 4.2, np.linspace(0.5, 30.0, 2000), "xlarge_array"),
        ]

        for formula, density, energies, size_desc in test_cases:
            test_name = f"memory_baseline_{size_desc}"

            with profiler.profile_operation(
                operation_name=f"baseline_memory_{formula}_{size_desc}",
                material=formula,
                energy_points=len(energies),
            ) as profile:
                result = calculate_single_material_properties(
                    formula, energies, density
                )

            # Record memory metrics
            record_performance_metric(
                name="baseline_memory_peak_mb",
                value=profile.peak_memory_mb,
                unit="MB",
                context={
                    "formula": formula,
                    "energy_points": len(energies),
                    "size_category": size_desc,
                    "memory_per_calc": profile.peak_memory_mb / len(energies),
                },
            )

            record_performance_metric(
                name="baseline_memory_per_calculation",
                value=(profile.peak_memory_mb / len(energies)) * 1024,  # Convert to KB
                unit="KB/calc",
                context={
                    "formula": formula,
                    "energy_points": len(energies),
                    "size_category": size_desc,
                    "total_memory_mb": profile.peak_memory_mb,
                },
            )

            self.baseline_data[test_name] = {
                "peak_memory_mb": profile.peak_memory_mb,
                "memory_per_calc_kb": (profile.peak_memory_mb / len(energies)) * 1024,
                "energy_points": len(energies),
                "formula": formula,
            }

            assert result is not None, f"Memory test failed for {test_name}"

    def teardown_method(self):
        """Save baseline summary and clean up."""
        super().teardown_method()

        # Create summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "baseline_establishment_summary": {
                "total_tests": len(self.baseline_data),
                "performance_targets": {
                    "current_minimum": "150,000 calc/sec aggregate",
                    "optimization_target": "300,000 calc/sec aggregate",
                    "stretch_target": "500,000 calc/sec aggregate",
                },
                "test_categories": {
                    "elements": len([k for k in self.baseline_data if "element_" in k]),
                    "compounds": len(
                        [k for k in self.baseline_data if "compound_" in k]
                    ),
                    "complex_materials": len(
                        [k for k in self.baseline_data if "complex_" in k]
                    ),
                    "energy_scaling": len(
                        [k for k in self.baseline_data if "energy_scaling_" in k]
                    ),
                    "memory_tests": len(
                        [k for k in self.baseline_data if "memory_baseline_" in k]
                    ),
                },
            },
            "detailed_results": self.baseline_data,
        }

        # Save summary
        summary_file = Path("performance_baseline_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nBaseline establishment completed. Summary saved to {summary_file}")
        print(f"Total baseline tests: {len(self.baseline_data)}")

        # Export detector data for CI integration
        ci_report = self.detector.export_for_ci(Path("baseline_ci_report.json"))
        print(f"CI report saved with {len(ci_report['baselines'])} baseline metrics")


@pytest.mark.performance
class TestBaselineValidation(BasePerformanceTest):
    """Validate that baseline establishment process is working correctly."""

    def test_baseline_data_integrity(self):
        """Verify baseline data is being recorded correctly."""
        import tempfile

        # Use temporary file for test isolation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            detector = PerformanceRegressionDetector(data_file=temp_path)

            # Record enough test metrics to establish a baseline (minimum_samples = 5)
            for _i in range(5):
                detector.record_metric(
                    name="test_validation_metric",
                    value=12345.0,  # Same value to get median of 12345.0
                    unit="calc/sec",
                    context={"test": "validation"},
                )

            # Verify it was recorded
            baseline = detector.get_baseline("test_validation_metric")
            assert baseline is not None
            assert baseline == 12345.0

            # Test regression detection
            alert = detector.check_regression("test_validation_metric", 10000.0)
            assert alert is not None
            assert alert.severity == "warning"
            assert alert.regression_percentage > 15
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)

    def test_baseline_persistence(self):
        """Test that baselines persist across detector instances."""
        import tempfile

        # Use temporary file for test isolation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Create detector and record enough metrics to establish baseline
            detector1 = PerformanceRegressionDetector(data_file=temp_path)
            for _i in range(5):
                detector1.record_metric("persistence_test", 9999.0, "calc/sec")

            # Create new detector instance and verify data persists
            detector2 = PerformanceRegressionDetector(data_file=temp_path)
            baseline = detector2.get_baseline("persistence_test")
            assert baseline == 9999.0
        finally:
            # Clean up
            temp_path.unlink(missing_ok=True)
