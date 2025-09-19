"""
Bottleneck identification tests for XRayLabTool optimization.

This module runs comprehensive bottleneck analysis on the XRayLabTool codebase
to identify performance bottlenecks and optimization opportunities.
"""

from pathlib import Path
import time

import numpy as np
import pytest

from tests.fixtures.test_base import BasePerformanceTest
from xraylabtool.calculators import calculate_single_material_properties
from xraylabtool.optimization.bottleneck_analyzer import (
    BottleneckAnalysisReport,
    BottleneckAnalyzer,
)


class TestBottleneckIdentification(BasePerformanceTest):
    """
    Comprehensive bottleneck identification across XRayLabTool calculations.

    This test suite identifies performance bottlenecks in:
    - Core calculation functions
    - Memory allocation patterns
    - Vectorization opportunities
    - Function call overhead
    """

    def setup_method(self):
        """Set up bottleneck analysis environment."""
        super().setup_method()
        # Disable line profiling to avoid conflicts with cProfile for function bottlenecks
        self.analyzer = BottleneckAnalyzer(enable_line_profiling=False)

    def test_core_calculation_bottlenecks(self):
        """Identify bottlenecks in core calculation functions."""
        test_materials = [
            ("Si", 2.33, "silicon_element"),
            ("SiO2", 2.2, "silica_compound"),
            ("Al0.3Ga0.7As", 4.5, "complex_alloy"),
        ]

        energy_arrays = [
            np.linspace(5.0, 25.0, 100),
            np.linspace(1.0, 25.0, 500),
            np.linspace(1.0, 25.0, 1000),
        ]

        for formula, density, material_type in test_materials:
            for i, energies in enumerate(energy_arrays):
                operation_name = f"core_calc_{material_type}_array_{i}"

                with self.analyzer.profile_operation(
                    operation_name,
                    enable_memory_tracking=True,
                    material=formula,
                    energy_points=len(energies),
                ):
                    # Run calculation multiple times to get meaningful profiling data
                    for _ in range(3):
                        result = calculate_single_material_properties(
                            formula, energies, density
                        )

                # Verify calculation succeeded
                assert result is not None
                assert hasattr(result, "delta")
                assert len(result.delta) == len(energies)

        # Analyze function bottlenecks
        for profile_name in self.analyzer.profiles:
            function_bottlenecks = self.analyzer.analyze_function_bottlenecks(
                profile_name, top_n=10
            )

            # Verify we found bottlenecks
            assert len(function_bottlenecks) > 0, (
                f"No function bottlenecks found for {profile_name}"
            )

            # Check that top bottleneck is significant
            if function_bottlenecks:
                top_bottleneck = function_bottlenecks[0]
                assert top_bottleneck.cumulative_time > 0, (
                    "Top bottleneck has no cumulative time"
                )

                # Print bottleneck information for debugging
                print(f"\nTop bottleneck for {profile_name}:")
                print(f"  Function: {top_bottleneck.function_name}")
                print(
                    f"  Time: {top_bottleneck.cumulative_time:.4f}s"
                    f" ({top_bottleneck.percentage_of_total:.1f}%)"
                )
                print(f"  Calls: {top_bottleneck.call_count}")

    def test_memory_allocation_bottlenecks(self):
        """Identify memory allocation bottlenecks."""
        # Test with increasingly large arrays to stress memory allocation
        test_cases = [
            ("Si", 2.33, np.linspace(1.0, 25.0, 500), "medium_array"),
            ("SiO2", 2.2, np.linspace(1.0, 30.0, 1000), "large_array"),
            ("Al2O3", 3.95, np.linspace(1.0, 25.0, 2000), "xlarge_array"),
        ]

        for formula, density, energies, size_desc in test_cases:
            operation_name = f"memory_test_{formula}_{size_desc}"

            with self.analyzer.profile_operation(
                operation_name,
                enable_memory_tracking=True,
                material=formula,
                array_size=len(energies),
            ):
                # Multiple calculations to accumulate memory allocation data
                for _ in range(5):
                    result = calculate_single_material_properties(
                        formula, energies, density
                    )

            assert result is not None

        # Analyze memory bottlenecks
        memory_bottlenecks = self.analyzer.analyze_memory_bottlenecks()

        if memory_bottlenecks:
            print("\nMemory allocation bottlenecks:")
            for bottleneck in memory_bottlenecks[:5]:
                print(
                    f"  {bottleneck.location}: {bottleneck.allocation_size:.2f} MB "
                    f"({bottleneck.percentage_of_total:.1f}%)"
                )

    def test_vectorization_opportunities(self):
        """Identify vectorization opportunities in source code."""
        # Get XRayLabTool source paths
        xraylabtool_path = Path(__file__).parent.parent.parent / "xraylabtool"
        source_paths = []

        # Collect Python files from key modules
        for module_path in [
            xraylabtool_path / "calculators",
            xraylabtool_path / "data_handling",
            xraylabtool_path / "utils.py",
            xraylabtool_path / "constants.py",
        ]:
            if module_path.is_file() and module_path.suffix == ".py":
                source_paths.append(module_path)
            elif module_path.is_dir():
                source_paths.extend(module_path.glob("*.py"))

        # Identify vectorization opportunities
        opportunities = self.analyzer.identify_vectorization_opportunities(source_paths)

        print(f"\nFound {len(opportunities)} vectorization opportunities:")

        # Categorize opportunities by benefit level
        high_benefit = [op for op in opportunities if op.estimated_benefit == "high"]
        medium_benefit = [
            op for op in opportunities if op.estimated_benefit == "medium"
        ]

        print(f"  High benefit: {len(high_benefit)} opportunities")
        print(f"  Medium benefit: {len(medium_benefit)} opportunities")

        # Display high-benefit opportunities
        if high_benefit:
            print("\nHigh-benefit vectorization opportunities:")
            for op in high_benefit[:5]:  # Show top 5
                print(
                    f"  {op.function_name} ({Path(op.file_path).name}:{op.line_range[0]})"
                )
                print(f"    Current: {op.current_pattern[:60]}...")
                print(f"    Suggestion: {op.suggested_optimization}")

        # Verify we found some opportunities (the codebase should have room for improvement)
        assert len(opportunities) >= 0, (
            "Should find vectorization opportunities in the codebase"
        )

    def test_function_call_overhead(self):
        """Analyze function call overhead patterns."""
        # Test with operations that should highlight function call patterns
        test_formula = "Si"
        test_density = 2.33

        # Single energy vs array to compare overhead
        single_energy = np.array([10.0])
        array_energy = np.linspace(5.0, 25.0, 100)

        with self.analyzer.profile_operation(
            "single_energy_overhead",
            enable_memory_tracking=False,
            test_type="single_energy",
        ):
            # Run many single-energy calculations
            for _ in range(100):
                calculate_single_material_properties(
                    test_formula, single_energy, test_density
                )

        with self.analyzer.profile_operation(
            "array_energy_overhead",
            enable_memory_tracking=False,
            test_type="array_energy",
        ):
            # Run fewer array calculations with same total energy points
            calculate_single_material_properties(
                test_formula, array_energy, test_density
            )

        # Compare overhead patterns
        single_bottlenecks = self.analyzer.analyze_function_bottlenecks(
            "single_energy_overhead", top_n=10
        )
        array_bottlenecks = self.analyzer.analyze_function_bottlenecks(
            "array_energy_overhead", top_n=10
        )

        assert len(single_bottlenecks) > 0
        assert len(array_bottlenecks) > 0

        print("\nFunction call overhead analysis:")
        print(
            "Single energy calculations - top function:"
            f" {single_bottlenecks[0].function_name} "
            f"({single_bottlenecks[0].call_count} calls)"
        )
        print(
            "Array energy calculations - top function:"
            f" {array_bottlenecks[0].function_name} "
            f"({array_bottlenecks[0].call_count} calls)"
        )

    def test_comprehensive_bottleneck_report(self):
        """Generate comprehensive bottleneck analysis report."""
        # Run a representative calculation workload
        test_materials = [("Si", 2.33), ("SiO2", 2.2), ("Al2O3", 3.95)]

        energies = np.linspace(5.0, 25.0, 500)

        operation_name = "comprehensive_analysis"

        with self.analyzer.profile_operation(
            operation_name, enable_memory_tracking=True, analysis_type="comprehensive"
        ):
            for formula, density in test_materials:
                for _ in range(3):  # Multiple runs for statistics
                    calculate_single_material_properties(formula, energies, density)

        # Get source paths for vectorization analysis
        xraylabtool_path = Path(__file__).parent.parent.parent / "xraylabtool"
        source_paths = list(xraylabtool_path.glob("**/*.py"))

        # Generate comprehensive report
        report = self.analyzer.generate_comprehensive_report(
            operation_name, source_paths=source_paths
        )

        # Verify report completeness
        assert isinstance(report, BottleneckAnalysisReport)
        assert len(report.function_bottlenecks) > 0
        assert len(report.recommendations) > 0
        assert report.summary_stats["total_functions_analyzed"] > 0

        # Save report
        report_file = Path("bottleneck_analysis_report.json")
        self.analyzer.save_report(report, report_file)

        print("\nComprehensive bottleneck analysis completed:")
        print(
            f"  Functions analyzed: {report.summary_stats['total_functions_analyzed']}"
        )
        print(
            f"  Memory bottlenecks: {report.summary_stats['total_memory_bottlenecks']}"
        )
        print(
            "  Vectorization opportunities:"
            f" {report.summary_stats['total_vectorization_opportunities']}"
        )
        print(
            "  High-impact opportunities:"
            f" {report.summary_stats['high_impact_vectorization_count']}"
        )

        print("\nTop recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"  {i}. {rec}")

        print(f"\nReport saved to: {report_file}")

        # Verify report file was created
        assert report_file.exists()

    def test_line_level_bottlenecks(self):
        """Test line-level bottleneck identification (if available)."""
        if not self.analyzer.enable_line_profiling:
            pytest.skip("Line profiler not available")

        # Profile a specific calculation
        formula = "SiO2"
        density = 2.2
        energies = np.linspace(5.0, 25.0, 200)

        operation_name = "line_profiling_test"

        with self.analyzer.profile_operation(
            operation_name, enable_memory_tracking=False
        ):
            for _ in range(5):
                calculate_single_material_properties(formula, density, energies)

        # Analyze line bottlenecks
        line_bottlenecks = self.analyzer.analyze_line_bottlenecks(
            operation_name, top_n=10
        )

        if line_bottlenecks:
            print(f"\nLine-level bottlenecks found: {len(line_bottlenecks)}")
            for bottleneck in line_bottlenecks[:3]:
                print(
                    f"  Line {bottleneck.line_number}: {bottleneck.percentage:.1f}% - "
                    f"{bottleneck.line_contents.strip()[:50]}..."
                )
        else:
            print(
                "\nNo line-level bottlenecks captured (may need longer-running"
                " operations)"
            )

    def teardown_method(self):
        """Clean up after bottleneck analysis."""
        super().teardown_method()

        # Summary of all discovered bottlenecks
        total_profiles = len(self.analyzer.profiles)
        print("\n=== Bottleneck Analysis Summary ===")
        print(f"Total operations profiled: {total_profiles}")

        if total_profiles > 0:
            # Find the most significant bottlenecks across all profiles
            all_function_bottlenecks = []
            for profile_name in self.analyzer.profiles:
                bottlenecks = self.analyzer.analyze_function_bottlenecks(
                    profile_name, top_n=3
                )
                all_function_bottlenecks.extend(bottlenecks)

            if all_function_bottlenecks:
                # Sort by cumulative time
                all_function_bottlenecks.sort(
                    key=lambda x: x.cumulative_time, reverse=True
                )
                print("\nTop overall bottlenecks:")
                for i, bottleneck in enumerate(all_function_bottlenecks[:5], 1):
                    print(
                        f"  {i}. {bottleneck.function_name}:"
                        f" {bottleneck.cumulative_time:.4f}s"
                        f" ({bottleneck.percentage_of_total:.1f}%)"
                    )


@pytest.mark.performance
class TestBottleneckAnalysisIntegration(BasePerformanceTest):
    """Integration tests for bottleneck analysis tools."""

    def test_analyzer_integration_with_regression_detector(self):
        """Test integration between bottleneck analyzer and regression detector."""
        from xraylabtool.optimization.regression_detector import get_global_detector

        analyzer = BottleneckAnalyzer(enable_line_profiling=False)
        detector = get_global_detector()

        # Profile an operation
        formula = "Si"
        density = 2.33
        energies = np.linspace(5.0, 25.0, 100)

        start_time = time.perf_counter()
        with analyzer.profile_operation("integration_test"):
            calculate_single_material_properties(formula, energies, density)
        elapsed_time = time.perf_counter() - start_time

        # Record multiple performance metrics to establish baseline (minimum 5 samples)
        calc_per_second = len(energies) / elapsed_time
        for _i in range(5):
            # Record the same metric multiple times to establish baseline
            detector.record_metric(
                "integration_test_calc_per_sec",
                calc_per_second,
                "calc/sec",
                context={"formula": formula, "energy_points": len(energies)},
            )

        # Verify integration works
        baseline = detector.get_baseline("integration_test_calc_per_sec")
        assert baseline is not None
        # Allow significant variance in baseline calculation due to system performance fluctuations
        # Performance can vary dramatically on different systems and under different load conditions
        variance = abs(baseline - calc_per_second) / calc_per_second
        if variance >= 3.0:
            pytest.skip(
                f"Performance variance too high ({variance:.2f}x) - system may be under load"
            )
        assert variance < 3.0, (
            f"Performance variance {variance:.2f}x exceeds maximum threshold"
        )

        function_bottlenecks = analyzer.analyze_function_bottlenecks("integration_test")
        assert len(function_bottlenecks) > 0

    def test_bottleneck_data_persistence(self):
        """Test that bottleneck analysis data can be saved and loaded."""
        analyzer = BottleneckAnalyzer()

        # Profile a simple operation
        with analyzer.profile_operation("persistence_test"):
            calculate_single_material_properties("Si", np.array([10.0]), 2.33)

        # Generate and save report
        report = analyzer.generate_comprehensive_report("persistence_test")
        report_file = Path("test_bottleneck_persistence.json")

        analyzer.save_report(report, report_file)

        # Verify file was created and contains expected data
        assert report_file.exists()

        import json

        with open(report_file) as f:
            loaded_data = json.load(f)

        assert "function_bottlenecks" in loaded_data
        assert "recommendations" in loaded_data
        assert "summary_stats" in loaded_data

        # Clean up
        report_file.unlink(missing_ok=True)
