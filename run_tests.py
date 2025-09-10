#!/usr/bin/env python3
"""
Comprehensive test runner for xraylabtool package.

This script runs a comprehensive test suite including unit tests, integration tests,
performance tests, and benchmarks. It provides detailed reporting and handles
test environment setup.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class TestRunner:
    """Comprehensive test runner for xraylabtool."""

    def __init__(self, verbose: bool = True, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.results: dict[str, Any] = {}
        self.start_time = time.time()

    def run_command(self, cmd: list[str], description: str) -> bool:
        """Run a command and return success status."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"🔍 {description}")
            print(f"{'=' * 60}")
            print(f"Command: {' '.join(cmd)}")
            print()

        try:
            result = subprocess.run(  # nosec B603
                cmd, check=True, capture_output=not self.verbose, text=True
            )

            if self.verbose:
                print(f"✅ {description} - PASSED")

            self.results[description] = {
                "status": "PASSED",
                "returncode": result.returncode,
            }
            return True

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"❌ {description} - FAILED")
                print(f"Return code: {e.returncode}")
                if e.stdout:
                    print(f"STDOUT:\n{e.stdout}")
                if e.stderr:
                    print(f"STDERR:\n{e.stderr}")

            self.results[description] = {
                "status": "FAILED",
                "returncode": e.returncode,
                "error": str(e),
            }
            return False

    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        # Run core functionality tests using the unit test directory
        cmd = ["pytest", "tests/unit/", "-v", "--tb=short"]
        if self.parallel:
            cmd.extend(["-n", "auto"])
        return self.run_command(cmd, "Unit Tests")

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        # Run tests that test cross-module functionality
        cmd = ["pytest", "tests/integration/", "-v", "--tb=short"]
        if self.parallel:
            cmd.extend(["-n", "auto"])
        return self.run_command(cmd, "Integration Tests")

    def run_performance_tests(self) -> bool:
        """Run performance tests."""
        cmd = ["pytest", "tests/performance/", "-v", "--tb=short"]
        return self.run_command(cmd, "Performance Tests")

    def run_memory_tests(self) -> bool:
        """Run memory management tests."""
        cmd = [
            "pytest",
            "tests/performance/test_memory_management.py",
            "-v",
            "--tb=short",
        ]
        return self.run_command(cmd, "Memory Tests")

    def run_stability_tests(self) -> bool:
        """Run numerical stability tests."""
        cmd = ["pytest", "tests/unit/test_numerical_stability.py", "-v", "--tb=short"]
        return self.run_command(cmd, "Stability Tests")

    def run_optimization_tests(self) -> bool:
        """Run optimization validation tests."""
        # Run performance and optimization related tests
        test_files = [
            "tests/performance/test_performance_benchmarks.py",
            "tests/performance/test_performance_optimizations.py",
            "tests/performance/test_performance_optimizations_enhanced.py",
            "tests/performance/test_memory_management.py",
            "tests/unit/test_numerical_stability.py",
        ]
        cmd = ["pytest"] + test_files + ["-v", "--tb=short"]
        return self.run_command(cmd, "Optimization Tests")

    def run_coverage_tests(self) -> bool:
        """Run tests with coverage reporting."""
        cmd = [
            "pytest",
            "tests/unit/",
            "tests/integration/",
            "tests/performance/",
            "--cov=xraylabtool",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml",
            "--cov-fail-under=44",
            "-v",
        ]
        return self.run_command(cmd, "Coverage Tests")

    def run_benchmarks(self) -> bool:
        """Run performance benchmarks."""
        # Check if pytest-benchmark is available
        try:
            subprocess.run(  # nosec B603
                ["python", "-c", "import pytest_benchmark"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            if self.verbose:
                print("⚠️  pytest-benchmark not available, skipping benchmarks")
            return True

        cmd = [
            "pytest",
            "tests/performance/",
            "tests/integration/",
            "--benchmark-only",
            "-v",
        ]
        return self.run_command(cmd, "Performance Benchmarks")

    def run_smoke_tests(self) -> bool:
        """Run smoke tests for quick validation."""
        # Run basic functionality tests to ensure the package works
        test_files = [
            "tests/unit/test_atomic_data.py::TestAtomicDataLookup::test_atomic_number_valid_elements",
            "tests/integration/test_integration.py::TestBasicSetupAndInitialization::test_basic_setup_and_initialization",
            "tests/unit/test_core.py",  # Basic core functionality
        ]
        cmd = ["pytest"] + test_files + ["-v", "--tb=line"]
        return self.run_command(cmd, "Smoke Tests")

    def run_cli_tests(self) -> bool:
        """Test CLI functionality."""
        tests = [
            (["xraylabtool", "--version"], "CLI Version Check"),
            (["xraylabtool", "--help"], "CLI Help"),
            (
                ["xraylabtool", "calc", "SiO2", "-e", "10.0", "-d", "2.2"],
                "CLI Basic Calculation",
            ),
        ]

        all_passed = True
        for cmd, description in tests:
            success = self.run_command(cmd, description)
            all_passed = all_passed and success

        return all_passed

    def run_comprehensive_suite(self) -> bool:
        """Run the comprehensive test suite."""
        test_phases = [
            ("Smoke Tests", self.run_smoke_tests),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Memory Tests", self.run_memory_tests),
            ("Stability Tests", self.run_stability_tests),
            ("Optimization Tests", self.run_optimization_tests),
            ("CLI Tests", self.run_cli_tests),
            ("Coverage Tests", self.run_coverage_tests),
            ("Benchmarks", self.run_benchmarks),
        ]

        passed_phases = 0
        total_phases = len(test_phases)

        for phase_name, phase_func in test_phases:
            if self.verbose:
                print(f"\n🚀 Starting {phase_name}...")

            success = phase_func()
            if success:
                passed_phases += 1
                if self.verbose:
                    print(f"✅ {phase_name} completed successfully")
            elif self.verbose:
                print(f"❌ {phase_name} failed")

        return passed_phases == total_phases

    def print_summary(self) -> None:
        """Print test summary."""
        end_time = time.time()
        duration = end_time - self.start_time

        print(f"\n{'=' * 60}")
        print("📊 TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total runtime: {duration:.2f} seconds")
        print()

        passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        total = len(self.results)

        print(f"Results: {passed} passed, {failed} failed, {total} total")
        print()

        if failed > 0:
            print("❌ FAILED TESTS:")
            for name, result in self.results.items():
                if result["status"] == "FAILED":
                    print(f"  - {name}")
            print()

        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")

        print(f"{'=' * 60}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive xraylabtool tests")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Run quietly (less verbose output)"
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel where possible",
    )
    parser.add_argument(
        "--phase",
        choices=[
            "smoke",
            "unit",
            "integration",
            "performance",
            "memory",
            "stability",
            "optimization",
            "cli",
            "coverage",
            "benchmarks",
            "all",
        ],
        default="all",
        help="Run specific test phase",
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: Must be run from the project root directory")
        sys.exit(1)

    # Create test runner
    runner = TestRunner(verbose=not args.quiet, parallel=args.parallel)

    print("🧪 XRayLabTool Comprehensive Test Suite")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print()

    # Run specific phase or all phases
    if args.phase == "all":
        success = runner.run_comprehensive_suite()
    else:
        phase_map = {
            "smoke": runner.run_smoke_tests,
            "unit": runner.run_unit_tests,
            "integration": runner.run_integration_tests,
            "performance": runner.run_performance_tests,
            "memory": runner.run_memory_tests,
            "stability": runner.run_stability_tests,
            "optimization": runner.run_optimization_tests,
            "cli": runner.run_cli_tests,
            "coverage": runner.run_coverage_tests,
            "benchmarks": runner.run_benchmarks,
        }
        success = phase_map[args.phase]()

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
