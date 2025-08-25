#!/usr/bin/env python3
"""
Test runner script for XRayLabTool pytest suite.

This script runs the complete test suite including integration tests,
performance benchmarks, and generates coverage reports.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, optional=False):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        # Check if this is a missing dependency error
        if "unrecognized arguments" in e.stderr and optional:
            print(f"⚠️  SKIPPED: {description} (optional dependency not installed)")
            return True
        else:
            print(f"❌ FAILED: {description}")
            print(f"Exit code: {e.returncode}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            return False
    except FileNotFoundError:
        print(f"⚠️  SKIPPED: {description} (command not found)")
        return True


def main():
    """Main test runner."""
    print("XRayLabTool Test Suite Runner")
    print("=" * 60)

    # Change to project directory
    project_dir = Path(__file__).parent.absolute()
    os.chdir(project_dir)
    print(f"Working directory: {project_dir}")

    success_count = 0
    total_count = 0

    # List of test commands to run
    test_commands = [
        # Basic test discovery
        (["python", "-m", "pytest", "--collect-only", "-q"], "Test Discovery", False),
        # Run main integration tests with coverage
        (
            [
                "python",
                "-m",
                "pytest",
                "tests/test_integration.py",
                "-v",
                "--cov=xraylabtool",
                "--cov-report=term-missing",
            ],
            "Integration Tests with Coverage",
            True,
        ),
        # Fallback integration tests without coverage
        (
            ["python", "-m", "pytest", "tests/test_integration.py", "-v"],
            "Integration Tests (without coverage)",
            False,
        ),
        # Run formula parsing tests
        (
            ["python", "-m", "pytest", "tests/test_formula_parsing.py", "-v"],
            "Formula Parsing Tests",
            False,
        ),
        # Run utility tests (both versions)
        (
            ["python", "-m", "pytest", "tests/test_utils.py", "-v"],
            "Utility Functions Tests",
            False,
        ),
        (
            ["python", "-m", "pytest", "tests/test_utils_enhanced.py", "-v"],
            "Enhanced Utility Tests",
            False,
        ),
        # Run atomic data tests
        (
            ["python", "-m", "pytest", "tests/test_atomic_data.py", "-v"],
            "Atomic Data Tests",
            False,
        ),
        # Run scattering factor tests
        (
            ["python", "-m", "pytest", "tests/test_scattering_factors.py", "-v"],
            "Scattering Factor Tests",
            False,
        ),
        # Run core physics tests
        (
            ["python", "-m", "pytest", "tests/test_core_physics.py", "-v"],
            "Core Physics Tests",
            False,
        ),
        # Run core tests
        (
            ["python", "-m", "pytest", "tests/test_core.py", "-v"],
            "Core Module Tests",
            False,
        ),
        # Run robustness tests
        (
            ["python", "-m", "pytest", "tests/test_robustness.py", "-v"],
            "Robustness Tests",
            False,
        ),
        # Run smooth data tests
        (
            ["python", "-m", "pytest", "tests/test_smooth_data.py", "-v"],
            "Smooth Data Tests",
            False,
        ),
        # Run benchmarks separately
        (
            [
                "python",
                "-m",
                "pytest",
                "tests/test_integration.py::TestPerformanceBenchmarks",
                "--benchmark-only",
                "-v",
            ],
            "Performance Benchmarks",
            True,
        ),
        # Generate HTML coverage report
        (
            [
                "python",
                "-m",
                "pytest",
                "tests/",
                "--cov=xraylabtool",
                "--cov-report=html",
                "--cov-report=xml",
            ],
            "Full Test Suite with HTML Coverage Report",
            True,
        ),
    ]

    # Run each test command
    integration_with_coverage_succeeded = False

    for cmd, description, optional in test_commands:
        # Skip fallback integration test if coverage version succeeded
        if (
            description == "Integration Tests (without coverage)"
            and integration_with_coverage_succeeded
        ):
            continue

        total_count += 1
        success = run_command(cmd, description, optional)
        if success:
            success_count += 1
            # Track if integration tests with coverage succeeded
            if description == "Integration Tests with Coverage":
                integration_with_coverage_succeeded = True

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total test suites: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")

    if success_count == total_count:
        print("🎉 All test suites completed successfully!")
        exit_code = 0
    else:
        print("⚠️  Some test suites failed. Check output above for details.")
        exit_code = 1

    # Check for coverage report
    if Path("htmlcov/index.html").exists():
        print(
            f"\n📊 Coverage report generated: file://{project_dir}/htmlcov/index.html"
        )

    if Path("coverage.xml").exists():
        print(f"📊 XML coverage report: {project_dir}/coverage.xml")

    print(f"\n{'=' * 60}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
