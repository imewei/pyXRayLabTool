#!/usr/bin/env python3
"""
Automated testing workflows for the cleanup system.

This module provides automated testing orchestration, test discovery,
parallel execution, and intelligent test selection based on code changes.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test execution result"""

    test_name: str
    test_type: str
    duration: float
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    coverage_data: dict[str, Any] | None = None
    performance_data: dict[str, Any] | None = None


@dataclass
class TestConfiguration:
    """Test execution configuration"""

    test_types: list[str]
    python_versions: list[str]
    parallel_workers: int
    timeout_seconds: int
    coverage_enabled: bool
    performance_enabled: bool
    fail_fast: bool
    verbose: bool


class TestDiscovery:
    """Discover and categorize tests"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"

    def discover_tests(self) -> dict[str, list[Path]]:
        """Discover all tests categorized by type"""
        test_categories = {
            "unit": [],
            "integration": [],
            "performance": [],
            "safety": [],
            "end_to_end": [],
        }

        if not self.test_dir.exists():
            logger.warning(f"Test directory not found: {self.test_dir}")
            return test_categories

        # Discover unit tests
        unit_dir = self.test_dir / "unit"
        if unit_dir.exists():
            test_categories["unit"] = list(unit_dir.glob("test_*.py"))

        # Discover integration tests
        integration_dir = self.test_dir / "integration"
        if integration_dir.exists():
            test_categories["integration"] = list(integration_dir.glob("test_*.py"))

        # Discover performance tests
        performance_dir = self.test_dir / "performance"
        if performance_dir.exists():
            test_categories["performance"] = list(performance_dir.glob("test_*.py"))

        # Special test files
        safety_test = self.test_dir / "unit" / "test_safety_mechanisms.py"
        if safety_test.exists():
            test_categories["safety"] = [safety_test]

        e2e_test = self.test_dir / "integration" / "test_end_to_end_workflows.py"
        if e2e_test.exists():
            test_categories["end_to_end"] = [e2e_test]

        return test_categories

    def get_affected_tests(self, changed_files: list[Path]) -> set[str]:
        """Determine which test categories should run based on changed files"""
        affected_categories = set()

        for changed_file in changed_files:
            file_str = str(changed_file)

            # Check if cleanup system files changed
            if "xraylabtool/cleanup/" in file_str:
                affected_categories.update(["unit", "integration", "safety"])

                # If core safety components changed, run all tests
                if any(
                    component in file_str
                    for component in [
                        "safety_integration",
                        "backup_manager",
                        "emergency_manager",
                        "audit_logger",
                    ]
                ):
                    affected_categories.update(
                        ["unit", "integration", "performance", "safety", "end_to_end"]
                    )

            # Check if test files changed
            elif "tests/" in file_str:
                if "unit/" in file_str:
                    affected_categories.add("unit")
                elif "integration/" in file_str:
                    affected_categories.add("integration")
                elif "performance/" in file_str:
                    affected_categories.add("performance")

            # Check if CI/CD files changed
            elif any(
                ci_file in file_str
                for ci_file in [".github/workflows/", "scripts/ci-", "Makefile"]
            ):
                affected_categories.update(["unit", "integration"])

        # If no specific tests affected, run unit tests as minimum
        if not affected_categories:
            affected_categories.add("unit")

        return affected_categories


class TestExecutor:
    """Execute tests with various configurations"""

    def __init__(self, project_root: Path, config: TestConfiguration):
        self.project_root = project_root
        self.config = config
        self.results: list[TestResult] = []

    def run_test_category(
        self, category: str, test_files: list[Path]
    ) -> list[TestResult]:
        """Run all tests in a category"""
        if not test_files:
            logger.info(f"No tests found for category: {category}")
            return []

        logger.info(f"Running {category} tests: {len(test_files)} files")

        category_results = []

        if self.config.parallel_workers > 1 and len(test_files) > 1:
            # Run tests in parallel
            with ThreadPoolExecutor(
                max_workers=self.config.parallel_workers
            ) as executor:
                futures = []

                for test_file in test_files:
                    future = executor.submit(
                        self._execute_single_test, category, test_file
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        category_results.append(result)

                        if not result.success and self.config.fail_fast:
                            logger.error(
                                f"Test failed, stopping due to fail_fast: {result.test_name}"
                            )
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break

                    except Exception as e:
                        logger.error(f"Test execution error: {e}")

        else:
            # Run tests sequentially
            for test_file in test_files:
                result = self._execute_single_test(category, test_file)
                category_results.append(result)

                if not result.success and self.config.fail_fast:
                    logger.error(
                        f"Test failed, stopping due to fail_fast: {result.test_name}"
                    )
                    break

        return category_results

    def _execute_single_test(self, category: str, test_file: Path) -> TestResult:
        """Execute a single test file"""
        test_name = f"{category}::{test_file.name}"
        logger.info(f"Executing test: {test_name}")

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", str(test_file)]

        # Add coverage if enabled
        if self.config.coverage_enabled:
            cmd.extend(
                [
                    "--cov=xraylabtool.cleanup",
                    "--cov-report=json:coverage.json",
                    "--cov-report=term-missing",
                ]
            )

        # Add verbosity
        if self.config.verbose:
            cmd.append("-v")

        # Add timeout
        cmd.extend(["--timeout", str(self.config.timeout_seconds)])

        # Add JUnit XML output
        junit_file = self.project_root / f"test-results-{category}-{test_file.stem}.xml"
        cmd.extend(["--junit-xml", str(junit_file)])

        # Special handling for performance tests
        if category == "performance":
            os.environ["CLEANUP_PERF_MODE"] = "true"
            cmd.append("-s")  # Don't capture output for performance tests

        # Special handling for safety tests
        if category == "safety":
            os.environ["CLEANUP_SAFETY_STRICT"] = "true"

        # Execute test
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            # Load coverage data if available
            coverage_data = None
            if (
                self.config.coverage_enabled
                and (self.project_root / "coverage.json").exists()
            ):
                try:
                    with open(self.project_root / "coverage.json") as f:
                        coverage_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load coverage data: {e}")

            # Extract performance data for performance tests
            performance_data = None
            if category == "performance":
                performance_data = self._extract_performance_data(result.stdout)

            return TestResult(
                test_name=test_name,
                test_type=category,
                duration=duration,
                success=success,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                coverage_data=coverage_data,
                performance_data=performance_data,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Test timed out after {duration:.1f}s: {test_name}")

            return TestResult(
                test_name=test_name,
                test_type=category,
                duration=duration,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Test timed out after {duration:.1f}s",
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Test execution failed: {test_name}: {e}")

            return TestResult(
                test_name=test_name,
                test_type=category,
                duration=duration,
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
            )

    def _extract_performance_data(self, stdout: str) -> dict[str, Any]:
        """Extract performance metrics from test output"""
        performance_data = {}

        # Look for performance indicators in output
        lines = stdout.split("\n")
        for line in lines:
            if "duration:" in line.lower():
                try:
                    duration = float(line.split(":")[1].strip().replace("s", ""))
                    performance_data["duration"] = duration
                except:
                    pass

            if "throughput:" in line.lower():
                try:
                    throughput = float(line.split(":")[1].strip().split()[0])
                    performance_data["throughput"] = throughput
                except:
                    pass

            if "memory peak:" in line.lower():
                try:
                    memory = float(line.split(":")[1].strip().replace("MB", ""))
                    performance_data["memory_peak_mb"] = memory
                except:
                    pass

        return performance_data


class TestReporter:
    """Generate test reports and summaries"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def generate_summary_report(self, results: list[TestResult]) -> dict[str, Any]:
        """Generate comprehensive test summary"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests

        # Calculate durations by category
        category_stats = {}
        for result in results:
            category = result.test_type
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "duration": 0.0,
                }

            category_stats[category]["total"] += 1
            category_stats[category]["duration"] += result.duration

            if result.success:
                category_stats[category]["passed"] += 1
            else:
                category_stats[category]["failed"] += 1

        # Aggregate coverage data
        coverage_summary = self._aggregate_coverage_data(results)

        # Aggregate performance data
        performance_summary = self._aggregate_performance_data(results)

        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (
                    (passed_tests / total_tests * 100) if total_tests > 0 else 0
                ),
                "total_duration": sum(r.duration for r in results),
            },
            "categories": category_stats,
            "coverage": coverage_summary,
            "performance": performance_summary,
            "failed_tests": [
                {
                    "name": r.test_name,
                    "type": r.test_type,
                    "exit_code": r.exit_code,
                    "stderr": r.stderr[:500],  # Truncate for summary
                }
                for r in results
                if not r.success
            ],
        }

    def _aggregate_coverage_data(self, results: list[TestResult]) -> dict[str, Any]:
        """Aggregate coverage data from all test results"""
        coverage_data = {}

        for result in results:
            if result.coverage_data and "totals" in result.coverage_data:
                totals = result.coverage_data["totals"]
                coverage_data[result.test_name] = {
                    "covered_lines": totals.get("covered_lines", 0),
                    "num_statements": totals.get("num_statements", 0),
                    "percent_covered": totals.get("percent_covered", 0.0),
                    "missing_lines": totals.get("missing_lines", 0),
                }

        # Calculate overall coverage if we have data
        if coverage_data:
            total_statements = sum(
                data["num_statements"] for data in coverage_data.values()
            )
            total_covered = sum(
                data["covered_lines"] for data in coverage_data.values()
            )

            overall_coverage = (
                (total_covered / total_statements * 100) if total_statements > 0 else 0
            )

            return {
                "overall_percent": overall_coverage,
                "total_statements": total_statements,
                "total_covered": total_covered,
                "by_test": coverage_data,
            }

        return {}

    def _aggregate_performance_data(self, results: list[TestResult]) -> dict[str, Any]:
        """Aggregate performance data from performance tests"""
        performance_data = {}

        for result in results:
            if result.test_type == "performance" and result.performance_data:
                performance_data[result.test_name] = result.performance_data

        if performance_data:
            # Calculate aggregate metrics
            durations = [data.get("duration", 0) for data in performance_data.values()]
            throughputs = [
                data.get("throughput", 0) for data in performance_data.values()
            ]
            memory_peaks = [
                data.get("memory_peak_mb", 0) for data in performance_data.values()
            ]

            return {
                "tests_run": len(performance_data),
                "average_duration": sum(durations) / len(durations) if durations else 0,
                "max_duration": max(durations) if durations else 0,
                "average_throughput": (
                    sum(throughputs) / len(throughputs) if throughputs else 0
                ),
                "max_memory_mb": max(memory_peaks) if memory_peaks else 0,
                "by_test": performance_data,
            }

        return {}

    def save_reports(self, results: list[TestResult], output_dir: Path):
        """Save detailed test reports"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate summary report
        summary = self.generate_summary_report(results)

        # Save JSON report
        json_report_path = output_dir / "test-report.json"
        with open(json_report_path, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "summary": summary,
                    "detailed_results": [asdict(r) for r in results],
                },
                f,
                indent=2,
            )

        # Save human-readable report
        text_report_path = output_dir / "test-report.txt"
        with open(text_report_path, "w") as f:
            self._write_text_report(f, summary, results)

        logger.info(f"Reports saved to {output_dir}")

    def _write_text_report(
        self, file, summary: dict[str, Any], results: list[TestResult]
    ):
        """Write human-readable test report"""
        file.write("=" * 80 + "\n")
        file.write("AUTOMATED TEST EXECUTION REPORT\n")
        file.write("=" * 80 + "\n\n")

        # Summary section
        file.write("SUMMARY\n")
        file.write("-" * 40 + "\n")
        file.write(f"Total Tests: {summary['summary']['total_tests']}\n")
        file.write(f"Passed: {summary['summary']['passed']}\n")
        file.write(f"Failed: {summary['summary']['failed']}\n")
        file.write(f"Success Rate: {summary['summary']['success_rate']:.1f}%\n")
        file.write(f"Total Duration: {summary['summary']['total_duration']:.2f}s\n\n")

        # Category breakdown
        file.write("CATEGORY BREAKDOWN\n")
        file.write("-" * 40 + "\n")
        for category, stats in summary["categories"].items():
            file.write(f"{category.upper()}:\n")
            file.write(f"  Total: {stats['total']}\n")
            file.write(f"  Passed: {stats['passed']}\n")
            file.write(f"  Failed: {stats['failed']}\n")
            file.write(f"  Duration: {stats['duration']:.2f}s\n\n")

        # Coverage information
        if summary.get("coverage"):
            file.write("COVERAGE SUMMARY\n")
            file.write("-" * 40 + "\n")
            coverage = summary["coverage"]
            file.write(f"Overall Coverage: {coverage.get('overall_percent', 0):.1f}%\n")
            file.write(f"Total Statements: {coverage.get('total_statements', 0)}\n")
            file.write(f"Covered Statements: {coverage.get('total_covered', 0)}\n\n")

        # Performance information
        if summary.get("performance"):
            file.write("PERFORMANCE SUMMARY\n")
            file.write("-" * 40 + "\n")
            perf = summary["performance"]
            file.write(f"Performance Tests: {perf.get('tests_run', 0)}\n")
            file.write(f"Average Duration: {perf.get('average_duration', 0):.2f}s\n")
            file.write(f"Max Duration: {perf.get('max_duration', 0):.2f}s\n")
            file.write(
                f"Average Throughput: {perf.get('average_throughput', 0):.1f} files/s\n"
            )
            file.write(f"Max Memory: {perf.get('max_memory_mb', 0):.1f}MB\n\n")

        # Failed tests
        if summary["failed_tests"]:
            file.write("FAILED TESTS\n")
            file.write("-" * 40 + "\n")
            for failed_test in summary["failed_tests"]:
                file.write(f"Test: {failed_test['name']}\n")
                file.write(f"Type: {failed_test['type']}\n")
                file.write(f"Exit Code: {failed_test['exit_code']}\n")
                file.write(f"Error: {failed_test['stderr']}\n")
                file.write("-" * 20 + "\n")


def load_ci_config(config_path: Path) -> dict[str, Any]:
    """Load CI configuration from YAML file"""
    if not config_path.exists():
        logger.warning(f"CI config file not found: {config_path}")
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load CI config: {e}")
        return {}


def main():
    """Main automated testing workflow"""
    parser = argparse.ArgumentParser(
        description="Automated testing workflow for cleanup system"
    )

    parser.add_argument(
        "--test-types",
        nargs="+",
        choices=["unit", "integration", "performance", "safety", "end_to_end", "all"],
        default=["unit"],
        help="Test types to run",
    )

    parser.add_argument(
        "--python-versions",
        nargs="+",
        default=[f"{sys.version_info.major}.{sys.version_info.minor}"],
        help="Python versions to test",
    )

    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of parallel workers"
    )

    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds"
    )

    parser.add_argument(
        "--coverage", action="store_true", help="Enable coverage reporting"
    )

    parser.add_argument(
        "--performance", action="store_true", help="Enable performance monitoring"
    )

    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-results"),
        help="Output directory for reports",
    )

    parser.add_argument(
        "--changed-files",
        nargs="*",
        help="List of changed files for intelligent test selection",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/ci-config.yml"),
        help="CI configuration file",
    )

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    logger.info("Starting automated testing workflow")
    logger.info(f"Project root: {project_root}")

    # Load CI configuration
    load_ci_config(args.config)

    # Create test configuration
    config = TestConfiguration(
        test_types=args.test_types,
        python_versions=args.python_versions,
        parallel_workers=args.parallel,
        timeout_seconds=args.timeout,
        coverage_enabled=args.coverage,
        performance_enabled=args.performance,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
    )

    # Discover tests
    discovery = TestDiscovery(project_root)
    all_tests = discovery.discover_tests()

    # Intelligent test selection based on changed files
    if args.changed_files:
        changed_file_paths = [Path(f) for f in args.changed_files]
        affected_categories = discovery.get_affected_tests(changed_file_paths)
        logger.info(
            f"Affected test categories based on changed files: {affected_categories}"
        )

        # Filter test types to only affected ones
        if "all" not in config.test_types:
            config.test_types = [
                t for t in config.test_types if t in affected_categories
            ]

    # Expand "all" test type
    if "all" in config.test_types:
        config.test_types = list(all_tests.keys())

    logger.info(f"Running test types: {config.test_types}")

    # Execute tests
    executor = TestExecutor(project_root, config)
    all_results = []

    for test_type in config.test_types:
        if test_type not in all_tests:
            logger.warning(f"Unknown test type: {test_type}")
            continue

        test_files = all_tests[test_type]
        if not test_files:
            logger.info(f"No tests found for type: {test_type}")
            continue

        category_results = executor.run_test_category(test_type, test_files)
        all_results.extend(category_results)

        # Check for failures and fail-fast
        if config.fail_fast and any(not r.success for r in category_results):
            logger.error(
                f"Tests failed in category {test_type}, stopping due to fail-fast"
            )
            break

    # Generate reports
    reporter = TestReporter(project_root)
    reporter.save_reports(all_results, args.output_dir)

    # Print summary
    summary = reporter.generate_summary_report(all_results)
    logger.info("Test execution completed:")
    logger.info(f"  Total: {summary['summary']['total_tests']}")
    logger.info(f"  Passed: {summary['summary']['passed']}")
    logger.info(f"  Failed: {summary['summary']['failed']}")
    logger.info(f"  Success Rate: {summary['summary']['success_rate']:.1f}%")

    # Exit with error code if tests failed
    if summary["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
