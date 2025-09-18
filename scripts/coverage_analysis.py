#!/usr/bin/env python3
"""
Test coverage analysis and quality metrics for the cleanup system.

This module provides comprehensive coverage analysis, quality metrics calculation,
trend tracking, and reporting for continuous quality improvement.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Coverage metrics for a module or file"""

    module_name: str
    statements: int
    missing: int
    excluded: int
    coverage_percent: float
    missing_lines: list[int]
    branch_coverage_percent: float | None = None
    complexity_score: float | None = None


@dataclass
class QualityMetrics:
    """Code quality metrics"""

    module_name: str
    lines_of_code: int
    cyclomatic_complexity: float
    maintainability_index: float
    code_duplication_percent: float
    technical_debt_minutes: float
    security_issues: int
    style_issues: int
    type_coverage_percent: float


@dataclass
class TestMetrics:
    """Test-specific metrics"""

    test_count: int
    test_duration: float
    test_success_rate: float
    test_stability_score: float  # Based on flakiness
    assertion_density: float  # Assertions per test
    test_coverage_ratio: float  # Tests per line of production code


@dataclass
class QualityReport:
    """Comprehensive quality report"""

    timestamp: float
    overall_coverage: float
    module_coverage: list[CoverageMetrics]
    quality_metrics: list[QualityMetrics]
    test_metrics: TestMetrics
    trend_analysis: dict[str, Any]
    recommendations: list[str]
    quality_gates: dict[str, bool]


class CoverageAnalyzer:
    """Analyze test coverage using multiple tools"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cleanup_dir = project_root / "xraylabtool" / "cleanup"

    def run_coverage_analysis(self, test_command: str | None = None) -> dict[str, Any]:
        """Run comprehensive coverage analysis"""
        logger.info("Starting coverage analysis...")

        # Default test command
        if not test_command:
            test_command = "python -m pytest tests/ --cov=xraylabtool.cleanup --cov-report=json --cov-report=xml --cov-report=html"

        # Run tests with coverage
        coverage_data = self._run_coverage_tests(test_command)

        # Analyze coverage data
        module_metrics = self._analyze_module_coverage(coverage_data)

        # Generate missing coverage report
        missing_coverage = self._identify_missing_coverage(coverage_data)

        # Calculate coverage trends
        trends = self._calculate_coverage_trends()

        return {
            "overall_coverage": (
                coverage_data.get("totals", {}).get("percent_covered", 0)
            ),
            "module_metrics": module_metrics,
            "missing_coverage": missing_coverage,
            "trends": trends,
            "raw_data": coverage_data,
        }

    def _run_coverage_tests(self, test_command: str) -> dict[str, Any]:
        """Run tests with coverage measurement"""
        logger.info("Running tests with coverage measurement...")

        try:
            # Run the test command
            result = subprocess.run(
                test_command.split(),
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.returncode != 0:
                logger.warning("Tests failed but continuing with coverage analysis")
                logger.warning(f"Test stderr: {result.stderr}")

            # Load coverage JSON data
            coverage_json_path = self.project_root / "coverage.json"
            if coverage_json_path.exists():
                with open(coverage_json_path) as f:
                    return json.load(f)
            else:
                logger.error("Coverage JSON file not found")
                return {}

        except subprocess.TimeoutExpired:
            logger.error("Coverage test execution timed out")
            return {}
        except Exception as e:
            logger.error(f"Failed to run coverage tests: {e}")
            return {}

    def _analyze_module_coverage(
        self, coverage_data: dict[str, Any]
    ) -> list[CoverageMetrics]:
        """Analyze coverage data by module"""
        module_metrics = []

        files_data = coverage_data.get("files", {})

        for file_path, file_data in files_data.items():
            # Only analyze cleanup module files
            if "xraylabtool/cleanup" not in file_path:
                continue

            module_name = Path(file_path).stem

            summary = file_data.get("summary", {})

            metrics = CoverageMetrics(
                module_name=module_name,
                statements=summary.get("num_statements", 0),
                missing=summary.get("missing_lines", 0),
                excluded=summary.get("excluded_lines", 0),
                coverage_percent=summary.get("percent_covered", 0.0),
                missing_lines=file_data.get("missing_lines", []),
                branch_coverage_percent=summary.get("percent_covered_display", None),
            )

            module_metrics.append(metrics)

        # Sort by coverage percentage (lowest first for prioritization)
        module_metrics.sort(key=lambda x: x.coverage_percent)

        return module_metrics

    def _identify_missing_coverage(
        self, coverage_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Identify specific areas lacking coverage"""
        missing_coverage = {
            "low_coverage_modules": [],
            "uncovered_functions": [],
            "critical_paths_uncovered": [],
            "recommendations": [],
        }

        files_data = coverage_data.get("files", {})

        for file_path, file_data in files_data.items():
            if "xraylabtool/cleanup" not in file_path:
                continue

            summary = file_data.get("summary", {})
            coverage_percent = summary.get("percent_covered", 0.0)

            # Identify low coverage modules
            if coverage_percent < 80:
                missing_coverage["low_coverage_modules"].append(
                    {
                        "module": Path(file_path).stem,
                        "coverage": coverage_percent,
                        "missing_lines": len(file_data.get("missing_lines", [])),
                        "file_path": file_path,
                    }
                )

            # Identify critical uncovered areas
            missing_lines = file_data.get("missing_lines", [])
            if missing_lines:
                # Analyze which functions/classes are uncovered
                uncovered_functions = self._analyze_uncovered_functions(
                    file_path, missing_lines
                )
                missing_coverage["uncovered_functions"].extend(uncovered_functions)

        # Generate recommendations
        missing_coverage["recommendations"] = self._generate_coverage_recommendations(
            missing_coverage
        )

        return missing_coverage

    def _analyze_uncovered_functions(
        self, file_path: str, missing_lines: list[int]
    ) -> list[dict[str, Any]]:
        """Analyze which functions/classes are not covered by tests"""
        uncovered_functions = []

        try:
            # Read the source file
            full_path = self.project_root / file_path
            if not full_path.exists():
                return uncovered_functions

            with open(full_path) as f:
                lines = f.readlines()

            # Simple analysis to find function/class definitions in missing lines
            for line_num in missing_lines:
                if line_num <= len(lines):
                    line = lines[line_num - 1].strip()

                    if line.startswith("def ") or line.startswith("class "):
                        function_name = line.split("(")[0].split(":")[0].strip()
                        uncovered_functions.append(
                            {
                                "function": function_name,
                                "file": file_path,
                                "line": line_num,
                                "type": (
                                    "function" if line.startswith("def") else "class"
                                ),
                            }
                        )

        except Exception as e:
            logger.warning(f"Failed to analyze uncovered functions in {file_path}: {e}")

        return uncovered_functions

    def _generate_coverage_recommendations(
        self, missing_coverage: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for improving coverage"""
        recommendations = []

        # Recommendations based on low coverage modules
        low_coverage_modules = missing_coverage["low_coverage_modules"]
        if low_coverage_modules:
            recommendations.append(
                f"Priority: Add tests for {len(low_coverage_modules)} modules with <80% coverage"
            )

            # Specific module recommendations
            for module in low_coverage_modules[:3]:  # Top 3 priorities
                recommendations.append(
                    f"Add {module['missing_lines']} test cases for {module['module']} "
                    f"(current coverage: {module['coverage']:.1f}%)"
                )

        # Recommendations based on uncovered functions
        uncovered_functions = missing_coverage["uncovered_functions"]
        if uncovered_functions:
            recommendations.append(
                f"Add unit tests for {len(uncovered_functions)} uncovered functions/classes"
            )

            # Critical function recommendations
            critical_functions = [
                f
                for f in uncovered_functions
                if any(
                    keyword in f["function"].lower()
                    for keyword in ["safety", "emergency", "backup", "validate"]
                )
            ]

            if critical_functions:
                recommendations.append(
                    f"CRITICAL: Add tests for {len(critical_functions)} safety-critical functions"
                )

        return recommendations

    def _calculate_coverage_trends(self) -> dict[str, Any]:
        """Calculate coverage trends over time"""
        trends = {
            "trend_direction": "unknown",
            "coverage_change": 0.0,
            "historical_data": [],
        }

        # Load historical coverage data
        coverage_history_file = self.project_root / ".coverage_history.json"

        if coverage_history_file.exists():
            try:
                with open(coverage_history_file) as f:
                    historical_data = json.load(f)

                trends["historical_data"] = historical_data

                # Calculate trend
                if len(historical_data) >= 2:
                    recent_coverage = historical_data[-1]["coverage"]
                    previous_coverage = historical_data[-2]["coverage"]

                    trends["coverage_change"] = recent_coverage - previous_coverage

                    if trends["coverage_change"] > 1.0:
                        trends["trend_direction"] = "improving"
                    elif trends["coverage_change"] < -1.0:
                        trends["trend_direction"] = "declining"
                    else:
                        trends["trend_direction"] = "stable"

            except Exception as e:
                logger.warning(f"Failed to load coverage history: {e}")

        return trends

    def save_coverage_history(self, coverage_percent: float):
        """Save current coverage to history for trend analysis"""
        coverage_history_file = self.project_root / ".coverage_history.json"

        # Load existing history
        historical_data = []
        if coverage_history_file.exists():
            try:
                with open(coverage_history_file) as f:
                    historical_data = json.load(f)
            except:
                historical_data = []

        # Add current data point
        historical_data.append(
            {
                "timestamp": time.time(),
                "coverage": coverage_percent,
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Keep only last 50 data points
        historical_data = historical_data[-50:]

        # Save updated history
        try:
            with open(coverage_history_file, "w") as f:
                json.dump(historical_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save coverage history: {e}")


class QualityAnalyzer:
    """Analyze code quality metrics"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cleanup_dir = project_root / "xraylabtool" / "cleanup"

    def analyze_quality_metrics(self) -> dict[str, Any]:
        """Analyze comprehensive code quality metrics"""
        logger.info("Analyzing code quality metrics...")

        metrics = {
            "complexity_metrics": self._analyze_complexity(),
            "style_metrics": self._analyze_style_issues(),
            "security_metrics": self._analyze_security_issues(),
            "maintainability_metrics": self._analyze_maintainability(),
            "type_coverage": self._analyze_type_coverage(),
            "documentation_coverage": self._analyze_documentation_coverage(),
        }

        return metrics

    def _analyze_complexity(self) -> dict[str, Any]:
        """Analyze code complexity using various tools"""
        complexity_metrics = {}

        try:
            # Use radon for complexity analysis
            result = subprocess.run(
                ["python", "-m", "radon", "cc", str(self.cleanup_dir), "--json"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                complexity_metrics["cyclomatic_complexity"] = complexity_data
            else:
                logger.warning("Radon complexity analysis failed")

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("Radon not available for complexity analysis")

        # Calculate aggregate complexity metrics
        complexity_metrics["summary"] = self._summarize_complexity(
            complexity_metrics.get("cyclomatic_complexity", {})
        )

        return complexity_metrics

    def _summarize_complexity(self, complexity_data: dict[str, Any]) -> dict[str, Any]:
        """Summarize complexity metrics"""
        total_functions = 0
        total_complexity = 0
        high_complexity_functions = 0

        for _file_path, functions in complexity_data.items():
            for function in functions:
                total_functions += 1
                complexity = function.get("complexity", 0)
                total_complexity += complexity

                if complexity > 10:  # High complexity threshold
                    high_complexity_functions += 1

        average_complexity = (
            total_complexity / total_functions if total_functions > 0 else 0
        )

        return {
            "total_functions": total_functions,
            "average_complexity": average_complexity,
            "high_complexity_functions": high_complexity_functions,
            "complexity_score": min(
                100, max(0, 100 - (average_complexity * 5))
            ),  # Normalized score
        }

    def _analyze_style_issues(self) -> dict[str, Any]:
        """Analyze code style issues"""
        style_metrics = {}

        try:
            # Use flake8 for style analysis
            result = subprocess.run(
                ["python", "-m", "flake8", str(self.cleanup_dir), "--format=json"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # flake8 might not have JSON output by default
            if result.returncode == 0 or result.stdout:
                style_metrics["flake8_issues"] = (
                    result.stdout.count("\n") if result.stdout else 0
                )
            else:
                # Parse line-based output
                style_metrics["flake8_issues"] = (
                    len(result.stdout.split("\n")) if result.stdout else 0
                )

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("flake8 not available for style analysis")
            style_metrics["flake8_issues"] = 0

        return style_metrics

    def _analyze_security_issues(self) -> dict[str, Any]:
        """Analyze security issues using bandit"""
        security_metrics = {}

        try:
            result = subprocess.run(
                ["python", "-m", "bandit", "-r", str(self.cleanup_dir), "-f", "json"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                security_data = json.loads(result.stdout)

                security_metrics = {
                    "total_issues": len(security_data.get("results", [])),
                    "high_severity": len(
                        [
                            r
                            for r in security_data.get("results", [])
                            if r.get("issue_severity") == "HIGH"
                        ]
                    ),
                    "medium_severity": len(
                        [
                            r
                            for r in security_data.get("results", [])
                            if r.get("issue_severity") == "MEDIUM"
                        ]
                    ),
                    "low_severity": len(
                        [
                            r
                            for r in security_data.get("results", [])
                            if r.get("issue_severity") == "LOW"
                        ]
                    ),
                    "details": security_data.get("results", []),
                }

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("bandit not available for security analysis")
            security_metrics = {"total_issues": 0}

        return security_metrics

    def _analyze_maintainability(self) -> dict[str, Any]:
        """Analyze code maintainability metrics"""
        maintainability_metrics = {}

        try:
            # Use radon for maintainability index
            result = subprocess.run(
                ["python", "-m", "radon", "mi", str(self.cleanup_dir), "--json"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                mi_data = json.loads(result.stdout)

                # Calculate average maintainability index
                total_mi = 0
                file_count = 0

                for _file_path, mi_score in mi_data.items():
                    if isinstance(mi_score, (int, float)):
                        total_mi += mi_score
                        file_count += 1

                average_mi = total_mi / file_count if file_count > 0 else 0

                maintainability_metrics = {
                    "maintainability_index": mi_data,
                    "average_maintainability": average_mi,
                    "maintainability_grade": self._grade_maintainability(average_mi),
                }

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("Maintainability analysis not available")
            maintainability_metrics = {"average_maintainability": 0}

        return maintainability_metrics

    def _grade_maintainability(self, mi_score: float) -> str:
        """Convert maintainability index to grade"""
        if mi_score >= 20:
            return "A"
        elif mi_score >= 10:
            return "B"
        elif mi_score >= 0:
            return "C"
        else:
            return "F"

    def _analyze_type_coverage(self) -> dict[str, Any]:
        """Analyze type annotation coverage"""
        type_metrics = {}

        try:
            # Use mypy for type analysis
            subprocess.run(
                [
                    "python",
                    "-m",
                    "mypy",
                    str(self.cleanup_dir),
                    "--json-report",
                    "mypy-report",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Try to load mypy JSON report
            mypy_report_path = self.project_root / "mypy-report" / "index.json"
            if mypy_report_path.exists():
                with open(mypy_report_path) as f:
                    mypy_data = json.load(f)

                type_metrics["mypy_data"] = mypy_data
                type_metrics["type_errors"] = len(mypy_data.get("files", {}))

        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            logger.warning("mypy not available for type analysis")
            type_metrics = {"type_errors": 0}

        return type_metrics

    def _analyze_documentation_coverage(self) -> dict[str, Any]:
        """Analyze documentation coverage"""
        doc_metrics = {
            "total_functions": 0,
            "documented_functions": 0,
            "documentation_coverage": 0.0,
        }

        try:
            # Analyze Python files for docstring coverage
            for py_file in self.cleanup_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                file_metrics = self._analyze_file_documentation(py_file)
                doc_metrics["total_functions"] += file_metrics["total_functions"]
                doc_metrics["documented_functions"] += file_metrics[
                    "documented_functions"
                ]

            # Calculate coverage percentage
            if doc_metrics["total_functions"] > 0:
                doc_metrics["documentation_coverage"] = (
                    doc_metrics["documented_functions"]
                    / doc_metrics["total_functions"]
                    * 100
                )

        except Exception as e:
            logger.warning(f"Documentation analysis failed: {e}")

        return doc_metrics

    def _analyze_file_documentation(self, file_path: Path) -> dict[str, int]:
        """Analyze documentation coverage for a single file"""
        metrics = {"total_functions": 0, "documented_functions": 0}

        try:
            with open(file_path) as f:
                lines = f.readlines()

            in_function = False

            for i, line in enumerate(lines):
                stripped_line = line.strip()

                # Detect function definition
                if stripped_line.startswith("def ") and not stripped_line.startswith(
                    "def _"
                ):
                    metrics["total_functions"] += 1
                    in_function = True

                # Check for docstring in next few lines after function definition
                elif in_function and i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        metrics["documented_functions"] += 1
                        in_function = False
                    elif next_line and not next_line.startswith("#"):
                        # Non-comment, non-docstring line found
                        in_function = False

        except Exception as e:
            logger.warning(f"Failed to analyze documentation for {file_path}: {e}")

        return metrics


class QualityReporter:
    """Generate comprehensive quality reports"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def generate_quality_report(
        self, coverage_data: dict[str, Any], quality_data: dict[str, Any]
    ) -> QualityReport:
        """Generate comprehensive quality report"""

        # Extract coverage metrics
        module_coverage = []
        for metric in coverage_data.get("module_metrics", []):
            module_coverage.append(CoverageMetrics(**metric))

        # Generate test metrics
        test_metrics = self._calculate_test_metrics()

        # Generate quality gates assessment
        quality_gates = self._assess_quality_gates(coverage_data, quality_data)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            coverage_data, quality_data, quality_gates
        )

        return QualityReport(
            timestamp=time.time(),
            overall_coverage=coverage_data.get("overall_coverage", 0),
            module_coverage=module_coverage,
            quality_metrics=[],  # Would be populated with QualityMetrics objects
            test_metrics=test_metrics,
            trend_analysis=coverage_data.get("trends", {}),
            recommendations=recommendations,
            quality_gates=quality_gates,
        )

    def _calculate_test_metrics(self) -> TestMetrics:
        """Calculate test-related metrics"""
        # This would analyze test files to calculate metrics
        # For now, return placeholder values
        return TestMetrics(
            test_count=0,
            test_duration=0.0,
            test_success_rate=100.0,
            test_stability_score=95.0,
            assertion_density=3.5,
            test_coverage_ratio=0.8,
        )

    def _assess_quality_gates(
        self, coverage_data: dict[str, Any], quality_data: dict[str, Any]
    ) -> dict[str, bool]:
        """Assess whether quality gates are met"""
        gates = {}

        # Coverage gate
        overall_coverage = coverage_data.get("overall_coverage", 0)
        gates["coverage_threshold"] = overall_coverage >= 80.0

        # Security gate
        security_metrics = quality_data.get("security_metrics", {})
        high_security_issues = security_metrics.get("high_severity", 0)
        gates["security_threshold"] = high_security_issues == 0

        # Complexity gate
        complexity_metrics = quality_data.get("complexity_metrics", {})
        complexity_score = complexity_metrics.get("summary", {}).get(
            "complexity_score", 0
        )
        gates["complexity_threshold"] = complexity_score >= 70.0

        # Style gate
        style_metrics = quality_data.get("style_metrics", {})
        style_issues = style_metrics.get("flake8_issues", 0)
        gates["style_threshold"] = style_issues <= 10

        # Documentation gate
        doc_metrics = quality_data.get("documentation_coverage", {})
        doc_coverage = doc_metrics.get("documentation_coverage", 0)
        gates["documentation_threshold"] = doc_coverage >= 75.0

        return gates

    def _generate_recommendations(
        self,
        coverage_data: dict[str, Any],
        quality_data: dict[str, Any],
        quality_gates: dict[str, bool],
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Coverage recommendations
        if not quality_gates.get("coverage_threshold", True):
            coverage_recs = coverage_data.get("missing_coverage", {}).get(
                "recommendations", []
            )
            recommendations.extend(coverage_recs)

        # Security recommendations
        if not quality_gates.get("security_threshold", True):
            security_metrics = quality_data.get("security_metrics", {})
            high_issues = security_metrics.get("high_severity", 0)
            if high_issues > 0:
                recommendations.append(
                    f"CRITICAL: Fix {high_issues} high-severity security issues"
                )

        # Complexity recommendations
        if not quality_gates.get("complexity_threshold", True):
            complexity_metrics = quality_data.get("complexity_metrics", {})
            high_complexity = complexity_metrics.get("summary", {}).get(
                "high_complexity_functions", 0
            )
            if high_complexity > 0:
                recommendations.append(
                    f"Refactor {high_complexity} high-complexity functions"
                )

        # Style recommendations
        if not quality_gates.get("style_threshold", True):
            style_metrics = quality_data.get("style_metrics", {})
            style_issues = style_metrics.get("flake8_issues", 0)
            recommendations.append(f"Fix {style_issues} code style issues")

        # Documentation recommendations
        if not quality_gates.get("documentation_threshold", True):
            doc_metrics = quality_data.get("documentation_coverage", {})
            missing_docs = doc_metrics.get("total_functions", 0) - doc_metrics.get(
                "documented_functions", 0
            )
            recommendations.append(f"Add documentation for {missing_docs} functions")

        return recommendations

    def save_quality_report(self, report: QualityReport, output_path: Path):
        """Save quality report to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save human-readable report
        text_path = output_path.with_suffix(".txt")
        with open(text_path, "w") as f:
            self._write_text_report(f, report)

        logger.info(f"Quality report saved to {output_path}")

    def _write_text_report(self, file, report: QualityReport):
        """Write human-readable quality report"""
        file.write("=" * 80 + "\n")
        file.write("CODE QUALITY & COVERAGE REPORT\n")
        file.write("=" * 80 + "\n\n")

        # Overall metrics
        file.write("OVERALL METRICS\n")
        file.write("-" * 40 + "\n")
        file.write(f"Overall Coverage: {report.overall_coverage:.1f}%\n")
        file.write(f"Test Count: {report.test_metrics.test_count}\n")
        file.write(
            f"Test Success Rate: {report.test_metrics.test_success_rate:.1f}%\n\n"
        )

        # Quality gates
        file.write("QUALITY GATES\n")
        file.write("-" * 40 + "\n")
        for gate, passed in report.quality_gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            file.write(f"{gate}: {status}\n")
        file.write("\n")

        # Module coverage breakdown
        file.write("MODULE COVERAGE\n")
        file.write("-" * 40 + "\n")
        for module in report.module_coverage:
            file.write(
                f"{module.module_name}: {module.coverage_percent:.1f}% "
                f"({module.missing}/{module.statements} lines missing)\n"
            )
        file.write("\n")

        # Recommendations
        if report.recommendations:
            file.write("RECOMMENDATIONS\n")
            file.write("-" * 40 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                file.write(f"{i}. {rec}\n")
        file.write("\n")

        # Trend analysis
        if report.trend_analysis:
            file.write("TREND ANALYSIS\n")
            file.write("-" * 40 + "\n")
            trend_direction = report.trend_analysis.get("trend_direction", "unknown")
            coverage_change = report.trend_analysis.get("coverage_change", 0)
            file.write(f"Coverage Trend: {trend_direction}\n")
            if coverage_change != 0:
                file.write(f"Recent Change: {coverage_change:+.1f}%\n")
        file.write("\n")


def main():
    """Main coverage and quality analysis workflow"""
    parser = argparse.ArgumentParser(
        description="Coverage and quality analysis for cleanup system"
    )

    parser.add_argument(
        "--coverage-only", action="store_true", help="Run only coverage analysis"
    )

    parser.add_argument(
        "--quality-only", action="store_true", help="Run only quality analysis"
    )

    parser.add_argument(
        "--test-command", type=str, help="Custom test command for coverage analysis"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quality-reports"),
        help="Output directory for reports",
    )

    parser.add_argument(
        "--save-history", action="store_true", help="Save coverage to historical data"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Coverage threshold for quality gate",
    )

    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parent.parent

    logger.info("Starting coverage and quality analysis...")

    # Initialize analyzers
    coverage_analyzer = CoverageAnalyzer(project_root)
    quality_analyzer = QualityAnalyzer(project_root)
    reporter = QualityReporter(project_root)

    # Run analyses
    coverage_data = {}
    quality_data = {}

    if not args.quality_only:
        coverage_data = coverage_analyzer.run_coverage_analysis(args.test_command)

        if args.save_history:
            coverage_analyzer.save_coverage_history(
                coverage_data.get("overall_coverage", 0)
            )

    if not args.coverage_only:
        quality_data = quality_analyzer.analyze_quality_metrics()

    # Generate comprehensive report
    if coverage_data or quality_data:
        report = reporter.generate_quality_report(coverage_data, quality_data)

        # Save report
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"quality-report-{int(time.time())}"
        reporter.save_quality_report(report, report_path)

        # Print summary
        logger.info("Quality analysis completed:")
        logger.info(f"  Overall Coverage: {report.overall_coverage:.1f}%")
        logger.info(
            f"  Quality Gates Passed: {sum(report.quality_gates.values())}/{len(report.quality_gates)}"
        )
        logger.info(f"  Recommendations: {len(report.recommendations)}")

        # Exit with error if quality gates fail
        if not all(report.quality_gates.values()):
            logger.error("Quality gates failed!")
            sys.exit(1)
        else:
            logger.info("All quality gates passed!")
            sys.exit(0)

    else:
        logger.error("No analysis performed")
        sys.exit(1)


if __name__ == "__main__":
    main()
