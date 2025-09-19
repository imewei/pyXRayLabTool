#!/usr/bin/env python3
"""
Automated Style Guide Validation Script for XRayLabTool

This script provides comprehensive validation of Python style guide compliance
across the entire codebase, with detailed reporting and actionable feedback.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@dataclass
class StyleViolation:
    """Represents a style guide violation."""

    category: str
    severity: str  # 'error', 'warning', 'info'
    file_path: str
    line_number: int | None = None
    column: int | None = None
    message: str = ""
    rule: str = ""
    suggestion: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    timestamp: datetime = field(default_factory=datetime.now)
    total_files_checked: int = 0
    violations: list[StyleViolation] = field(default_factory=list)
    summary: dict[str, int] = field(default_factory=dict)
    tool_results: dict[str, dict] = field(default_factory=dict)
    compliance_score: float = 0.0


class StyleGuideValidator:
    """Comprehensive style guide validator for XRayLabTool."""

    def __init__(self, project_root: Path, verbose: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.violations: list[StyleViolation] = []

        # Define validation rules
        self.validation_rules = {
            "imports": self._validate_imports,
            "naming": self._validate_naming_conventions,
            "type_hints": self._validate_type_hints,
            "docstrings": self._validate_docstrings,
            "error_handling": self._validate_error_handling,
            "dataclasses": self._validate_dataclass_usage,
            "performance": self._validate_performance_patterns,
            "module_organization": self._validate_module_organization,
        }

    def validate_all(self) -> ValidationReport:
        """Run comprehensive style guide validation."""
        print(f"{Colors.HEADER}🔍 XRayLabTool Style Guide Validation{Colors.ENDC}")
        print(f"{Colors.BOLD}Project root: {self.project_root}{Colors.ENDC}")
        print("-" * 80)

        self.violations = []
        python_files = self._get_python_files()

        print(f"📁 Found {len(python_files)} Python files to analyze")

        # Run custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            print(f"\n🔎 Running {rule_name} validation...")
            try:
                rule_violations = rule_func(python_files)
                self.violations.extend(rule_violations)
                print(f"   Found {len(rule_violations)} violations")
            except Exception as e:
                print(
                    f"   {Colors.WARNING}Warning: {rule_name} validation failed: {e}{Colors.ENDC}"
                )

        # Run external tool validations
        tool_results = {}

        print("\n🛠️  Running external tool validations...")

        # Black formatting check
        tool_results["black"] = self._run_black_validation()

        # Ruff linting check
        tool_results["ruff"] = self._run_ruff_validation()

        # MyPy type checking
        tool_results["mypy"] = self._run_mypy_validation()

        # Generate report
        report = self._generate_report(python_files, tool_results)

        return report

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in the project."""
        python_files = []

        # Core xraylabtool module
        xraylabtool_dir = self.project_root / "xraylabtool"
        if xraylabtool_dir.exists():
            python_files.extend(xraylabtool_dir.rglob("*.py"))

        # Test files
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            python_files.extend(tests_dir.rglob("*.py"))

        # Scripts
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists():
            python_files.extend(scripts_dir.rglob("*.py"))

        # Filter out __pycache__ and other temporary files
        python_files = [
            f
            for f in python_files
            if "__pycache__" not in str(f) and ".pyc" not in str(f)
        ]

        return python_files

    def _validate_imports(self, python_files: list[Path]) -> list[StyleViolation]:
        """Validate import patterns according to style guide."""
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for relative imports
                    if re.match(r"from\s+\.+\w*\s+import", line):
                        violations.append(
                            StyleViolation(
                                category="imports",
                                severity="error",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                message=f"Relative import found: {line}",
                                rule="absolute_imports",
                                suggestion="Use absolute imports: from xraylabtool.module import ...",
                            )
                        )

                    # Check for star imports
                    if re.match(r"from\s+\w+.*\s+import\s+\*", line):
                        violations.append(
                            StyleViolation(
                                category="imports",
                                severity="warning",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                message=f"Star import found: {line}",
                                rule="no_star_imports",
                                suggestion="Import specific functions/classes instead of using *",
                            )
                        )

                    # Check for deprecated typing imports (Python 3.12+)
                    if re.search(
                        r"from typing import.*\b(Dict|List|Tuple|Set)\b", line
                    ):
                        violations.append(
                            StyleViolation(
                                category="imports",
                                severity="warning",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                message=f"Deprecated typing import: {line}",
                                rule="modern_typing",
                                suggestion="Use built-in types: dict, list, tuple, set for Python 3.12+",
                            )
                        )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking imports in {file_path}: {e}")
                continue

        return violations

    def _validate_naming_conventions(
        self, python_files: list[Path]
    ) -> list[StyleViolation]:
        """Validate naming conventions according to style guide."""
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(file_path))

                for node in ast.walk(tree):
                    # Check function names
                    if isinstance(node, ast.FunctionDef):
                        if not self._is_snake_case(
                            node.name
                        ) and not node.name.startswith("_"):
                            violations.append(
                                StyleViolation(
                                    category="naming",
                                    severity="error",
                                    file_path=str(
                                        file_path.relative_to(self.project_root)
                                    ),
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' should use snake_case",
                                    rule="function_naming",
                                    suggestion=f"Rename to: {self._to_snake_case(node.name)}",
                                )
                            )

                    # Check class names
                    elif isinstance(node, ast.ClassDef):
                        if not self._is_camel_case(node.name):
                            violations.append(
                                StyleViolation(
                                    category="naming",
                                    severity="error",
                                    file_path=str(
                                        file_path.relative_to(self.project_root)
                                    ),
                                    line_number=node.lineno,
                                    message=f"Class '{node.name}' should use CamelCase",
                                    rule="class_naming",
                                    suggestion=f"Rename to: {self._to_camel_case(node.name)}",
                                )
                            )

                    # Check variable assignments in function scope
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                if (
                                    not self._is_snake_case(target.id)
                                    and not target.id.isupper()
                                    and not target.id.startswith("_")
                                    and target.id not in ["MW", "SLD"]
                                ):  # Allow scientific abbreviations
                                    violations.append(
                                        StyleViolation(
                                            category="naming",
                                            severity="warning",
                                            file_path=str(
                                                file_path.relative_to(self.project_root)
                                            ),
                                            line_number=node.lineno,
                                            message=f"Variable '{target.id}' should use snake_case",
                                            rule="variable_naming",
                                            suggestion=f"Rename to: {self._to_snake_case(target.id)}",
                                        )
                                    )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking naming in {file_path}: {e}")
                continue

        return violations

    def _validate_type_hints(self, python_files: list[Path]) -> list[StyleViolation]:
        """Validate type hint usage according to style guide."""
        violations = []

        for file_path in python_files:
            # Skip certain directories where type hints may be optional
            if any(skip_dir in str(file_path) for skip_dir in ["__pycache__", "build"]):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions and special methods
                        if node.name.startswith("_"):
                            continue

                        # Check return type annotation
                        if node.returns is None:
                            violations.append(
                                StyleViolation(
                                    category="type_hints",
                                    severity="warning",
                                    file_path=str(
                                        file_path.relative_to(self.project_root)
                                    ),
                                    line_number=node.lineno,
                                    message=f"Function '{node.name}' missing return type hint",
                                    rule="return_type_hints",
                                    suggestion="Add return type annotation: -> ReturnType",
                                )
                            )

                        # Check parameter type annotations
                        for arg in node.args.args:
                            if arg.annotation is None and arg.arg not in [
                                "self",
                                "cls",
                            ]:
                                violations.append(
                                    StyleViolation(
                                        category="type_hints",
                                        severity="warning",
                                        file_path=str(
                                            file_path.relative_to(self.project_root)
                                        ),
                                        line_number=node.lineno,
                                        message=f"Parameter '{arg.arg}' in '{node.name}' missing type hint",
                                        rule="parameter_type_hints",
                                        suggestion=f"Add type annotation: {arg.arg}: ParameterType",
                                    )
                                )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking type hints in {file_path}: {e}")
                continue

        return violations

    def _validate_docstrings(self, python_files: list[Path]) -> list[StyleViolation]:
        """Validate docstring patterns according to style guide."""
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # Skip private functions/classes for docstring requirements
                        if node.name.startswith("_"):
                            continue

                        docstring = ast.get_docstring(node)
                        node_type = (
                            "Function" if isinstance(node, ast.FunctionDef) else "Class"
                        )

                        if not docstring:
                            violations.append(
                                StyleViolation(
                                    category="docstrings",
                                    severity="warning",
                                    file_path=str(
                                        file_path.relative_to(self.project_root)
                                    ),
                                    line_number=node.lineno,
                                    message=f"{node_type} '{node.name}' missing docstring",
                                    rule="docstring_required",
                                    suggestion="Add NumPy-style docstring with description and parameters",
                                )
                            )
                        elif len(docstring.strip()) < 10:
                            violations.append(
                                StyleViolation(
                                    category="docstrings",
                                    severity="info",
                                    file_path=str(
                                        file_path.relative_to(self.project_root)
                                    ),
                                    line_number=node.lineno,
                                    message=f"{node_type} '{node.name}' has minimal docstring",
                                    rule="docstring_length",
                                    suggestion="Expand docstring with more detailed description",
                                )
                            )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking docstrings in {file_path}: {e}")
                continue

        return violations

    def _validate_error_handling(
        self, python_files: list[Path]
    ) -> list[StyleViolation]:
        """Validate error handling patterns according to style guide."""
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    line_stripped = line.strip()

                    # Check for bare except clauses
                    if re.match(r"except\s*:", line_stripped):
                        violations.append(
                            StyleViolation(
                                category="error_handling",
                                severity="error",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                message="Bare except clause found",
                                rule="specific_exceptions",
                                suggestion="Catch specific exceptions: except SpecificException:",
                            )
                        )

                    # Check for overly broad Exception catches
                    if re.match(r"except\s+Exception\s*:", line_stripped):
                        violations.append(
                            StyleViolation(
                                category="error_handling",
                                severity="warning",
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                message="Generic Exception catch found",
                                rule="specific_exceptions",
                                suggestion="Consider catching more specific exceptions",
                            )
                        )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking error handling in {file_path}: {e}")
                continue

        return violations

    def _validate_dataclass_usage(
        self, python_files: list[Path]
    ) -> list[StyleViolation]:
        """Validate dataclass usage patterns according to style guide."""
        violations = []

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Look for classes with __init__ methods that might benefit from dataclass
                class_pattern = (
                    r'class\s+(\w+).*?:\s*\n(?:\s*""".*?"""\s*\n)?\s*def\s+__init__'
                )
                potential_dataclasses = re.findall(class_pattern, content, re.DOTALL)

                if potential_dataclasses and "@dataclass" not in content:
                    violations.append(
                        StyleViolation(
                            category="dataclasses",
                            severity="info",
                            file_path=str(file_path.relative_to(self.project_root)),
                            message=f"Classes {potential_dataclasses} might benefit from @dataclass",
                            rule="dataclass_usage",
                            suggestion="Consider using @dataclass for structured data classes",
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking dataclass usage in {file_path}: {e}")
                continue

        return violations

    def _validate_performance_patterns(
        self, python_files: list[Path]
    ) -> list[StyleViolation]:
        """Validate performance optimization patterns."""
        violations = []

        for file_path in python_files:
            # Focus on performance-critical modules
            if not any(
                perf_dir in str(file_path)
                for perf_dir in ["data_handling", "calculators"]
            ):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Check for potential performance issues
                if "numpy" in content or "np." in content:
                    if ".tolist()" in content:
                        violations.append(
                            StyleViolation(
                                category="performance",
                                severity="warning",
                                file_path=str(file_path.relative_to(self.project_root)),
                                message="Potential numpy array to list conversion (performance concern)",
                                rule="numpy_efficiency",
                                suggestion="Keep data as numpy arrays when possible",
                            )
                        )

                # Check for string concatenation in loops (basic pattern)
                if re.search(
                    r'for\s+\w+.*:\s*\n\s*\w+\s*\+=\s*["\']', content, re.MULTILINE
                ):
                    violations.append(
                        StyleViolation(
                            category="performance",
                            severity="info",
                            file_path=str(file_path.relative_to(self.project_root)),
                            message="Potential string concatenation in loop",
                            rule="string_efficiency",
                            suggestion="Consider using join() or f-strings for string building",
                        )
                    )

            except Exception as e:
                if self.verbose:
                    print(f"Error checking performance patterns in {file_path}: {e}")
                continue

        return violations

    def _validate_module_organization(
        self, python_files: list[Path]
    ) -> list[StyleViolation]:
        """Validate module organization according to style guide."""
        violations = []

        xraylabtool_dir = self.project_root / "xraylabtool"

        # Check for expected module structure
        expected_modules = {
            "calculators/core.py": "Core calculation functionality",
            "interfaces/cli.py": "CLI implementation",
            "validation/exceptions.py": "Custom exception hierarchy",
            "data_handling/atomic_cache.py": "Atomic data caching",
        }

        for module_path, description in expected_modules.items():
            full_path = xraylabtool_dir / module_path
            if not full_path.exists():
                violations.append(
                    StyleViolation(
                        category="module_organization",
                        severity="warning",
                        file_path=module_path,
                        message=f"Expected module missing: {description}",
                        rule="module_structure",
                        suggestion=f"Ensure {module_path} exists with {description}",
                    )
                )

        # Check for legacy modules that should be deprecated
        legacy_modules = ["core.py", "cli.py", "exceptions.py"]
        for legacy in legacy_modules:
            legacy_path = xraylabtool_dir / legacy
            if legacy_path.exists():
                with open(legacy_path) as f:
                    content = f.read()
                    if len(content) > 200 and "deprecated" not in content.lower():
                        violations.append(
                            StyleViolation(
                                category="module_organization",
                                severity="info",
                                file_path=legacy,
                                message="Legacy module exists but may not be properly deprecated",
                                rule="legacy_deprecation",
                                suggestion="Add deprecation warning or move functionality to new module structure",
                            )
                        )

        return violations

    def _run_black_validation(self) -> dict:
        """Run Black formatting validation."""
        try:
            result = subprocess.run(
                ["black", "--check", "--diff", "xraylabtool", "tests"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "issues_count": (
                    result.stdout.count("would reformat") if result.stdout else 0
                ),
            }

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            return {"status": "error", "error": str(e), "issues_count": 0}

    def _run_ruff_validation(self) -> dict:
        """Run Ruff linting validation."""
        try:
            result = subprocess.run(
                ["ruff", "check", "xraylabtool", "tests", "--output-format=json"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            violations = []
            if result.stdout:
                try:
                    violations = json.loads(result.stdout)
                except json.JSONDecodeError:
                    violations = []

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "violations": violations,
                "issues_count": len(violations),
            }

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            return {"status": "error", "error": str(e), "issues_count": 0}

    def _run_mypy_validation(self) -> dict:
        """Run MyPy type checking validation."""
        try:
            # Focus on core modules for type checking
            result = subprocess.run(
                [
                    "mypy",
                    "xraylabtool/calculators",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                ],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            error_count = result.stdout.count("error:") if result.stdout else 0

            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "issues_count": error_count,
            }

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            return {"status": "error", "error": str(e), "issues_count": 0}

    def _generate_report(
        self, python_files: list[Path], tool_results: dict
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        # Calculate summary statistics
        summary = defaultdict(int)
        for violation in self.violations:
            summary[f"{violation.category}_{violation.severity}"] += 1
            summary[violation.severity] += 1

        # Calculate compliance score
        total_checks = len(python_files) * 5  # Rough estimate of checks per file
        total_violations = len(self.violations)
        compliance_score = max(
            0, (total_checks - total_violations) / total_checks * 100
        )

        report = ValidationReport(
            total_files_checked=len(python_files),
            violations=self.violations,
            summary=dict(summary),
            tool_results=tool_results,
            compliance_score=compliance_score,
        )

        return report

    def print_report(self, report: ValidationReport):
        """Print formatted validation report."""
        print(f"\n{Colors.HEADER}📊 Style Guide Validation Report{Colors.ENDC}")
        print(
            f"{Colors.BOLD}Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}"
        )
        print("=" * 80)

        # Summary statistics
        print("\n📈 Summary:")
        print(f"  Files analyzed: {report.total_files_checked}")
        print(f"  Total violations: {len(report.violations)}")
        print(f"  Compliance score: {report.compliance_score:.1f}%")

        # Compliance score color coding
        if report.compliance_score >= 90:
            score_color = Colors.OKGREEN
        elif report.compliance_score >= 70:
            score_color = Colors.WARNING
        else:
            score_color = Colors.FAIL

        print(
            f"  Overall status: {score_color}{'EXCELLENT' if report.compliance_score >= 90 else 'GOOD' if report.compliance_score >= 70 else 'NEEDS_IMPROVEMENT'}{Colors.ENDC}"
        )

        # Violation breakdown
        if report.summary:
            print("\n🚨 Violations by category:")
            for category, count in sorted(report.summary.items()):
                if "error" in category:
                    color = Colors.FAIL
                elif "warning" in category:
                    color = Colors.WARNING
                else:
                    color = Colors.OKCYAN
                print(f"  {color}{category}: {count}{Colors.ENDC}")

        # Tool results
        print("\n🛠️  External tools:")
        for tool, results in report.tool_results.items():
            status = results.get("status", "unknown")
            issues = results.get("issues_count", 0)

            if status == "passed":
                color = Colors.OKGREEN
                status_text = "✅ PASSED"
            elif status == "failed":
                color = Colors.WARNING
                status_text = f"⚠️  FAILED ({issues} issues)"
            else:
                color = Colors.FAIL
                status_text = "❌ ERROR"

            print(f"  {tool}: {color}{status_text}{Colors.ENDC}")

        # Top violations (limit to first 10)
        if report.violations:
            print("\n🔍 Top violations:")
            for i, violation in enumerate(report.violations[:10], 1):
                severity_color = {
                    "error": Colors.FAIL,
                    "warning": Colors.WARNING,
                    "info": Colors.OKCYAN,
                }.get(violation.severity, Colors.ENDC)

                location = f"{violation.file_path}"
                if violation.line_number:
                    location += f":{violation.line_number}"

                print(
                    f"  {i:2d}. {severity_color}[{violation.severity.upper()}]{Colors.ENDC} {location}"
                )
                print(f"      {violation.message}")
                if violation.suggestion:
                    print(f"      💡 {violation.suggestion}")
                print()

        # Recommendations
        print("\n💡 Recommendations:")
        if report.compliance_score < 70:
            print(
                f"  • Focus on fixing {Colors.FAIL}error{Colors.ENDC} severity violations first"
            )
            print(
                f"  • Run {Colors.BOLD}make format{Colors.ENDC} to auto-fix formatting issues"
            )
            print("  • Review naming conventions in most violated files")
        elif report.compliance_score < 90:
            print(
                f"  • Address remaining {Colors.WARNING}warning{Colors.ENDC} severity violations"
            )
            print("  • Add missing type hints to public functions")
            print("  • Enhance docstrings for better documentation")
        else:
            print(
                f"  • {Colors.OKGREEN}Excellent compliance!{Colors.ENDC} Consider addressing remaining info-level items"
            )
            print("  • Maintain current standards in new code")

        print(
            f"\n{Colors.BOLD}For detailed fixes, run: python scripts/validate_style_guide.py --fix{Colors.ENDC}"
        )

    def save_report(self, report: ValidationReport, output_file: Path):
        """Save validation report to JSON file."""
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "total_files_checked": report.total_files_checked,
            "compliance_score": report.compliance_score,
            "summary": report.summary,
            "tool_results": report.tool_results,
            "violations": [
                {
                    "category": v.category,
                    "severity": v.severity,
                    "file_path": v.file_path,
                    "line_number": v.line_number,
                    "message": v.message,
                    "rule": v.rule,
                    "suggestion": v.suggestion,
                }
                for v in report.violations
            ],
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"\n💾 Report saved to: {output_file}")

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        return re.match(r"^[a-z][a-z0-9_]*$", name) is not None

    def _is_camel_case(self, name: str) -> bool:
        """Check if name follows CamelCase convention."""
        return re.match(r"^[A-Z][a-zA-Z0-9]*$", name) is not None

    def _to_snake_case(self, name: str) -> str:
        """Convert name to snake_case."""
        # Insert underscores before capital letters
        s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return s1.lower()

    def _to_camel_case(self, name: str) -> str:
        """Convert name to CamelCase."""
        components = name.split("_")
        return "".join(x.capitalize() for x in components)


def main():
    """Main entry point for style guide validation script."""
    parser = argparse.ArgumentParser(
        description="Validate XRayLabTool Python style guide compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_style_guide.py                    # Run full validation
  python scripts/validate_style_guide.py --verbose          # Verbose output
  python scripts/validate_style_guide.py --output report.json  # Save report
  python scripts/validate_style_guide.py --categories imports,naming  # Specific checks
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--output", "-o", type=Path, help="Save detailed report to JSON file"
    )

    parser.add_argument(
        "--categories",
        type=str,
        help="Comma-separated list of categories to check (imports,naming,type_hints,docstrings,error_handling,dataclasses,performance,module_organization)",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    args = parser.parse_args()

    # Validate project root
    if not args.project_root.exists():
        print(
            f"{Colors.FAIL}Error: Project root '{args.project_root}' does not exist{Colors.ENDC}"
        )
        sys.exit(1)

    # Initialize validator
    validator = StyleGuideValidator(args.project_root, verbose=args.verbose)

    # Filter categories if specified
    if args.categories:
        requested_categories = [cat.strip() for cat in args.categories.split(",")]
        validator.validation_rules = {
            cat: func
            for cat, func in validator.validation_rules.items()
            if cat in requested_categories
        }

    try:
        # Run validation
        report = validator.validate_all()

        # Print report
        validator.print_report(report)

        # Save report if requested
        if args.output:
            validator.save_report(report, args.output)

        # Exit with appropriate code
        if report.compliance_score < 50:
            sys.exit(2)  # Major issues
        elif report.compliance_score < 80:
            sys.exit(1)  # Minor issues
        else:
            sys.exit(0)  # Good compliance

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Validation interrupted by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.FAIL}Validation failed with error: {e}{Colors.ENDC}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
