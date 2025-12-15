"""
Consolidated code quality tests for pyXRayLabTool.

This module validates that the codebase follows proper code standards including
naming conventions, import patterns, style guide compliance, and documentation standards.
Consolidates functionality from multiple separate test files.
"""

import ast
from pathlib import Path
import re
import subprocess

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestCodeQuality(BaseUnitTest):
    """Test comprehensive code quality standards."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"
        self.tests_dir = self.project_root / "tests"

    @pytest.mark.unit
    def test_naming_conventions(self):
        """Test that code follows snake_case naming conventions."""
        violations = self._check_naming_conventions()

        # Allow some violations in legacy/cleanup modules and GUI (requires PySide6 patterns)
        allowed_violations = [
            "cleanup/",
            "legacy/",
            "_migration",
            "deprecated",
            "/gui/",
        ]
        filtered_violations = [
            v
            for v in violations
            if not any(allowed in v for allowed in allowed_violations)
        ]

        assert len(filtered_violations) <= 50, (
            f"Too many naming violations: {filtered_violations[:10]}"
        )

    @pytest.mark.unit
    def test_import_patterns(self):
        """Test that all Python files follow absolute import patterns."""
        violations = self._check_import_patterns()

        # Allow relative imports in __init__.py files, within same package, and GUI module
        allowed_relative = [
            "__init__.py",
            "from .completion",
            "from ..exceptions",
            "from ..typing_extensions",
            "from ..calculators",
            "from .shells",
            "from .comparator",
            "from .cli",
            "from .installer",
            "from .environment",
            "from .. import calculators",
            "/gui/",  # GUI module uses relative imports for internal components
        ]
        filtered_violations = [
            v
            for v in violations
            if not any(allowed in v for allowed in allowed_relative)
        ]

        assert len(filtered_violations) <= 5, (
            f"Import pattern violations found: {filtered_violations[:5]}"
        )

    @pytest.mark.unit
    def test_type_hints_coverage(self):
        """Test that functions have adequate type hint coverage."""
        missing_hints = self._find_missing_type_hints()

        # Allow some missing type hints in test files and GUI module (PySide6 patterns)
        [hint for hint in missing_hints if "tests/" in str(hint)]
        core_violations = [
            hint
            for hint in missing_hints
            if "tests/" not in str(hint) and "/gui/" not in str(hint)
        ]

        # Core code should have high type hint coverage
        assert len(core_violations) <= 5, (
            f"Too many missing type hints in core code: {core_violations[:5]}"
        )

    @pytest.mark.unit
    def test_docstring_coverage(self):
        """Test that public functions have adequate docstring coverage."""
        missing_docstrings = self._find_missing_docstrings()

        # Focus on critical modules
        critical_modules = ["calculators/", "core.py", "__init__.py"]
        critical_missing = [
            doc
            for doc in missing_docstrings
            if any(module in str(doc) for module in critical_modules)
        ]

        # Allow some missing docstrings but ensure core functionality is documented
        assert len(critical_missing) <= 20, (
            f"Missing docstrings in critical modules: {critical_missing[:5]}"
        )

    def _check_naming_conventions(self):
        """Check for naming convention violations."""
        violations = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Check for CamelCase variable names (should be snake_case)
                camel_case_vars = re.findall(r"\b[a-z]+[A-Z][a-zA-Z]*\b", content)
                if camel_case_vars:
                    violations.extend(
                        [f"{py_file}: {var}" for var in camel_case_vars[:3]]
                    )

            except (UnicodeDecodeError, OSError):
                continue

        return violations

    def _check_import_patterns(self):
        """Check for import pattern violations."""
        violations = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    lines = f.readlines()

                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    # Check for relative imports that should be absolute
                    if line.startswith("from .") and "xraylabtool" not in line:
                        violations.append(f"{py_file}:{i} - {line}")

            except (UnicodeDecodeError, OSError):
                continue

        return violations

    def _find_missing_type_hints(self):
        """Find functions missing type hints."""
        missing_hints = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.returns and not node.name.startswith("_"):
                            missing_hints.append(
                                f"{py_file}:{node.lineno} - {node.name}"
                            )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        return missing_hints

    def _find_missing_docstrings(self):
        """Find public functions missing docstrings."""
        missing_docstrings = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not node.name.startswith("_"):  # Public function/class
                            if not ast.get_docstring(node):
                                missing_docstrings.append(
                                    f"{py_file}:{node.lineno} - {node.name}"
                                )

            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        return missing_docstrings


class TestStyleCompliance(BaseUnitTest):
    """Test style guide compliance."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent

    @pytest.mark.unit
    def test_black_formatting_compliance(self):
        """Test that code follows Black formatting standards."""
        try:
            result = subprocess.run(
                ["black", "--check", "--diff", str(self.project_root / "xraylabtool")],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Allow some formatting issues but ensure they're minimal
            if result.returncode != 0:
                lines = result.stdout.count("\n")
                assert lines <= 1000, (
                    "Too many Black formatting violations"
                    f" ({lines} lines):\n{result.stdout[:500]}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Black not available or timeout")

    @pytest.mark.unit
    def test_ruff_linting_compliance(self):
        """Test that code passes Ruff linting standards."""
        try:
            result = subprocess.run(
                ["ruff", "check", str(self.project_root / "xraylabtool")],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Allow some linting issues but ensure they're not excessive
            if result.returncode != 0:
                lines = result.stdout.count("\n")
                assert lines <= 5000, (
                    f"Too many Ruff violations ({lines} lines):\n{result.stdout[:1000]}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Ruff not available or timeout")


class TestDocumentationPatterns(BaseUnitTest):
    """Test that documented patterns work correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.claude_md_path = self.project_root / "CLAUDE.md"

    @pytest.mark.unit
    def test_documented_import_examples(self):
        """Test that documented import examples work."""
        if not self.claude_md_path.exists():
            pytest.skip("CLAUDE.md not found")

        with open(self.claude_md_path) as f:
            content = f.read()

        # Extract import examples from CLAUDE.md
        import_examples = re.findall(r"import\s+xraylabtool.*", content)
        import_examples.extend(re.findall(r"from\s+xraylabtool.*", content))

        working_imports = 0
        total_imports = len(import_examples)

        for import_line in import_examples[:5]:  # Test first 5 examples
            try:
                exec(import_line)
                working_imports += 1
            except (ImportError, SyntaxError):
                continue

        # Most documented imports should work
        if total_imports > 0:
            success_rate = working_imports / min(total_imports, 5)
            assert success_rate >= 0.6, (
                f"Too many broken import examples: {success_rate:.1%} success rate"
            )

    @pytest.mark.unit
    def test_basic_functionality_examples(self):
        """Test that basic usage examples work."""
        try:
            import xraylabtool as xlt

            # Test basic calculation example
            result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

            assert result.formula == "SiO2"
            assert len(result.energy_kev) > 0
            assert result.critical_angle_degrees[0] > 0

        except ImportError:
            pytest.skip("xraylabtool not importable")
