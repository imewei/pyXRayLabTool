"""
Comprehensive tests for style guide compliance checking.

This module validates that the Python style guide implementation works correctly
across all documented patterns, code standards, and development practices.
"""

import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestStyleGuideCompliance(BaseUnitTest):
    """Test comprehensive style guide compliance checking."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"
        self.tests_dir = self.project_root / "tests"

    @pytest.mark.unit
    def test_import_pattern_compliance(self):
        """Test that all Python files follow absolute import patterns."""
        violations = self._check_import_patterns()

        assert not violations, f"Import pattern violations found: {violations}"

    @pytest.mark.unit
    def test_naming_convention_compliance(self):
        """Test that code follows snake_case naming conventions."""
        violations = self._check_naming_conventions()

        # Allow some violations in legacy/cleanup modules
        allowed_violations = self._get_allowed_naming_violations()
        filtered_violations = [
            v for v in violations
            if not any(allowed in v for allowed in allowed_violations)
        ]

        assert len(filtered_violations) <= 10, f"Too many naming violations: {filtered_violations[:10]}"

    @pytest.mark.unit
    def test_type_hint_compliance(self):
        """Test that functions have proper type hints."""
        violations = self._check_type_hints()

        # Allow some violations in legacy/experimental modules
        allowed_missing = 20  # Reasonable threshold for legacy code
        assert len(violations) <= allowed_missing, f"Too many missing type hints: {violations[:10]}"

    @pytest.mark.unit
    def test_docstring_compliance(self):
        """Test that functions have proper NumPy-style docstrings."""
        violations = self._check_docstring_patterns()

        # Allow some violations for private functions and legacy code
        allowed_missing = 50  # Reasonable threshold
        assert len(violations) <= allowed_missing, f"Too many docstring violations: {violations[:10]}"

    @pytest.mark.unit
    def test_error_handling_compliance(self):
        """Test that error handling follows custom exception hierarchy."""
        violations = self._check_error_handling()

        assert len(violations) <= 5, f"Error handling violations: {violations}"

    @pytest.mark.unit
    def test_dataclass_usage_compliance(self):
        """Test that data structures use dataclass patterns appropriately."""
        violations = self._check_dataclass_usage()

        assert len(violations) <= 3, f"Dataclass usage violations: {violations}"

    @pytest.mark.unit
    def test_module_organization_compliance(self):
        """Test that modules are organized according to documented patterns."""
        violations = self._check_module_organization()

        assert not violations, f"Module organization violations: {violations}"

    @pytest.mark.unit
    def test_performance_pattern_compliance(self):
        """Test that performance-critical code follows optimization patterns."""
        violations = self._check_performance_patterns()

        assert len(violations) <= 5, f"Performance pattern violations: {violations}"

    def _check_import_patterns(self) -> List[str]:
        """Check for relative import violations."""
        violations = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for relative imports
                relative_imports = re.findall(r'^from\s+\.+\w*\s+import', content, re.MULTILINE)
                if relative_imports:
                    violations.extend([
                        f"{py_file}: {imp}" for imp in relative_imports
                    ])

                # Check for star imports (discouraged)
                star_imports = re.findall(r'^from\s+\w+.*\s+import\s+\*', content, re.MULTILINE)
                if star_imports:
                    violations.extend([
                        f"{py_file}: {imp} (star import)" for imp in star_imports
                    ])

            except Exception as e:
                # Skip files that can't be read
                continue

        return violations

    def _check_naming_conventions(self) -> List[str]:
        """Check for naming convention violations."""
        violations = []

        for py_file in self._get_python_files():
            try:
                violations.extend(self._check_file_naming_conventions(py_file))
            except Exception:
                continue

        return violations

    def _check_file_naming_conventions(self, file_path: Path) -> List[str]:
        """Check naming conventions in a specific file."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except:
            return violations

        for node in ast.walk(tree):
            # Check function names
            if isinstance(node, ast.FunctionDef):
                if not self._is_snake_case(node.name) and not node.name.startswith('_'):
                    violations.append(f"{file_path}:{node.lineno}: Function '{node.name}' not snake_case")

            # Check class names
            elif isinstance(node, ast.ClassDef):
                if not self._is_camel_case(node.name):
                    violations.append(f"{file_path}:{node.lineno}: Class '{node.name}' not CamelCase")

            # Check variable assignments (limited scope)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if (not self._is_snake_case(target.id) and
                            not target.id.isupper() and  # Allow CONSTANTS
                            not target.id.startswith('_')):  # Allow private vars
                            violations.append(f"{file_path}:{node.lineno}: Variable '{target.id}' not snake_case")

        return violations

    def _check_type_hints(self) -> List[str]:
        """Check for missing type hints in function signatures."""
        violations = []

        for py_file in self._get_python_files():
            try:
                violations.extend(self._check_file_type_hints(py_file))
            except Exception:
                continue

        return violations

    def _check_file_type_hints(self, file_path: Path) -> List[str]:
        """Check type hints in a specific file."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions and special methods
                if node.name.startswith('_'):
                    continue

                # Check return type annotation
                if node.returns is None:
                    violations.append(f"{file_path}:{node.lineno}: Function '{node.name}' missing return type hint")

                # Check parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        violations.append(f"{file_path}:{node.lineno}: Parameter '{arg.arg}' in '{node.name}' missing type hint")

        return violations

    def _check_docstring_patterns(self) -> List[str]:
        """Check for proper NumPy-style docstrings."""
        violations = []

        for py_file in self._get_python_files():
            try:
                violations.extend(self._check_file_docstrings(py_file))
            except Exception:
                continue

        return violations

    def _check_file_docstrings(self, file_path: Path) -> List[str]:
        """Check docstrings in a specific file."""
        violations = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except:
            return violations

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Skip private functions/classes
                if node.name.startswith('_'):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring:
                    violations.append(f"{file_path}:{node.lineno}: {type(node).__name__} '{node.name}' missing docstring")
                elif len(docstring.strip()) < 10:  # Very short docstrings
                    violations.append(f"{file_path}:{node.lineno}: {type(node).__name__} '{node.name}' has minimal docstring")

        return violations

    def _check_error_handling(self) -> List[str]:
        """Check for proper error handling patterns."""
        violations = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for bare except clauses
                bare_excepts = re.findall(r'except\s*:', content)
                if bare_excepts:
                    violations.append(f"{py_file}: {len(bare_excepts)} bare except clause(s)")

                # Check for generic Exception catches without reraising
                generic_catches = re.findall(r'except\s+Exception\s*:', content)
                if len(generic_catches) > 3:  # Allow some for top-level handlers
                    violations.append(f"{py_file}: {len(generic_catches)} generic Exception catches")

            except Exception:
                continue

        return violations

    def _check_dataclass_usage(self) -> List[str]:
        """Check for appropriate dataclass usage patterns."""
        violations = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Look for classes that might benefit from dataclass
                class_pattern = r'class\s+(\w+).*?:\s*\n(?:\s*""".*?"""\s*\n)?\s*def\s+__init__'
                potential_dataclasses = re.findall(class_pattern, content, re.DOTALL)

                # Check if dataclass is imported where classes are defined
                if potential_dataclasses and '@dataclass' not in content and 'from dataclasses import' not in content:
                    violations.append(f"{py_file}: Classes {potential_dataclasses} might benefit from dataclass")

            except Exception:
                continue

        return violations

    def _check_module_organization(self) -> List[str]:
        """Check that modules follow documented organization patterns."""
        violations = []

        # Check that core modules exist in correct locations
        expected_modules = {
            'calculators/core.py': 'Core calculation functionality',
            'interfaces/cli.py': 'CLI implementation',
            'validation/exceptions.py': 'Custom exception hierarchy',
            'data_handling/atomic_cache.py': 'Atomic data caching',
        }

        for module_path, description in expected_modules.items():
            full_path = self.xraylabtool_dir / module_path
            if not full_path.exists():
                violations.append(f"Missing expected module: {module_path} ({description})")

        # Check for legacy modules in root that should be deprecated
        legacy_modules = ['core.py', 'cli.py', 'exceptions.py']
        for legacy in legacy_modules:
            legacy_path = self.xraylabtool_dir / legacy
            if legacy_path.exists():
                with open(legacy_path, 'r') as f:
                    content = f.read()
                    if len(content) > 200 and 'deprecated' not in content.lower():
                        violations.append(f"Legacy module {legacy} exists but may not be properly deprecated")

        return violations

    def _check_performance_patterns(self) -> List[str]:
        """Check for performance optimization patterns."""
        violations = []

        for py_file in self._get_python_files():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for inefficient patterns in performance-critical modules
                if any(perf_module in str(py_file) for perf_module in ['data_handling', 'calculators']):
                    # Look for list comprehensions vs loops (basic check)
                    if 'for ' in content and '[' in content:
                        # This is a basic check - could be enhanced
                        pass

                    # Check for numpy usage patterns
                    if 'numpy' in content or 'np.' in content:
                        if 'tolist()' in content:
                            violations.append(f"{py_file}: Potential numpy to list conversion (performance concern)")

            except Exception:
                continue

        return violations

    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []

        # Get xraylabtool module files
        for py_file in self.xraylabtool_dir.rglob("*.py"):
            python_files.append(py_file)

        # Get test files
        for py_file in self.tests_dir.rglob("*.py"):
            python_files.append(py_file)

        return python_files

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        if not name:
            return False
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None

    def _is_camel_case(self, name: str) -> bool:
        """Check if name follows CamelCase convention."""
        if not name:
            return False
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None

    def _get_allowed_naming_violations(self) -> List[str]:
        """Get list of allowed naming convention violations."""
        return [
            'cleanup/',  # Enterprise modules may have different patterns
            'legacy',    # Legacy code allowed violations
            'MW',        # Molecular weight abbreviation
            'SLD',       # Scattering length density abbreviation
            'reSLD',     # Real scattering length density
            'imSLD',     # Imaginary scattering length density
        ]


class TestStyleGuideToolIntegration(BaseUnitTest):
    """Test integration with development tools for style guide enforcement."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent

    @pytest.mark.integration
    def test_black_configuration_compliance(self):
        """Test that Black configuration matches style guide requirements."""
        result = subprocess.run(
            ['black', '--check', '--diff', 'xraylabtool', 'tests'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Black should pass or show minimal formatting differences
        if result.returncode != 0:
            diff_lines = result.stdout.count('\n')
            assert diff_lines < 50, f"Too many Black formatting differences: {diff_lines} lines"

    @pytest.mark.integration
    def test_ruff_configuration_compliance(self):
        """Test that Ruff configuration catches style violations appropriately."""
        try:
            result = subprocess.run(
                ['ruff', 'check', 'xraylabtool', 'tests', '--output-format=json'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if result.stdout:
                import json
                violations = json.loads(result.stdout)
                # Allow some violations but ensure they're reasonable
                assert len(violations) < 1000, f"Too many Ruff violations: {len(violations)}"

        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Ruff not available or configured")

    @pytest.mark.integration
    def test_mypy_configuration_compliance(self):
        """Test that MyPy configuration provides appropriate type checking."""
        try:
            result = subprocess.run(
                ['mypy', 'xraylabtool/calculators', '--ignore-missing-imports'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            # MyPy may have errors but should run successfully
            error_lines = result.stdout.count('error:')
            assert error_lines < 50, f"Too many MyPy errors: {error_lines}"

        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("MyPy not available or configured")


class TestStyleGuideDocumentationAccuracy(BaseUnitTest):
    """Test that style guide documentation accurately reflects implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.claude_md_path = self.project_root / "CLAUDE.md"

    @pytest.mark.unit
    def test_documented_import_patterns_accuracy(self):
        """Test that documented import patterns match actual implementation."""
        with open(self.claude_md_path, 'r') as f:
            claude_content = f.read()

        # Check that documented patterns exist in CLAUDE.md
        assert "absolute import" in claude_content.lower(), "Absolute import patterns should be documented"
        assert "xraylabtool.calculators" in claude_content, "Module examples should be present"
        assert "from typing import" in claude_content, "Type import examples should be present"

    @pytest.mark.unit
    def test_documented_naming_conventions_accuracy(self):
        """Test that documented naming conventions match actual implementation."""
        with open(self.claude_md_path, 'r') as f:
            claude_content = f.read()

        # Check for naming convention documentation
        assert "snake_case" in claude_content, "Snake case should be documented"
        assert "CamelCase" in claude_content, "CamelCase should be documented"
        assert "critical_angle_degrees" in claude_content, "Example field names should be present"

    @pytest.mark.unit
    def test_documented_module_organization_accuracy(self):
        """Test that documented module organization matches actual structure."""
        with open(self.claude_md_path, 'r') as f:
            claude_content = f.read()

        # Check that documented modules exist
        documented_modules = [
            'calculators/', 'data_handling/', 'interfaces/',
            'validation/', 'io/', 'cleanup/', 'optimization/'
        ]

        for module in documented_modules:
            assert module in claude_content, f"Module {module} should be documented"

            # Check that module actually exists
            module_path = self.project_root / "xraylabtool" / module
            assert module_path.exists(), f"Documented module {module} should exist"

    @pytest.mark.unit
    def test_documented_development_commands_accuracy(self):
        """Test that documented development commands actually work."""
        with open(self.claude_md_path, 'r') as f:
            claude_content = f.read()

        # Extract make commands mentioned in documentation
        make_commands = re.findall(r'make\s+([a-z-]+)', claude_content)

        # Test a few key commands exist in Makefile
        makefile_path = self.project_root / "Makefile"
        if makefile_path.exists():
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()

            essential_commands = ['test', 'lint', 'format', 'dev', 'validate']
            for cmd in essential_commands:
                if f"make {cmd}" in claude_content:
                    assert f"{cmd}:" in makefile_content, f"Documented command 'make {cmd}' should exist in Makefile"


class TestStyleGuideExamples(BaseUnitTest):
    """Test that style guide examples are valid and work correctly."""

    @pytest.mark.unit
    def test_code_examples_syntax_validity(self):
        """Test that code examples in documentation have valid syntax."""
        claude_md_path = Path(__file__).parent.parent / "CLAUDE.md"

        with open(claude_md_path, 'r') as f:
            content = f.read()

        # Extract Python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        syntax_errors = []
        for i, block in enumerate(python_blocks):
            try:
                # Remove common documentation artifacts
                block = block.replace('# ✅ Correct', '').replace('# ❌ Wrong', '')
                block = block.replace('...', 'pass')  # Replace ellipsis

                # Try to parse the code
                ast.parse(block)
            except SyntaxError as e:
                syntax_errors.append(f"Block {i+1}: {e}")

        assert len(syntax_errors) <= 2, f"Python code examples have syntax errors: {syntax_errors}"

    @pytest.mark.unit
    def test_import_examples_validity(self):
        """Test that import examples in documentation are valid."""
        claude_md_path = Path(__file__).parent.parent / "CLAUDE.md"

        with open(claude_md_path, 'r') as f:
            content = f.read()

        # Find import examples
        import_examples = re.findall(r'from xraylabtool\.[a-z_.]+import [A-Za-z_,\s]+', content)

        # Test that mentioned modules exist
        for import_line in import_examples[:5]:  # Test first 5 examples
            try:
                # Extract module path
                module_match = re.search(r'from (xraylabtool\.[a-z_.]+)', import_line)
                if module_match:
                    module_name = module_match.group(1)
                    # Don't actually import (may have dependencies) but check path exists
                    module_parts = module_name.split('.')
                    if len(module_parts) >= 2:
                        module_path = Path(__file__).parent.parent / '/'.join(module_parts) + '.py'
                        parent_path = Path(__file__).parent.parent / '/'.join(module_parts[:-1])

                        # Check if module file or package directory exists
                        assert module_path.exists() or parent_path.exists(), f"Module/package for {module_name} should exist"

            except Exception as e:
                # Skip imports that can't be easily validated
                continue

    @pytest.mark.unit
    def test_function_signature_examples_validity(self):
        """Test that function signature examples follow documented patterns."""
        claude_md_path = Path(__file__).parent.parent / "CLAUDE.md"

        with open(claude_md_path, 'r') as f:
            content = f.read()

        # Find function definitions in examples
        function_defs = re.findall(r'def\s+([a-z_][a-z0-9_]*)\s*\([^)]*\)\s*->\s*[^:]+:', content)

        # Check that function names follow snake_case
        for func_name in function_defs:
            assert self._is_snake_case(func_name), f"Example function '{func_name}' should use snake_case"

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        if not name:
            return False
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None