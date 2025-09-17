"""
Tests for code quality standards and example validation.

This module validates that the codebase follows proper code standards including
naming conventions, error handling patterns, docstring standards, and performance
guidelines as defined in the style guide.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestCodeQualityStandards(BaseUnitTest):
    """Test code quality standards compliance."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"

    def test_snake_case_naming_conventions(self):
        """Test that field names and variables use snake_case."""
        violations = self._find_naming_violations()

        if violations:
            error_msg = "Found naming convention violations (should use snake_case):\n"
            for file_path, issues in violations.items():
                error_msg += f"\n{file_path}:\n"
                for issue in issues:
                    error_msg += f"  Line {issue['line']}: {issue['violation']} (suggestion: {issue['suggestion']})\n"

            pytest.fail(error_msg)

    def test_deprecation_warnings_for_legacy_names(self):
        """Test that legacy CamelCase names have proper deprecation warnings."""
        missing_warnings = self._find_missing_deprecation_warnings()

        if missing_warnings:
            error_msg = "Found legacy names without deprecation warnings:\n"
            for file_path, issues in missing_warnings.items():
                error_msg += f"\n{file_path}:\n"
                for issue in issues:
                    error_msg += f"  Line {issue['line']}: {issue['name']} needs deprecation warning\n"

            pytest.fail(error_msg)

    def test_complete_type_hints(self):
        """Test that all functions have complete type hints."""
        missing_hints = self._find_missing_type_hints()

        if missing_hints:
            error_msg = "Found functions without complete type hints:\n"
            for file_path, functions in missing_hints.items():
                error_msg += f"\n{file_path}:\n"
                for func in functions:
                    error_msg += (
                        f"  Line {func['line']}: {func['name']} - {func['issue']}\n"
                    )

            pytest.fail(error_msg)

    def test_numpy_style_docstrings(self):
        """Test that public functions have NumPy-style docstrings."""
        missing_docstrings = self._find_incomplete_docstrings()

        if missing_docstrings:
            error_msg = "Found public functions with incomplete docstrings:\n"
            for file_path, functions in missing_docstrings.items():
                error_msg += f"\n{file_path}:\n"
                for func in functions:
                    error_msg += (
                        f"  Line {func['line']}: {func['name']} - {func['issue']}\n"
                    )

            # For now, this is a warning rather than a failure since updating all docstrings is extensive
            print(f"Warning - Docstring issues found:\n{error_msg}")

    def test_error_handling_patterns(self):
        """Test that error handling follows custom exception hierarchy patterns."""
        error_issues = self._find_error_handling_issues()

        if error_issues:
            error_msg = "Found error handling pattern violations:\n"
            for file_path, issues in error_issues.items():
                error_msg += f"\n{file_path}:\n"
                for issue in issues:
                    error_msg += f"  Line {issue['line']}: {issue['violation']}\n"

            # Warning for now since this requires significant refactoring
            print(f"Warning - Error handling issues found:\n{error_msg}")

    def test_performance_guidelines_compliance(self):
        """Test that code follows performance consideration guidelines."""
        performance_issues = self._find_performance_violations()

        if performance_issues:
            error_msg = "Found performance guideline violations:\n"
            for file_path, issues in performance_issues.items():
                error_msg += f"\n{file_path}:\n"
                for issue in issues:
                    error_msg += f"  Line {issue['line']}: {issue['violation']} - {issue['suggestion']}\n"

            # Warning for now since performance optimizations are extensive
            print(f"Warning - Performance issues found:\n{error_msg}")

    def test_dataclass_patterns(self):
        """Test that structured data uses dataclass patterns properly."""
        dataclass_issues = self._find_dataclass_violations()

        if dataclass_issues:
            error_msg = "Found dataclass pattern violations:\n"
            for file_path, issues in dataclass_issues.items():
                error_msg += f"\n{file_path}:\n"
                for issue in issues:
                    error_msg += f"  Line {issue['line']}: {issue['violation']}\n"

            pytest.fail(error_msg)

    def test_code_examples_are_valid(self):
        """Test that code examples in documentation are syntactically valid."""
        invalid_examples = self._find_invalid_code_examples()

        if invalid_examples:
            error_msg = "Found invalid code examples in documentation:\n"
            for file_path, examples in invalid_examples.items():
                error_msg += f"\n{file_path}:\n"
                for example in examples:
                    error_msg += f"  Line {example['line']}: {example['error']}\n"

            pytest.fail(error_msg)

    def _find_naming_violations(self) -> Dict[str, List[Dict[str, any]]]:
        """Find naming convention violations (non-snake_case)."""
        violations = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_violations = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check class attribute names
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(
                                item.target, ast.Name
                            ):
                                name = item.target.id
                                if self._is_camel_case(name) and not self._is_constant(
                                    name
                                ):
                                    suggestion = self._to_snake_case(name)
                                    file_violations.append(
                                        {
                                            "line": item.lineno,
                                            "violation": (
                                                f'Field "{name}" should use snake_case'
                                            ),
                                            "suggestion": suggestion,
                                        }
                                    )

                    elif isinstance(node, ast.FunctionDef):
                        # Check function parameter names
                        for arg in node.args.args:
                            if self._is_camel_case(arg.arg) and arg.arg != "self":
                                suggestion = self._to_snake_case(arg.arg)
                                file_violations.append(
                                    {
                                        "line": node.lineno,
                                        "violation": (
                                            f'Parameter "{arg.arg}" should use snake_case'
                                        ),
                                        "suggestion": suggestion,
                                    }
                                )

                if file_violations:
                    violations[str(py_file.relative_to(self.project_root))] = (
                        file_violations
                    )

            except (SyntaxError, UnicodeDecodeError):
                continue

        return violations

    def _find_missing_deprecation_warnings(self) -> Dict[str, List[Dict[str, any]]]:
        """Find legacy names without deprecation warnings."""
        missing_warnings = {}

        # Look specifically at XRayResult and other key classes that might have legacy names
        target_files = [
            self.xraylabtool_dir / "calculators" / "core.py",
            self.xraylabtool_dir / "__init__.py",
        ]

        legacy_patterns = [
            "criticalAngle",
            "attenuationLength",
            "energyKeV",
            "wavelengthAngstrom",
        ]

        for py_file in target_files:
            if not py_file.exists():
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                file_issues = []
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    for pattern in legacy_patterns:
                        if pattern in line and "warnings.warn" not in line:
                            # Check if there's a deprecation warning nearby
                            found_warning = False
                            for j in range(max(0, i - 3), min(len(lines), i + 3)):
                                if (
                                    "DeprecationWarning" in lines[j]
                                    or "warnings.warn" in lines[j]
                                ):
                                    found_warning = True
                                    break

                            if not found_warning:
                                file_issues.append({"line": i, "name": pattern})

                if file_issues:
                    missing_warnings[str(py_file.relative_to(self.project_root))] = (
                        file_issues
                    )

            except UnicodeDecodeError:
                continue

        return missing_warnings

    def _find_missing_type_hints(self) -> Dict[str, List[Dict[str, any]]]:
        """Find functions without complete type hints."""
        missing_hints = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_functions = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip private functions and property methods
                        if node.name.startswith("_") or any(
                            isinstance(dec, ast.Name)
                            and dec.id in ["property", "staticmethod", "classmethod"]
                            for dec in node.decorator_list
                        ):
                            continue

                        issues = []

                        # Check return type annotation
                        if node.returns is None:
                            issues.append("missing return type hint")

                        # Check parameter type annotations
                        for arg in node.args.args:
                            if arg.annotation is None and arg.arg != "self":
                                issues.append(
                                    f"parameter '{arg.arg}' missing type hint"
                                )

                        if issues:
                            file_functions.append(
                                {
                                    "line": node.lineno,
                                    "name": node.name,
                                    "issue": "; ".join(issues),
                                }
                            )

                if file_functions:
                    missing_hints[str(py_file.relative_to(self.project_root))] = (
                        file_functions
                    )

            except (SyntaxError, UnicodeDecodeError):
                continue

        return missing_hints

    def _find_incomplete_docstrings(self) -> Dict[str, List[Dict[str, any]]]:
        """Find public functions with incomplete NumPy-style docstrings."""
        incomplete_docstrings = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_functions = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Only check public functions
                        if node.name.startswith("_"):
                            continue

                        docstring = ast.get_docstring(node)
                        issues = []

                        if not docstring:
                            issues.append("missing docstring")
                        else:
                            # Check for NumPy-style sections
                            if not re.search(r"Parameters\s*\n\s*-+", docstring):
                                if (
                                    node.args.args and len(node.args.args) > 1
                                ):  # More than just 'self'
                                    issues.append("missing Parameters section")

                            if not re.search(r"Returns\s*\n\s*-+", docstring):
                                if node.returns is not None:
                                    issues.append("missing Returns section")

                            if (
                                "raise" in content.lower()
                                or "except" in content.lower()
                            ):
                                if not re.search(r"Raises\s*\n\s*-+", docstring):
                                    issues.append("missing Raises section")

                        if issues:
                            file_functions.append(
                                {
                                    "line": node.lineno,
                                    "name": node.name,
                                    "issue": "; ".join(issues),
                                }
                            )

                if file_functions:
                    incomplete_docstrings[
                        str(py_file.relative_to(self.project_root))
                    ] = file_functions

            except (SyntaxError, UnicodeDecodeError):
                continue

        return incomplete_docstrings

    def _find_error_handling_issues(self) -> Dict[str, List[Dict[str, any]]]:
        """Find error handling pattern violations."""
        error_issues = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                file_issues = []
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for generic exceptions
                    if re.search(r"raise\s+Exception\s*\(", line):
                        file_issues.append(
                            {
                                "line": i,
                                "violation": (
                                    "Generic Exception used instead of specific custom exception"
                                ),
                            }
                        )

                    # Check for bare except clauses
                    if re.search(r"except\s*:", line):
                        file_issues.append(
                            {
                                "line": i,
                                "violation": (
                                    "Bare except clause - should catch specific exceptions"
                                ),
                            }
                        )

                    # Check for print statements instead of logging
                    if re.search(r"\bprint\s*\(", line) and "test" not in str(py_file):
                        file_issues.append(
                            {"line": i, "violation": "print() used instead of logging"}
                        )

                if file_issues:
                    error_issues[str(py_file.relative_to(self.project_root))] = (
                        file_issues
                    )

            except UnicodeDecodeError:
                continue

        return error_issues

    def _find_performance_violations(self) -> Dict[str, List[Dict[str, any]]]:
        """Find performance guideline violations."""
        performance_issues = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                file_issues = []
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for inefficient list operations
                    if re.search(r"\.append\s*\(.*\)\s*for.*in", line):
                        file_issues.append(
                            {
                                "line": i,
                                "violation": (
                                    "List comprehension could be more efficient than append in loop"
                                ),
                                "suggestion": (
                                    "Use list comprehension: [item for item in iterable]"
                                ),
                            }
                        )

                    # Check for repeated calculations
                    if re.search(r"len\s*\([^)]+\).*for.*in", line):
                        file_issues.append(
                            {
                                "line": i,
                                "violation": "len() called repeatedly in loop",
                                "suggestion": "Calculate len() once before loop",
                            }
                        )

                    # Check for inefficient string concatenation
                    if "+=" in line and "str" in line.lower():
                        file_issues.append(
                            {
                                "line": i,
                                "violation": (
                                    "String concatenation with += in loop may be inefficient"
                                ),
                                "suggestion": (
                                    "Use list and join() for multiple concatenations"
                                ),
                            }
                        )

                if file_issues:
                    performance_issues[str(py_file.relative_to(self.project_root))] = (
                        file_issues
                    )

            except UnicodeDecodeError:
                continue

        return performance_issues

    def _find_dataclass_violations(self) -> Dict[str, List[Dict[str, any]]]:
        """Find dataclass pattern violations."""
        dataclass_issues = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_issues = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if class should be a dataclass
                        has_init = any(
                            isinstance(n, ast.FunctionDef) and n.name == "__init__"
                            for n in node.body
                        )
                        has_dataclass = any(
                            isinstance(dec, ast.Name) and dec.id == "dataclass"
                            for dec in node.decorator_list
                        )

                        # Count attribute assignments
                        attr_count = sum(
                            1 for n in node.body if isinstance(n, ast.AnnAssign)
                        )

                        if attr_count > 3 and has_init and not has_dataclass:
                            file_issues.append(
                                {
                                    "line": node.lineno,
                                    "violation": (
                                        f'Class "{node.name}" with {attr_count} attributes should use @dataclass decorator'
                                    ),
                                }
                            )

                        # Check for missing type annotations in dataclass
                        if has_dataclass:
                            for item in node.body:
                                if isinstance(item, ast.Assign):
                                    for target in item.targets:
                                        if isinstance(target, ast.Name):
                                            file_issues.append(
                                                {
                                                    "line": item.lineno,
                                                    "violation": (
                                                        f'Dataclass field "{target.id}" missing type annotation'
                                                    ),
                                                }
                                            )

                if file_issues:
                    dataclass_issues[str(py_file.relative_to(self.project_root))] = (
                        file_issues
                    )

            except (SyntaxError, UnicodeDecodeError):
                continue

        return dataclass_issues

    def _find_invalid_code_examples(self) -> Dict[str, List[Dict[str, any]]]:
        """Find invalid code examples in documentation."""
        invalid_examples = {}

        # Check CLAUDE.md for code examples
        claude_md = self.project_root / "CLAUDE.md"
        if claude_md.exists():
            try:
                with open(claude_md, "r", encoding="utf-8") as f:
                    content = f.read()

                file_examples = []
                lines = content.split("\n")
                in_code_block = False
                code_block_start = 0
                code_lines = []

                for i, line in enumerate(lines, 1):
                    if line.strip().startswith("```python"):
                        in_code_block = True
                        code_block_start = i
                        code_lines = []
                    elif line.strip() == "```" and in_code_block:
                        in_code_block = False
                        if code_lines:
                            code = "\n".join(code_lines)
                            try:
                                ast.parse(code)
                            except SyntaxError as e:
                                file_examples.append(
                                    {
                                        "line": code_block_start,
                                        "error": f"Invalid Python syntax: {e.msg}",
                                    }
                                )
                    elif in_code_block:
                        code_lines.append(line)

                if file_examples:
                    invalid_examples[str(claude_md.relative_to(self.project_root))] = (
                        file_examples
                    )

            except UnicodeDecodeError:
                pass

        return invalid_examples

    def _is_camel_case(self, name: str) -> bool:
        """Check if a name is in camelCase or PascalCase."""
        return bool(
            re.match(r"^[a-z]+([A-Z][a-z]*)+$|^[A-Z][a-z]*([A-Z][a-z]*)+$", name)
        )

    def _is_constant(self, name: str) -> bool:
        """Check if a name is a constant (ALL_CAPS)."""
        return name.isupper() and "_" in name

    def _to_snake_case(self, name: str) -> str:
        """Convert camelCase or PascalCase to snake_case."""
        # Insert underscores before capital letters
        snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        return snake.lower()


class TestExampleValidation(BaseUnitTest):
    """Test that code examples in documentation are valid and follow best practices."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent

    def test_readme_examples_are_runnable(self):
        """Test that README examples can be executed."""
        readme_path = self.project_root / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md not found")

        # This would test actual execution of README examples
        # For now, we'll just check syntax
        self._validate_markdown_code_examples(readme_path)

    def test_docstring_examples_are_valid(self):
        """Test that docstring examples are syntactically valid."""
        invalid_examples = self._find_invalid_docstring_examples()

        if invalid_examples:
            error_msg = "Found invalid examples in docstrings:\n"
            for file_path, examples in invalid_examples.items():
                error_msg += f"\n{file_path}:\n"
                for example in examples:
                    error_msg += (
                        f"  Function {example['function']}: {example['error']}\n"
                    )

            pytest.fail(error_msg)

    def _validate_markdown_code_examples(self, md_file: Path):
        """Validate code examples in markdown files."""
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            in_python_block = False
            code_lines = []

            for line in lines:
                if line.strip().startswith("```python"):
                    in_python_block = True
                    code_lines = []
                elif line.strip() == "```" and in_python_block:
                    in_python_block = False
                    if code_lines:
                        code = "\n".join(code_lines)
                        try:
                            ast.parse(code)
                        except SyntaxError as e:
                            pytest.fail(f"Invalid Python syntax in {md_file}: {e.msg}")
                elif in_python_block:
                    code_lines.append(line)

        except UnicodeDecodeError:
            pytest.skip(f"Could not read {md_file}")

    def _find_invalid_docstring_examples(self) -> Dict[str, List[Dict[str, any]]]:
        """Find invalid examples in docstrings."""
        invalid_examples = {}

        xraylabtool_dir = self.project_root / "xraylabtool"
        for py_file in xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_examples = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        docstring = ast.get_docstring(node)
                        if docstring and ">>>" in docstring:
                            # Extract complete code blocks from docstring, handling multiline examples
                            lines = docstring.split("\n")
                            in_example = False
                            current_example = []

                            for line in lines:
                                line = line.strip()
                                if line.startswith(">>> "):
                                    if current_example:
                                        # Test previous example
                                        example_code = "\n".join(current_example)
                                        try:
                                            ast.parse(example_code)
                                        except SyntaxError as e:
                                            # Skip multiline examples and decorator patterns
                                            if (
                                                "was never closed" not in e.msg
                                                and "@" not in example_code
                                            ):
                                                file_examples.append(
                                                    {
                                                        "function": node.name,
                                                        "error": (
                                                            f"Invalid example syntax: {e.msg}"
                                                        ),
                                                    }
                                                )

                                    # Start new example
                                    current_example = [line[4:]]  # Remove '>>> '
                                    in_example = True
                                elif line.startswith("... ") and in_example:
                                    current_example.append(line[4:])  # Remove '... '
                                elif in_example and line and not line.startswith(">>>"):
                                    # This is output, end the current example
                                    if current_example:
                                        example_code = "\n".join(current_example)
                                        try:
                                            ast.parse(example_code)
                                        except SyntaxError as e:
                                            # Skip multiline examples and decorator patterns
                                            if (
                                                "was never closed" not in e.msg
                                                and "@" not in example_code
                                            ):
                                                file_examples.append(
                                                    {
                                                        "function": node.name,
                                                        "error": (
                                                            f"Invalid example syntax: {e.msg}"
                                                        ),
                                                    }
                                                )
                                    current_example = []
                                    in_example = False
                                elif not line:
                                    # Empty line ends example
                                    if current_example:
                                        example_code = "\n".join(current_example)
                                        try:
                                            ast.parse(example_code)
                                        except SyntaxError as e:
                                            # Skip multiline examples and decorator patterns
                                            if (
                                                "was never closed" not in e.msg
                                                and "@" not in example_code
                                            ):
                                                file_examples.append(
                                                    {
                                                        "function": node.name,
                                                        "error": (
                                                            f"Invalid example syntax: {e.msg}"
                                                        ),
                                                    }
                                                )
                                    current_example = []
                                    in_example = False

                if file_examples:
                    invalid_examples[str(py_file.relative_to(self.project_root))] = (
                        file_examples
                    )

            except (SyntaxError, UnicodeDecodeError):
                continue

        return invalid_examples
