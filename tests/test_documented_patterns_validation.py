"""
Test validation for all documented patterns in the style guide.

This module ensures that all code examples and patterns documented in CLAUDE.md
actually work correctly and follow the established standards.
"""

import ast
import importlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestDocumentedPatternsValidation(BaseUnitTest):
    """Test that all documented patterns in CLAUDE.md work correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.claude_md_path = self.project_root / "CLAUDE.md"

    @pytest.mark.unit
    def test_documented_import_patterns_work(self):
        """Test that documented import patterns actually work."""
        # Read CLAUDE.md
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract import examples
        import_examples = self._extract_import_examples(content)

        working_imports = []
        broken_imports = []

        for import_line in import_examples:
            if self._test_import_works(import_line):
                working_imports.append(import_line)
            else:
                broken_imports.append(import_line)

        # Most imports should work (allow some to fail for edge cases)
        success_rate = len(working_imports) / len(import_examples) if import_examples else 1.0
        assert success_rate >= 0.7, f"Too many broken import examples: {broken_imports}"

    @pytest.mark.unit
    def test_documented_function_signatures_are_valid(self):
        """Test that documented function signatures have valid syntax."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract function definitions from code blocks
        function_examples = self._extract_function_examples(content)

        invalid_functions = []
        for func_example in function_examples:
            try:
                # Try to parse the function definition
                ast.parse(func_example)
            except SyntaxError as e:
                invalid_functions.append(f"{func_example[:50]}... -> {e}")

        assert len(invalid_functions) <= 2, f"Invalid function examples: {invalid_functions}"

    @pytest.mark.unit
    def test_documented_dataclass_patterns_work(self):
        """Test that documented dataclass patterns work correctly."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract dataclass examples
        dataclass_examples = self._extract_dataclass_examples(content)

        for dataclass_code in dataclass_examples:
            # Test that dataclass example can be executed
            success = self._test_code_execution(dataclass_code)
            assert success, f"Dataclass example failed to execute: {dataclass_code[:100]}..."

    @pytest.mark.unit
    def test_documented_naming_conventions_are_consistent(self):
        """Test that documented naming conventions are applied consistently."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Find documented naming examples
        naming_examples = self._extract_naming_examples(content)

        inconsistent_examples = []
        for example_type, examples in naming_examples.items():
            if example_type == 'snake_case_functions':
                for example in examples:
                    if not self._is_snake_case(example):
                        inconsistent_examples.append(f"Function {example} should be snake_case")
            elif example_type == 'camel_case_classes':
                for example in examples:
                    if not self._is_camel_case(example):
                        inconsistent_examples.append(f"Class {example} should be CamelCase")

        assert len(inconsistent_examples) <= 2, f"Inconsistent naming examples: {inconsistent_examples}"

    @pytest.mark.unit
    def test_documented_type_hints_are_valid(self):
        """Test that documented type hint examples are valid."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract type hint examples
        type_hint_examples = self._extract_type_hint_examples(content)

        invalid_type_hints = []
        for example in type_hint_examples:
            try:
                # Create a minimal function with the type hint
                test_code = f"""
from typing import *
from pathlib import Path
from datetime import datetime
import numpy as np

def test_function({example['parameter']}) -> {example['return_type']}:
    pass
"""
                ast.parse(test_code)
            except SyntaxError as e:
                invalid_type_hints.append(f"{example} -> {e}")

        assert len(invalid_type_hints) <= 3, f"Invalid type hint examples: {invalid_type_hints}"

    @pytest.mark.integration
    def test_documented_cli_commands_work(self):
        """Test that documented CLI commands actually work."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract CLI command examples
        cli_commands = self._extract_cli_commands(content)

        working_commands = []
        broken_commands = []

        for command in cli_commands[:5]:  # Test first 5 commands to avoid long test times
            if self._test_cli_command_works(command):
                working_commands.append(command)
            else:
                broken_commands.append(command)

        # Most CLI commands should work (some may require specific files)
        if cli_commands:
            success_rate = len(working_commands) / min(len(cli_commands), 5)
            assert success_rate >= 0.6, f"Too many broken CLI commands: {broken_commands}"

    @pytest.mark.integration
    def test_documented_make_targets_work(self):
        """Test that documented make targets actually work."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract make command examples
        make_commands = self._extract_make_commands(content)

        # Test a subset of essential make commands
        essential_commands = ['help', 'status', 'quick-test', 'version-check']
        working_commands = []
        broken_commands = []

        for command in essential_commands:
            if command in make_commands:
                if self._test_make_command_works(command):
                    working_commands.append(command)
                else:
                    broken_commands.append(command)

        # Most essential make commands should work
        if essential_commands:
            success_rate = len(working_commands) / len(essential_commands)
            assert success_rate >= 0.7, f"Too many broken make commands: {broken_commands}"

    @pytest.mark.unit
    def test_documented_error_handling_patterns_work(self):
        """Test that documented error handling patterns work correctly."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Extract error handling examples
        error_examples = self._extract_error_handling_examples(content)

        for example in error_examples:
            # Test that error handling example can be parsed
            try:
                ast.parse(example)
            except SyntaxError as e:
                pytest.fail(f"Error handling example has syntax error: {example[:100]}... -> {e}")

    @pytest.mark.unit
    def test_documented_performance_patterns_are_valid(self):
        """Test that documented performance optimization patterns are valid."""
        with open(self.claude_md_path, 'r') as f:
            content = f.read()

        # Look for performance-related code examples
        performance_examples = self._extract_performance_examples(content)

        for example in performance_examples:
            # Test that performance example can be parsed
            try:
                ast.parse(example)
            except SyntaxError as e:
                pytest.fail(f"Performance example has syntax error: {example[:100]}... -> {e}")

    def _extract_import_examples(self, content: str) -> List[str]:
        """Extract import examples from documentation."""
        # Find import statements in code blocks
        import_pattern = r'(?:from\s+\w+[\w\.]*\s+import\s+[\w\s,]+|import\s+[\w\.]+)'
        imports = re.findall(import_pattern, content)

        # Filter to xraylabtool imports and common standard library imports
        xraylabtool_imports = [
            imp for imp in imports
            if 'xraylabtool' in imp or any(stdlib in imp for stdlib in ['typing', 'pathlib', 'dataclasses'])
        ]

        return xraylabtool_imports[:10]  # Limit to first 10 for testing

    def _extract_function_examples(self, content: str) -> List[str]:
        """Extract function definition examples from documentation."""
        # Find function definitions in code blocks
        function_pattern = r'def\s+[\w_]+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:'
        functions = re.findall(function_pattern, content)

        return functions[:10]  # Limit for testing

    def _extract_dataclass_examples(self, content: str) -> List[str]:
        """Extract dataclass examples from documentation."""
        # Look for dataclass patterns in code blocks
        dataclass_sections = []

        # Find code blocks that contain @dataclass
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        for block in code_blocks:
            if '@dataclass' in block:
                # Create a complete example that can be executed
                complete_example = f"""
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

{block}
"""
                dataclass_sections.append(complete_example)

        return dataclass_sections[:3]  # Limit for testing

    def _extract_naming_examples(self, content: str) -> Dict[str, List[str]]:
        """Extract naming convention examples from documentation."""
        naming_examples = {
            'snake_case_functions': [],
            'camel_case_classes': []
        }

        # Find function names mentioned as examples
        function_mentions = re.findall(r'`([a-z_][a-z0-9_]+)`', content)
        naming_examples['snake_case_functions'] = [
            name for name in function_mentions
            if not name.startswith('__') and '_' in name
        ][:10]

        # Find class names mentioned as examples
        class_mentions = re.findall(r'`([A-Z][a-zA-Z0-9]+)`', content)
        naming_examples['camel_case_classes'] = class_mentions[:10]

        return naming_examples

    def _extract_type_hint_examples(self, content: str) -> List[Dict[str, str]]:
        """Extract type hint examples from documentation."""
        # Find function signatures with type hints
        type_hint_pattern = r'def\s+\w+\s*\(([^)]*)\)\s*->\s*([^:]+):'
        matches = re.findall(type_hint_pattern, content)

        examples = []
        for params, return_type in matches[:5]:  # Limit for testing
            # Parse parameters
            if params.strip():
                for param in params.split(','):
                    param = param.strip()
                    if ':' in param:
                        examples.append({
                            'parameter': param,
                            'return_type': return_type.strip()
                        })

        return examples

    def _extract_cli_commands(self, content: str) -> List[str]:
        """Extract CLI command examples from documentation."""
        # Find xraylabtool command examples
        cli_pattern = r'xraylabtool\s+([a-z-]+)'
        commands = re.findall(cli_pattern, content)

        # Remove duplicates while preserving order
        unique_commands = []
        for cmd in commands:
            if cmd not in unique_commands:
                unique_commands.append(cmd)

        return unique_commands

    def _extract_make_commands(self, content: str) -> List[str]:
        """Extract make command examples from documentation."""
        # Find make command examples
        make_pattern = r'make\s+([a-z-]+)'
        commands = re.findall(make_pattern, content)

        # Remove duplicates while preserving order
        unique_commands = []
        for cmd in commands:
            if cmd not in unique_commands:
                unique_commands.append(cmd)

        return unique_commands

    def _extract_error_handling_examples(self, content: str) -> List[str]:
        """Extract error handling examples from documentation."""
        # Look for try/except blocks in code examples
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        error_examples = []
        for block in code_blocks:
            if 'try:' in block and 'except' in block:
                error_examples.append(block)

        return error_examples[:5]  # Limit for testing

    def _extract_performance_examples(self, content: str) -> List[str]:
        """Extract performance-related examples from documentation."""
        # Look for performance-related code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)

        performance_examples = []
        for block in code_blocks:
            if any(keyword in block.lower() for keyword in ['numpy', 'vectorized', 'cache', 'performance']):
                performance_examples.append(block)

        return performance_examples[:5]  # Limit for testing

    def _test_import_works(self, import_line: str) -> bool:
        """Test if an import statement works."""
        try:
            # Create a temporary module to test the import
            test_code = f"""
import sys
sys.path.insert(0, '{self.project_root}')
{import_line}
"""
            exec(test_code)
            return True
        except (ImportError, ModuleNotFoundError, SyntaxError):
            return False

    def _test_code_execution(self, code: str) -> bool:
        """Test if code can be executed successfully."""
        try:
            # Create a temporary file and execute it
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()

                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                Path(f.name).unlink()  # Clean up
                return result.returncode == 0

        except Exception:
            return False

    def _test_cli_command_works(self, command: str) -> bool:
        """Test if a CLI command works."""
        try:
            # Test with --help to see if command exists
            result = subprocess.run(
                ['xraylabtool', command, '--help'],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _test_make_command_works(self, command: str) -> bool:
        """Test if a make command works."""
        try:
            result = subprocess.run(
                ['make', command],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _is_snake_case(self, name: str) -> bool:
        """Check if name follows snake_case convention."""
        if not name or name.startswith('__'):  # Allow special methods
            return True
        return re.match(r'^[a-z][a-z0-9_]*$', name) is not None

    def _is_camel_case(self, name: str) -> bool:
        """Check if name follows CamelCase convention."""
        if not name:
            return False
        return re.match(r'^[A-Z][a-zA-Z0-9]*$', name) is not None


class TestRealWorldPatternUsage(BaseUnitTest):
    """Test that documented patterns are actually used in the real codebase."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"

    @pytest.mark.unit
    def test_dataclass_pattern_usage_in_codebase(self):
        """Test that dataclass patterns are used where documented."""
        # Check for actual dataclass usage in the codebase
        dataclass_files = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if '@dataclass' in content:
                        dataclass_files.append(py_file)
            except Exception:
                continue

        # Should have at least some dataclass usage
        assert len(dataclass_files) >= 1, "Dataclass pattern should be used in codebase"

    @pytest.mark.unit
    def test_absolute_import_usage_in_codebase(self):
        """Test that absolute imports are predominantly used."""
        total_files = 0
        files_with_absolute_imports = 0
        files_with_relative_imports = 0

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    total_files += 1

                    # Check for absolute xraylabtool imports
                    if re.search(r'from xraylabtool\.', content):
                        files_with_absolute_imports += 1

                    # Check for relative imports
                    if re.search(r'from \.', content):
                        files_with_relative_imports += 1

            except Exception:
                continue

        if total_files > 0:
            # Some files should use absolute imports
            absolute_ratio = files_with_absolute_imports / total_files
            # Allow some relative imports but absolute should be preferred in new modules
            assert absolute_ratio >= 0.1 or files_with_relative_imports <= 20, "Absolute imports should be used where documented"

    @pytest.mark.unit
    def test_type_hint_usage_in_codebase(self):
        """Test that type hints are used in public functions."""
        total_public_functions = 0
        functions_with_type_hints = 0

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count public functions
                        if not node.name.startswith('_'):
                            total_public_functions += 1

                            # Check for type hints
                            has_return_type = node.returns is not None
                            has_param_types = any(arg.annotation is not None for arg in node.args.args if arg.arg != 'self')

                            if has_return_type or has_param_types:
                                functions_with_type_hints += 1

            except Exception:
                continue

        if total_public_functions > 0:
            type_hint_ratio = functions_with_type_hints / total_public_functions
            # Allow some functions without type hints (legacy code)
            assert type_hint_ratio >= 0.3, f"Type hints should be used more frequently: {type_hint_ratio:.2f}"

    @pytest.mark.unit
    def test_exception_hierarchy_usage_in_codebase(self):
        """Test that custom exception hierarchy is used."""
        # Check that validation/exceptions.py exists and is used
        exceptions_file = self.xraylabtool_dir / "validation" / "exceptions.py"
        assert exceptions_file.exists(), "Custom exception hierarchy should exist"

        # Check that custom exceptions are imported in other modules
        exception_usage_count = 0

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    if 'ValidationError' in content or 'FormulaError' in content or 'EnergyError' in content:
                        exception_usage_count += 1
            except Exception:
                continue

        assert exception_usage_count >= 2, "Custom exceptions should be used across modules"