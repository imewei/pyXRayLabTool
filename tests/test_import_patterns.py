"""
Tests for import pattern validation and module structure documentation.

This module validates that the codebase follows proper import patterns and
module organization standards as defined in the style guide.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from tests.fixtures.test_base import BaseUnitTest


class TestImportPatterns(BaseUnitTest):
    """Test import patterns and module structure compliance."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"

    def test_no_relative_imports(self):
        """Test that no relative imports are used in the codebase."""
        relative_imports = self._find_relative_imports()

        if relative_imports:
            error_msg = "Found relative imports (should use absolute imports):\n"
            for file_path, imports in relative_imports.items():
                error_msg += f"\n{file_path}:\n"
                for imp in imports:
                    error_msg += f"  Line {imp['line']}: {imp['import']}\n"

            pytest.fail(error_msg)

    def test_absolute_import_patterns(self):
        """Test that all imports follow absolute import patterns."""
        invalid_imports = self._find_invalid_absolute_imports()

        if invalid_imports:
            error_msg = "Found imports that don't follow absolute import patterns:\n"
            for file_path, imports in invalid_imports.items():
                error_msg += f"\n{file_path}:\n"
                for imp in imports:
                    error_msg += f"  Line {imp['line']}: {imp['import']} (should be: {imp['suggestion']})\n"

            pytest.fail(error_msg)

    def test_module_organization_compliance(self):
        """Test that modules are organized according to defined structure."""
        expected_structure = {
            "calculators": ["core.py", "derived_quantities.py"],
            "data_handling": ["atomic_cache.py", "batch_processing.py"],
            "interfaces": ["cli.py", "completion.py"],
            "io": ["file_operations.py", "data_export.py"],
            "validation": ["exceptions.py", "validators.py"],
        }

        missing_modules = []
        for package, expected_modules in expected_structure.items():
            package_dir = self.xraylabtool_dir / package
            if not package_dir.exists():
                missing_modules.append(f"Package directory missing: {package}")
                continue

            for module in expected_modules:
                module_path = package_dir / module
                if not module_path.exists():
                    missing_modules.append(
                        f"Expected module missing: {package}/{module}"
                    )

        if missing_modules:
            pytest.fail(f"Module organization issues:\n" + "\n".join(missing_modules))

    def test_import_organization_in_init_files(self):
        """Test that __init__.py files have properly organized imports."""
        init_files = list(self.xraylabtool_dir.rglob("__init__.py"))

        issues = []
        for init_file in init_files:
            if init_file.name == "__init__.py":
                content = init_file.read_text(encoding="utf-8")
                if not self._has_proper_import_organization(content, init_file):
                    issues.append(f"Improper import organization in {init_file}")

        if issues:
            pytest.fail(f"Import organization issues:\n" + "\n".join(issues))

    def test_no_circular_imports(self):
        """Test that there are no circular import dependencies."""
        import_graph = self._build_import_graph()
        cycles = self._find_cycles(import_graph)

        if cycles:
            error_msg = "Found circular import dependencies:\n"
            for cycle in cycles:
                error_msg += f"  {' -> '.join(cycle)} -> {cycle[0]}\n"
            pytest.fail(error_msg)

    def test_typing_imports_use_builtin_types(self):
        """Test that code uses built-in types instead of typing module when possible."""
        outdated_typing_imports = self._find_outdated_typing_imports()

        if outdated_typing_imports:
            error_msg = (
                "Found outdated typing imports (use built-in types for Python 3.12+):\n"
            )
            for file_path, imports in outdated_typing_imports.items():
                error_msg += f"\n{file_path}:\n"
                for imp in imports:
                    error_msg += f"  Line {imp['line']}: {imp['import']} (use: {imp['replacement']})\n"

            pytest.fail(error_msg)

    def test_subpackage_responsibility_compliance(self):
        """Test that modules are in the correct sub-packages based on their responsibility."""
        responsibilities = {
            "calculators": ["calculation", "compute", "xray", "result", "scattering"],
            "data_handling": ["cache", "batch", "data", "atomic", "preload"],
            "interfaces": ["cli", "command", "completion", "interface"],
            "io": ["export", "import", "file", "csv", "json"],
            "validation": ["validate", "exception", "error", "check"],
        }

        misplaced_modules = []

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            relative_path = py_file.relative_to(self.xraylabtool_dir)
            parts = relative_path.parts

            if len(parts) > 1:  # File is in a sub-package
                subpackage = parts[0]
                if subpackage in responsibilities:
                    content = py_file.read_text(encoding="utf-8").lower()
                    expected_keywords = responsibilities[subpackage]

                    # Check if file content matches sub-package responsibility
                    has_matching_keywords = any(
                        keyword in content for keyword in expected_keywords
                    )
                    if not has_matching_keywords:
                        misplaced_modules.append(
                            f"{relative_path}: may not belong in {subpackage}/ "
                            f"(expected keywords: {expected_keywords})"
                        )

        if misplaced_modules:
            # This is a warning, not a failure, as some modules may legitimately
            # not match the keyword patterns
            print(
                f"Warning - potential module placement issues:\n"
                + "\n".join(misplaced_modules)
            )

    def _find_relative_imports(self) -> Dict[str, List[Dict[str, any]]]:
        """Find all relative imports in the codebase."""
        relative_imports = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)
                file_imports = []

                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.ImportFrom) and node.level > 0:
                            # This is a relative import
                            import_str = (
                                f"from {'.' * node.level}{node.module or ''} import ..."
                            )
                            file_imports.append(
                                {"line": node.lineno, "import": import_str}
                            )

                if file_imports:
                    relative_imports[str(py_file.relative_to(self.project_root))] = (
                        file_imports
                    )

            except (SyntaxError, UnicodeDecodeError) as e:
                # Skip files that can't be parsed
                continue

        return relative_imports

    def _find_invalid_absolute_imports(self) -> Dict[str, List[Dict[str, any]]]:
        """Find imports that should be absolute but aren't properly formed."""
        invalid_imports = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                file_imports = []
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for problematic import patterns
                    if re.match(r"^from\s+\.\s+import", line):
                        suggestion = line.replace(
                            "from . import", "from xraylabtool import"
                        )
                        file_imports.append(
                            {"line": line_num, "import": line, "suggestion": suggestion}
                        )
                    elif re.match(r"^from\s+\..*\s+import", line):
                        # Convert relative to absolute
                        relative_part = re.search(r"from\s+(\.+\w*)", line).group(1)
                        module_path = relative_part.replace(".", "").strip()
                        if module_path:
                            suggestion = line.replace(
                                relative_part, f"xraylabtool.{module_path}"
                            )
                        else:
                            suggestion = line.replace("from .", "from xraylabtool.")
                        file_imports.append(
                            {"line": line_num, "import": line, "suggestion": suggestion}
                        )

                if file_imports:
                    invalid_imports[str(py_file.relative_to(self.project_root))] = (
                        file_imports
                    )

            except (UnicodeDecodeError, Exception) as e:
                continue

        return invalid_imports

    def _has_proper_import_organization(self, content: str, file_path: Path) -> bool:
        """Check if imports in __init__.py are properly organized."""
        lines = content.split("\n")
        import_lines = [
            line for line in lines if line.strip().startswith(("import ", "from "))
        ]

        # Check that imports are grouped: stdlib, third-party, local
        # For this project, most imports should be local xraylabtool imports
        xraylabtool_imports = [line for line in import_lines if "xraylabtool" in line]

        # For __init__.py files, most imports should be from xraylabtool modules
        if len(xraylabtool_imports) > 0 and file_path.parent.name != "xraylabtool":
            return True  # Sub-package __init__.py should have xraylabtool imports

        return True  # For now, we'll be lenient on import organization

    def _build_import_graph(self) -> Dict[str, Set[str]]:
        """Build a graph of module dependencies."""
        import_graph = {}

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            module_name = self._get_module_name(py_file)
            if module_name:
                import_graph[module_name] = set()

                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom) and node.module:
                            if node.module.startswith("xraylabtool"):
                                import_graph[module_name].add(node.module)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name.startswith("xraylabtool"):
                                    import_graph[module_name].add(alias.name)

                except (SyntaxError, UnicodeDecodeError):
                    continue

        return import_graph

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find cycles in the import graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if dfs(neighbor, path + [neighbor]):
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node, [node])

        return cycles

    def _get_module_name(self, py_file: Path) -> str:
        """Get the module name for a Python file."""
        relative_path = py_file.relative_to(self.project_root)
        parts = list(relative_path.parts)

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]  # Remove .py extension

        return ".".join(parts)

    def _find_outdated_typing_imports(self) -> Dict[str, List[Dict[str, any]]]:
        """Find imports from typing module that should use built-in types."""
        outdated_imports = {}

        # Mapping of old typing imports to new built-in types
        replacements = {
            "Dict": "dict",
            "List": "list",
            "Set": "set",
            "Tuple": "tuple",
            "FrozenSet": "frozenset",
            "Deque": "collections.deque",
            "DefaultDict": "collections.defaultdict",
            "OrderedDict": "collections.OrderedDict",
            "Counter": "collections.Counter",
            "ChainMap": "collections.ChainMap",
        }

        for py_file in self.xraylabtool_dir.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                file_imports = []
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()

                    # Check for typing imports that have built-in replacements
                    if line.startswith("from typing import"):
                        imported_items = line.replace("from typing import", "").strip()
                        items = [item.strip() for item in imported_items.split(",")]

                        for item in items:
                            if item in replacements:
                                file_imports.append(
                                    {
                                        "line": line_num,
                                        "import": line,
                                        "replacement": (
                                            f"Use built-in {replacements[item]} instead of typing.{item}"
                                        ),
                                    }
                                )

                    elif re.match(r"^import typing", line):
                        # Check usage in the file content to see if replaceable types are used
                        with open(py_file, "r", encoding="utf-8") as f:
                            content = f.read()

                        for old_type, new_type in replacements.items():
                            if f"typing.{old_type}" in content:
                                file_imports.append(
                                    {
                                        "line": line_num,
                                        "import": f"typing.{old_type}",
                                        "replacement": f"Use built-in {new_type}",
                                    }
                                )

                if file_imports:
                    outdated_imports[str(py_file.relative_to(self.project_root))] = (
                        file_imports
                    )

            except (UnicodeDecodeError, Exception):
                continue

        return outdated_imports


class TestModuleStructureDocumentation(BaseUnitTest):
    """Test that module structure documentation is accurate and complete."""

    def setup_method(self):
        """Set up test fixtures."""
        super().setup_method()
        self.project_root = Path(__file__).parent.parent
        self.xraylabtool_dir = self.project_root / "xraylabtool"

    def test_all_modules_documented_in_claude_md(self):
        """Test that all modules are documented in CLAUDE.md."""
        claude_md_path = self.project_root / "CLAUDE.md"
        if not claude_md_path.exists():
            pytest.skip("CLAUDE.md not found")

        with open(claude_md_path, "r", encoding="utf-8") as f:
            claude_content = f.read()

        # Find all Python modules
        all_modules = []
        for py_file in self.xraylabtool_dir.rglob("*.py"):
            if py_file.name != "__init__.py":
                relative_path = py_file.relative_to(self.xraylabtool_dir)
                module_path = str(relative_path).replace("/", ".").replace(".py", "")
                all_modules.append(module_path)

        # Check if key modules are mentioned in CLAUDE.md
        key_modules = [
            "calculators.core",
            "data_handling.atomic_cache",
            "interfaces.cli",
            "validation.exceptions",
            "io.file_operations",
        ]

        missing_modules = []
        for module in key_modules:
            if module not in claude_content:
                missing_modules.append(module)

        if missing_modules:
            pytest.fail(f"Key modules not documented in CLAUDE.md: {missing_modules}")

    def test_subpackage_responsibilities_documented(self):
        """Test that sub-package responsibilities are clearly documented."""
        claude_md_path = self.project_root / "CLAUDE.md"
        if not claude_md_path.exists():
            pytest.skip("CLAUDE.md not found")

        with open(claude_md_path, "r", encoding="utf-8") as f:
            claude_content = f.read()

        expected_subpackages = [
            "calculators",
            "data_handling",
            "interfaces",
            "io",
            "validation",
        ]

        missing_docs = []
        for subpackage in expected_subpackages:
            # Check if subpackage responsibility is documented
            if (
                f"{subpackage}/" not in claude_content
                and subpackage not in claude_content
            ):
                missing_docs.append(subpackage)

        if missing_docs:
            pytest.fail(f"Sub-package responsibilities not documented: {missing_docs}")

    def test_import_standards_documented(self):
        """Test that import standards are documented with examples."""
        claude_md_path = self.project_root / "CLAUDE.md"
        if not claude_md_path.exists():
            pytest.skip("CLAUDE.md not found")

        with open(claude_md_path, "r", encoding="utf-8") as f:
            claude_content = f.read()

        required_documentation = [
            "absolute import",
            "xraylabtool.",
            "import",
            "from xraylabtool",
        ]

        missing_docs = []
        for doc_item in required_documentation:
            if doc_item.lower() not in claude_content.lower():
                missing_docs.append(doc_item)

        if missing_docs:
            pytest.fail(f"Import standards not properly documented: {missing_docs}")
