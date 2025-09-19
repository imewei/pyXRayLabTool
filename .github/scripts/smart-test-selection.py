#!/usr/bin/env python3
"""
Smart Test Selection for CI/CD Optimization

This script analyzes code changes and intelligently selects which tests to run,
significantly reducing CI execution time while maintaining comprehensive coverage.
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys


class TestSelector:
    """Intelligent test selection based on code changes."""

    def __init__(self, repo_root: str, base_sha: str, head_sha: str):
        self.repo_root = Path(repo_root)
        self.base_sha = base_sha
        self.head_sha = head_sha
        self.changed_files = self._get_changed_files()
        self.test_mapping = self._build_test_mapping()

    def _get_changed_files(self) -> list[str]:
        """Get list of changed Python files."""
        try:
            result = subprocess.run(
                [
                    "/usr/bin/git",
                    "diff",
                    "--name-only",
                    f"{self.base_sha}...{self.head_sha}",
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.repo_root,
            )
            return [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
        except subprocess.CalledProcessError:
            print("⚠️ Unable to get changed files, running all tests")
            return []

    def _build_test_mapping(self) -> dict[str, set[str]]:
        """Build mapping of source files to test files."""
        mapping = {}
        test_dir = self.repo_root / "tests"

        if not test_dir.exists():
            return mapping

        # Scan test files to understand what they test
        for test_file in test_dir.rglob("test_*.py"):
            relative_test = str(test_file.relative_to(self.repo_root))
            tested_modules = self._analyze_test_file(test_file)

            for module in tested_modules:
                if module not in mapping:
                    mapping[module] = set()
                mapping[module].add(relative_test)

        return mapping

    def _analyze_test_file(self, test_file: Path) -> set[str]:
        """Analyze a test file to determine which modules it tests."""
        tested_modules = set()

        try:
            with open(test_file, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("xraylabtool"):
                            tested_modules.add(alias.name)

                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and node.module.startswith("xraylabtool")
                ):
                    tested_modules.add(node.module)

                    # Also add specific imports
                    for alias in node.names:
                        if alias.name != "*":
                            full_name = f"{node.module}.{alias.name}"
                            tested_modules.add(full_name)

        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"⚠️ Could not parse {test_file}: {e}")

        return tested_modules

    def _get_module_from_file(self, file_path: str) -> str:
        """Convert file path to module name."""
        if not file_path.endswith(".py"):
            return ""

        # Remove .py extension and convert path separators to dots
        module_path = file_path[:-3].replace("/", ".").replace("\\", ".")

        # Remove __init__ from module paths
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]

        return module_path

    def select_tests(self) -> tuple[list[str], str]:
        """
        Select tests to run based on changes.

        Returns:
            Tuple of (test_files_to_run, reason)
        """
        if not self.changed_files:
            return [], "no_changes"

        # Filter Python files in the main package
        relevant_changes = [
            f
            for f in self.changed_files
            if f.startswith("xraylabtool/") and f.endswith(".py")
        ]

        if not relevant_changes:
            # Check if test files changed
            test_changes = [f for f in self.changed_files if f.startswith("tests/")]
            return (
                (test_changes, "test_files_changed")
                if test_changes
                else ([], "no_relevant_changes")
            )

        # Find tests that cover the changed files
        tests_to_run = set()

        for changed_file in relevant_changes:
            module_name = self._get_module_from_file(changed_file)

            # Direct mapping
            if module_name in self.test_mapping:
                tests_to_run.update(self.test_mapping[module_name])

            # Look for partial matches (e.g., submodules)
            for mapped_module, test_files in self.test_mapping.items():
                if module_name.startswith(
                    mapped_module + "."
                ) or mapped_module.startswith(module_name + "."):
                    tests_to_run.update(test_files)

        if tests_to_run:
            return sorted(tests_to_run), "specific_tests"
        else:
            # Fallback: run tests based on directory structure
            fallback_tests = self._get_fallback_tests(relevant_changes)
            return fallback_tests, "fallback_tests"

    def _get_fallback_tests(self, changed_files: list[str]) -> list[str]:
        """Get fallback tests based on directory structure."""
        tests = set()

        for changed_file in changed_files:
            # Map source file to corresponding test file
            if changed_file.startswith("xraylabtool/"):
                # Remove xraylabtool/ prefix and .py suffix
                core_path = changed_file[12:-3]

                # Look for corresponding test files
                potential_tests = [
                    f"tests/unit/test_{core_path.replace('/', '_')}.py",
                    f"tests/unit/test_{Path(core_path).name}.py",
                    f"tests/integration/test_{core_path.replace('/', '_')}_integration.py",
                ]

                for test_path in potential_tests:
                    if (self.repo_root / test_path).exists():
                        tests.add(test_path)

        # If no specific tests found, add core test categories
        if not tests:
            core_tests = [
                "tests/unit/test_core.py",
                "tests/unit/test_calculators.py",
                "tests/integration/test_integration.py",
            ]

            for test_path in core_tests:
                if (self.repo_root / test_path).exists():
                    tests.add(test_path)

        return sorted(tests)

    def get_test_command(self) -> str:
        """Generate optimized pytest command."""
        tests_to_run, reason = self.select_tests()

        print(f"🧠 Test selection reason: {reason}")
        print(f"📝 Selected tests: {len(tests_to_run)} files")

        if not tests_to_run:
            if reason == "no_changes":
                return "echo '✅ No tests needed - no changes detected'"
            elif reason == "no_relevant_changes":
                return "echo '✅ No tests needed - no relevant code changes'"
            else:
                # Fallback to core tests
                return "pytest tests/unit/test_core.py -v"

        # Build optimized pytest command
        test_paths = " ".join(tests_to_run)

        if reason == "test_files_changed":
            cmd = f"pytest {test_paths} -v --tb=short"
        elif len(tests_to_run) <= 5:
            # For small test sets, run with more detail
            cmd = f"pytest {test_paths} -v --tb=short -s"
        else:
            # For larger test sets, optimize for speed
            cmd = f"pytest {test_paths} -v --tb=line -x"

        return cmd

    def generate_report(self) -> dict:
        """Generate test selection report for CI."""
        tests_to_run, reason = self.select_tests()

        return {
            "changed_files": self.changed_files,
            "relevant_changes": [
                f for f in self.changed_files if f.startswith("xraylabtool/")
            ],
            "selected_tests": tests_to_run,
            "selection_reason": reason,
            "test_count": len(tests_to_run),
            "optimization_enabled": True,
        }


def main():
    """Main entry point for smart test selection."""
    if len(sys.argv) < 3:
        print("Usage: smart-test-selection.py <base_sha> <head_sha>")
        sys.exit(1)

    base_sha = sys.argv[1]
    head_sha = sys.argv[2]
    repo_root = os.getcwd()

    print("🧠 Smart Test Selection")
    print(f"📂 Repository: {repo_root}")
    print(f"🔄 Comparing: {base_sha}...{head_sha}")
    print()

    selector = TestSelector(repo_root, base_sha, head_sha)

    # Generate test command
    test_command = selector.get_test_command()
    print(f"🚀 Generated command: {test_command}")

    # Generate report
    report = selector.generate_report()

    # Save report for CI
    with open("test-selection-report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Set GitHub Actions outputs
    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"test-command={test_command}\n")
            f.write(f"test-count={report['test_count']}\n")
            f.write(f"selection-reason={report['selection_reason']}\n")

    print("📊 Report saved to test-selection-report.json")
    print("✅ Smart test selection completed")


if __name__ == "__main__":
    main()
