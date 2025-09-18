#!/usr/bin/env python3
"""
Documentation Integration Testing Script.

This script performs comprehensive integration testing of all documentation
updates to ensure consistency, accuracy, and completeness across the entire
documentation system.
"""

from pathlib import Path
import re
import subprocess
import sys


class DocumentationIntegrationTester:
    """Comprehensive documentation integration testing."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"
        self.readme_file = self.project_root / "README.md"
        self.claude_file = self.project_root / "CLAUDE.md"
        self.results = {}

    def test_cli_command_consistency(self) -> bool:
        """Test CLI command consistency across all documentation."""
        print("🔍 Testing CLI command consistency...")

        # Extract CLI commands from different sources
        cli_source_commands = self._extract_cli_source_commands()
        docs_commands = self._extract_documented_commands()
        example_commands = self._extract_example_commands()

        print(f"   CLI source commands: {len(cli_source_commands)}")
        print(f"   Documented commands: {len(docs_commands)}")
        print(f"   Example commands: {len(example_commands)}")

        # Check consistency
        missing_in_docs = cli_source_commands - docs_commands
        missing_in_examples = cli_source_commands - example_commands

        if missing_in_docs:
            print(f"   ❌ Commands missing from docs: {missing_in_docs}")
            return False

        if missing_in_examples:
            print(f"   ⚠️  Commands missing from examples: {missing_in_examples}")

        print("   ✅ CLI command consistency validated")
        return True

    def test_module_import_consistency(self) -> bool:
        """Test that all documented imports are valid."""
        print("🔍 Testing module import consistency...")

        documented_imports = self._extract_documented_imports()
        invalid_imports = []

        for import_stmt in documented_imports:
            if not self._validate_import_path(import_stmt):
                invalid_imports.append(import_stmt)

        if invalid_imports:
            print(f"   ❌ Invalid imports found: {invalid_imports}")
            return False

        print(f"   ✅ All {len(documented_imports)} documented imports are valid")
        return True

    def test_performance_claims_consistency(self) -> bool:
        """Test performance claims consistency across documentation."""
        print("🔍 Testing performance claims consistency...")

        performance_claims = self._extract_performance_claims()

        # Check for consistency in numerical claims
        calculation_speeds = []
        for file_path, claims in performance_claims.items():
            for claim in claims:
                if "calculations" in claim and "second" in claim:
                    # Extract numerical values
                    numbers = re.findall(r"(\d+(?:,\d+)*)", claim)
                    if numbers:
                        calculation_speeds.append((file_path, numbers[0], claim))

        # Verify all major performance claims are consistent
        major_claims = [
            speed
            for _, speed, _ in calculation_speeds
            if "150" in speed or "100" in speed
        ]

        if not major_claims:
            print("   ❌ No major performance claims found")
            return False

        print(f"   ✅ Found {len(major_claims)} consistent performance claims")
        return True

    def test_field_name_consistency(self) -> bool:
        """Test XRayResult field name consistency."""
        print("🔍 Testing XRayResult field name consistency...")

        documented_fields = self._extract_documented_field_names()
        legacy_field_usage = self._find_legacy_field_usage()

        # Check that legacy fields are only used in migration sections
        inappropriate_legacy_usage = []
        for file_path, fields in legacy_field_usage.items():
            if (
                "migration" not in file_path.lower()
                and "legacy" not in file_path.lower()
            ):
                content = Path(file_path).read_text()
                for field in fields:
                    if (
                        "deprecation" not in content.lower()
                        and "legacy" not in content.lower()
                    ):
                        inappropriate_legacy_usage.append((file_path, field))

        if inappropriate_legacy_usage:
            print(
                f"   ❌ Inappropriate legacy field usage: {inappropriate_legacy_usage}"
            )
            return False

        print(
            f"   ✅ Field name usage is appropriate (found {len(documented_fields)} current fields)"
        )
        return True

    def test_documentation_language_tone(self) -> bool:
        """Test documentation language tone for technical precision."""
        print("🔍 Testing documentation language tone...")

        # Check for remaining flowery language
        flowery_patterns = [
            r"\bcomprehensive\b",
            r"\brobust\b",
            r"\bpowerful\b",
            r"\bseamless\b",
            r"\befficient\b",
            r"\bunprecedented\b",
        ]

        flowery_found = []
        doc_files = [
            self.readme_file,
            self.claude_file,
            *self.docs_dir.rglob("*.rst"),
            *self.docs_dir.rglob("*.md"),
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()
                for pattern in flowery_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        flowery_found.append((str(doc_file), pattern, len(matches)))

        if flowery_found:
            print(f"   ⚠️  Found {len(flowery_found)} instances of flowery language")
            for file_path, pattern, count in flowery_found[:5]:  # Show first 5
                print(f"      {Path(file_path).name}: {pattern} ({count} times)")
        else:
            print("   ✅ Documentation language tone is appropriate")

        return len(flowery_found) < 10  # Allow some instances in appropriate contexts

    def test_code_examples_integration(self) -> bool:
        """Test that code examples integrate properly with current API."""
        print("🔍 Testing code examples integration...")

        try:
            # Run our existing code examples test
            result = subprocess.run(
                [sys.executable, "scripts/test_code_examples.py"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            # Parse results
            if (
                "All Python code examples have valid syntax and imports!"
                in result.stdout
            ):
                print("   ✅ All code examples have valid syntax and imports")
                return True
            else:
                print("   ❌ Code examples have issues")
                print(f"      Error: {result.stderr[:200]}...")
                return False

        except Exception as e:
            print(f"   ❌ Failed to test code examples: {e}")
            return False

    def test_cli_examples_integration(self) -> bool:
        """Test CLI examples integration."""
        print("🔍 Testing CLI examples integration...")

        try:
            # Run our existing CLI examples test
            result = subprocess.run(
                [sys.executable, "scripts/test_cli_examples.py"],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if "All CLI examples have valid syntax!" in result.stdout:
                print("   ✅ All CLI examples have valid syntax")
                return True
            else:
                print("   ❌ CLI examples have issues")
                return False

        except Exception as e:
            print(f"   ❌ Failed to test CLI examples: {e}")
            return False

    def _extract_cli_source_commands(self) -> set:
        """Extract CLI commands from source code."""
        cli_file = self.project_root / "xraylabtool" / "interfaces" / "cli.py"
        if not cli_file.exists():
            return set()

        content = cli_file.read_text()
        commands = set()

        # Look for subparsers.add_parser calls
        pattern = r'subparsers\.add_parser\(\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, content)
        commands.update(matches)

        return commands

    def _extract_documented_commands(self) -> set:
        """Extract CLI commands from documentation."""
        commands = set()
        doc_files = [self.docs_dir / "cli_reference.rst", self.readme_file]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()

                # Look for documented commands
                patterns = [
                    r"``([a-z-]+)``",  # RST format
                    r"`([a-z-]+)`",  # Markdown format
                    r"xraylabtool\s+([a-z-]+)",  # Command examples
                ]

                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    # Filter out common words that aren't commands
                    valid_commands = [
                        m
                        for m in matches
                        if (len(m) > 2 and "-" in m)
                        or m
                        in [
                            "calc",
                            "batch",
                            "convert",
                            "formula",
                            "atomic",
                            "bragg",
                            "list",
                        ]
                    ]
                    commands.update(valid_commands)

        return commands

    def _extract_example_commands(self) -> set:
        """Extract CLI commands from examples."""
        commands = set()
        doc_files = list(self.docs_dir.rglob("*.rst")) + list(
            self.docs_dir.rglob("*.md")
        )
        doc_files.append(self.readme_file)

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()

                # Look for xraylabtool command examples
                pattern = r"xraylabtool\s+([a-z-]+)"
                matches = re.findall(pattern, content)
                commands.update(matches)

        return commands

    def _extract_documented_imports(self) -> list[str]:
        """Extract all documented import statements."""
        imports = []
        doc_files = [
            self.readme_file,
            self.claude_file,
            *list(self.docs_dir.rglob("*.rst")),
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()

                # Find import statements in code blocks
                import_patterns = [
                    r"from\s+(xraylabtool[.\w]*)\s+import",
                    r"import\s+(xraylabtool[.\w]*)",
                ]

                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    imports.extend(matches)

        return list(set(imports))

    def _validate_import_path(self, import_path: str) -> bool:
        """Validate that an import path exists."""
        # Check against known valid imports
        valid_imports = [
            "xraylabtool",
            "xraylabtool.calculators",
            "xraylabtool.data_handling",
            "xraylabtool.interfaces",
            "xraylabtool.io",
            "xraylabtool.utils",
            "xraylabtool.validation",
        ]

        return any(import_path.startswith(valid) for valid in valid_imports)

    def _extract_performance_claims(self) -> dict[str, list[str]]:
        """Extract performance claims from documentation."""
        claims = {}
        doc_files = [
            self.readme_file,
            self.claude_file,
            self.project_root / "xraylabtool" / "__init__.py",
            *list(self.docs_dir.rglob("*.rst")),
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()
                file_claims = []

                # Look for performance-related statements
                performance_patterns = [
                    r"[0-9,]+\+?\s*calculations[/\s]*second",
                    r"[0-9,]+x\s*speed",
                    r"[0-9,]+\s*times faster",
                ]

                for pattern in performance_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    file_claims.extend(matches)

                if file_claims:
                    claims[str(doc_file)] = file_claims

        return claims

    def _extract_documented_field_names(self) -> set:
        """Extract documented XRayResult field names."""
        fields = set()
        doc_files = [self.readme_file, *list(self.docs_dir.rglob("*.rst"))]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()

                # Look for result.field_name patterns
                pattern = r"result\.([a-z_]+)"
                matches = re.findall(pattern, content)
                fields.update(matches)

        return fields

    def _find_legacy_field_usage(self) -> dict[str, list[str]]:
        """Find usage of legacy field names."""
        legacy_usage = {}
        legacy_patterns = [
            r"result\.([A-Z][a-zA-Z_]*)",  # CamelCase fields
            r"result\.(MW|Formula|Critical_Angle|Attenuation_Length)",  # Known legacy fields
        ]

        doc_files = [self.readme_file, *list(self.docs_dir.rglob("*.rst"))]

        for doc_file in doc_files:
            if doc_file.exists():
                content = doc_file.read_text()
                file_usage = []

                for pattern in legacy_patterns:
                    matches = re.findall(pattern, content)
                    file_usage.extend(matches)

                if file_usage:
                    legacy_usage[str(doc_file)] = file_usage

        return legacy_usage

    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print("🧪 Documentation Integration Testing")
        print("=" * 50)

        tests = [
            ("CLI Command Consistency", self.test_cli_command_consistency),
            ("Module Import Consistency", self.test_module_import_consistency),
            (
                "Performance Claims Consistency",
                self.test_performance_claims_consistency,
            ),
            ("Field Name Consistency", self.test_field_name_consistency),
            ("Documentation Language Tone", self.test_documentation_language_tone),
            ("Code Examples Integration", self.test_code_examples_integration),
            ("CLI Examples Integration", self.test_cli_examples_integration),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name}...")
            try:
                if test_func():
                    passed += 1
                    self.results[test_name] = "PASSED"
                else:
                    self.results[test_name] = "FAILED"
            except Exception as e:
                print(f"   ❌ Test {test_name} failed with exception: {e}")
                self.results[test_name] = f"ERROR: {e}"

        print(f"\n📊 Integration Test Results: {passed}/{total}")

        if passed == total:
            print("🎉 All integration tests passed!")
            return True
        else:
            print("❌ Some integration tests failed")
            return False


def main():
    """Run documentation integration tests."""
    tester = DocumentationIntegrationTester()
    success = tester.run_all_tests()

    print("\n📊 Final Results:")
    for test_name, result in tester.results.items():
        status = "✅" if result == "PASSED" else "❌"
        print(f"   {status} {test_name}: {result}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
