#!/usr/bin/env python3
"""
CLI documentation examples testing script.

This script tests that CLI examples in documentation have correct syntax
and structure without executing them.
"""

from pathlib import Path
import re
import sys


def extract_cli_examples_from_docs(docs_dir: Path) -> list[str]:
    """Extract CLI command examples from documentation files."""
    examples = []

    # Files that contain CLI examples
    doc_files = [
        docs_dir / "index.rst",
        docs_dir / "getting_started.rst",
        docs_dir / "cli_reference.rst",
        docs_dir / "examples" / "index.rst",
    ]

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        try:
            content = doc_file.read_text(encoding="utf-8")

            # Find CLI command examples
            # Pattern 1: xraylabtool commands in code blocks
            bash_blocks = re.findall(
                r".. code-block:: bash\s*\n\n(.*?)\n\n", content, re.DOTALL
            )
            for block in bash_blocks:
                lines = block.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("xraylabtool"):
                        examples.append(line)

            # Pattern 2: Direct xraylabtool commands
            direct_commands = re.findall(
                r"^\s*xraylabtool\s+[^\n]+", content, re.MULTILINE
            )
            examples.extend([cmd.strip() for cmd in direct_commands])

        except Exception as e:
            print(f"Warning: Could not read {doc_file}: {e}")

    return list(set(examples))  # Remove duplicates


def validate_cli_example_syntax(example: str) -> bool:
    """Validate that a CLI example has correct syntax."""
    # Basic validation of CLI command structure
    if not example.startswith("xraylabtool"):
        return False

    # Split into parts
    parts = example.split()
    if len(parts) < 2:
        return False

    command = parts[1]

    # Special handling for placeholder syntax (documentation examples)
    if command.startswith("[") and command.endswith("]"):
        return True

    # Special handling for uppercase placeholders like COMMAND, OPTIONS
    if command.isupper() or command == "COMMAND":
        return True

    # Valid commands and global options
    valid_commands = [
        "calc",
        "batch",
        "convert",
        "formula",
        "atomic",
        "bragg",
        "list",
        "install-completion",
        "uninstall-completion",
    ]

    valid_global_options = ["--help", "--version"]

    # Special handling for global options
    if command in valid_global_options:
        return True

    if command not in valid_commands:
        return False

    # Check for obvious syntax errors
    return not (
        "--" in example and not any(part.startswith("--") for part in parts[2:])
    )


def test_cli_examples_in_documentation():
    """Test CLI examples found in documentation."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    print("🔍 Extracting CLI examples from documentation...")
    examples = extract_cli_examples_from_docs(docs_dir)

    print(f"📊 Found {len(examples)} CLI examples in documentation")

    if not examples:
        print("⚠️  No CLI examples found in documentation")
        return True

    passed = 0
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Testing: {example}")

        if validate_cli_example_syntax(example):
            print("   ✅ Valid syntax")
            passed += 1
        else:
            print("   ❌ Invalid syntax")
            # Debug info for failed cases
            parts = example.split()
            if len(parts) >= 2:
                print(f"      Command: '{parts[1]}'")
            else:
                print(f"      Not enough parts: {parts}")

    print(f"\n📊 CLI Examples Test Results: {passed}/{len(examples)}")

    if passed == len(examples):
        print("🎉 All CLI examples have valid syntax!")
        return True
    else:
        print("❌ Some CLI examples have invalid syntax")
        return False


def test_cli_command_coverage_in_docs():
    """Test that all CLI commands are covered in documentation."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    expected_commands = [
        "calc",
        "batch",
        "convert",
        "formula",
        "atomic",
        "bragg",
        "list",
        "install-completion",
        "uninstall-completion",
    ]

    # Check CLI reference documentation
    cli_ref_file = docs_dir / "cli_reference.rst"

    if not cli_ref_file.exists():
        print("❌ CLI reference documentation not found")
        return False

    content = cli_ref_file.read_text(encoding="utf-8")

    missing_commands = []
    for command in expected_commands:
        if f"``{command}``" not in content and f'"{command}"' not in content:
            missing_commands.append(command)

    if missing_commands:
        print(f"❌ Commands missing from CLI reference: {missing_commands}")
        return False

    print("✅ All CLI commands are documented in CLI reference")
    return True


def main():
    """Run all CLI documentation tests."""
    print("🧪 CLI Documentation Testing")
    print("=" * 40)

    tests = [
        test_cli_examples_in_documentation,
        test_cli_command_coverage_in_docs,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n🔍 Running {test.__name__}...")
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"\n📊 CLI Documentation Test Results: {passed}/{total}")

    if passed == total:
        print("🎉 All CLI documentation tests passed!")
        return True
    else:
        print("❌ Some CLI documentation tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
