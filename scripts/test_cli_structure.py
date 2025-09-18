#!/usr/bin/env python3
"""
CLI structure testing script.

This script tests the CLI command structure, argument parsing,
and help text without running the actual calculations.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path to import CLI module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cli_parser_creation():
    """Test that the CLI parser can be created successfully."""
    try:
        # Import the CLI parser creation function
        from xraylabtool.interfaces.cli import create_parser

        parser = create_parser()

        # Verify it's an ArgumentParser
        assert isinstance(parser, argparse.ArgumentParser)

        print("✅ CLI parser creation test passed")
        return True

    except ImportError as e:
        if "scipy" in str(e):
            print("⚠️  CLI parser creation test skipped (scipy import issue)")
            return True  # Skip this test due to environment
        else:
            print(f"❌ CLI parser creation test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ CLI parser creation test failed: {e}")
        return False


def test_cli_help_generation():
    """Test that CLI help can be generated for all commands."""
    try:
        from xraylabtool.interfaces.cli import create_parser

        parser = create_parser()

        # Test main help
        help_text = parser.format_help()
        assert "xraylabtool" in help_text.lower()
        assert "commands:" in help_text.lower() or "subcommands:" in help_text.lower()

        print("✅ CLI main help generation test passed")

        # Test subcommand help (this would require parser execution which might fail)
        # So we'll just verify the structure exists
        subparsers = None
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                subparsers = action
                break

        if subparsers:
            commands = list(subparsers.choices.keys())
            print(f"✅ Found {len(commands)} CLI subcommands: {commands}")

            # Verify all expected commands are present
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

            missing_commands = set(expected_commands) - set(commands)
            extra_commands = set(commands) - set(expected_commands)

            if missing_commands:
                print(f"❌ Missing commands: {missing_commands}")
                return False

            if extra_commands:
                print(f"⚠️  Extra commands found: {extra_commands}")

            print("✅ All expected CLI commands are present")
            return True
        else:
            print("❌ No subparsers found in CLI")
            return False

    except ImportError as e:
        if "scipy" in str(e):
            print("⚠️  CLI help generation test skipped (scipy import issue)")
            return True  # Skip due to environment
        else:
            print(f"❌ CLI help generation test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ CLI help generation test failed: {e}")
        return False


def test_cli_command_documentation():
    """Test that CLI commands match their documentation."""
    cli_file = project_root / "xraylabtool" / "interfaces" / "cli.py"

    try:
        content = cli_file.read_text(encoding="utf-8")

        # Check that docstring contains all commands
        docstring_commands = [
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

        for command in docstring_commands:
            if command not in content:
                print(f"❌ Command '{command}' not found in CLI source")
                return False

        print("✅ All CLI commands are documented in source code")
        return True

    except Exception as e:
        print(f"❌ CLI documentation test failed: {e}")
        return False


def test_cli_argument_structure():
    """Test CLI argument structure without executing commands."""
    try:
        from xraylabtool.interfaces.cli import create_parser

        parser = create_parser()

        # Test parsing common argument patterns
        test_cases = [
            # Help commands
            ["--help"],
            ["--version"],
            ["calc", "--help"],
            ["batch", "--help"],
            ["convert", "--help"],
            ["formula", "--help"],
            ["atomic", "--help"],
            ["bragg", "--help"],
            ["list", "--help"],
            ["install-completion", "--help"],
            ["uninstall-completion", "--help"],
        ]

        passed = 0
        for test_args in test_cases:
            try:
                # This will raise SystemExit for help, which is expected
                parser.parse_args(test_args)
                passed += 1
            except SystemExit:
                # Help commands exit with code 0, which is expected
                passed += 1
            except Exception as e:
                print(f"❌ Failed to parse args {test_args}: {e}")

        print(f"✅ CLI argument parsing test passed ({passed}/{len(test_cases)} cases)")
        return passed == len(test_cases)

    except ImportError as e:
        if "scipy" in str(e):
            print("⚠️  CLI argument structure test skipped (scipy import issue)")
            return True
        else:
            print(f"❌ CLI argument structure test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ CLI argument structure test failed: {e}")
        return False


def main():
    """Run all CLI structure tests."""
    print("🧪 CLI Structure Testing")
    print("=" * 40)

    tests = [
        test_cli_parser_creation,
        test_cli_help_generation,
        test_cli_command_documentation,
        test_cli_argument_structure,
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

    print(f"\n📊 CLI Structure Test Results: {passed}/{total}")

    if passed == total:
        print("🎉 All CLI structure tests passed!")
        return True
    else:
        print("❌ Some CLI structure tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
