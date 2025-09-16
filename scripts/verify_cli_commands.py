#!/usr/bin/env python3
"""
CLI command verification script.

This script analyzes the CLI source code to verify the actual number
and structure of commands without importing the module.
"""

import re
from pathlib import Path
from typing import List, Dict


def extract_cli_commands(cli_file_path: Path) -> List[Dict]:
    """Extract CLI commands from the source code."""
    if not cli_file_path.exists():
        raise FileNotFoundError(f"CLI file not found: {cli_file_path}")

    content = cli_file_path.read_text(encoding='utf-8')

    # Find all add_parser calls
    parser_pattern = r'parser = subparsers\.add_parser\(\s*["\']([^"\']+)["\']'
    commands = []

    matches = re.finditer(parser_pattern, content)
    for match in matches:
        command_name = match.group(1)
        commands.append({
            'name': command_name,
            'line': content[:match.start()].count('\n') + 1
        })

    return commands

def extract_documented_commands(cli_file_path: Path) -> List[str]:
    """Extract documented commands from the docstring."""
    if not cli_file_path.exists():
        raise FileNotFoundError(f"CLI file not found: {cli_file_path}")

    content = cli_file_path.read_text(encoding='utf-8')

    # Find the docstring with Available Commands
    docstring_pattern = r'Available Commands:(.*?)"""'
    match = re.search(docstring_pattern, content, re.DOTALL)

    if not match:
        return []

    commands_section = match.group(1)

    # Extract command names
    command_pattern = r'^\s+(\w+(?:-\w+)*)\s+'
    commands = []

    for line in commands_section.split('\n'):
        match = re.match(command_pattern, line)
        if match:
            commands.append(match.group(1))

    return commands

def main():
    """Main verification function."""
    project_root = Path(__file__).parent.parent
    cli_file = project_root / "xraylabtool" / "interfaces" / "cli.py"

    print("🔍 CLI Command Verification")
    print("=" * 40)

    try:
        # Extract actual commands from code
        actual_commands = extract_cli_commands(cli_file)

        # Extract documented commands
        documented_commands = extract_documented_commands(cli_file)

        print(f"\n📊 Actual Commands Found: {len(actual_commands)}")
        for i, cmd in enumerate(actual_commands, 1):
            print(f"  {i}. {cmd['name']} (line {cmd['line']})")

        print(f"\n📝 Documented Commands: {len(documented_commands)}")
        for i, cmd in enumerate(documented_commands, 1):
            print(f"  {i}. {cmd}")

        # Verify consistency
        actual_names = [cmd['name'] for cmd in actual_commands]

        print(f"\n✅ Command Count Verification:")
        print(f"   Actual commands: {len(actual_names)}")
        print(f"   Documented commands: {len(documented_commands)}")

        if len(actual_names) == len(documented_commands):
            print("   ✅ Command counts match!")
        else:
            print("   ❌ Command counts don't match!")

        # Check for missing commands
        missing_in_docs = set(actual_names) - set(documented_commands)
        missing_in_code = set(documented_commands) - set(actual_names)

        if missing_in_docs:
            print(f"\n❌ Commands missing from documentation: {missing_in_docs}")

        if missing_in_code:
            print(f"\n❌ Commands documented but not in code: {missing_in_code}")

        if not missing_in_docs and not missing_in_code and len(actual_names) == len(documented_commands):
            print("\n🎉 All CLI commands are properly documented!")
            return True
        else:
            print(f"\n❌ CLI command documentation needs updates")
            return False

    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)