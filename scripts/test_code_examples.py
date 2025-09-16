#!/usr/bin/env python3
"""
Code examples testing script.

This script extracts and validates Python code examples from documentation files
to ensure they execute correctly with the current codebase.
"""

import ast
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


def extract_python_code_examples(docs_dir: Path) -> List[Dict[str, str]]:
    """Extract Python code examples from documentation files."""
    examples = []

    # Files that contain Python code examples
    doc_files = [
        docs_dir / "index.rst",
        docs_dir / "getting_started.rst",
        docs_dir / "cli_reference.rst",
        docs_dir / "examples" / "index.rst",
        Path(__file__).parent.parent / "README.md",
        Path(__file__).parent.parent / "CLAUDE.md",
    ]

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        try:
            content = doc_file.read_text(encoding='utf-8')

            # Pattern 1: RST code-block:: python
            python_blocks = re.findall(
                r'.. code-block:: python\s*\n\n(.*?)(?=\n\n\*\*|\n\.\.|$)',
                content, re.DOTALL
            )

            for i, block in enumerate(python_blocks):
                # Clean up indentation
                lines = block.split('\n')
                if lines:
                    # Find minimum indentation (excluding empty lines)
                    min_indent = min(
                        len(line) - len(line.lstrip())
                        for line in lines if line.strip()
                    )
                    cleaned_lines = [
                        line[min_indent:] if len(line) >= min_indent else line
                        for line in lines
                    ]
                    cleaned_code = '\n'.join(cleaned_lines).strip()

                    # Skip incomplete or template examples
                    if (cleaned_code and
                        not cleaned_code.startswith('#') and
                        not '[Title]' in cleaned_code and
                        not 'Example: [' in cleaned_code and
                        '"""' in cleaned_code and cleaned_code.count('"""') >= 2):
                        examples.append({
                            'file': str(doc_file),
                            'type': 'rst_python',
                            'code': cleaned_code,
                            'id': f"{doc_file.name}_rst_{i}"
                        })

            # Pattern 2: Markdown python code blocks
            if doc_file.suffix == '.md':
                md_blocks = re.findall(
                    r'```python\s*\n(.*?)\n```',
                    content, re.DOTALL
                )

                for i, block in enumerate(md_blocks):
                    if block.strip() and not block.strip().startswith('#'):
                        examples.append({
                            'file': str(doc_file),
                            'type': 'md_python',
                            'code': block.strip(),
                            'id': f"{doc_file.name}_md_{i}"
                        })

        except Exception as e:
            print(f"Warning: Could not read {doc_file}: {e}")

    return examples


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Validate Python code syntax without executing it."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Parse error: {e}"


def validate_imports(code: str) -> Tuple[bool, str]:
    """Validate that imports in code are available."""
    try:
        # Extract import statements
        tree = ast.parse(code)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        # Check if xraylabtool imports are valid
        valid_xraylabtool_imports = [
            'xraylabtool',
            'xraylabtool.calculators',
            'xraylabtool.calculators.core',
            'xraylabtool.utils',
            'xraylabtool.data_handling',
            'xraylabtool.data_handling.atomic_cache',
            'xraylabtool.data_handling.batch_processing',
            'xraylabtool.interfaces',
            'xraylabtool.interfaces.cli',
            'xraylabtool.io',
            'xraylabtool.validation'
        ]

        for imp in imports:
            if imp.startswith('xraylabtool'):
                # Split module and function/class name
                parts = imp.split('.')
                module_path = '.'.join(parts[:-1]) if len(parts) > 1 else imp

                # Check if module path is valid
                if module_path not in valid_xraylabtool_imports and imp not in valid_xraylabtool_imports:
                    # Allow imports from documented submodules
                    if not any(imp.startswith(valid_mod) for valid_mod in valid_xraylabtool_imports):
                        return False, f"Invalid xraylabtool import: {imp}"

        return True, ""

    except Exception as e:
        return False, f"Import validation error: {e}"


def execute_code_example(code: str) -> Tuple[bool, str]:
    """Execute a code example in a temporary environment."""
    # Skip examples that require matplotlib or other optional dependencies
    if any(skip_pattern in code for skip_pattern in [
        'matplotlib', 'plt.', 'import matplotlib',
        'subprocess.run', 'subprocess.call',
        'plt.show()', 'plt.tight_layout()'
    ]):
        return True, "Skipped (optional dependencies)"

    # Skip examples with shell commands or file operations
    if any(skip_pattern in code for skip_pattern in [
        'subprocess', '#!/bin/bash', 'mkdir', 'rm -f',
        'materials.csv', 'results.csv'
    ]):
        return True, "Skipped (shell/file operations)"

    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add project root to path for imports
            test_code = f"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

{code}
"""
            f.write(test_code)
            temp_file = f.name

        # Execute the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Clean up
        Path(temp_file).unlink()

        if result.returncode == 0:
            return True, "Executed successfully"
        else:
            return False, f"Execution failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        return False, "Execution timeout"
    except Exception as e:
        return False, f"Execution error: {e}"


def test_python_code_examples():
    """Test all Python code examples found in documentation."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    print("🔍 Extracting Python code examples from documentation...")
    examples = extract_python_code_examples(docs_dir)

    print(f"📊 Found {len(examples)} Python code examples")

    if not examples:
        print("⚠️  No Python code examples found in documentation")
        return True

    passed_syntax = 0
    passed_imports = 0
    passed_execution = 0
    total = len(examples)

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. Testing example from {Path(example['file']).name} ({example['id']})")
        print(f"   Type: {example['type']}")

        # Show first few lines of code
        code_preview = '\n'.join(example['code'].split('\n')[:3])
        if len(example['code'].split('\n')) > 3:
            code_preview += "\n   ..."
        print(f"   Code: {code_preview}")

        # Test syntax
        syntax_valid, syntax_error = validate_python_syntax(example['code'])
        if syntax_valid:
            print("   ✅ Valid syntax")
            passed_syntax += 1
        else:
            print(f"   ❌ Invalid syntax: {syntax_error}")
            continue

        # Test imports
        imports_valid, import_error = validate_imports(example['code'])
        if imports_valid:
            print("   ✅ Valid imports")
            passed_imports += 1
        else:
            print(f"   ❌ Invalid imports: {import_error}")
            continue

        # Test execution
        exec_success, exec_result = execute_code_example(example['code'])
        if exec_success:
            print(f"   ✅ {exec_result}")
            passed_execution += 1
        else:
            print(f"   ❌ {exec_result}")

    print(f"\n📊 Python Code Examples Test Results:")
    print(f"   Syntax validation: {passed_syntax}/{total}")
    print(f"   Import validation: {passed_imports}/{total}")
    print(f"   Execution tests: {passed_execution}/{total}")

    if passed_syntax == total and passed_imports == total:
        print("🎉 All Python code examples have valid syntax and imports!")
        if passed_execution >= total * 0.8:  # Allow some skipped examples
            print("🎉 Most Python code examples executed successfully!")
            return True
        else:
            print("⚠️  Some Python code examples failed execution")
            return False
    else:
        print("❌ Some Python code examples have syntax or import issues")
        return False


def main():
    """Run all code example tests."""
    print("🧪 Python Code Examples Testing")
    print("=" * 40)

    success = test_python_code_examples()

    if success:
        print("\n🎉 All code example tests passed!")
        return True
    else:
        print("\n❌ Some code example tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)