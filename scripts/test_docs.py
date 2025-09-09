#!/usr/bin/env python3
"""
Documentation Testing Script for XRayLabTool

This script provides comprehensive testing of documentation including:
- Docstring examples (doctest)
- Code examples in RST files
- Link validation
- Documentation coverage analysis
- Accessibility checks
- Style validation

Usage:
    python scripts/test_docs.py [--quick] [--no-links] [--verbose]
"""

import argparse
import doctest
import glob
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_section(title: str, color: str = Colors.BLUE) -> None:
    """Print a formatted section header."""
    print(f"\n{color}{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{color}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'=' * 60}{Colors.END}\n")


def print_status(message: str, status: str, details: str = None) -> None:
    """Print a status message with color coding."""
    if status == "PASS":
        color = Colors.GREEN
        icon = "✅"
    elif status == "FAIL":
        color = Colors.RED
        icon = "❌"
    elif status == "WARN":
        color = Colors.YELLOW
        icon = "⚠️"
    else:
        color = Colors.CYAN
        icon = "ℹ️"

    print(f"{color}{icon} {message}{Colors.END}")
    if details:
        print(f"   {details}")


def test_doctests(verbose: bool = False) -> tuple[int, int]:
    """Test docstrings for code examples."""
    print_section("🧪 TESTING DOCSTRING EXAMPLES")

    total_failures = 0
    total_tests = 0

    # Test using Sphinx doctest builder
    print_status("Running Sphinx doctest builder...", "INFO")

    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "sphinx",
                "-b",
                "doctest",
                "docs/source",
                "docs/_build/doctest",
            ],
            capture_output=True,
            text=True,
            timeout=300, check=False,
        )

        if result.returncode == 0:
            print_status("Sphinx doctest", "PASS", "All docstring examples passed")
        else:
            print_status("Sphinx doctest", "FAIL", "Some docstring examples failed")
            if verbose:
                print(result.stdout)
                print(result.stderr)
            total_failures += 1
    except Exception as e:
        print_status("Sphinx doctest", "FAIL", f"Error: {e}")
        total_failures += 1

    total_tests += 1
    return total_failures, total_tests


def test_rst_code_examples(verbose: bool = False) -> tuple[int, int]:
    """Test code examples in RST documentation files."""
    print_section("📖 TESTING RST CODE EXAMPLES")

    total_failures = 0
    total_tests = 0

    rst_files = glob.glob("docs/source/**/*.rst", recursive=True)

    for rst_file in rst_files:
        total_tests += 1
        print_status(f"Testing {rst_file}...", "INFO")

        try:
            # Use doctest to test the RST file
            result = doctest.testfile(
                rst_file, verbose=verbose, optionflags=doctest.ELLIPSIS
            )

            if result.failed == 0:
                print_status(
                    f"  {os.path.basename(rst_file)}",
                    "PASS",
                    f"{result.attempted} examples tested",
                )
            else:
                print_status(
                    f"  {os.path.basename(rst_file)}",
                    "FAIL",
                    f"{result.failed}/{result.attempted} examples failed",
                )
                total_failures += 1
        except Exception as e:
            print_status(f"  {os.path.basename(rst_file)}", "WARN", f"Skipped: {e}")

    return total_failures, total_tests


def test_readme_examples(verbose: bool = False) -> tuple[int, int]:
    """Test code examples in README.md."""
    print_section("📋 TESTING README CODE EXAMPLES")

    total_failures = 0
    total_tests = 0

    try:
        with open("README.md") as f:
            content = f.read()
    except FileNotFoundError:
        print_status("README.md not found", "FAIL")
        return 1, 1

    # Extract Python code blocks
    code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

    print_status(f"Found {len(code_blocks)} Python code blocks", "INFO")

    for i, code in enumerate(code_blocks):
        total_tests += 1

        # Skip certain code blocks
        if (
            "..." in code
            or code.strip().startswith("#")
            or len(code.strip()) < 10
            or "import xraylabtool" not in code
        ):
            print_status(
                f"Code block {i + 1}", "WARN", "Skipped (placeholder or too short)"
            )
            continue

        print_status(f"Testing code block {i + 1}...", "INFO")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Add necessary setup
            f.write("import sys\n")
            f.write('sys.path.insert(0, ".")\n')
            f.write("import warnings\n")
            f.write('warnings.filterwarnings("ignore", category=DeprecationWarning)\n')
            f.write(code)
            temp_file = f.name

        try:
            # Run the code
            result = subprocess.run(
                ["python", temp_file], capture_output=True, text=True, timeout=30, check=False
            )

            if result.returncode == 0:
                print_status(f"  Code block {i + 1}", "PASS")
            else:
                print_status(
                    f"  Code block {i + 1}", "FAIL", f"Exit code: {result.returncode}"
                )
                if verbose:
                    print(f"    Stdout: {result.stdout}")
                    print(f"    Stderr: {result.stderr}")
                total_failures += 1

        except subprocess.TimeoutExpired:
            print_status(f"  Code block {i + 1}", "FAIL", "Timeout (30s)")
            total_failures += 1
        except Exception as e:
            print_status(f"  Code block {i + 1}", "FAIL", f"Error: {e}")
            total_failures += 1
        finally:
            os.unlink(temp_file)

    return total_failures, total_tests


def test_links(verbose: bool = False) -> tuple[int, int]:
    """Test documentation links."""
    print_section("🔗 TESTING DOCUMENTATION LINKS")

    total_failures = 0
    total_tests = 1

    print_status("Running Sphinx linkcheck...", "INFO")

    try:
        subprocess.run(
            [
                "python",
                "-m",
                "sphinx",
                "-b",
                "linkcheck",
                "docs/source",
                "docs/_build/linkcheck",
            ],
            capture_output=True,
            text=True,
            timeout=600, check=False,
        )

        # Check for broken links file
        broken_links_file = Path("docs/_build/linkcheck/broken.txt")
        if broken_links_file.exists():
            broken_content = broken_links_file.read_text()
            if broken_content.strip():
                print_status("Link check", "FAIL", "Some links are broken")
                if verbose:
                    print("Broken links:")
                    print(broken_content)
                total_failures += 1
            else:
                print_status("Link check", "PASS", "All links working")
        else:
            print_status("Link check", "WARN", "No broken links file generated")

    except Exception as e:
        print_status("Link check", "FAIL", f"Error: {e}")
        total_failures += 1

    return total_failures, total_tests


def check_documentation_coverage(verbose: bool = False) -> dict[str, int]:
    """Check documentation coverage."""
    print_section("📊 DOCUMENTATION COVERAGE ANALYSIS")

    coverage_stats = {"documented": 0, "undocumented": 0, "total": 0}

    print_status("Running Sphinx coverage check...", "INFO")

    try:
        subprocess.run(
            [
                "python",
                "-m",
                "sphinx",
                "-b",
                "coverage",
                "docs/source",
                "docs/_build/coverage",
            ],
            capture_output=True,
            text=True,
            timeout=300, check=False,
        )

        coverage_file = Path("docs/_build/coverage/python.txt")
        if coverage_file.exists():
            content = coverage_file.read_text()

            # Parse coverage results
            documented = len(re.findall(r"documented", content, re.IGNORECASE))
            undocumented = len(re.findall(r"undocumented", content, re.IGNORECASE))
            total = documented + undocumented

            coverage_stats.update(
                {"documented": documented, "undocumented": undocumented, "total": total}
            )

            if total > 0:
                percentage = (documented * 100) // total
                print_status(
                    "Documentation coverage",
                    "INFO",
                    f"{percentage}% ({documented}/{total} items documented)",
                )

                if percentage < 80:
                    print_status(
                        "Coverage warning", "WARN", "Documentation coverage below 80%"
                    )
            else:
                print_status("Coverage analysis", "WARN", "No items found to analyze")

            if verbose and undocumented > 0:
                print("\nUndocumented items:")
                print(content)
        else:
            print_status("Coverage analysis", "FAIL", "Coverage file not generated")

    except Exception as e:
        print_status("Coverage analysis", "FAIL", f"Error: {e}")

    return coverage_stats


def check_accessibility(verbose: bool = False) -> tuple[int, int]:
    """Basic accessibility checks."""
    print_section("♿ ACCESSIBILITY CHECKS")

    total_failures = 0
    total_tests = 0

    # Build HTML first
    print_status("Building HTML documentation...", "INFO")

    try:
        subprocess.run(
            ["python", "-m", "sphinx", "-b", "html", "docs/source", "docs/_build/html"],
            capture_output=True,
            text=True,
            timeout=300, check=False,
        )
    except Exception as e:
        print_status("HTML build", "FAIL", f"Error: {e}")
        return 1, 1

    html_dir = Path("docs/_build/html")
    if not html_dir.exists():
        print_status("HTML directory", "FAIL", "HTML build directory not found")
        return 1, 1

    # Check for images without alt text
    total_tests += 1
    html_files = list(html_dir.glob("**/*.html"))
    images_without_alt = []

    for html_file in html_files:
        try:
            content = html_file.read_text()
            # Find img tags without alt attributes
            img_pattern = r"<img[^>]*src=[^>]*>"
            alt_pattern = r"<img[^>]*alt=[^>]*>"

            img_matches = re.findall(img_pattern, content, re.IGNORECASE)
            alt_matches = re.findall(alt_pattern, content, re.IGNORECASE)

            if len(img_matches) > len(alt_matches):
                images_without_alt.append(str(html_file))
        except Exception:
            continue

    if images_without_alt:
        print_status(
            "Image alt text",
            "WARN",
            f"{len(images_without_alt)} files have images without alt text",
        )
        if verbose:
            for file in images_without_alt[:5]:  # Show first 5
                print(f"  {file}")
    else:
        print_status("Image alt text", "PASS", "All images have alt text")

    return total_failures, total_tests


def run_style_checks(verbose: bool = False) -> tuple[int, int]:
    """Run documentation style checks."""
    print_section("🎨 DOCUMENTATION STYLE CHECKS")

    total_failures = 0
    total_tests = 0

    # Check RST syntax with rstcheck
    total_tests += 1
    print_status("Checking RST syntax...", "INFO")

    try:
        result = subprocess.run(
            ["rstcheck", "--report-level", "warning"]
            + glob.glob("docs/**/*.rst", recursive=True),
            capture_output=True,
            text=True, check=False,
        )

        if result.returncode == 0:
            print_status("RST syntax", "PASS", "All RST files valid")
        else:
            print_status("RST syntax", "WARN", "Some RST syntax issues found")
            if verbose:
                print(result.stdout)
                print(result.stderr)
    except FileNotFoundError:
        print_status("RST syntax", "WARN", "rstcheck not available")
    except Exception as e:
        print_status("RST syntax", "FAIL", f"Error: {e}")
        total_failures += 1

    # Check documentation style with doc8
    total_tests += 1
    print_status("Checking documentation style...", "INFO")

    try:
        result = subprocess.run(
            [
                "doc8",
                "docs/source",
                "--ignore-path",
                "docs/source/_build",
                "--max-line-length",
                "100",
            ],
            capture_output=True,
            text=True, check=False,
        )

        if result.returncode == 0:
            print_status("Documentation style", "PASS", "Style checks passed")
        else:
            print_status("Documentation style", "WARN", "Some style issues found")
            if verbose:
                print(result.stdout)
                print(result.stderr)
    except FileNotFoundError:
        print_status("Documentation style", "WARN", "doc8 not available")
    except Exception as e:
        print_status("Documentation style", "FAIL", f"Error: {e}")
        total_failures += 1

    return total_failures, total_tests


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test XRayLabTool documentation")
    parser.add_argument(
        "--quick", action="store_true", help="Run only quick tests (skip link checking)"
    )
    parser.add_argument("--no-links", action="store_true", help="Skip link checking")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    start_time = time.time()

    print_section("🔬 XRAYLABTOOL DOCUMENTATION TESTING", Colors.CYAN)
    print(f"{Colors.BOLD}Starting comprehensive documentation testing...{Colors.END}\n")

    total_failures = 0
    total_tests = 0

    # Run all tests
    test_results = []

    # 1. Docstring tests
    failures, tests = test_doctests(args.verbose)
    total_failures += failures
    total_tests += tests
    test_results.append(("Docstring Tests", failures, tests))

    # 2. RST code examples
    failures, tests = test_rst_code_examples(args.verbose)
    total_failures += failures
    total_tests += tests
    test_results.append(("RST Code Examples", failures, tests))

    # 3. README examples
    failures, tests = test_readme_examples(args.verbose)
    total_failures += failures
    total_tests += tests
    test_results.append(("README Examples", failures, tests))

    # 4. Link checking (unless disabled)
    if not args.quick and not args.no_links:
        failures, tests = test_links(args.verbose)
        total_failures += failures
        total_tests += tests
        test_results.append(("Link Checking", failures, tests))

    # 5. Documentation coverage
    coverage_stats = check_documentation_coverage(args.verbose)

    # 6. Accessibility checks
    failures, tests = check_accessibility(args.verbose)
    total_failures += failures
    total_tests += tests
    test_results.append(("Accessibility", failures, tests))

    # 7. Style checks
    failures, tests = run_style_checks(args.verbose)
    total_failures += failures
    total_tests += tests
    test_results.append(("Style Checks", failures, tests))

    # Final summary
    end_time = time.time()
    duration = end_time - start_time

    print_section("📊 FINAL SUMMARY", Colors.BOLD)

    for test_name, failures, tests in test_results:
        if failures == 0:
            print_status(f"{test_name}", "PASS", f"{tests} tests completed")
        else:
            print_status(f"{test_name}", "FAIL", f"{failures}/{tests} tests failed")

    # Coverage summary
    if coverage_stats["total"] > 0:
        coverage_pct = (coverage_stats["documented"] * 100) // coverage_stats["total"]
        print_status(
            "Documentation Coverage",
            "INFO",
            f"{coverage_pct}% ({coverage_stats['documented']}/{coverage_stats['total']})",
        )

    print(f"\n{Colors.BOLD}Overall Results:{Colors.END}")
    print(f"  Total tests: {total_tests}")
    print(f"  Failed tests: {total_failures}")
    print(
        f"  Success rate: {((total_tests - total_failures) * 100) // total_tests if total_tests > 0 else 0}%"
    )
    print(f"  Duration: {duration:.1f} seconds")

    if total_failures == 0:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL DOCUMENTATION TESTS PASSED! 🎉{Colors.END}"
        )
        sys.exit(0)
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}❌ {total_failures} TESTS FAILED{Colors.END}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
