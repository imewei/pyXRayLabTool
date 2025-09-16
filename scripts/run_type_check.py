#!/usr/bin/env python3
"""
Enhanced type checking script for XRayLabTool.

This script provides comprehensive type checking functionality with
performance monitoring and reporting capabilities. It's designed to
be used both in development and CI/CD environments.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_mypy_check(
    target_paths: List[str],
    config_file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    strict_mode: bool = False,
    show_error_codes: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run MyPy type checking with enhanced configuration.

    Parameters
    ----------
    target_paths : List[str]
        Paths to check (files or directories)
    config_file : Optional[str]
        Path to MyPy configuration file
    cache_dir : Optional[str]
        Path to MyPy cache directory
    strict_mode : bool
        Enable strict mode checking
    show_error_codes : bool
        Show error codes in output
    verbose : bool
        Enable verbose output

    Returns
    -------
    Dict[str, Any]
        Type checking results with metrics
    """
    start_time = time.time()

    # Build MyPy command
    cmd = [sys.executable, "-m", "mypy"]

    if config_file:
        cmd.extend(["--config-file", config_file])

    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])

    if strict_mode:
        cmd.append("--strict")

    if show_error_codes:
        cmd.append("--show-error-codes")

    if verbose:
        cmd.append("--verbose")

    # Add target paths
    cmd.extend(target_paths)

    if verbose:
        print(f"Running command: {' '.join(cmd)}")

    # Run MyPy
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        duration = time.time() - start_time

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": duration,
            "command": " ".join(cmd),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "Type checking timed out after 5 minutes",
            "duration_seconds": time.time() - start_time,
            "command": " ".join(cmd),
        }
    except Exception as e:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Type checking failed: {e}",
            "duration_seconds": time.time() - start_time,
            "command": " ".join(cmd),
        }


def analyze_mypy_output(output: str) -> Dict[str, Any]:
    """
    Analyze MyPy output to extract metrics and insights.

    Parameters
    ----------
    output : str
        MyPy stdout output

    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    lines = output.strip().split("\n") if output.strip() else []

    # Count different types of issues
    error_count = 0
    warning_count = 0
    note_count = 0
    files_checked = set()

    for line in lines:
        if ":" in line and ("error:" in line or "note:" in line):
            parts = line.split(":")
            if len(parts) >= 2:
                file_path = parts[0].strip()
                files_checked.add(file_path)

                if "error:" in line:
                    error_count += 1
                elif "note:" in line:
                    note_count += 1

    # Extract success message if present
    success_line = None
    for line in lines:
        if "Success:" in line or "Found" in line:
            success_line = line.strip()
            break

    return {
        "total_lines": len(lines),
        "error_count": error_count,
        "warning_count": warning_count,
        "note_count": note_count,
        "files_checked_count": len(files_checked),
        "files_checked": list(files_checked),
        "success_message": success_line,
        "raw_output": output,
    }


def check_core_modules(project_root: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Check core modules with enhanced strict mode.

    Parameters
    ----------
    project_root : Path
        Project root directory
    verbose : bool
        Enable verbose output

    Returns
    -------
    Dict[str, Any]
        Core modules check results
    """
    core_modules = [
        "xraylabtool/calculators/core.py",
        "xraylabtool/utils.py",
        "xraylabtool/constants.py",
        "xraylabtool/typing_extensions.py",
    ]

    # Filter to existing modules
    existing_modules = []
    for module in core_modules:
        module_path = project_root / module
        if module_path.exists():
            existing_modules.append(str(module_path))

    if not existing_modules:
        return {
            "success": False,
            "error": "No core modules found to check",
        }

    result = run_mypy_check(
        target_paths=existing_modules,
        config_file="pyproject.toml",
        cache_dir=".mypy_cache",
        show_error_codes=True,
        verbose=verbose,
    )

    result["modules_checked"] = existing_modules
    result["analysis"] = analyze_mypy_output(result["stdout"])

    return result


def check_type_definitions(project_root: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Specifically check the typing_extensions module.

    Parameters
    ----------
    project_root : Path
        Project root directory
    verbose : bool
        Enable verbose output

    Returns
    -------
    Dict[str, Any]
        Type definitions check results
    """
    typing_module = project_root / "xraylabtool" / "typing_extensions.py"

    if not typing_module.exists():
        return {
            "success": False,
            "error": "typing_extensions.py not found",
        }

    result = run_mypy_check(
        target_paths=[str(typing_module)],
        config_file="pyproject.toml",
        strict_mode=True,
        verbose=verbose,
    )

    result["analysis"] = analyze_mypy_output(result["stdout"])
    return result


def validate_cache_performance(cache_dir: Path) -> Dict[str, Any]:
    """
    Validate MyPy cache performance and setup.

    Parameters
    ----------
    cache_dir : Path
        MyPy cache directory

    Returns
    -------
    Dict[str, Any]
        Cache validation results
    """
    results = {
        "cache_exists": cache_dir.exists(),
        "cache_readable": False,
        "cache_writable": False,
        "cache_size_mb": 0.0,
    }

    if cache_dir.exists():
        try:
            # Test read access
            list(cache_dir.iterdir())
            results["cache_readable"] = True

            # Test write access
            test_file = cache_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            results["cache_writable"] = True

            # Calculate cache size
            total_size = sum(
                f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
            )
            results["cache_size_mb"] = total_size / (1024 * 1024)

        except Exception as e:
            results["error"] = str(e)

    return results


def main():
    """Main function for type checking script."""
    parser = argparse.ArgumentParser(
        description="Enhanced type checking for XRayLabTool"
    )
    parser.add_argument(
        "--target",
        choices=["all", "core", "types", "tests"],
        default="core",
        help="Target modules to check",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode checking",
    )
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Show cache information",
    )

    args = parser.parse_args()

    # Get project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    print(f"🔍 XRayLabTool Type Checking")
    print(f"Project root: {project_root}")
    print(f"Target: {args.target}")
    print("-" * 50)

    # Cache validation
    cache_dir = project_root / ".mypy_cache"
    if args.cache_info:
        print("📁 Cache Information:")
        cache_info = validate_cache_performance(cache_dir)
        for key, value in cache_info.items():
            print(f"  {key}: {value}")
        print()

    success = True

    # Run type checking based on target
    if args.target == "core":
        print("🎯 Checking core modules...")
        result = check_core_modules(project_root, args.verbose)

        if result["success"]:
            print("✅ Core modules type checking passed")
            analysis = result.get("analysis", {})
            if analysis.get("success_message"):
                print(f"   {analysis['success_message']}")
        else:
            print("❌ Core modules type checking failed")
            if result.get("stderr"):
                print(f"Error: {result['stderr']}")
            success = False

    elif args.target == "types":
        print("🎯 Checking type definitions...")
        result = check_type_definitions(project_root, args.verbose)

        if result["success"]:
            print("✅ Type definitions checking passed")
        else:
            print("❌ Type definitions checking failed")
            if result.get("stderr"):
                print(f"Error: {result['stderr']}")
            success = False

    elif args.target == "all":
        print("🎯 Checking all modules...")
        all_result = run_mypy_check(
            target_paths=["xraylabtool"],
            config_file="pyproject.toml",
            cache_dir=".mypy_cache",
            strict_mode=args.strict,
            verbose=args.verbose,
        )

        if all_result["success"]:
            print("✅ All modules type checking passed")
        else:
            print("❌ Some modules have type checking issues")
            success = False

        # Show analysis
        analysis = analyze_mypy_output(all_result["stdout"])
        print(f"📊 Analysis:")
        print(f"   Files checked: {analysis['files_checked_count']}")
        print(f"   Errors: {analysis['error_count']}")
        print(f"   Notes: {analysis['note_count']}")

    if args.verbose and "result" in locals():
        print("\n📝 Detailed Output:")
        if result.get("stdout"):
            print(result["stdout"])
        if result.get("stderr"):
            print("STDERR:", result["stderr"])

    print("-" * 50)
    if success:
        print("🎉 Type checking completed successfully!")
        return 0
    else:
        print("💥 Type checking found issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
