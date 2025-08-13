#!/usr/bin/env python3
"""
Build script for creating a PyPI-ready distribution of XRayLabTool.

This script automates the process of:
1. Running tests to ensure package quality
2. Building the source distribution and wheel
3. Checking the distribution for common issues
4. Providing instructions for PyPI upload

Usage:
    python build_package.py [--skip-tests] [--upload-test] [--upload-pypi]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def clean_build_artifacts():
    """Clean up build artifacts."""
    print("\n🧹 Cleaning build artifacts...")
    artifacts = ["build/", "dist/", "*.egg-info/"]
    for pattern in artifacts:
        os.system(f"rm -rf {pattern}")

def run_tests():
    """Run the test suite."""
    if not run_command(["python", "run_tests.py"], "Running test suite"):
        print("❌ Tests failed. Please fix issues before building package.")
        return False
    print("✅ All tests passed!")
    return True

def build_package():
    """Build the package distribution."""
    clean_build_artifacts()
    
    # Build source distribution and wheel
    if not run_command(["python", "-m", "build"], "Building package distributions"):
        return False
    
    print("✅ Package built successfully!")
    return True

def check_package():
    """Check the built package for issues."""
    # Check with twine
    if not run_command(["python", "-m", "twine", "check", "dist/*"], "Checking package with twine"):
        print("⚠️  Package check found issues. Please review.")
        return False
    
    print("✅ Package passed all checks!")
    return True

def upload_to_test_pypi():
    """Upload to Test PyPI."""
    print("\n📤 Uploading to Test PyPI...")
    print("You'll need to configure your ~/.pypirc file or use tokens.")
    
    cmd = ["python", "-m", "twine", "upload", "--repository", "testpypi", "dist/*"]
    if run_command(cmd, "Uploading to Test PyPI"):
        print("✅ Uploaded to Test PyPI successfully!")
        print("🔗 Check: https://test.pypi.org/project/xraylabtool/")
        return True
    return False

def upload_to_pypi():
    """Upload to PyPI."""
    print("\n📤 Uploading to PyPI...")
    print("⚠️  WARNING: This will upload to the REAL PyPI!")
    
    confirm = input("Are you sure you want to upload to PyPI? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Upload cancelled.")
        return False
    
    cmd = ["python", "-m", "twine", "upload", "dist/*"]
    if run_command(cmd, "Uploading to PyPI"):
        print("✅ Uploaded to PyPI successfully!")
        print("🔗 Check: https://pypi.org/project/xraylabtool/")
        return True
    return False

def check_dependencies():
    """Check if required build tools are installed."""
    required_tools = ["build", "twine"]
    missing = []
    
    for tool in required_tools:
        try:
            subprocess.run([sys.executable, "-m", tool, "--help"], 
                         capture_output=True, check=True)
        except subprocess.CalledProcessError:
            missing.append(tool)
    
    if missing:
        print(f"❌ Missing required tools: {', '.join(missing)}")
        print("Install with: pip install build twine")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Build XRayLabTool package for PyPI")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip running tests before building")
    parser.add_argument("--upload-test", action="store_true",
                       help="Upload to Test PyPI after building")
    parser.add_argument("--upload-pypi", action="store_true",
                       help="Upload to PyPI after building")
    
    args = parser.parse_args()
    
    print("🚀 XRayLabTool Package Builder")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            sys.exit(1)
    else:
        print("⚠️  Skipping tests as requested")
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Check package
    if not check_package():
        sys.exit(1)
    
    # Upload if requested
    if args.upload_test:
        if not upload_to_test_pypi():
            sys.exit(1)
    
    if args.upload_pypi:
        if not upload_to_pypi():
            sys.exit(1)
    
    # Final instructions
    print("\n🎉 Package build completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test the package locally: pip install dist/*.whl")
    print("2. Test upload: python build_package.py --upload-test")
    print("3. Production upload: python build_package.py --upload-pypi")
    print("\n📂 Built files are in the 'dist/' directory")

if __name__ == "__main__":
    main()