#!/usr/bin/env python3
"""
Package validation script for XRayLabTool.

This script validates that all package setup files are correctly configured
and can be used for PyPI publishing.
"""

import ast
import re
import sys
from pathlib import Path


def validate_setup_py():
    """Validate setup.py file."""
    print("🔍 Validating setup.py...")

    setup_path = Path("setup.py")
    if not setup_path.exists():
        print("❌ setup.py not found")
        return False

    try:
        with open(setup_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic syntax check
        ast.parse(content)

        # Check required fields
        required_fields = [
            "PACKAGE_NAME",
            "VERSION",
            "AUTHOR",
            "AUTHOR_EMAIL",
            "DESCRIPTION",
            "URL",
            "INSTALL_REQUIRES",
        ]

        for field in required_fields:
            if field not in content:
                print(f"⚠️  Missing field in setup.py: {field}")
                return False

        print("✅ setup.py validation passed")
        return True
    except Exception as e:
        print(f"❌ setup.py validation failed: {e}")
        return False


def validate_pyproject_toml():
    """Validate pyproject.toml file."""
    print("🔍 Validating pyproject.toml...")

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False

    try:
        import tomllib
    except ImportError:
        # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            print("⚠️  Cannot validate TOML (tomllib/tomli not available)")
            return True

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        # Check required sections
        required_sections = ["build-system", "project"]
        for section in required_sections:
            if section not in data:
                print(f"❌ Missing section in pyproject.toml: {section}")
                return False

        # Check project metadata
        project = data["project"]
        required_fields = ["name", "version", "description", "dependencies"]
        for field in required_fields:
            if field not in project:
                print(f"❌ Missing field in [project]: {field}")
                return False

        print("✅ pyproject.toml validation passed")
        return True
    except Exception as e:
        print(f"❌ pyproject.toml validation failed: {e}")
        return False


def validate_requirements():
    """Validate requirements files."""
    print("🔍 Validating requirements files...")

    req_files = ["requirements.txt"]

    all_valid = True
    for req_file in req_files:
        req_path = Path(req_file)
        if not req_path.exists():
            print(f"⚠️  {req_file} not found")
            continue

        try:
            with open(req_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Basic format check - look for package>=version patterns
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip()
                and not line.startswith("#")
                and not line.startswith("-r")
            ]

            if lines:  # Only check non-empty requirement files
                for line in lines:
                    if not re.match(r"^[a-zA-Z0-9_.-]+([><=!]+[0-9.]+.*)?$", line):
                        print(
                            f"⚠️  Potentially invalid requirement in {req_file}: {line}"
                        )

            print(f"✅ {req_file} validation passed")
        except Exception as e:
            print(f"❌ {req_file} validation failed: {e}")
            all_valid = False

    return all_valid


def validate_manifest():
    """Validate MANIFEST.in file."""
    print("🔍 Validating MANIFEST.in...")

    manifest_path = Path("MANIFEST.in")
    if not manifest_path.exists():
        print("❌ MANIFEST.in not found")
        return False

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for essential includes
        essential_includes = ["README.md", "LICENSE", "pyproject.toml"]
        for include in essential_includes:
            if include not in content:
                print(f"⚠️  MANIFEST.in should include: {include}")

        print("✅ MANIFEST.in validation passed")
        return True
    except Exception as e:
        print(f"❌ MANIFEST.in validation failed: {e}")
        return False


def validate_package_structure():
    """Validate package directory structure."""
    print("🔍 Validating package structure...")

    required_files = [
        "xraylabtool/__init__.py",
        "xraylabtool/core.py",
        "xraylabtool/cli.py",
        "tests/",
        "README.md",
    ]

    all_exist = True
    for item in required_files:
        path = Path(item)
        if not path.exists():
            print(f"❌ Missing required file/directory: {item}")
            all_exist = False
        else:
            print(f"✅ Found: {item}")

    return all_exist


def validate_version_consistency():
    """Check version consistency across files."""
    print("🔍 Checking version consistency...")

    versions = {}

    # Check setup.py
    try:
        setup_path = Path("setup.py")
        if setup_path.exists():
            with open(setup_path, "r") as f:
                content = f.read()
            match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                versions["setup.py"] = match.group(1)
    except Exception as e:
        print(f"⚠️  Could not extract version from setup.py: {e}")

    # Check pyproject.toml
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            tomllib = None

    if tomllib:
        try:
            with open("pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            versions["pyproject.toml"] = data["project"]["version"]
        except Exception as e:
            print(f"⚠️  Could not extract version from pyproject.toml: {e}")

    # Check versions match
    if len(set(versions.values())) > 1:
        print(f"❌ Version mismatch found: {versions}")
        return False
    elif versions:
        version = list(versions.values())[0]
        print(f"✅ Version consistency check passed: v{version}")
        return True
    else:
        print("⚠️  Could not validate version consistency")
        return True


def main():
    """Main validation function."""
    print("🚀 XRayLabTool Package Validation")
    print("=" * 40)

    validations = [
        validate_package_structure,
        validate_setup_py,
        validate_pyproject_toml,
        validate_requirements,
        validate_manifest,
        validate_version_consistency,
    ]

    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation error: {e}")
            results.append(False)
        print()

    # Summary
    print("📋 Validation Summary")
    print("=" * 20)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All validations passed! ({passed}/{total})")
        print("\n✅ Package is ready for PyPI publishing!")
        return 0
    else:
        print(f"⚠️  {passed}/{total} validations passed")
        print(f"❌ {total - passed} validation(s) failed")
        print("\n🔧 Please fix the issues above before publishing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
