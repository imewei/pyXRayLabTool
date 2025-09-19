# Scripts Directory

This directory contains essential utility scripts for XRayLabTool development and maintenance.

## Available Scripts

### `test_docs.py` - Documentation Testing

Documentation testing script that validates:

- **Docstring Examples**: Tests all code examples in Python docstrings
- **RST Code Blocks**: Validates code examples in documentation files
- **README Examples**: Ensures README code samples work correctly
- **Link Validation**: Checks for broken links in documentation
- **Coverage Analysis**: Reports documentation coverage statistics
- **Accessibility**: Basic accessibility compliance checks
- **Style Validation**: RST syntax and style checking

#### Usage:

```bash
# Run all tests
python scripts/test_docs.py

# Quick tests (skip slow link checking)
python scripts/test_docs.py --quick

# Skip link checking only
python scripts/test_docs.py --no-links

# Verbose output with detailed error information
python scripts/test_docs.py --verbose

# Help and options
python scripts/test_docs.py --help
```

#### Requirements:

Install documentation testing dependencies:

```bash
pip install -e .[docs]
pip install doc8 rstcheck  # For style checking
```

#### Output:

The script provides color-coded output with clear pass/fail indicators:

- ✅ **PASS**: All tests passed
- ❌ **FAIL**: Test failures found
- ⚠️ **WARN**: Warnings or skipped tests
- ℹ️ **INFO**: Informational messages

#### Integration:

This script integrates with:

- **Pre-commit hooks**: Add to `.pre-commit-config.yaml`
- **CI/CD pipelines**: Used in GitHub Actions workflows
- **Local development**: Run before committing documentation changes
- **Release process**: Validate documentation before releases

#### Example Output:

```
🔬 XRAYLABTOOL DOCUMENTATION TESTING
====================================================================

🧪 TESTING DOCSTRING EXAMPLES
====================================================================

ℹ️ Running Sphinx doctest builder...
✅ Sphinx doctest: All docstring examples passed

📖 TESTING RST CODE EXAMPLES
====================================================================

ℹ️ Testing docs/source/examples.rst...
✅ docs/source/examples.rst: 12 examples tested

📊 FINAL SUMMARY
====================================================================

✅ Docstring Tests: 1 tests completed
✅ RST Code Examples: 5 tests completed
✅ README Examples: 8 tests completed
✅ Documentation Coverage: 85% (234/275)

🎉 ALL DOCUMENTATION TESTS PASSED! 🎉
```

### `run_type_check.py` - Type Checking

Advanced type checking script with scientific computing focus:

- **MyPy Integration**: Comprehensive static type analysis
- **Cache Management**: Optimized caching for faster subsequent runs
- **Target Selection**: Core packages, all packages, or strict mode
- **Scientific Libraries**: Enhanced type checking for NumPy, SciPy, etc.

#### Usage:
```bash
python scripts/run_type_check.py --target core    # Core packages only
python scripts/run_type_check.py --target all     # All packages
python scripts/run_type_check.py --strict         # Strict mode
python scripts/run_type_check.py --cache-info     # Cache information
```

### `validate_style_guide.py` - Style Guide Validation

Comprehensive style guide validation for scientific computing:

- **Import Organization**: Scientific library import patterns
- **Naming Conventions**: PEP 8 and scientific computing standards
- **Documentation**: NumPy-style docstring validation
- **Code Patterns**: Scientific computing best practices

#### Usage:
```bash
python scripts/validate_style_guide.py           # Full validation
python scripts/validate_style_guide.py --verbose # Detailed output
python scripts/validate_style_guide.py --fix     # Apply fixes
```

### `format-code.sh` - Code Formatting

Shell script for automated code formatting using modern tools.

### `setup-githooks.sh` - Git Hooks Setup

Sets up pre-commit and pre-push hooks for quality assurance.

## Maintenance Notes

This directory has been cleaned up to contain only essential scripts actively used in the development workflow. All scripts maintain compatibility with the Makefile targets and CI/CD pipelines.
