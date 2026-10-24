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

ℹ️ Testing docs/examples/index.rst...
✅ docs/examples/index.rst: 12 examples tested

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

### `check_version.py` - Version Consistency

Reads every file that records the package version as text (it does not import the
package) and exits non-zero if any disagree: `pyproject.toml`,
`xraylabtool/__init__.py`, `uv.lock`, `docs/conf.py` fallbacks, the BibTeX entries in
`README.md` and `docs/index.rst`, and the newest section of both changelogs.

```bash
python scripts/check_version.py                  # sources agree with each other
python scripts/check_version.py --expect 0.4.7   # ... and equal this release
python scripts/check_version.py --installed      # ... and the installed metadata
python scripts/check_version.py --print          # print the pyproject version
```

Wired into `make version-check` and the `check-version` pre-commit hook.

### `test_install.py` - Install Smoke Test

Installs xraylabtool into a throw-away virtual environment and runs one SiO2
calculation, so a broken wheel (missing `.nff` data, bad entry points) fails here
rather than for users.

```bash
python scripts/test_install.py wheel [PATH]      # newest dist/*.whl by default
python scripts/test_install.py testpypi [--version X.Y.Z]
python scripts/test_install.py pypi [--version X.Y.Z]
```

Used by `make test-install-local`, `test-install-testpypi` and `test-install-pypi`.

### `gpu_check.py` - GPU Diagnostics

Reports the JAX backend, visible devices and a small SVD sanity check, and fails on
conflicting `jax-cuda12-plugin` / `jax-cuda13-plugin` installs. `--diagnose` adds the
system CUDA version, GPU model and installed JAX/CUDA packages.

Used by `make gpu-check` and `make gpu-diagnose`.

### `gui_headless_smoke.py` - GUI Smoke Check

Exercises the single- and multi-material GUI flows offscreen and writes PNG/CSV
exports to temporary files.

```bash
QT_QPA_PLATFORM=offscreen python -m scripts.gui_headless_smoke
```

## Maintenance Notes

This directory has been cleaned up to contain only essential scripts actively used in the development workflow. All scripts maintain compatibility with the Makefile targets and CI/CD pipelines.
