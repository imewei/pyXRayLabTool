# Scripts Directory

This directory contains utility scripts for XRayLabTool development and maintenance.

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

### Future Scripts

Additional utility scripts may be added for:

- **Performance benchmarking**: `benchmark.py`
- **Release preparation**: `prepare_release.py`
- **Code generation**: `generate_api_docs.py`
- **Data validation**: `validate_atomic_data.py`
