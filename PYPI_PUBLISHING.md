# PyPI Publishing Guide for XRayLabTool

This guide provides step-by-step instructions for publishing XRayLabTool to PyPI.

## 📋 Prerequisites

### 1. Install Build Tools

```bash
pip install build twine
```

### 2. Create PyPI Accounts

- **Test PyPI Account**: https://test.pypi.org/account/register/
- **PyPI Account**: https://pypi.org/account/register/

### 3. Configure API Tokens (Recommended)

Create API tokens for secure uploads:

#### For Test PyPI:
1. Go to https://test.pypi.org/manage/account/token/
2. Create a new token with scope "Entire account"
3. Save the token securely

#### For PyPI:
1. Go to https://pypi.org/manage/account/token/
2. Create a new token with scope "Entire account"
3. Save the token securely

#### Configure ~/.pypirc (Optional)

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-api-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-api-token-here
```

## 🚀 Publishing Steps

### Step 1: Prepare and Test

```bash
# Run all tests
python run_tests.py

# Build the package
python build_package.py

# Test local installation
python test_installation.py --source local
```

### Step 2: Upload to Test PyPI

```bash
# Upload to Test PyPI
python build_package.py --upload-test

# Test installation from Test PyPI
python test_installation.py --source testpypi
```

### Step 3: Upload to PyPI (Production)

```bash
# Upload to PyPI (after testing)
python build_package.py --upload-pypi

# Test installation from PyPI
python test_installation.py --source pypi
```

## 🛠 Manual Commands

If you prefer to run commands manually:

### Build Package

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build source distribution and wheel
python -m build

# Check the package
python -m twine check dist/*
```

### Upload to Test PyPI

```bash
python -m twine upload --repository testpypi dist/*
```

### Upload to PyPI

```bash
python -m twine upload dist/*
```

## 📦 Package Structure

The package includes:

```
xraylabtool/
├── pyproject.toml          # Package configuration
├── MANIFEST.in             # Include/exclude files for distribution
├── README.md               # Package description (used by PyPI)
├── LICENSE                 # MIT License
├── build_package.py        # Automated build script
├── test_installation.py    # Installation testing script
├── xraylabtool/            # Main package
│   ├── __init__.py        # Package exports
│   ├── core.py            # Core functionality
│   ├── utils.py           # Utility functions
│   ├── constants.py       # Physical constants
│   └── data/              # Atomic scattering factor data
│       └── AtomicScatteringFactor/
│           └── *.nff      # Element data files
├── tests/                  # Test suite
├── docs/                   # Documentation
└── run_tests.py           # Test runner
```

## ✅ Pre-Publication Checklist

- [ ] All tests passing (13/13 test suites)
- [ ] Version number updated in `pyproject.toml` and `__init__.py`
- [ ] Changelog updated with release notes
- [ ] README.md is comprehensive and up-to-date
- [ ] License file is present and correct
- [ ] Package builds without errors
- [ ] Package passes `twine check`
- [ ] Local installation test passes
- [ ] Test PyPI upload and installation successful

## 🔍 Verification After Publishing

### Check Package on PyPI

- **Test PyPI**: https://test.pypi.org/project/xraylabtool/
- **PyPI**: https://pypi.org/project/xraylabtool/

### Test Installation

```bash
# Create a new virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install xraylabtool

# Test basic functionality
python -c "import xraylabtool as xlt; print(xlt.__version__)"
```

### Verify Package Contents

```bash
# Install and inspect
pip install xraylabtool
pip show xraylabtool
pip show -f xraylabtool  # Show all files
```

## 🔄 Version Management

### Semantic Versioning

XRayLabTool follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Process

1. Update version in `pyproject.toml` and `xraylabtool/__init__.py`
2. Update `docs/source/changelog.rst`
3. Update README.md with new features
4. Run full test suite
5. Build and test package
6. Upload to Test PyPI
7. Test installation from Test PyPI
8. Upload to PyPI
9. Create GitHub release tag
10. Update documentation

## 🐛 Troubleshooting

### Common Issues

1. **Upload Error: "File already exists"**
   - Solution: Increment version number

2. **Import Error After Installation**
   - Check package structure in `pyproject.toml`
   - Verify `MANIFEST.in` includes all necessary files

3. **Missing Data Files**
   - Ensure `*.nff` files are included via `package-data`
   - Check `MANIFEST.in` includes data directory

4. **Dependency Issues**
   - Verify all dependencies are available on PyPI
   - Check version constraints in `pyproject.toml`

### Package Size Optimization

The package includes ~100 atomic scattering factor files (~2MB total). This is necessary for functionality but keeps the package lightweight.

## 📞 Support

For issues with the publishing process:

1. Check the [Python Packaging Guide](https://packaging.python.org/)
2. Review [PyPI documentation](https://pypi.org/help/)
3. Open an issue on the project repository

## 🎯 Post-Publication Tasks

After successful publication:

1. **Update README badges** with PyPI version/download counts
2. **Announce release** on relevant forums/mailing lists
3. **Update Conda Forge** recipe (if applicable)
4. **Create GitHub release** with changelog
5. **Update documentation** with installation instructions
6. **Monitor PyPI statistics** and user feedback

## 📈 Analytics and Monitoring

Track package usage via:

- **PyPI Stats**: https://pypistats.org/packages/xraylabtool
- **GitHub Insights**: Repository traffic and clones
- **Download Statistics**: Monitor adoption and popular versions

## 🔐 Security Considerations

- Use API tokens instead of passwords
- Store tokens securely (never commit to repository)
- Use separate tokens for Test PyPI and PyPI
- Regularly rotate API tokens
- Consider using GitHub Actions for automated publishing