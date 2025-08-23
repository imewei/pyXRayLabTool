# Publishing Guide for XRayLabTool

## 📋 Pre-Publishing Checklist

### ✅ Code Quality
- [ ] All tests passing (`make test`)
- [ ] Code formatted with black (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking clean (`make type-check`)
- [ ] Full validation passes (`make validate`)

### ✅ Package Configuration
- [ ] Version number updated in both `setup.py` and `pyproject.toml`
- [ ] `CHANGELOG.md` updated with new version
- [ ] Dependencies correctly specified in `requirements.txt`
- [ ] CLI commands tested manually (`xraylabtool --help`)
- [ ] Package validation passes (`python3 validate_package.py`)

### ✅ Documentation
- [ ] `README.md` updated with any new features
- [ ] `CLI_REFERENCE.md` updated for CLI changes
- [ ] Examples tested and working
- [ ] Version badges updated

### ✅ Build Testing
- [ ] Package builds successfully (`make build`)
- [ ] Local installation works (`make test-install-local`)
- [ ] Wheel file structure looks correct

---

## 🚀 Publishing Methods

### Method 1: Automated via GitHub (Recommended)

#### For Test PyPI:
1. Push to `develop` branch
2. GitHub Actions will automatically publish to Test PyPI
3. Verify at: https://test.pypi.org/project/xraylabtool/

#### For Production PyPI:
1. Create a GitHub release with tag (e.g., `v0.1.5`)
2. GitHub Actions will automatically publish to PyPI
3. Verify at: https://pypi.org/project/xraylabtool/

### Method 2: Manual Publishing

#### Setup (one-time):
```bash
# Install publishing tools
pip install build twine

# Get API tokens from:
# - PyPI: https://pypi.org/manage/account/token/
# - Test PyPI: https://test.pypi.org/manage/account/token/
```

#### Test PyPI Publishing:
```bash
# Build package
make build

# Upload to Test PyPI
make upload-test
# OR manually:
# twine upload --repository testpypi dist/*

# Test installation
pip install -i https://test.pypi.org/simple/ xraylabtool
```

#### Production PyPI Publishing:
```bash
# Build package (clean build)
make clean
make build

# Upload to PyPI
make upload
# OR manually:
# twine upload dist/*
```

### Method 3: Using Enhanced Makefile Commands

```bash
# Complete release validation
make release-check

# Test PyPI workflow
make test-install-testpypi

# Production release
make upload  # Will prompt for confirmation
```

---

## 🔐 Security Setup

### GitHub Repository Secrets

Add these secrets in GitHub repository settings (`Settings → Secrets and variables → Actions`):

1. **`PYPI_API_TOKEN`**: Production PyPI API token
   - Get from: https://pypi.org/manage/account/token/
   - Scope: Entire account or specific to xraylabtool project

2. **`TEST_PYPI_API_TOKEN`**: Test PyPI API token
   - Get from: https://test.pypi.org/manage/account/token/
   - Scope: Entire account or specific to xraylabtool project

### Local `.pypirc` Configuration (Optional)

```ini
[distutils]
index-servers = 
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <your-pypi-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = <your-test-pypi-token>
```

---

## 📊 Post-Publishing Verification

### Test PyPI Verification:
```bash
# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ xraylabtool

# Test CLI
xraylabtool --version
xraylabtool calc SiO2 -e 10.0 -d 2.2

# Test API
python -c "import xraylabtool; print(xraylabtool.__version__)"
```

### Production PyPI Verification:
```bash
# Install from PyPI
pip install xraylabtool

# Run comprehensive tests
xraylabtool --help
xraylabtool list examples
xraylabtool calc SiO2 -e 5-15:5 -d 2.2 --format json
```

### Monitoring:
- [ ] Package appears on PyPI: https://pypi.org/project/xraylabtool/
- [ ] Download statistics at: https://pypistats.org/packages/xraylabtool
- [ ] GitHub repository badges updated
- [ ] Documentation links working

---

## 🐛 Common Issues & Solutions

### Build Issues:
```bash
# Clean build artifacts
make clean

# Reinstall in development mode
pip uninstall xraylabtool
pip install -e .

# Check package structure
python3 validate_package.py
```

### Upload Issues:
```bash
# Check package before upload
twine check dist/*

# Verify API tokens are correct
# Re-generate tokens if needed

# Check for existing version conflicts
# PyPI doesn't allow re-uploading same version
```

### Testing Issues:
```bash
# Clear package caches
pip cache purge

# Use virtual environment for clean testing
python -m venv test_env
source test_env/bin/activate  # or test_env\\Scripts\\activate on Windows
pip install xraylabtool
```

---

## 🔄 Version Management

### Semantic Versioning:
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes

### Update Locations:
1. `setup.py` → `VERSION = "x.y.z"`
2. `pyproject.toml` → `version = "x.y.z"`
3. `xraylabtool/__init__.py` → `__version__ = "x.y.z"`
4. `CHANGELOG.md` → Add new version section

### Release Notes Template:
```markdown
## [x.y.z] - YYYY-MM-DD

### Added
- New feature descriptions

### Changed
- Modified functionality

### Fixed
- Bug fixes

### Deprecated
- Features marked for removal
```

---

## 📞 Support

- **GitHub Issues**: https://github.com/imewei/pyXRayLabTool/issues
- **PyPI Project**: https://pypi.org/project/xraylabtool/
- **Email**: wchen@anl.gov
