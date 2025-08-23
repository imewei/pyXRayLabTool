# Development Environment Setup for XRayLabTool

## 🎉 Virtual Environment Successfully Created!

### ✅ Environment Details

**Location:** `/Users/b80985/Projects/pyXRayLabTool/venv/`  
**Python Version:** 3.13.7  
**Package Version:** XRayLabTool 0.1.8

### 📦 Installed Dependencies

#### Core Dependencies:
- **pandas** 2.3.2 - Data manipulation
- **numpy** 2.3.2 - Numerical computing  
- **scipy** 1.16.1 - Scientific computing
- **matplotlib** 3.10.5 - Plotting support
- **mendeleev** 1.1.0 - Atomic data
- **tqdm** 4.67.1 - Progress bars

#### Development Tools:
- **pytest** 8.4.1 - Testing framework
- **pytest-cov** 6.2.1 - Coverage reporting
- **pytest-benchmark** 5.1.0 - Performance benchmarks
- **black** 25.1.0 - Code formatting
- **flake8** 7.3.0 - Code linting
- **mypy** 1.17.1 - Type checking

#### Documentation Tools:
- **sphinx** 8.2.3 - Documentation generation
- **sphinx-rtd-theme** 3.0.2 - ReadTheDocs theme
- **sphinxcontrib-napoleon** 0.7 - Google/NumPy docstring support

#### Build & Publishing Tools:
- **build** 1.3.0 - Modern package builder
- **twine** 6.1.0 - PyPI upload tool

---

## 🚀 Quick Start Guide

### Activate Environment:
```bash
# Method 1: Using activation script (recommended)
source activate_venv.sh

# Method 2: Manual activation
source venv/bin/activate
```

### Verify Installation:
```bash
# Check CLI functionality
xraylabtool --version
xraylabtool calc SiO2 -e 10.0 -d 2.2

# Test API functionality
python -c "import xraylabtool as xlt; print(xlt.__version__)"
```

---

## 🧪 Testing & Validation

### Quick Tests:
```bash
# Package validation
python validate_package.py

# Basic functionality test
pytest tests/test_integration.py::TestBasicSetupAndInitialization -v

# CLI examples
xraylabtool list examples
```

### Full Test Suite:
```bash
# All tests with coverage
pytest tests/ -v --cov=xraylabtool

# Performance benchmarks only
pytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only -v

# Integration tests (Julia compatibility)
pytest tests/test_integration.py -k \"SiO2 or H2O\" -v
```

### Code Quality:
```bash
# Format code
black xraylabtool/ tests/

# Check formatting
black --check xraylabtool/ tests/

# Lint code
flake8 xraylabtool/ --max-line-length=88

# Type checking
mypy xraylabtool/ --ignore-missing-imports
```

---

## 📦 Package Building & Distribution

### Build Package:
```bash
# Modern build (recommended)
python -m build

# Verify build integrity
twine check dist/*

# View built files
ls -la dist/
```

### Package Validation:
```bash
# Comprehensive validation
python validate_package.py

# Manual checks
python setup.py check --restructuredtext --strict
```

---

## 📚 Documentation

### Build Documentation:
```bash
# Build Sphinx docs
cd docs && make html

# Serve docs locally (if sphinx-autobuild is installed)
cd docs && make livehtml

# Clean docs build
cd docs && make clean
```

### API Documentation:
```bash
# Generate API docs
sphinx-apidoc -o docs/source/api xraylabtool/

# Build with API docs
cd docs && make html
```

---

## 🔄 Development Workflow

### Using Enhanced Makefile:
```bash
# Quick development cycle
make dev                # Format + lint + fast tests

# Full validation
make validate          # Complete pre-commit checks

# Individual operations
make test              # Full test suite with coverage
make test-fast         # Quick tests without coverage
make lint              # Code linting
make format           # Auto-format code
make docs             # Build documentation
make clean            # Clean build artifacts
```

### Manual Workflow:
```bash
# 1. Make code changes
# 2. Format and lint
black xraylabtool/ tests/
flake8 xraylabtool/

# 3. Run tests
pytest tests/ -v

# 4. Test CLI functionality
xraylabtool calc SiO2 -e 10.0 -d 2.2

# 5. Build and validate
python -m build
python validate_package.py
```

---

## 🐍 Python API Usage

### Basic Usage:
```python
import xraylabtool as xlt
import numpy as np

# Single calculation
result = xlt.calculate_single_material_properties(\"SiO2\", 10.0, 2.2)
print(f\"Critical angle: {result.critical_angle_degrees[0]:.3f}°\")

# Multiple materials
materials = [\"SiO2\", \"Si\", \"Al2O3\"]
densities = [2.2, 2.33, 3.95]
energies = np.linspace(5, 15, 11)

results = xlt.calculate_xray_properties(materials, energies, densities)
for formula, result in results.items():
    print(f\"{formula}: {len(result.energy_kev)} energy points\")\n```

### Advanced Features:
```python
# Using new snake_case field names (recommended)
result = xlt.calculate_single_material_properties(\"Si\", 10.0, 2.33)
print(f\"Dispersion: {result.dispersion_delta[0]:.2e}\")\nprint(f\"Attenuation: {result.attenuation_length_cm[0]:.3f} cm\")\n\n# Energy range calculations\nenergies = np.logspace(np.log10(1), np.log10(30), 100)\nresult = xlt.calculate_single_material_properties(\"SiO2\", energies, 2.2)\nprint(f\"Energy range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV\")\n```\n\n---\n\n## 🖥️ CLI Usage Examples\n\n### Single Material:\n```bash\n# Basic calculation\nxraylabtool calc SiO2 -e 10.0 -d 2.2\n\n# Multiple energies\nxraylabtool calc Si -e 5.0,10.0,15.0 -d 2.33\n\n# Energy range with CSV output\nxraylabtool calc Al2O3 -e 5-15:11 -d 3.95 -o results.csv\n```\n\n### Batch Processing:\n```bash\n# Create materials file\ncat > materials.csv << EOF\nformula,density,energy\nSiO2,2.2,10.0\nSi,2.33,\"5.0,10.0,15.0\"\nAl2O3,3.95,10.0\nEOF\n\n# Process batch\nxraylabtool batch materials.csv -o results.csv --workers 4\n```\n\n### Utilities:\n```bash\n# Unit conversions\nxraylabtool convert energy 8.048,10.0,12.4 --to wavelength\n\n# Formula analysis\nxraylabtool formula Ca10P6O26H2 --verbose\nxraylabtool atomic Si,Al,Fe\n\n# Bragg diffraction\nxraylabtool bragg -d 3.14,2.45,1.92 -e 8.048\n\n# Reference information\nxraylabtool list constants\nxraylabtool list fields\nxraylabtool list examples\n```\n\n---\n\n## 🔧 Troubleshooting\n\n### Common Issues:\n\n1. **Virtual Environment Not Activated:**\n   ```bash\n   source venv/bin/activate  # or use activate_venv.sh\n   ```\n\n2. **Missing Dependencies:**\n   ```bash\n   pip install -e .[dev,docs]  # Reinstall with all extras\n   ```\n\n3. **CLI Not Found:**\n   ```bash\n   pip install -e .  # Reinstall in editable mode\n   which xraylabtool  # Should show path in venv\n   ```\n\n4. **Import Errors:**\n   ```bash\n   python -c \"import xraylabtool; print(xraylabtool.__file__)\"\n   # Should show path in project directory\n   ```\n\n5. **Test Failures:**\n   ```bash\n   pytest tests/ --tb=short  # Show shorter traceback\n   pytest tests/ -x  # Stop at first failure\n   ```\n\n### Performance Issues:\n```bash\n# Clear caches\npython -c \"from xraylabtool.core import clear_scattering_factor_cache; clear_scattering_factor_cache()\"\n\n# Benchmark performance\npytest tests/test_integration.py::TestPerformanceBenchmarks --benchmark-only -v\n```\n\n---\n\n## 📋 Package Status\n\n**✅ Ready for Development**  \n**✅ Ready for Testing**  \n**✅ Ready for Documentation**  \n**✅ Ready for PyPI Publishing**  \n\n**Package Validation Results:**\n- ✅ Package structure validated\n- ✅ setup.py configuration correct\n- ✅ pyproject.toml configuration correct\n- ✅ Requirements files validated\n- ✅ MANIFEST.in validated\n- ✅ Version consistency verified\n- ✅ Build process successful\n- ✅ Package integrity verified with twine\n\n**Next Steps:**\n1. Continue development and testing\n2. Update documentation as needed\n3. Prepare for PyPI publishing when ready\n4. Set up GitHub Actions workflows\n\n---\n\n## 💡 Tips for Development\n\n1. **Use the activation script:** `source activate_venv.sh` provides a nice overview\n2. **Leverage the Makefile:** Use `make dev`, `make test`, `make validate` for common tasks\n3. **Validate frequently:** Run `python validate_package.py` before major changes\n4. **Test CLI regularly:** Quick functionality checks with `xraylabtool calc SiO2 -e 10.0 -d 2.2`\n5. **Monitor performance:** Use benchmark tests to catch performance regressions\n\n**Happy developing! 🔬✨**"
