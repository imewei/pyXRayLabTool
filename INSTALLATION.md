# XRayLabTool Installation Guide

This guide covers all installation options for XRayLabTool, including the new comprehensive [all] option.

## Quick Installation

### Basic Installation (Core Dependencies Only)
```bash
pip install xraylabtool
```

### Development Installation (All Dependencies)
```bash
git clone https://github.com/imewei/pyXRayLabTool.git
cd pyXRayLabTool
pip install -e .[all]
```

## Installation Options

### Core Package Only
```bash
pip install xraylabtool
# or from source:
pip install -e .
```

### Available Extras

#### Development Environment
```bash
pip install -e .[dev]
# Includes: testing, linting, type checking, security analysis, pre-commit
```

#### Documentation Building
```bash
pip install -e .[docs]
# Includes: Sphinx, themes, extensions, notebook support
```

#### Testing Only
```bash
pip install -e .[test]
# Includes: pytest and essential testing tools
```

#### Linting and Formatting
```bash
pip install -e .[lint]
# Includes: ruff, mypy
```

#### Enhanced Export Features
```bash
pip install -e .[export]
# Includes: Excel export, interactive plotting, enhanced visualizations
```

#### API Server Functionality
```bash
pip install -e .[api]
# Includes: FastAPI, authentication, file upload support
```

#### Advanced Analysis
```bash
pip install -e .[analysis]
# Includes: scikit-learn, statistical modeling tools
```

#### Performance Profiling
```bash
pip install -e .[perf]
# Includes: memory profiler, line profiler, py-spy
```

#### **NEW: Complete Environment (All Dependencies)**
```bash
pip install -e .[all]
# Includes: ALL of the above + Jupyter/notebook support
```

### Combining Multiple Extras
```bash
# Multiple specific extras
pip install -e .[dev,docs,export]

# Development + documentation
pip install -e .[dev,docs]

# Everything at once
pip install -e .[all]
```

## Dependency Details

### Core Runtime Dependencies
- **numpy** ≥1.26.0, <2.0.0 - Numerical computing
- **pandas** ≥2.1.0, <3.0.0 - Data manipulation
- **scipy** ≥1.11.0, <2.0.0 - Scientific computing
- **matplotlib** ≥3.8.0 - Plotting
- **mendeleev** ≥0.15.0 - Atomic data
- **tqdm** ≥4.66.0 - Progress bars
- **psutil** ≥5.9.0 - Memory monitoring

### [all] Extra Includes (51 total dependencies):

**Testing & Quality (8 packages):**
- pytest, pytest-cov, pytest-benchmark, pytest-xdist, pytest-timeout
- ruff, mypy, pre-commit

**Documentation (10 packages):**
- sphinx, furo, sphinx-autodoc-typehints, sphinx-copybutton
- sphinx-design, sphinx-tabs, sphinx-togglebutton
- myst-parser, doc8, rstcheck, Pygments, docutils

**Type Checking (3 packages):**
- pandas-stubs, types-tqdm, types-psutil

**Enhanced Features (12 packages):**
- openpyxl, seaborn, plotly, jinja2 (export functionality)
- fastapi, uvicorn, python-multipart, python-jose, passlib, python-dateutil (API)
- scikit-learn, statsmodels (analysis)

**Performance & Profiling (3 packages):**
- memory-profiler, line-profiler, py-spy

**Jupyter & Notebooks (5 packages):**
- jupyter, jupyterlab, ipywidgets, nbsphinx, ipykernel

**Security & Coverage (2 packages):**
- codecov, pbr

## Requirements Files

### For Direct Installation
```bash
# Core dependencies only
pip install -r requirements.txt

# Documentation building
pip install -r docs/requirements.txt

# Binder/online notebooks
pip install -r requirements-binder.txt
```

### For Development
```bash
# Recommended: Use pyproject.toml extras
pip install -e .[all]

# Alternative: Use requirements files + extras
pip install -r requirements.txt
pip install -e .[dev,docs]
```

## Verification

### Test Installation
```bash
# Test core functionality
xraylabtool --version
python -c "import xraylabtool; print('✅ Core installation successful')"

# Test CLI
xraylabtool calc Si --density 2.33 --energy 8000

# Test shell completion (if installed)
xraylabtool completion status
```

### Test Development Environment
```bash
# Test development tools
black --version
pytest --version
mypy --version
sphinx-build --version

# Run tests
pytest tests/unit -v

# Build documentation
cd docs && make html
```

## Virtual Environment Setup

### Recommended Development Setup
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install with all dependencies
pip install -e .[all]

# Install shell completion
xraylabtool completion install

# Verify setup
pytest tests/unit/test_calculators.py -v
```

### Conda Environment
```bash
# Create conda environment
conda create -n xraylabtool python=3.12
conda activate xraylabtool

# Install XRayLabTool with all features
pip install -e .[all]

# Install completion in conda environment
xraylabtool completion install
```

## Troubleshooting

### Common Issues

**ImportError after installation:**
```bash
# Ensure you're in the right environment
which python
python -c "import sys; print(sys.path)"

# Reinstall in editable mode
pip install -e .[all]
```

**Completion not working:**
```bash
# Check completion status
xraylabtool completion status

# Reinstall completion
xraylabtool completion install --force
```

**Missing development tools:**
```bash
# Install development dependencies
pip install -e .[dev]

# Or install everything
pip install -e .[all]
```

**Documentation build fails:**
```bash
# Install documentation dependencies
pip install -e .[docs]

# Check for missing packages
sphinx-build --version
```

### Performance Notes

The [all] extra installs 51 packages totaling ~500MB. For specific use cases, consider targeted extras:

- **Basic development**: `[dev]` (13 packages)
- **Documentation only**: `[docs]` (10 packages)
- **Testing only**: `[test]` (5 packages)
- **Enhanced features**: `[export,api,analysis]` (12 packages)

## See Also

- [Shell Completion Guide](docs/completion_guide.rst) - Comprehensive completion setup
- [CLI Reference](docs/cli_reference.rst) - Command-line interface guide
- [Contributing Guide](CONTRIBUTING.md) - Development workflow
- [README.md](README.md) - Quick start and usage examples
