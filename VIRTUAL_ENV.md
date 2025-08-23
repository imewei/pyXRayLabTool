# Virtual Environment Setup for XRayLabTool

This guide provides instructions for setting up a virtual environment for testing and developing XRayLabTool.

## Quick Setup

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
./setup_test_env.sh
```

This script will:
- Create a virtual environment
- Install the package with development dependencies
- Test all CLI functionality
- Run unit tests
- Provide usage examples

### Option 2: Manual Setup

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

4. **Install package with development dependencies:**
   ```bash
   pip install -e .[dev]
   ```

5. **Test installation:**
   ```bash
   xraylabtool --version
   ```

## CLI Usage Examples

Once the virtual environment is set up and activated, you can use the CLI:

### Basic Calculations
```bash
# Single material calculation
xraylabtool calc SiO2 -e 10.0 -d 2.2

# Multiple energies
xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33

# Energy range (linear)
xraylabtool calc Al2O3 -e 5-15:11 -d 3.95

# Energy range (logarithmic)
xraylabtool calc C -e 1-30:100:log -d 3.52
```

### File Output
```bash
# Save to CSV
xraylabtool calc SiO2 -e 8.0,10.0,12.0 -d 2.2 -o results.csv

# Save to JSON
xraylabtool calc Si -e 10.0 -d 2.33 -o results.json --format json
```

### Batch Processing
```bash
# Create a materials CSV file
cat > materials.csv << EOF
formula,density,energy
SiO2,2.2,10.0
Si,2.33,"8.0,12.0"
Al2O3,3.95,10.0
EOF

# Process batch
xraylabtool batch materials.csv -o batch_results.csv
```

### Utility Commands
```bash
# Unit conversions
xraylabtool convert energy 10.0 --to wavelength
xraylabtool convert wavelength 1.24 --to energy

# Formula parsing
xraylabtool formula SiO2
xraylabtool formula Al2O3

# Atomic data lookup
xraylabtool atomic Si
xraylabtool atomic H,C,N,O,Si

# Bragg angle calculations
xraylabtool bragg -d 3.14 -e 8.0
xraylabtool bragg -d 3.14,2.45,1.92 -w 1.54

# List available information
xraylabtool list fields
xraylabtool list constants
xraylabtool list examples
```

## Development Commands

### Testing
```bash
# Run CLI-specific tests
python -m pytest tests/test_cli.py -v

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=xraylabtool --cov-report=html
```

### Code Quality
```bash
# Format code
make format

# Check linting
make lint

# Run fast tests
make test-fast
```

### Package Building
```bash
# Build package
python build_package.py

# Run comprehensive tests
python run_tests.py
```

## Python API Usage

You can also use the package directly in Python:

```python
import xraylabtool as xlt
import numpy as np

# Single material calculation
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
print(f"Dispersion: {result.dispersion_delta[0]:.2e}")

# Multiple materials
formulas = ["SiO2", "Si", "Al2O3"]
densities = [2.2, 2.33, 3.95]
results = xlt.calculate_xray_properties(formulas, 10.0, densities)

# Energy sweep
energies = np.logspace(np.log10(1), np.log10(30), 100)
result = xlt.calculate_single_material_properties("Si", energies, 2.33)
```

## Deactivating the Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Common Issues

1. **"xraylabtool: command not found"**
   - Make sure the virtual environment is activated
   - Verify the package was installed with: `pip list | grep xraylabtool`

2. **Import errors for atomic data**
   - The mendeleev package should be installed automatically
   - Try reinstalling: `pip install --upgrade mendeleev`

3. **Permission errors**
   - Make sure the setup script is executable: `chmod +x setup_test_env.sh`

### Reinstalling

To completely reinstall:

```bash
rm -rf venv
./setup_test_env.sh
```

## Package Information

- **Version:** 0.1.5
- **Python Requirements:** ≥ 3.12
- **License:** MIT
- **Dependencies:** NumPy, SciPy, Pandas, Mendeleev, tqdm, matplotlib

For more detailed information, see the main [README.md](README.md) and [WARP.md](WARP.md) files.
