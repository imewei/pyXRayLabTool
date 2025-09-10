# XRayLabTool

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/xraylabtool.svg)](https://badge.fury.io/py/xraylabtool)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pyxraylabtool/badge/?version=latest)](https://pyxraylabtool.readthedocs.io/en/latest/?badge=latest)

XRayLabTool is a Python package and command-line tool for calculating X-ray optical properties of materials based on their chemical formulas and densities.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Input Parameters](#input-parameters)
- [Output: XRayResult Dataclass](#output-xrayresult-dataclass)
- [Usage Examples](#usage-examples)
- [Migration Guide](#migration-guide)
- [Supported Calculations](#supported-calculations)
- [Performance Features](#performance-features)
- [Testing and Validation](#testing-and-validation)
- [API Reference](#api-reference)
- [Documentation & Support](#documentation--support)
- [Citation](#citation)

## Installation

### From PyPI (Recommended)

```bash
pip install xraylabtool
```

### From Source (Development)

```bash
git clone https://github.com/imewei/pyXRayLabTool.git
cd pyXRayLabTool
pip install -e .
```

### Shell Completion Setup

After installation, enable tab completion for enhanced CLI productivity:

```bash
# Install shell completion (auto-detects shell)
xraylabtool install-completion

# Alternative: using flag syntax
xraylabtool --install-completion

# Test if completion is working
xraylabtool install-completion --test
```

**Prerequisites by Shell:**

**Bash users:**
```bash
# macOS (Homebrew)
brew install bash-completion@2

# Add to ~/.bash_profile or ~/.bashrc:
[[ -r "/opt/homebrew/etc/profile.d/bash_completion.sh" ]] && . "/opt/homebrew/etc/profile.d/bash_completion.sh"

# Linux (Ubuntu/Debian)
sudo apt install bash-completion

# Linux (RHEL/CentOS)
sudo yum install bash-completion
```

**Zsh users:**
```bash
# macOS (Homebrew)
brew install zsh-completions

# Add to ~/.zshrc:
if type brew &>/dev/null; then
  FPATH="$(brew --prefix)/share/zsh-completions:${FPATH}"
  autoload -U compinit
  compinit
fi

# Linux (Ubuntu/Debian)
sudo apt install zsh-autosuggestions zsh-syntax-highlighting

# Linux (RHEL/CentOS)
sudo yum install zsh-autosuggestions
```

**Fish and PowerShell users:**
- Fish: No additional prerequisites (built-in completion system)
- PowerShell: No additional prerequisites (built-in completion system)

**Uninstalling completion:**
```bash
# Remove shell completion
xraylabtool uninstall-completion
```

Restart your shell or source your config file after installation.

> **Note**: Shell completion is supported for **bash, zsh, fish, and PowerShell**. Use `xraylabtool install-completion <shell>` to install completion for your preferred shell, or `xraylabtool install-completion` for auto-detection.

### Requirements

- **Python** ≥ 3.12
- **NumPy** ≥ 1.20.0
- **SciPy** ≥ 1.7.0
- **Pandas** ≥ 1.3.0
- **Mendeleev** ≥ 0.10.0
- **tqdm** ≥ 4.60.0
- **matplotlib** ≥ 3.4.0 (optional, for plotting)

---

## Quick Start

```bash
# Install from PyPI
pip install xraylabtool

# Calculate X-ray properties for silicon at 10 keV
python -c "import xraylabtool as xlt; result = xlt.calculate_single_material_properties('Si', 10.0, 2.33); print(f'Critical angle: {result.critical_angle_degrees[0]:.3f}°')"

# Or use the command-line interface
xraylabtool calc Si -e 10.0 -d 2.33
```

## Usage Examples

### Single Material Analysis

```python
import xraylabtool as xlt
import numpy as np

# Calculate properties for quartz at 10 keV
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
print(f"Formula: {result.formula}")
print(f"Molecular Weight: {result.molecular_weight_g_mol:.2f} g/mol")
print(f"Critical Angle: {result.critical_angle_degrees[0]:.3f}°")
print(f"Attenuation Length: {result.attenuation_length_cm[0]:.2f} cm")
```

### Multiple Materials Comparison

```python
# Compare common X-ray optics materials
materials = {
    "SiO2": 2.2,      # Fused silica
    "Si": 2.33,       # Silicon
    "Al2O3": 3.95,    # Sapphire
    "C": 3.52,        # Diamond
}

formulas = list(materials.keys())
densities = list(materials.values())
energy = 10.0  # keV (Cu Kα)

results = xlt.calculate_xray_properties(formulas, energy, densities)

# Display results (using new field names)
for formula, result in results.items():
    print(f"{formula:6}: θc = {result.critical_angle_degrees[0]:.3f}°, "
          f"δ = {result.dispersion_delta[0]:.2e}")
```

### Energy Range Analysis

```python
# Energy sweep for material characterization
energies = np.logspace(np.log10(1), np.log10(30), 100)  # 1-30 keV
result = xlt.calculate_single_material_properties("Si", energies, 2.33)

print(f"Energy range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV")
print(f"Data points: {len(result.energy_kev)}")
```

---

## Command-Line Interface (CLI)

### Installation & Verification

```bash
# Install with CLI support
pip install xraylabtool

# Verify CLI installation
xraylabtool --version

# Install shell completion (auto-detects shell)
xraylabtool install-completion

# Test completion is working
xraylabtool install-completion --test
```

### Quick CLI Examples

#### Single Material Calculation
```bash
# Calculate properties for quartz at 10 keV
xraylabtool calc SiO2 -e 10.0 -d 2.2
```

#### Energy Range Scan
```bash
# Energy sweep from 5-15 keV (11 points)
xraylabtool calc Si -e 5-15:11 -d 2.33 -o silicon_scan.csv
```

#### Batch Processing
```bash
# Create materials file
cat > materials.csv << EOF
formula,density,energy
SiO2,2.2,10.0
Si,2.33,"5.0,10.0,15.0"
Al2O3,3.95,10.0
EOF

# Process batch
xraylabtool batch materials.csv -o results.csv
```

#### Unit Conversions
```bash
# Convert energy to wavelength
xraylabtool convert energy 8.048,10.0,12.4 --to wavelength
```

#### Formula Analysis
```bash
# Parse chemical formulas
xraylabtool formula Ca10P6O26H2
xraylabtool atomic Si,Al,Fe
```

#### Bragg Diffraction Angles
```bash
# Calculate Bragg angles
xraylabtool bragg -d 3.14,2.45,1.92 -e 8.048
```

### Available CLI Commands

| Command | Purpose | Example |
|---------|---------|--------|
| `calc` | Single material calculations | `xraylabtool calc SiO2 -e 10.0 -d 2.2` |
| `batch` | Process multiple materials | `xraylabtool batch materials.csv -o results.csv` |
| `convert` | Energy/wavelength conversion | `xraylabtool convert energy 10.0 --to wavelength` |
| `formula` | Chemical formula analysis | `xraylabtool formula Al2O3` |
| `atomic` | Atomic data lookup | `xraylabtool atomic Si,Al,Fe` |
| `bragg` | Diffraction angle calculations | `xraylabtool bragg -d 3.14 -e 8.0` |
| `list` | Show constants/fields/examples | `xraylabtool list constants` |
| `install-completion` | Install shell completion | `xraylabtool install-completion` |
| `uninstall-completion` | Remove shell completion | `xraylabtool uninstall-completion` |

### Shell Completion Usage

Both command and flag syntaxes are supported:

```bash
# Subcommand syntax (recommended)
xraylabtool install-completion           # Install shell completion (auto-detect)
xraylabtool install-completion --test    # Test installation
xraylabtool uninstall-completion         # Remove completion

# Flag syntax (alternative)
xraylabtool --install-completion         # Install shell completion (auto-detect)

# Install for specific shells
xraylabtool install-completion bash      # Bash completion
xraylabtool install-completion zsh       # Zsh completion (requires zsh-completions)
xraylabtool install-completion fish      # Fish completion
xraylabtool install-completion powershell # PowerShell completion
```

> **💡 Shell Requirements**: Make sure to install the shell-specific prerequisites above before installing completion. Zsh users especially need `zsh-completions` for proper functionality.

**Tab Completion Features:**
- **Command completion**: Auto-complete all 9 available commands
- **Option completion**: Complete command-line options and flags
- **File path completion**: Automatic file path completion for input/output files
- **Chemical formulas**: Suggestions for common chemical formulas
- **Energy values**: Common X-ray energy suggestions (8.048, 10.0, 12.4 keV)

### Output Formats

- **Table** (default): Human-readable console output
- **CSV**: Spreadsheet-compatible format
- **JSON**: Structured data for programming

### Advanced Features

- **Energy Input Formats**: Single values, ranges, logarithmic spacing
- **Parallel Processing**: Multi-core batch processing with `--workers`
- **Field Selection**: Choose specific output fields with `--fields`
- **Precision Control**: Set decimal places with `--precision`
- **File Output**: Save results to CSV or JSON files
- **Multi-Shell Tab Completion**: Intelligent completion for bash, zsh, fish, and PowerShell
  - **Context-aware**: Suggests appropriate values based on current command
  - **File completion**: Automatic file path completion for input/output files
  - **Chemical formulas**: Common materials and element suggestions
  - **Energy values**: Typical X-ray energy suggestions (Cu Kα, Mo Kα, etc.)
  - **Cross-platform**: Works on macOS, Linux, and Windows (with WSL/Cygwin)

### 📖 CLI Help and Documentation

Get comprehensive help for any command:

```bash
# General help
xraylabtool --help

# Command-specific help
xraylabtool calc --help
xraylabtool batch --help
xraylabtool install-completion --help

# List available options and examples
xraylabtool list --help
```

**CLI Features:**
- 9 commands for X-ray analysis
- Energy input formats: Single values, ranges, lists, and logarithmic spacing
- Batch processing from CSV files
- Output formats: Table, CSV, and JSON
- Shell completion for bash, zsh, fish, and PowerShell
- Cross-platform support

---

## Input Parameters

| Parameter    | Type                                  | Description                                                    |
| ------------ | ------------------------------------- | -------------------------------------------------------------- |
| `formula(s)` | `str` or `List[str]`                  | Case-sensitive chemical formula(s), e.g., `"CO"` vs `"Co"`     |
| `energy`     | `float`, `List[float]`, or `np.array` | X-ray photon energies in keV (valid range: **0.03–30 keV**)   |
| `density`    | `float` or `List[float]`              | Mass density in g/cm³ (one per formula)                       |

---

## Output: `XRayResult` Dataclass

The `XRayResult` dataclass contains all computed X-ray optical properties with clear, descriptive field names:

### Material Properties
- **`formula: str`** – Chemical formula
- **`molecular_weight_g_mol: float`** – Molecular weight (g/mol)
- **`total_electrons: float`** – Total electrons per molecule
- **`density_g_cm3: float`** – Mass density (g/cm³)
- **`electron_density_per_ang3: float`** – Electron density (electrons/Å³)

### X-ray Properties (Arrays)
- **`energy_kev: np.ndarray`** – X-ray energies (keV)
- **`wavelength_angstrom: np.ndarray`** – X-ray wavelengths (Å)
- **`dispersion_delta: np.ndarray`** – Dispersion coefficient δ
- **`absorption_beta: np.ndarray`** – Absorption coefficient β
- **`scattering_factor_f1: np.ndarray`** – Real part of atomic scattering factor
- **`scattering_factor_f2: np.ndarray`** – Imaginary part of atomic scattering factor

### Derived Quantities (Arrays)
- **`critical_angle_degrees: np.ndarray`** – Critical angles (degrees)
- **`attenuation_length_cm: np.ndarray`** – Attenuation lengths (cm)
- **`real_sld_per_ang2: np.ndarray`** – Real scattering length density (Å⁻²)
- **`imaginary_sld_per_ang2: np.ndarray`** – Imaginary scattering length density (Å⁻²)

> **📝 Note**: Legacy field names (e.g., `Formula`, `MW`, `Critical_Angle`) are still supported for backward compatibility but will emit deprecation warnings. Use the new descriptive field names for clearer, more maintainable code.

---

## Usage Examples

### Recommended: Using New Field Names

```python
# Calculate properties for silicon dioxide at 10 keV
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.33)

# Use new descriptive field names (recommended)
print(f"Formula: {result.formula}")                                      # "SiO2"
print(f"Molecular weight: {result.molecular_weight_g_mol:.2f} g/mol")     # 60.08 g/mol
print(f"Dispersion: {result.dispersion_delta[0]:.2e}")                   # δ value
print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")        # θc
print(f"Attenuation: {result.attenuation_length_cm[0]:.1f} cm")          # Attenuation length
```

### Legacy Field Names (Still Supported)

```python
# Legacy field names still work but emit deprecation warnings
print(f"Formula: {result.Formula}")                    # ⚠️ DeprecationWarning
print(f"Molecular weight: {result.MW:.2f} g/mol")     # ⚠️ DeprecationWarning
print(f"Dispersion: {result.Dispersion[0]:.2e}")       # ⚠️ DeprecationWarning
print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")  # ⚠️ DeprecationWarning
```

### Energy Range Analysis

```python
# Energy sweep for material characterization
energies = np.linspace(8.0, 12.0, 21)  # 21 points from 8-12 keV
result = xlt.calculate_single_material_properties("SiO2", energies, 2.33)

# Using new field names
print(f"Energy range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV")
print(f"Number of points: {len(result.energy_kev)}")
print(f"Dispersion range: {result.dispersion_delta.min():.2e} to {result.dispersion_delta.max():.2e}")
```

### Multiple Materials Comparison

```python
# Compare common X-ray optics materials
materials = {
    "SiO2": 2.2,      # Fused silica
    "Si": 2.33,       # Silicon
    "Al2O3": 3.95,    # Sapphire
    "C": 3.52,        # Diamond
}

formulas = list(materials.keys())
densities = list(materials.values())
energy = 10.0  # keV (Cu Kα)

results = xlt.calculate_xray_properties(formulas, energy, densities)

# Compare using new field names
for formula, result in results.items():
    print(f"{formula:8}: θc = {result.critical_angle_degrees[0]:.3f}°, "
          f"δ = {result.dispersion_delta[0]:.2e}, "
          f"μ = {result.attenuation_length_cm[0]:.1f} cm")
```

### Enhanced Plotting Example

```python
import matplotlib.pyplot as plt

# Energy-dependent properties with new field names
energies = np.logspace(np.log10(1), np.log10(20), 100)
result = xlt.calculate_single_material_properties("Si", energies, 2.33)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot using new descriptive field names
ax1.loglog(result.energy_kev, result.dispersion_delta, 'b-',
           label='δ (dispersion)', linewidth=2)
ax1.loglog(result.energy_kev, result.absorption_beta, 'r-',
           label='β (absorption)', linewidth=2)
ax1.set_xlabel('Energy (keV)')
ax1.set_ylabel('Optical constants')
ax1.set_title('Silicon: Dispersion & Absorption')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot critical angle with new field name
ax2.semilogx(result.energy_kev, result.critical_angle_degrees, 'g-', linewidth=2)
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Critical angle (°)')
ax2.set_title('Silicon: Critical Angle')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Migration Guide: Legacy to New Field Names

To help users transition from legacy CamelCase field names to the new descriptive snake_case names, here's a comprehensive mapping:

### Field Name Migration Table

| **Legacy Name**                    | **New Name**                       | **Description**                                   |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------- |
| `result.Formula`                   | `result.formula`                   | Chemical formula string                          |
| `result.MW`                        | `result.molecular_weight_g_mol`    | Molecular weight (g/mol)                         |
| `result.Number_Of_Electrons`       | `result.total_electrons`           | Total electrons per molecule                     |
| `result.Density`                   | `result.density_g_cm3`             | Mass density (g/cm³)                             |
| `result.Electron_Density`          | `result.electron_density_per_ang3` | Electron density (electrons/Å³)                  |
| `result.Energy`                    | `result.energy_kev`                | X-ray energies (keV)                             |
| `result.Wavelength`                | `result.wavelength_angstrom`       | X-ray wavelengths (Å)                            |
| `result.Dispersion`                | `result.dispersion_delta`          | Dispersion coefficient δ                         |
| `result.Absorption`                | `result.absorption_beta`           | Absorption coefficient β                         |
| `result.f1`                        | `result.scattering_factor_f1`      | Real part of atomic scattering factor            |
| `result.f2`                        | `result.scattering_factor_f2`      | Imaginary part of atomic scattering factor       |
| `result.Critical_Angle`            | `result.critical_angle_degrees`    | Critical angles (degrees)                        |
| `result.Attenuation_Length`        | `result.attenuation_length_cm`     | Attenuation lengths (cm)                         |
| `result.reSLD`                     | `result.real_sld_per_ang2`         | Real scattering length density (Å⁻²)             |
| `result.imSLD`                     | `result.imaginary_sld_per_ang2`    | Imaginary scattering length density (Å⁻²)        |

### Quick Migration Examples

```python
# ❌ OLD (deprecated, but still works)
print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")     # Emits warning
print(f"Attenuation: {result.Attenuation_Length[0]:.1f} cm")  # Emits warning
print(f"MW: {result.MW:.2f} g/mol")                           # Emits warning

# ✅ NEW (recommended)
print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
print(f"Attenuation: {result.attenuation_length_cm[0]:.1f} cm")
print(f"MW: {result.molecular_weight_g_mol:.2f} g/mol")
```

### Suppressing Deprecation Warnings (Temporary)

If you need to temporarily suppress deprecation warnings during migration:

```python
import warnings

# Suppress only XRayLabTool deprecation warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                          message=".*deprecated.*")
    # Your legacy code here
    print(f"Result: {result.Critical_Angle[0]}")
```

### Migration Strategy

1. **Identify Usage**: Search your codebase for the legacy field names
2. **Update Gradually**: Replace legacy names with new ones section by section
3. **Test**: Ensure your code works with new field names
4. **Clean Up**: Remove any deprecation warning suppressions

---

## Supported Calculations

### Optical Constants
- **Dispersion coefficient (δ)**: Real part of refractive index decrement
- **Absorption coefficient (β)**: Imaginary part of refractive index decrement
- **Complex refractive index**: n = 1 - δ - iβ

### Scattering Factors
- **f1, f2**: Atomic scattering factors from CXRO/NIST databases
- **Total scattering factors**: Sum over all atoms in the formula

### Derived Quantities
- **Critical angle**: Total external reflection angle
- **Attenuation length**: 1/e penetration depth
- **Scattering length density (SLD)**: Real and imaginary parts

---

## Scientific Background

XRayLabTool uses atomic scattering factor data from the [Center for X-ray Optics (CXRO)](https://henke.lbl.gov/optical_constants/) and NIST databases. The calculations are based on:

1. **Atomic Scattering Factors**: Henke, Gullikson, and Davis tabulations
2. **Optical Constants**: Classical dispersion relations
3. **Critical Angles**: Fresnel reflection theory
4. **Attenuation**: Beer-Lambert law

### Key Equations

- **Refractive Index**: n = 1 - δ - iβ
- **Dispersion**: δ = (r₀λ²/2π) × ρₑ × f₁
- **Absorption**: β = (r₀λ²/2π) × ρₑ × f₂
- **Critical Angle**: θc = √(2δ)

Where r₀ is the classical electron radius, λ is wavelength, and ρₑ is electron density.

---

## Performance Features

XRayLabTool has been extensively optimized for high-performance calculations. Here are the key performance improvements:

### Performance Cache System

#### Preloaded Atomic Data Cache
- 92 elements preloaded for instant access to atomic data
- Eliminates database queries to Mendeleev for common elements
- Fast access for Si, O, Al, Fe, and other common elements
- Fallback to Mendeleev for uncommon elements with runtime caching

```python
# Check cache statistics
from xraylabtool.atomic_data_cache import get_cache_stats
print(get_cache_stats())
# {'preloaded_elements': 92, 'runtime_cached_elements': 0, 'total_cached_elements': 92}
```

#### Advanced Caching Infrastructure
- Interpolator caching: Reuses PCHIP interpolators across calculations
- LRU caches: Memory management for frequently accessed data
- Bulk loading: Optimized atomic data loading for multiple elements

### Vectorized Mathematical Operations

#### Matrix Operations for Multi-Element Materials
- Vectorized computations: Matrix operations instead of loops for multi-element materials
- NumPy optimizations: Proper dtypes and memory-contiguous arrays
- Batch interpolation: Process multiple elements simultaneously
- Faster mathematical computations compared to previous versions

#### Smart Single vs Multi-Element Optimization
```python
# Single element materials use optimized direct computation
result_single = xlt.calculate_single_material_properties("Si", energies, 2.33)

# Multi-element materials use vectorized matrix operations
result_multi = xlt.calculate_single_material_properties("SiO2", energies, 2.2)
```

### Memory-Efficient Batch Processing

#### High-Performance Batch API
For large-scale calculations, use the optimized batch processor:

```python
from xraylabtool.batch_processor import calculate_batch_properties, BatchConfig

# Configure for optimal performance
config = BatchConfig(
    chunk_size=100,        # Process in chunks of 100
    max_workers=8,         # Use 8 parallel workers
    memory_limit_gb=4.0,   # Limit memory usage
    enable_progress=True   # Show progress bar
)

# Process large batches efficiently
formulas = ["SiO2", "Al2O3", "Fe2O3"] * 100  # 300 materials
energies = np.linspace(5, 15, 50)            # 50 energy points
densities = [2.2, 3.95, 5.24] * 100

results = calculate_batch_properties(formulas, energies, densities, config)
```

#### Memory Management Features
- Chunked processing: Handles datasets larger than available RAM
- Automatic garbage collection: Prevents memory leaks during large calculations
- Memory monitoring: Real-time memory usage tracking
- Progress tracking: Visual feedback for long-running calculations

### Performance Benchmarks

#### Real-World Performance (Modern Hardware)

**Single Material Calculations:**
- Single energy point: ~0.03 ms
- 100 energy points: ~0.3 ms
- 1000 energy points: ~3 ms

**Batch Processing:**
- High throughput for multiple materials
- 50 materials × 50 energies = 2,500 calculations in ~17ms
- Average: 0.33 ms per material

**Memory Efficiency:**
- 150 materials × 100 energies = 15,000 calculations
- Memory usage: <1 MB additional RAM
- No memory leaks during extended calculations

#### Performance Comparison

| Operation | Before Optimization | After Optimization | Improvement |
|-----------|--------------------|--------------------|-------------|
| Atomic data access | ~200ms (DB query) | ~0.001ms (cache) | 200,000x |
| Single calculation | ~1.07s | ~0.003s | 350x |
| Mathematical ops | Baseline | Vectorized | 2-3x |
| Memory usage | High allocation | Chunked/optimized | 5-10x |
| Batch processing | Sequential | Parallel+chunked | 5-15x |

### Performance Best Practices

#### For Maximum Speed
```python
# 1. Use common elements (preloaded in cache)
common_materials = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]  # ✅ Fast
uncommon_materials = ["Uuo", "Fl", "Mc"]  # ⚠️ Slower (Mendeleev fallback)

# 2. Reuse energy arrays when possible
energies = np.linspace(5, 15, 100)
for formula in formulas:
    result = xlt.calculate_single_material_properties(formula, energies, density)

# 3. Use batch processing for many materials
results = xlt.calculate_xray_properties(formulas, energies, densities)  # ✅ Parallel

# Instead of:
# results = {f: xlt.calculate_single_material_properties(f, energies, d)
#           for f, d in zip(formulas, densities)}  # ❌ Sequential
```

#### For Large Datasets
```python
# Use the optimized batch processor for very large datasets
from xraylabtool.batch_processor import calculate_batch_properties, BatchConfig

# Configure for your system
config = BatchConfig(
    chunk_size=min(100, len(formulas) // 4),  # Adapt to dataset size
    max_workers=os.cpu_count() // 2,          # Use half of CPU cores
    memory_limit_gb=8.0,                      # Set appropriate memory limit
    enable_progress=True                       # Monitor progress
)

results = calculate_batch_properties(formulas, energies, densities, config)
```

### Performance Monitoring

```python
# Monitor cache performance
from xraylabtool.atomic_data_cache import get_cache_stats, is_element_preloaded

print(f"Cache stats: {get_cache_stats()}")
print(f"Silicon preloaded: {is_element_preloaded('Si')}")  # True
print(f"Unobtainium preloaded: {is_element_preloaded('Uo')}")  # False

# Monitor memory usage during batch processing
from xraylabtool.batch_processor import MemoryMonitor

monitor = MemoryMonitor(limit_gb=4.0)
print(f"Current memory usage: {monitor.get_memory_usage_mb():.1f} MB")
print(f"Within limits: {monitor.check_memory()}")
```

---

## Testing and Validation

XRayLabTool includes a comprehensive test suite with:

- **Unit Tests**: Individual function validation
- **Integration Tests**: End-to-end workflows
- **Physics Tests**: Consistency with known relationships
- **Performance Tests**: Regression monitoring
- **Robustness Tests**: Edge cases and error handling

Run tests with:
```bash
pytest tests/ -v
```

---

## API Reference

### Main Functions

#### `calculate_single_material_properties(formula, energy, density)`
Calculate X-ray properties for a single material.

**Parameters:**
- `formula` (str): Chemical formula
- `energy` (float/array): X-ray energies in keV
- `density` (float): Mass density in g/cm³

**Returns:** `XRayResult` object

#### `calculate_xray_properties(formulas, energies, densities)`
Calculate X-ray properties for multiple materials.

**Parameters:**
- `formulas` (List[str]): List of chemical formulas
- `energies` (float/array): X-ray energies in keV
- `densities` (List[float]): Mass densities in g/cm³

**Returns:** `Dict[str, XRayResult]`

### Utility Functions

- `energy_to_wavelength(energy)`: Convert energy (keV) to wavelength (Å)
- `wavelength_to_energy(wavelength)`: Convert wavelength (Å) to energy (keV)
- `parse_formula(formula)`: Parse chemical formula into elements and counts
- `get_atomic_number(symbol)`: Get atomic number for element symbol
- `get_atomic_weight(symbol)`: Get atomic weight for element symbol

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **CXRO**: Atomic scattering factor databases
- **NIST**: Reference data and validation
- **NumPy/SciPy**: Scientific computing foundation

---

## Documentation & Support

### 📖 Documentation
- **Main README**: Overview and Python API examples
- **CLI Reference**: [CLI_REFERENCE.md](CLI_REFERENCE.md) - Comprehensive command-line interface guide
- **Virtual Environment Setup**: [VIRTUAL_ENV.md](VIRTUAL_ENV.md) - Development environment setup
- **Changelog**: [CHANGELOG.md](CHANGELOG.md) - Version history and updates
- **Online Docs**: [https://pyxraylabtool.readthedocs.io](https://pyxraylabtool.readthedocs.io)

### 🔍 Getting Help
- **Issues**: [GitHub Issues](https://github.com/imewei/pyXRayLabTool/issues) - Bug reports and feature requests
- **Discussions**: [GitHub Discussions](https://github.com/imewei/pyXRayLabTool/discussions) - Questions and community support
- **CLI Help**: `xraylabtool --help` or `xraylabtool <command> --help` for command-specific help

---

## Citation

If you use XRayLabTool in your research, please cite:

```bibtex
@software{xraylabtool,
  title = {XRayLabTool: High-Performance X-ray Optical Properties Calculator},
  author = {Wei Chen},
  url = {https://github.com/imewei/pyXRayLabTool},
  year = {2024},
  version = {0.1.10}
}
```

---

<!-- SEO Meta Tags -->
<!--
Primary Keywords: X-ray optical properties, atomic scattering factors, synchrotron calculations, Python X-ray tools
Secondary Keywords: CXRO NIST database, X-ray reflectometry, materials characterization, critical angle calculator
Long-tail Keywords: high-performance X-ray property calculator, Python synchrotron beamline tools, X-ray diffraction analysis software
-->

<!-- GitHub Topics for Discovery -->
<!-- Topics: xray, synchrotron, crystallography, materials-science, optics, physics, scientific-computing, python-package, cli-tool -->
