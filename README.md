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
- [GUI](#gui)
- [GUI UX Checklist](#gui-ux-checklist)
- [Performance Features](#performance-features)
- [Testing and Validation](#testing-and-validation)
- [API Reference](#api-reference)
- [Documentation & Support](#documentation--support)
- [Citation](#citation)

## Installation

### Using uv (Recommended)

`uv` provides a fast, reliable, and deterministic environment for XRayLabTool.

```bash
# 1. Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install from PyPI
uv pip install xraylabtool[all]

# OR: Development Setup
git clone https://github.com/imewei/pyXRayLabTool.git
cd pyXRayLabTool
uv sync --all-extras
```

### Using pip (Standard)

```bash
pip install xraylabtool[all]
```

### Shell Completion Setup (Virtual Environment-Centric)

XRayLabTool features a modern virtual environment-centric completion system that automatically activates/deactivates with your Python environments.

#### Quick Setup
```bash
# Install in current virtual environment
xraylabtool completion install

# List all environments with completion status
xraylabtool completion list

# Show status for current environment
xraylabtool completion status
```

#### Key Features
- **Virtual Environment Isolation**: Completion only available when environment is active
- **Multiple Environment Support**: venv, conda, Poetry, Pipenv
- **Multi-Shell Support**: Native completion for bash, zsh, fish, PowerShell
- **No System-Wide Changes**: No sudo required, environment-specific installation
- **Auto-Activation**: Completion activates/deactivates with environment changes

#### Installation Commands
```bash
# New completion system (recommended)
xraylabtool completion install              # Current environment, auto-detect shell
xraylabtool completion install --shell zsh  # Install for specific shell
xraylabtool completion list                 # List all environments
xraylabtool completion status               # Show current environment status
xraylabtool completion uninstall            # Remove from current environment
xraylabtool completion uninstall --all      # Remove from all environments
xraylabtool completion info                 # Show system information

# Legacy commands (still supported)
xraylabtool install-completion              # Uses new system backend
xraylabtool uninstall-completion            # Uses new system backend
```

#### Prerequisites by Shell

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

#### Virtual Environment Workflow
```bash
# 1. Activate your environment
conda activate myproject
# or: source venv/bin/activate
# or: poetry shell

# 2. Install completion in the environment
xraylabtool completion install

# 3. Completion is now available when environment is active
xraylabtool <TAB>  # Shows available commands

# 4. Deactivate environment - completion automatically unavailable
conda deactivate
xraylabtool <TAB>  # No completion (unless installed in base environment)
```

> **Migration Note**: If you previously used system-wide completion, the new system provides better isolation and no longer requires sudo. Run `xraylabtool completion install` in each virtual environment where you want completion available.

### Requirements

- **Python** ≥ 3.12
- **NumPy** ≥ 1.20.0
- **SciPy** ≥ 1.7.0
- **Pandas** ≥ 2.3.3
- **Mendeleev** ≥ 0.10.0
- **tqdm** ≥ 4.60.0
- **matplotlib** ≥ 3.10.8 (optional, for plotting)

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

## Graphical User Interface (GUI)

- Requirements: PySide6 and matplotlib (install via `pip install xraylabtool[all]` or ensure those are present in your venv).
- Launch: `python -m xraylabtool.gui`
- Tabs: **Single Material Analysis** (one material with presets, linear/log energy grids, plots, CSV/PNG export) and **Multiple Materials Comparison** (add/remove materials, presets, log/linear grids, overlay plots, CSV/PNG export).
- Headless smoke/CI: `python -m xraylabtool.gui --test-launch --platform offscreen` (offscreen Qt backend).

### Quick Walkthrough
- **Single tab:** enter formula + density, choose linear or log energy grid, optional presets; compute to see summary table, energy sweep plots, f1/f2 curves, and export plot/CSV.
- **Multiple tab:** add materials (manual or presets), set energy grid, pick property to compare; compute to see overlay plot and summary/comparator tables; export plot/CSV (per-property and comparator table).

![GUI overview](docs/_static/gui_main_offscreen.png)

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

# Install shell completion in current environment
xraylabtool completion install

# Verify completion status
xraylabtool completion status
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
| `completion` | Manage virtual environment completion | `xraylabtool completion install` |
| `install-completion` | Install shell completion (legacy) | `xraylabtool install-completion` |
| `uninstall-completion` | Remove shell completion (legacy) | `xraylabtool uninstall-completion` |

### Shell Completion Usage

The new virtual environment-centric completion system provides better isolation and management:

```bash
# New completion system (recommended)
xraylabtool completion install              # Install in current environment
xraylabtool completion install --shell zsh  # Install for specific shell
xraylabtool completion list                 # List all environments with status
xraylabtool completion status               # Show current environment status
xraylabtool completion uninstall            # Remove from current environment
xraylabtool completion uninstall --all      # Remove from all environments
xraylabtool completion info                 # Show system information

# Legacy commands (still supported via new backend)
xraylabtool install-completion              # Install in current environment
xraylabtool uninstall-completion            # Remove from current environment

# Flag syntax (legacy compatibility)
xraylabtool --install-completion            # Install in current environment
```

> **Virtual Environment Benefits**: The new system installs completion per environment, so it's only available when the relevant environment is active. This eliminates conflicts and provides better project isolation.

**Tab Completion Features:**
- **Command completion**: Complete all available commands including new `completion` command
- **Option completion**: Complete command-line options and flags
- **File path completion**: Complete file paths for input/output files
- **Chemical formulas**: Complete common chemical formulas
- **Energy values**: Complete common X-ray energies (8.048, 10.0, 12.4 keV)
- **Environment awareness**: Completion only active when virtual environment is active

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
- **Virtual Environment-Centric Completion**: Modern completion system that activates/deactivates with environments
  - **Multi-Shell Support**: Native completion for bash, zsh, fish, and PowerShell
  - **Environment Isolation**: Completion only available when virtual environment is active
  - **Context-aware**: Suggests values based on current command
  - **File completion**: Complete file paths for input/output files
  - **Chemical formulas**: Complete common materials and elements
  - **Energy values**: Complete X-ray energies (Cu Kα, Mo Kα, etc.)
  - **Cross-platform**: Works on macOS, Linux, and Windows (with WSL/Cygwin)

### CLI Help and Documentation

Get help for any command:

```bash
# General help
xraylabtool --help

# Command-specific help
xraylabtool calc --help
xraylabtool batch --help
xraylabtool completion --help
xraylabtool install-completion --help

# List available options and examples
xraylabtool list --help
```

**CLI Features:**
- 10+ commands for X-ray analysis and completion management
- Energy input formats: Single values, ranges, lists, and logarithmic spacing
- Batch processing from CSV files
- Output formats: Table, CSV, and JSON
- Virtual environment-centric shell completion for bash, zsh, fish, and PowerShell
- Cross-platform support with environment isolation

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

> **Note**: Legacy field names (e.g., `Formula`, `MW`, `Critical_Angle`) are supported for backward compatibility but emit deprecation warnings. Use the new descriptive field names for clearer code.

---

## Usage Examples

### Recommended: Using New Field Names

```python
# Calculate properties for silicon dioxide at 10 keV
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

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

### Plotting Example

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

To help users transition from legacy CamelCase field names to the new descriptive snake_case names, here's a mapping:

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
import xraylabtool as xlt

# Calculate result first
result = xlt.calculate_single_material_properties("Si", 8.0, 2.33)

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

XRayLabTool v0.2.5 includes performance optimizations that reduce cold start times and improve cache efficiency.

### Smart Cache Warming (v0.2.5)

Smart cache warming loads only the atomic data needed for a specific calculation instead of all priority elements.

```python
# Automatic smart warming - loads only Si and O for SiO2
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

# First calculation triggers warming, subsequent calculations reuse cache
result2 = xlt.calculate_single_material_properties("SiO2", 12.0, 2.2)  # Fast
```

**Smart cache features:**
- Formula-specific element loading
- 90% faster cold start than v0.2.4
- Background priority warming for complex cases
- Automatic fallback to full warming when needed

### Adaptive Batch Processing (v0.2.5)

Batch processing automatically switches between sequential and parallel modes based on workload size.

```python
# Small batches (<20 items) use sequential processing
small_batch = ["Si", "SiO2", "Al2O3"]  # 3 items - sequential
energies = [10.0] * 3
densities = [2.33, 2.2, 3.95]
results = xlt.calculate_xray_properties(small_batch, energies, densities)

# Large batches (≥20 items) use parallel processing
large_batch = ["Si"] * 25  # 25 items - parallel with ThreadPoolExecutor
energies = [10.0] * 25
densities = [2.33] * 25
results = xlt.calculate_xray_properties(large_batch, energies, densities)
```

**Adaptive processing features:**
- 20-item threshold for parallel activation
- Optimal CPU utilization for small and large workloads
- Reduces overhead for small calculations
- Maximizes throughput for large datasets

### Environment-Controlled Features (v0.2.5)

Performance monitoring features are disabled by default and can be enabled via environment variables.

```bash
# Enable cache metrics tracking
export XRAYLABTOOL_CACHE_METRICS=true

# Enable memory profiling
export XRAYLABTOOL_MEMORY_PROFILING=true
```

```python
# Check cache statistics when enabled
from xraylabtool.data_handling.cache_metrics import get_cache_stats

# Returns {} when disabled (default), stats when enabled
stats = get_cache_stats()
print(stats)  # {'hits': 45, 'misses': 5, 'total': 50, 'hit_rate': 0.9}
```

**Environment controls:**
- `XRAYLABTOOL_CACHE_METRICS`: Enable/disable cache statistics
- `XRAYLABTOOL_MEMORY_PROFILING`: Enable/disable memory profiling
- Disabled by default for maximum performance
- Enable only when debugging or optimizing

### Memory Optimizations (v0.2.5)

Memory profiling structures use lazy initialization and are only created when needed.

```python
# Memory profiling structures are None until activated
from xraylabtool.optimization.memory_profiler import _memory_snapshots
print(_memory_snapshots)  # None (lazy loading)

# Only initialized when memory profiling is enabled
# No memory overhead when disabled
```

### Performance Benchmarks (v0.2.5)

#### Cold Start Performance
- v0.2.3: ~60ms baseline
- v0.2.4: ~912ms (15x regression)
- v0.2.5: ~130ms (86% improvement from v0.2.4)

#### Cache Efficiency
- v0.2.3: 13x speedup baseline
- v0.2.4: 8.5x speedup (degradation)
- v0.2.5: 13.4x speedup (exceeds target)

#### Batch Processing
- v0.2.3: ~7ms baseline
- v0.2.4: ~20ms (regression)
- v0.2.5: ~1.7ms (exceeds baseline)

#### Memory Usage
- v0.2.3: ~0.006MB baseline
- v0.2.4: ~2.31MB (bloat)
- v0.2.5: ~0MB (minimal overhead)

### Legacy Cache System

For compatibility, the preloaded atomic data cache is still available:

```python
# Check cache statistics (legacy)
from xraylabtool.data_handling import get_cache_stats
print(get_cache_stats())
# {'preloaded_elements': 92, 'runtime_cached_elements': 0, 'total_cached_elements': 92}
```

**Legacy cache features:**
- 92 elements preloaded for instant access
- PCHIP interpolator caching
- LRU memory management
- Bulk atomic data loading

### Performance Best Practices

#### Maximum Speed Configuration
```python
# 1. Use smart cache warming (automatic in v0.2.5)
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

# 2. Disable metrics for production (default in v0.2.5)
# No environment variables needed - metrics disabled by default

# 3. Use adaptive batch processing (automatic threshold)
# Small batches: sequential, large batches: parallel

# 4. Reuse calculations when possible
result1 = xlt.calculate_single_material_properties("Si", 10.0, 2.33)  # Warms cache
result2 = xlt.calculate_single_material_properties("Si", 12.0, 2.33)  # Fast
```

#### Debugging Performance
```bash
# Enable metrics only when debugging
export XRAYLABTOOL_CACHE_METRICS=true
export XRAYLABTOOL_MEMORY_PROFILING=true
```

```python
# Monitor performance with metrics enabled
from xraylabtool.data_handling.cache_metrics import get_cache_stats
stats = get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
```

---

## Testing and Validation

XRayLabTool includes a test suite with:

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
- **NumPy/SciPy**: Scientific computing libraries

---

## Documentation & Support

### Documentation
- **Main README**: Overview and Python API examples
- **Performance Guide**: [PERFORMANCE.md](PERFORMANCE.md) - v0.2.5 optimization features and benchmarks
- **CLI Reference**: [CLI_REFERENCE.md](CLI_REFERENCE.md) - Comprehensive command-line interface guide
- **Virtual Environment Setup**: [VIRTUAL_ENV.md](VIRTUAL_ENV.md) - Development environment setup
- **Changelog**: [CHANGELOG.md](CHANGELOG.md) - Version history and updates
- **Online Docs**: [https://pyxraylabtool.readthedocs.io](https://pyxraylabtool.readthedocs.io)

### Getting Help
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
Long-tail Keywords: fast X-ray property calculator, Python synchrotron beamline tools, X-ray diffraction analysis software
-->

<!-- GitHub Topics: xray, synchrotron, crystallography, materials-science, optics, physics, scientific-computing, python-package, cli-tool -->
