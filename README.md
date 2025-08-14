# 🔬 XRayLabTool

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/xraylabtool.svg)](https://badge.fury.io/py/xraylabtool)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/imewei/pyXRayLabTool/workflows/Tests/badge.svg)](https://github.com/imewei/pyXRayLabTool/actions)

**High-Performance X-ray Optical Properties Calculator for Materials Science**

XRayLabTool is a comprehensive Python package for calculating X-ray optical properties of materials based on their chemical formulas and densities. Designed for synchrotron scientists, materials researchers, and X-ray optics developers, it provides fast, accurate calculations using CXRO/NIST atomic scattering factor data.

---

## 📦 Installation

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

### Requirements

- **Python** ≥ 3.12
- **NumPy** ≥ 1.20.0
- **SciPy** ≥ 1.7.0
- **Pandas** ≥ 1.3.0
- **Mendeleev** ≥ 0.10.0
- **tqdm** ≥ 4.60.0
- **matplotlib** ≥ 3.4.0 (optional, for plotting)

---

## 🚀 Quick Start

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

## 📥 Input Parameters

| Parameter    | Type                                  | Description                                                    |
| ------------ | ------------------------------------- | -------------------------------------------------------------- |
| `formula(s)` | `str` or `List[str]`                  | Case-sensitive chemical formula(s), e.g., `"CO"` vs `"Co"`     |
| `energy`     | `float`, `List[float]`, or `np.array` | X-ray photon energies in keV (valid range: **0.03–30 keV**)   |
| `density`    | `float` or `List[float]`              | Mass density in g/cm³ (one per formula)                       |

---

## 📤 Output: `XRayResult` Dataclass

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

## 💡 Usage Examples

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

## 🔄 Migration Guide: Legacy to New Field Names

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

## 🧮 Supported Calculations

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

## 🎯 Application Areas

- **Synchrotron Beamline Design**: Mirror and monochromator calculations
- **X-ray Optics**: Reflectivity and transmission analysis
- **Materials Science**: Characterization of thin films and multilayers
- **Crystallography**: Structure factor calculations
- **Small-Angle Scattering**: Contrast calculations
- **Medical Imaging**: Tissue contrast optimization

---

## 🔬 Scientific Background

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

## ⚡ Performance Features

### Caching System
- **Atomic Data Caching**: LRU cache for scattering factor files
- **Interpolator Caching**: Reuse PCHIP interpolators
- **Smart Loading**: Only load required atomic data

### Vectorization
- **NumPy Operations**: Vectorized calculations for arrays
- **Parallel Processing**: Multi-material calculations
- **Memory Efficient**: Optimized data structures

### Benchmarks
Typical performance on modern hardware:
- Single material, single energy: ~0.1 ms
- Single material, 100 energies: ~1 ms
- 10 materials, 100 energies: ~50 ms

---

## 🧪 Testing and Validation

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

## 📚 API Reference

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CXRO**: Atomic scattering factor databases
- **NIST**: Reference data and validation
- **NumPy/SciPy**: Scientific computing foundation

---

## 📞 Support

- **Documentation**: [https://xraylabtool.readthedocs.io](https://xraylabtool.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/imewei/pyXRayLabTool/issues)
- **Discussions**: [GitHub Discussions](https://github.com/imewei/pyXRayLabTool/discussions)

---

## 📈 Citation

If you use XRayLabTool in your research, please cite:

```bibtex
@software{xraylabtool,
  title = {XRayLabTool: High-Performance X-ray Optical Properties Calculator},
  author = {Wei Chen},
  url = {https://github.com/imewei/pyXRayLabTool},
  year = {2024},
  version = {0.1.5}
}
```

---

*Made with ❤️ for the X-ray science community*