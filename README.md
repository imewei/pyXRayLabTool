# 🔬 XRayLabTool

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/xraylabtool.svg)](https://badge.fury.io/py/xraylabtool)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/imewei/pyXRayLabTool/workflows/Tests/badge.svg)](https://github.com/imewei/pyXRayLabTool/actions)

**High-Performance X-ray Optical Properties Calculator for Materials Science**

XRayLabTool is a comprehensive Python package for calculating X-ray optical properties of materials based on their chemical formulas and densities. Designed for synchrotron scientists, materials researchers, and X-ray optics developers, it provides fast, accurate calculations using CXRO/NIST atomic scattering factor data.

## ✨ Key Features

- 🚀 **High Performance**: Vectorized NumPy calculations with intelligent caching
- 🎯 **Accurate**: Based on CXRO/NIST atomic scattering factor databases
- 🔧 **Easy to Use**: Simple API with dataclass-based results
- 📊 **Comprehensive**: Calculate refractive indices, critical angles, attenuation lengths, and more
- 🧪 **Materials Focus**: Support for both single materials and multi-material analysis
- 🔄 **Robust**: Enhanced error handling and type safety
- 📈 **Scalable**: Efficient parallel processing for multiple materials

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

- **Python** ≥ 3.8
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
print(f"Formula: {result.Formula}")
print(f"Molecular Weight: {result.MW:.2f} g/mol")
print(f"Critical Angle: {result.Critical_Angle[0]:.3f}°")
print(f"Attenuation Length: {result.Attenuation_Length[0]:.2f} cm")
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

# Display results
for formula, result in results.items():
    print(f"{formula:6}: θc = {result.Critical_Angle[0]:.3f}°, "
          f"δ = {result.Dispersion[0]:.2e}")
```

### Energy Range Analysis

```python
# Energy sweep for material characterization
energies = np.logspace(np.log10(1), np.log10(30), 100)  # 1-30 keV
result = xlt.calculate_single_material_properties("Si", energies, 2.33)

print(f"Energy range: {result.Energy[0]:.1f} - {result.Energy[-1]:.1f} keV")
print(f"Data points: {len(result.Energy)}")
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

The `XRayResult` dataclass contains all computed X-ray optical properties:

### Material Properties
- **`Formula: str`** – Chemical formula
- **`MW: float`** – Molecular weight (g/mol)
- **`Number_Of_Electrons: float`** – Total electrons per molecule
- **`Density: float`** – Mass density (g/cm³)
- **`Electron_Density: float`** – Electron density (electrons/Å³)

### X-ray Properties (Arrays)
- **`Energy: np.ndarray`** – X-ray energies (keV)
- **`Wavelength: np.ndarray`** – X-ray wavelengths (Å)
- **`Dispersion: np.ndarray`** – Dispersion coefficient δ
- **`Absorption: np.ndarray`** – Absorption coefficient β
- **`f1: np.ndarray`** – Real part of atomic scattering factor
- **`f2: np.ndarray`** – Imaginary part of atomic scattering factor

### Derived Quantities (Arrays)
- **`Critical_Angle: np.ndarray`** – Critical angles (degrees)
- **`Attenuation_Length: np.ndarray`** – Attenuation lengths (cm)
- **`reSLD: np.ndarray`** – Real scattering length density (Å⁻²)
- **`imSLD: np.ndarray`** – Imaginary scattering length density (Å⁻²)

---

## 💡 Usage Examples

### Single Energy Calculation

```python
# Calculate properties for silicon dioxide at 10 keV
result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.33)
print(f"Formula: {result.Formula}")                    # "SiO2"
print(f"Molecular weight: {result.MW:.2f} g/mol")     # 60.08 g/mol
print(f"Dispersion: {result.Dispersion[0]:.2e}")       # δ value
print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")  # θc
```

### Energy Range Scan

```python
# Energy range with numpy
energies = np.linspace(8.0, 12.0, 21)  # 21 points from 8-12 keV
result = xlt.calculate_single_material_properties("SiO2", energies, 2.33)

print(f"Energy range: {result.Energy[0]:.1f} - {result.Energy[-1]:.1f} keV")
print(f"Number of points: {len(result.Energy)}")
```

### Multiple Materials Analysis

```python
# Common X-ray optics materials
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

# Compare critical angles
for formula, result in results.items():
    print(f"{formula:8}: θc = {result.Critical_Angle[0]:.3f}°, "
          f"δ = {result.Dispersion[0]:.2e}")
```

### Plotting Results

```python
import matplotlib.pyplot as plt

# Energy-dependent properties
energies = np.logspace(np.log10(1), np.log10(20), 100)
result = xlt.calculate_single_material_properties("Si", energies, 2.33)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot dispersion and absorption
ax1.loglog(result.Energy, result.Dispersion, 'b-', label='δ (dispersion)')
ax1.loglog(result.Energy, result.Absorption, 'r-', label='β (absorption)')
ax1.set_xlabel('Energy (keV)')
ax1.set_ylabel('Optical constants')
ax1.legend()
ax1.grid(True)

# Plot critical angle
ax2.semilogx(result.Energy, result.Critical_Angle, 'g-')
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Critical angle (°)')
ax2.grid(True)

plt.tight_layout()
plt.show()
```

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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/imewei/pyXRayLabTool.git
cd pyXRayLabTool
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **CXRO**: Atomic scattering factor databases
- **NIST**: Reference data and validation
- **NumPy/SciPy**: Scientific computing foundation
- **Contributors**: See [CONTRIBUTORS.md](CONTRIBUTORS.md)

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
  version = {0.1.4}
}
```

---

*Made with ❤️ for the X-ray science community*