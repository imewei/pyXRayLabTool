# XRayLabTool

[![PyPI version](https://badge.fury.io/py/xraylabtool.svg)](https://badge.fury.io/py/xraylabtool)
[![Python versions](https://img.shields.io/pypi/pyversions/xraylabtool.svg)](https://pypi.org/project/xraylabtool/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/xraylabtool)](https://pepy.tech/project/xraylabtool)

**Material Property Calculations for X-ray Interactions**

`XRayLabTool` is a Python package that provides functions to calculate X-ray optical properties of materials based on their chemical formulas and densities. It is particularly useful for synchrotron scientists, materials researchers, and X-ray optics developers.

---

## 📆 Features

- Compute optical constants (δ, β), scattering factors (f1, f2), and other X-ray interaction parameters
- Support for both single and multiple material calculations
- Easy-to-use dataclass-based output
- Based on CXRO/NIST data tables
- Vectorized calculations using NumPy for high performance
- Built-in caching system for atomic scattering factor data
- **NEW**: Enhanced robustness with complex number handling
- **NEW**: Improved type safety and error handling
- **NEW**: Updated pandas compatibility for modern versions
- **NEW**: PCHIP interpolation for accurate scattering factor calculations
- **NEW**: Comprehensive test suite with 100% coverage

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

- Python ≥ 3.12
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Pandas ≥ 1.3.0
- Mendeleev ≥ 0.10.0
- tqdm ≥ 4.60.0
- matplotlib ≥ 3.4.0 (optional, for plotting)

---

## 🚀 Quick Start

### Single Material

```python
import xraylabtool as xlt

# Calculate properties for quartz at multiple X-ray energies
result = xlt.SubRefrac("SiO2", [8.0, 10.0, 12.0], 2.2)
print(f"Molecular weight: {result.MW:.2f} g/mol")
print(f"Critical angles: {result.Critical_Angle} degrees")
```

### Multiple Materials

```python
import xraylabtool as xlt

# Compare properties of different materials
formulas = ["SiO2", "Al2O3", "Fe2O3"]
densities = [2.2, 3.95, 5.24]
energies = [8.0, 10.0, 12.0]

results = xlt.Refrac(formulas, energies, densities)
for formula, result in results.items():
    print(f"{formula}: MW = {result.MW:.2f} g/mol")
```

### Accessing Results

```python
# Access individual properties
result = xlt.SubRefrac("SiO2", 10.0, 2.2)
print(f"Formula: {result.Formula}")
print(f"Molecular Weight: {result.MW:.2f} g/mol")
print(f"Electron Density: {result.Electron_Density:.4f} electrons/Å³")
print(f"Dispersion coefficient: {result.Dispersion[0]:.2e}")
print(f"Absorption coefficient: {result.Absorption[0]:.2e}")
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

- `Formula: str` – Chemical formula
- `MW: float` – Molecular weight (g/mol)
- `Number_Of_Electrons: float` – Total electrons per molecule
- `Density: float` – Mass density (g/cm³)
- `Electron_Density: float` – Electron density (electrons/Å³)
- `Energy: np.ndarray` – X-ray energies (keV)
- `Wavelength: np.ndarray` – X-ray wavelengths (Å)
- `Dispersion: np.ndarray` – Dispersion coefficient δ
- `Absorption: np.ndarray` – Absorption coefficient β
- `f1: np.ndarray` – Real part of atomic scattering factor
- `f2: np.ndarray` – Imaginary part of atomic scattering factor
- `Critical_Angle: np.ndarray` – Critical angles (degrees)
- `Attenuation_Length: np.ndarray` – Attenuation lengths (cm)
- `reSLD: np.ndarray` – Real scattering length density (Å⁻²)
- `imSLD: np.ndarray` – Imaginary scattering length density (Å⁻²)

---

## 📘 Detailed Examples

### Basic Usage

```python
import xraylabtool as xlt
import numpy as np

# Single energy calculation
result = xlt.SubRefrac("SiO2", 10.0, 2.33)
print(f"Formula: {result.Formula}")                    # "SiO2"
print(f"Molecular weight: {result.MW:.2f} g/mol")     # 60.08 g/mol
print(f"Dispersion: {result.Dispersion[0]:.2e}")       # δ value
print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")  # θc
```

### Energy Range Scan

```python
# Energy range with numpy
energies = np.linspace(8.0, 12.0, 21)  # 21 points from 8-12 keV
result = xlt.SubRefrac("SiO2", energies, 2.33)

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
    "Diamond": 3.52,  # Diamond
}

formulas = list(materials.keys())
densities = list(materials.values())
energy = 10.0  # keV (Cu Kα)

results = xlt.Refrac(formulas, energy, densities)

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
result = xlt.SubRefrac("Si", energies, 2.33)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot optical constants
ax1.loglog(result.Energy, np.abs(result.Dispersion), label='|δ|')
ax1.loglog(result.Energy, result.Absorption, label='β')
ax1.set_xlabel('Energy (keV)')
ax1.set_ylabel('Optical constants')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot critical angle
ax2.loglog(result.Energy, result.Critical_Angle)
ax2.set_xlabel('Energy (keV)')
ax2.set_ylabel('Critical angle (°)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 🔗 References

- [CXRO - Center for X-ray Optics](http://www.cxro.lbl.gov)
- [NIST - National Institute of Standards and Technology](http://www.nist.gov)

> This Python package is ported from a Julia module, which was translated from a MATLAB script originally developed by **Zhang Jiang** at the **Advanced Photon Source**, Argonne National Laboratory.

---

## 🔄 Recent Updates

### Version 0.1.1 (Latest)

**Enhanced Robustness & Compatibility:**
- ✅ Fixed complex number handling in energy conversion functions
- ✅ Improved type safety with comprehensive type hints and checking
- ✅ Updated pandas method calls for modern compatibility (`fillna` → `bfill`/`ffill`)
- ✅ Enhanced atomic data handling with robust type conversions
- ✅ Fixed numpy deprecation warnings (`trapz` → `trapezoid`)

**New Features:**
- ✅ PCHIP interpolation for atomic scattering factors
- ✅ Enhanced caching system for scattering factor data
- ✅ Comprehensive error handling and validation
- ✅ Improved smooth data function with edge case handling

**Testing & Quality:**
- ✅ 100% test suite coverage with 13/13 test suites passing
- ✅ Robust integration tests matching Julia implementation
- ✅ Performance benchmarks and regression testing
- ✅ Enhanced error message clarity and debugging

**Developer Experience:**
- ✅ Improved type annotations for better IDE support
- ✅ Enhanced test utilities and robustness testing
- ✅ Better documentation and examples
- ✅ Cross-platform compatibility verified

---

## 🧪 License

MIT License
