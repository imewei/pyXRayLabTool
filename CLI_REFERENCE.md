# XRayLabTool CLI Reference

**Version:** 0.1.5  
**Author:** Wei Chen  
**License:** MIT  

This document provides a comprehensive reference for all XRayLabTool command-line interface (CLI) commands, their functionalities, parameters, and usage examples.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Global Options](#global-options)
3. [Command Reference](#command-reference)
   - [calc](#calc---single-material-calculations)
   - [batch](#batch---batch-processing)
   - [convert](#convert---unit-conversions)
   - [formula](#formula---chemical-formula-analysis)
   - [atomic](#atomic---atomic-data-lookup)
   - [bragg](#bragg---diffraction-angle-calculations)
   - [list](#list---reference-information)
4. [Output Formats](#output-formats)
5. [Energy Input Formats](#energy-input-formats)
6. [Common Use Cases](#common-use-cases)
7. [Tips & Best Practices](#tips--best-practices)

---

## Installation & Setup

### Prerequisites
- Python ≥ 3.12
- Virtual environment (recommended)

### Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .[dev]

# Verify installation
xraylabtool --version
```

### Quick Test
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2
```

---

## Global Options

These options are available for the main `xraylabtool` command:

| Option | Description |
|--------|-------------|
| `--help`, `-h` | Show help message and exit |
| `--version` | Show program version and exit |
| `--verbose`, `-v` | Enable verbose output (global) |

**Example:**
```bash
xraylabtool --version
# Output: XRayLabTool 0.1.5
```

---

## Command Reference

## `calc` - Single Material Calculations

Calculate X-ray optical properties for a single material composition.

### Syntax
```bash
xraylabtool calc FORMULA -e ENERGY -d DENSITY [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `FORMULA` | Chemical formula (case-sensitive) | `SiO2`, `Al2O3`, `Fe2O3` |
| `-e, --energy ENERGY` | X-ray energy in keV | `10.0`, `5,10,15`, `5-15:11` |
| `-d, --density DENSITY` | Material density in g/cm³ | `2.2`, `3.95` |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `-o, --output OUTPUT` | Output filename | Console | `results.csv`, `data.json` |
| `--format FORMAT` | Output format | `table` | `table`, `csv`, `json` |
| `--fields FIELDS` | Comma-separated field list | All fields | `formula,energy_kev,dispersion_delta` |
| `--precision PRECISION` | Decimal places | `6` | `3`, `8` |

### Energy Input Formats

| Format | Description | Example |
|--------|-------------|---------|
| Single value | Single energy point | `10.0` |
| Comma-separated | Multiple discrete energies | `5.0,10.0,15.0,20.0` |
| Linear range | Linearly spaced points | `5-15:11` (11 points from 5 to 15 keV) |
| Logarithmic range | Log-spaced points | `1-30:100:log` (100 log-spaced points) |

### Output Fields

The calculation returns the following properties:

#### Material Properties
- `formula` — Chemical formula string
- `molecular_weight_g_mol` — Molecular weight (g/mol)
- `total_electrons` — Total electrons per molecule
- `density_g_cm3` — Mass density (g/cm³)
- `electron_density_per_ang3` — Electron density (electrons/Å³)

#### X-ray Properties (Arrays)
- `energy_kev` — X-ray energies (keV)
- `wavelength_angstrom` — X-ray wavelengths (Å)
- `dispersion_delta` — Dispersion coefficient δ
- `absorption_beta` — Absorption coefficient β
- `scattering_factor_f1` — Real atomic scattering factor
- `scattering_factor_f2` — Imaginary atomic scattering factor
- `critical_angle_degrees` — Critical angles (degrees)
- `attenuation_length_cm` — Attenuation lengths (cm)
- `real_sld_per_ang2` — Real scattering length density (Å⁻²)
- `imaginary_sld_per_ang2` — Imaginary scattering length density (Å⁻²)

### Examples

#### Basic Calculation
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2
```
**Output:**
```
Material Properties:
  Formula: SiO2
  Molecular Weight: 60.083000 g/mol
  Total Electrons: 30.000000
  Density: 2.200000 g/cm³
  Electron Density: 6.615205e-01 electrons/Å³

X-ray Properties:
  Energy: 10.000000 keV
  Wavelength: 1.239842 Å
  Dispersion (δ): 4.613309e-06
  Absorption (β): 3.887268e-08
  Critical Angle: 0.174038°
  Attenuation Length: 0.025381 cm
```

#### Multiple Energies
```bash
xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33
```

#### Energy Range (Linear)
```bash
xraylabtool calc Al2O3 -e 5-15:11 -d 3.95
```

#### Energy Range (Logarithmic)
```bash
xraylabtool calc C -e 1-30:100:log -d 3.52
```

#### CSV Output
```bash
xraylabtool calc SiO2 -e 8.0,10.0,12.0 -d 2.2 -o results.csv --format csv
```

#### JSON Output with Selected Fields
```bash
xraylabtool calc Si -e 10.0 -d 2.33 \
  --format json \
  --fields formula,energy_kev,dispersion_delta,critical_angle_degrees \
  -o results.json
```

#### High Precision Output
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2 --precision 10
```

---

## `batch` - Batch Processing

Process multiple materials from a CSV input file with support for parallel processing.

### Syntax
```bash
xraylabtool batch INPUT_FILE -o OUTPUT_FILE [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `INPUT_FILE` | Input CSV file path | `materials.csv` |
| `-o, --output OUTPUT` | Output filename | `results.csv`, `batch.json` |

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--format FORMAT` | Output format | Auto-detect | `csv`, `json` |
| `--workers WORKERS` | Number of parallel workers | Auto | `4`, `8` |
| `--fields FIELDS` | Output field selection | All fields | `formula,energy_kev,dispersion_delta` |

### Input CSV Format

The input CSV file must have the following columns:

| Column | Description | Required | Example |
|--------|-------------|----------|---------|
| `formula` | Chemical formula | Yes | `SiO2`, `Al2O3` |
| `density` | Mass density (g/cm³) | Yes | `2.2`, `3.95` |
| `energy` | Energy specification | Yes | `10.0` or `"5.0,10.0,15.0"` |

### Example Input CSV
```csv
formula,density,energy
SiO2,2.2,10.0
Al2O3,3.95,"5.0,10.0,15.0"
Si,2.33,8.0
Fe2O3,5.24,"8.0,12.0"
C,3.52,15.0
```

### Examples

#### Basic Batch Processing
```bash
# Create input file
cat > materials.csv << EOF
formula,density,energy
SiO2,2.2,10.0
Si,2.33,8.0
Al2O3,3.95,10.0
EOF

# Process batch
xraylabtool batch materials.csv -o results.csv
```

#### JSON Output
```bash
xraylabtool batch materials.csv -o results.json --format json
```

#### Parallel Processing
```bash
xraylabtool batch large_materials.csv -o results.csv --workers 8
```

#### Selected Fields Output
```bash
xraylabtool batch materials.csv -o results.csv \
  --fields formula,energy_kev,dispersion_delta,critical_angle_degrees
```

---

## `convert` - Unit Conversions

Convert between X-ray energy (keV) and wavelength (Å) units.

### Syntax
```bash
xraylabtool convert INPUT_UNIT VALUES --to OUTPUT_UNIT [OPTIONS]
```

### Required Parameters

| Parameter | Description | Options | Example |
|-----------|-------------|---------|---------|
| `INPUT_UNIT` | Input unit type | `energy`, `wavelength` | `energy` |
| `VALUES` | Values to convert | Comma-separated | `10.0` or `5.0,10.0,15.0` |
| `--to OUTPUT_UNIT` | Output unit type | `energy`, `wavelength` | `wavelength` |

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-o, --output OUTPUT` | Save to CSV file | `conversions.csv` |

### Examples

#### Energy to Wavelength
```bash
xraylabtool convert energy 10.0 --to wavelength
```
**Output:**
```
Energy to Wavelength Conversion:
----------------------------------------
   10.0000 →     1.2398 Å
```

#### Wavelength to Energy
```bash
xraylabtool convert wavelength 1.24 --to energy
```
**Output:**
```
Wavelength to Energy Conversion:
----------------------------------------
    1.2400 →    10.0000 keV
```

#### Multiple Values
```bash
xraylabtool convert energy 5.0,10.0,15.0,20.0 --to wavelength
```

#### Save to File
```bash
xraylabtool convert energy 5.0,10.0,15.0,20.0 --to wavelength -o conversions.csv
```

---

## `formula` - Chemical Formula Analysis

Parse and analyze chemical formulas to show elemental composition and atomic data.

### Syntax
```bash
xraylabtool formula FORMULAS [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `FORMULAS` | Chemical formula(s) | `SiO2` or `SiO2,Al2O3,Fe2O3` |

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-o, --output OUTPUT` | Save to JSON file | `formulas.json` |

### Examples

#### Single Formula
```bash
xraylabtool formula SiO2
```
**Output:**
```
Formula: SiO2
Elements: Si, O
Counts: 1.0, 2.0
Total atoms: 3.0
```

#### Multiple Formulas
```bash
xraylabtool formula SiO2,Al2O3,Fe2O3
```

#### Save to File
```bash
xraylabtool formula SiO2,Al2O3 -o formulas.json
```

#### Complex Formula
```bash
xraylabtool formula Ca10P6O26H2
```

---

## `atomic` - Atomic Data Lookup

Look up atomic numbers, weights, and other properties for chemical elements.

### Syntax
```bash
xraylabtool atomic ELEMENTS [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ELEMENTS` | Element symbol(s) | `Si` or `H,C,N,O,Si` |

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-o, --output OUTPUT` | Output file | `atomic.csv`, `atomic.json` |

### Output Fields

- `element` — Element symbol
- `atomic_number` — Atomic number (Z)
- `atomic_weight` — Atomic weight (u)

### Examples

#### Single Element
```bash
xraylabtool atomic Si
```
**Output:**
```
Atomic Data:
------------------------------
 Element   Z     MW (u)
------------------------------
      Si  14     28.085
```

#### Multiple Elements
```bash
xraylabtool atomic H,C,N,O,Si
```

#### Save to CSV
```bash
xraylabtool atomic Si,Al,Fe -o atomic_data.csv
```

#### Save to JSON
```bash
xraylabtool atomic H,C,N,O,Si -o atomic_data.json
```

---

## `bragg` - Diffraction Angle Calculations

Calculate Bragg diffraction angles using Bragg's law: nλ = 2d sin(θ).

### Syntax
```bash
xraylabtool bragg -d DSPACING (-w WAVELENGTH | -e ENERGY) [OPTIONS]
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-d, --dspacing DSPACING` | d-spacing in Å | `3.14` or `3.14,2.45,1.92` |
| `-w, --wavelength WAVELENGTH` | X-ray wavelength in Å | `1.54` |
| `-e, --energy ENERGY` | X-ray energy in keV | `8.0` |

**Note:** Must specify either `-w` OR `-e`, not both.

### Optional Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--order ORDER` | Diffraction order (n) | `1` | `2`, `3` |
| `-o, --output OUTPUT` | Save to CSV file | Console | `bragg.csv` |

### Output Fields

- `d_spacing_angstrom` — d-spacing (Å)
- `wavelength_angstrom` — X-ray wavelength (Å)
- `order` — Diffraction order
- `bragg_angle_degrees` — Bragg angle θ (degrees)
- `two_theta_degrees` — Scattering angle 2θ (degrees)

### Examples

#### Single Calculation with Wavelength
```bash
xraylabtool bragg -d 3.14 -w 1.54
```
**Output:**
```
Bragg Angle Calculations:
--------------------------------------------------
   d (Å)    θ (°)   2θ (°)
--------------------------------------------------
   3.140   14.177   28.354
```

#### Single Calculation with Energy
```bash
xraylabtool bragg -d 3.14 -e 8.0
```

#### Multiple d-spacings
```bash
xraylabtool bragg -d 3.14,2.45,1.92 -w 1.54
```

#### Higher Order Diffraction
```bash
xraylabtool bragg -d 3.14 -w 1.54 --order 2
```

#### Save to File
```bash
xraylabtool bragg -d 3.14,2.45,1.92 -e 8.0 -o bragg_results.csv
```

---

## `list` - Reference Information

Display reference information including physical constants, available output fields, and usage examples.

### Syntax
```bash
xraylabtool list TYPE
```

### Required Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `TYPE` | Information type | `constants`, `fields`, `examples` |

### Examples

#### Physical Constants
```bash
xraylabtool list constants
```
**Output:**
```
Physical Constants:
========================================
THOMPSON                 : 2.8179403227e-15
SPEED_OF_LIGHT           : 299792458.0
PLANCK                   : 6.626068e-34
ELEMENT_CHARGE           : 1.60217646e-19
AVOGADRO                 : 6.02214199e+23
ENERGY_TO_WAVELENGTH_FACTOR: 1.2398417166827828e-09
PI                       : 3.141592653589793
TWO_PI                   : 6.283185307179586
```

#### Available Output Fields
```bash
xraylabtool list fields
```
Shows all available field names for the `--fields` parameter.

#### Usage Examples
```bash
xraylabtool list examples
```
Shows practical usage examples for all commands.

---

## Output Formats

### Table Format (Default)
Human-readable console output with aligned columns and clear headers.

**Usage:** Default format or `--format table`
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2
```

### CSV Format
Comma-separated values suitable for spreadsheets and data analysis.

**Usage:** `--format csv` or output file with `.csv` extension
```bash
xraylabtool calc Si -e 5,10,15 -d 2.33 --format csv
xraylabtool calc Si -e 5,10,15 -d 2.33 -o data.csv  # Auto-detected
```

**CSV Structure:**
- Headers in first row
- One row per energy point
- All fields included unless `--fields` specified

### JSON Format
Structured data format ideal for programmatic processing.

**Usage:** `--format json` or output file with `.json` extension
```bash
xraylabtool calc SiO2 -e 10.0 -d 2.2 --format json
xraylabtool calc SiO2 -e 10.0 -d 2.2 -o data.json  # Auto-detected
```

**JSON Structure:**
```json
{
  "formula": "SiO2",
  "molecular_weight_g_mol": 60.083,
  "energy_kev": [10.0],
  "dispersion_delta": [4.613309e-06],
  "critical_angle_degrees": [0.174038]
}
```

---

## Energy Input Formats

XRayLabTool supports flexible energy input formats for the `calc` command:

### Single Energy
```bash
-e 10.0                    # Single energy point at 10 keV
```

### Multiple Discrete Energies
```bash
-e 5.0,10.0,15.0,20.0      # Four specific energies
-e 8.0,10.0,12.0           # Three energies around Cu Kα
```

### Linear Energy Range
```bash
-e 5-15:11                 # 11 evenly spaced points from 5 to 15 keV
-e 1-30:300                # 300 points from 1 to 30 keV
-e 0.5-2.0:16              # 16 points for soft X-ray range
```

### Logarithmic Energy Range
```bash
-e 1-30:100:log            # 100 log-spaced points from 1 to 30 keV
-e 0.1-10:50:log           # 50 log-spaced points for wide range
-e 5-20:25:log             # 25 log-spaced points in hard X-ray range
```

### Energy Range Guidelines

| Energy Range | Recommended Format | Use Case |
|--------------|-------------------|----------|
| 0.03-30 keV | `0.1-30:200:log` | Full X-ray spectrum |
| 5-20 keV | `5-20:50` | Hard X-ray analysis |
| 8-12 keV | `8-12:21` | Around Cu Kα line |
| 1-10 keV | `1-10:100:log` | Soft X-ray range |

---

## Common Use Cases

### 1. Single Material Analysis
Calculate properties for one material at specific energies:
```bash
# Silicon at Cu Kα energy
xraylabtool calc Si -e 8.048 -d 2.33

# Quartz across energy range
xraylabtool calc SiO2 -e 5-20:50 -d 2.2 -o quartz_sweep.csv
```

### 2. Material Comparison
Compare multiple materials at the same energy:
```bash
# Create batch file
cat > comparison.csv << EOF
formula,density,energy
SiO2,2.2,10.0
Si,2.33,10.0
Al2O3,3.95,10.0
Fe2O3,5.24,10.0
EOF

xraylabtool batch comparison.csv -o material_comparison.csv
```

### 3. Energy Scan Analysis
Analyze energy-dependent properties:
```bash
# Log-spaced energy sweep for absorption edge analysis
xraylabtool calc Fe -e 7-9:100:log -d 7.87 -o iron_edge.csv

# Linear sweep around specific energy
xraylabtool calc Si -e 8-8.1:101 -d 2.33 -o silicon_fine.csv
```

### 4. Crystallography Analysis
Calculate diffraction angles:
```bash
# Silicon crystal d-spacings at Cu Kα
xraylabtool bragg -d 3.135,1.920,1.637 -e 8.048

# Multiple orders
xraylabtool bragg -d 3.14 -w 1.54 --order 1
xraylabtool bragg -d 3.14 -w 1.54 --order 2
```

### 5. Unit Conversions
Convert between energy and wavelength:
```bash
# Common X-ray lines
xraylabtool convert energy 8.048,17.478,59.318 --to wavelength

# Wavelength to energy for laser
xraylabtool convert wavelength 632.8 --to energy  # He-Ne laser
```

### 6. Formula Analysis
Analyze complex chemical formulas:
```bash
# Biological molecules
xraylabtool formula C6H12O6           # Glucose
xraylabtool formula Ca10P6O26H2       # Hydroxyapatite

# Save analysis
xraylabtool formula SiO2,TiO2,Al2O3,Fe2O3 -o oxides.json
```

---

## Tips & Best Practices

### Performance Optimization

1. **Energy Range Selection**
   - Use logarithmic spacing for wide energy ranges
   - Use linear spacing for fine scans around specific features
   - Limit points to what you actually need for analysis

2. **Batch Processing**
   - Use `--workers` parameter for large datasets
   - Process similar materials together for cache efficiency
   - Save intermediate results to avoid recalculation

3. **File Formats**
   - Use CSV for spreadsheet analysis
   - Use JSON for programmatic processing
   - Use table format for quick visual inspection

### Accuracy Considerations

1. **Energy Range Limits**
   - Valid range: 0.03–30 keV (X-ray regime)
   - Accuracy decreases near absorption edges
   - Use finer energy spacing near edges for accuracy

2. **Density Values**
   - Use published density values when available
   - Consider temperature and pressure effects
   - Verify formula matches expected stoichiometry

### Error Handling

1. **Common Errors**
   ```bash
   # Invalid energy range
   xraylabtool calc SiO2 -e 0.01 -d 2.2  # Error: too low
   
   # Invalid formula
   xraylabtool calc XYZ123 -e 10 -d 1.0  # Error: unknown elements
   
   # Missing parameters
   xraylabtool calc SiO2 -e 10.0         # Error: missing density
   ```

2. **Validation**
   ```bash
   # Test formula parsing first
   xraylabtool formula Ca10P6O26H2
   
   # Verify atomic data availability
   xraylabtool atomic Ca,P,O,H
   ```

### File Management

1. **Output Organization**
   ```bash
   # Organized output structure
   mkdir -p results/{calculations,conversions,batch}
   
   xraylabtool calc SiO2 -e 5-20:50 -d 2.2 -o results/calculations/sio2_scan.csv
   xraylabtool batch materials.csv -o results/batch/materials_batch.csv
   ```

2. **Naming Conventions**
   ```bash
   # Descriptive filenames
   xraylabtool calc Si -e 8.048 -d 2.33 -o si_cu_ka_8keV.csv
   xraylabtool calc SiO2 -e 1-30:100:log -d 2.2 -o sio2_full_spectrum.csv
   ```

### Integration with Other Tools

1. **Python Integration**
   ```python
   # After CLI calculation, load results in Python
   import pandas as pd
   data = pd.read_csv('results.csv')
   ```

2. **Spreadsheet Analysis**
   ```bash
   # Generate CSV for Excel/LibreOffice
   xraylabtool calc SiO2 -e 5-20:50 -d 2.2 --format csv -o for_excel.csv
   ```

3. **Plotting and Visualization**
   ```bash
   # Generate data for plotting software
   xraylabtool calc Si -e 1-30:200:log -d 2.33 \
     --fields energy_kev,dispersion_delta,absorption_beta \
     -o si_optical_constants.csv
   ```

---

## Command Summary Table

| Command | Purpose | Key Parameters | Output |
|---------|---------|----------------|--------|
| `calc` | Single material calculations | `formula`, `-e`, `-d` | Properties table/CSV/JSON |
| `batch` | Multiple materials | `input.csv`, `-o` | Batch results CSV/JSON |
| `convert` | Unit conversions | `unit`, `values`, `--to` | Conversion table |
| `formula` | Chemical analysis | `formulas` | Element breakdown |
| `atomic` | Element data | `elements` | Atomic properties |
| `bragg` | Diffraction angles | `-d`, `-w/-e` | Bragg angle table |
| `list` | Reference info | `type` | Constants/fields/examples |

---

## Version History

- **v0.1.5** - Current version with full CLI functionality
- **Python Requirements** - ≥ 3.12
- **License** - MIT

For more information, see:
- Main documentation: `README.md`
- Development guide: `WARP.md`
- Virtual environment setup: `VIRTUAL_ENV.md`

---

*This documentation was generated for XRayLabTool v0.1.5. For the latest updates, check the project repository.*
