CLI Reference
=============

XRayLabTool provides a comprehensive command-line interface with 9 commands for X-ray optical property calculations.

Overview
--------

.. code-block:: bash

   xraylabtool [COMMAND] [OPTIONS]

Available Commands:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``calc``
     - Calculate X-ray properties for a single material
   * - ``batch``
     - Process multiple materials from CSV file
   * - ``convert``
     - Convert between energy and wavelength units
   * - ``formula``
     - Parse and analyze chemical formulas
   * - ``atomic``
     - Look up atomic scattering factor data
   * - ``bragg``
     - Calculate Bragg diffraction angles
   * - ``list``
     - Display reference information and constants
   * - ``install-completion``
     - Install shell completion (bash, zsh, fish, PowerShell)
   * - ``uninstall-completion``
     - Remove shell completion

Global Options
--------------

.. option:: --version

   Show version information and exit.

.. option:: --help

   Show help message and exit.

.. option:: --verbose, -v

   Increase output verbosity (can be used multiple times).

.. option:: --quiet, -q

   Suppress non-essential output.

calc - Single Material Calculation
-----------------------------------

Calculate X-ray optical properties for a single material.

**Syntax:**

.. code-block:: bash

   xraylabtool calc FORMULA [OPTIONS]

**Arguments:**

.. option:: FORMULA

   Chemical formula of the material (e.g., "Si", "SiO2", "Al2O3").

**Required Options:**

.. option:: --density FLOAT

   Material density in g/cm³.

.. option:: --energy ENERGY_SPEC

   X-ray energy specification. Can be:

   - Single value: ``8000``
   - Multiple values: ``5000,8000,10000``
   - Range: ``1000-20000:1000`` (start-stop:step)
   - Mixed: ``5000,8000-12000:1000,15000``

**Optional Options:**

.. option:: --output {table,csv,json}

   Output format (default: table).

.. option:: --save FILENAME

   Save results to file.

.. option:: --precision INTEGER

   Number of decimal places in output (default: 3).

**Examples:**

.. code-block:: bash

   # Basic calculation
   xraylabtool calc Si --density 2.33 --energy 8000

   # Multiple energies
   xraylabtool calc SiO2 --density 2.20 --energy 5000,8000,10000

   # Energy range
   xraylabtool calc Al --density 2.70 --energy 1000-20000:1000

   # Save to CSV
   xraylabtool calc Cu --density 8.96 --energy 8000 --output csv --save copper_8keV.csv

   # JSON output with high precision
   xraylabtool calc Si --density 2.33 --energy 8000 --output json --precision 6

**Output Example:**

.. code-block:: text

   Material: Si (density = 2.33 g/cm³)
   Energy: 8000.0 eV (λ = 1.550 Å)

   Property                          Value        Unit
   ────────────────────────────────────────────────────
   Delta (δ)                        1.234e-05    -
   Beta (β)                         1.678e-07    -
   Critical angle                   0.158        degrees
   Critical angle                   2.76         mrad
   Attenuation length               9.84         cm
   Linear absorption coefficient    0.102        cm⁻¹
   Mass absorption coefficient      0.044        cm²/g

batch - Batch Processing
-------------------------

Process multiple materials from a CSV input file.

**Syntax:**

.. code-block:: bash

   xraylabtool batch INPUT_FILE [OPTIONS]

**Arguments:**

.. option:: INPUT_FILE

   Path to CSV file containing material specifications.

**Options:**

.. option:: --output FILENAME

   Output file path (default: stdout).

.. option:: --format {csv,json}

   Output format (default: csv).

.. option:: --energy-column NAME

   Name of energy column if different from 'energy' or 'Energy'.

.. option:: --show-progress

   Display progress bar during processing.

.. option:: --chunk-size INTEGER

   Process materials in chunks (default: 1000).

**Input CSV Format:**

The input CSV file should contain columns for Formula, Density, and optionally Energy:

.. code-block:: text

   Formula,Density,Energy
   Si,2.33,8000
   SiO2,2.20,8000
   Al,2.70,5000
   Cu,8.96,10000

**Alternative column names are supported:**

- **Formula**: "formula", "Formula", "material", "Material"
- **Density**: "density", "Density", "rho", "ρ"
- **Energy**: "energy", "Energy", "E", "keV" (with automatic unit conversion)

**Examples:**

.. code-block:: bash

   # Basic batch processing
   xraylabtool batch materials.csv --output results.csv

   # JSON output with progress
   xraylabtool batch large_dataset.csv --format json --show-progress --output results.json

   # Custom energy column
   xraylabtool batch data.csv --energy-column "Energy (keV)" --output results.csv

   # Process in smaller chunks
   xraylabtool batch huge_dataset.csv --chunk-size 500 --show-progress

convert - Unit Conversion
-------------------------

Convert between X-ray energy and wavelength units.

**Syntax:**

.. code-block:: bash

   xraylabtool convert [OPTIONS]

**Options:**

.. option:: --energy FLOAT_LIST

   Energy value(s) in eV. Can be single value or comma-separated list.

.. option:: --wavelength FLOAT_LIST

   Wavelength value(s) in Angstroms. Can be single value or comma-separated list.

.. option:: --to {wavelength,energy}

   Target unit for conversion.

.. option:: --precision INTEGER

   Number of decimal places (default: 4).

**Examples:**

.. code-block:: bash

   # Energy to wavelength
   xraylabtool convert --energy 8000 --to wavelength
   # Output: 8000.0 eV = 1.5498 Å

   # Wavelength to energy
   xraylabtool convert --wavelength 1.55 --to energy
   # Output: 1.55 Å = 8000.3 eV

   # Multiple values
   xraylabtool convert --energy 5000,8000,10000 --to wavelength
   # Output:
   # 5000.0 eV = 2.4797 Å
   # 8000.0 eV = 1.5498 Å
   # 10000.0 eV = 1.2398 Å

formula - Formula Analysis
--------------------------

Parse and analyze chemical formulas.

**Syntax:**

.. code-block:: bash

   xraylabtool formula FORMULA [OPTIONS]

**Arguments:**

.. option:: FORMULA

   Chemical formula to analyze.

**Options:**

.. option:: --molecular-weight

   Calculate and display molecular weight.

.. option:: --composition

   Show detailed elemental composition.

.. option:: --normalize

   Display normalized formula format.

**Examples:**

.. code-block:: bash

   # Basic formula parsing
   xraylabtool formula SiO2
   # Output:
   # Formula: SiO2
   # Elements: Si (1), O (2)

   # With molecular weight
   xraylabtool formula "Ca5(PO4)3F" --molecular-weight
   # Output:
   # Formula: Ca5(PO4)3F
   # Elements: Ca (5), P (3), O (12), F (1)
   # Molecular weight: 504.30 g/mol

   # Complex hydrated compound
   xraylabtool formula "CuSO4·5H2O" --composition
   # Output:
   # Formula: CuSO4·5H2O
   # Elements: Cu (1), S (1), O (9), H (10)
   # Composition by mass:
   #   Cu: 25.5%
   #   S:  12.8%
   #   O:  57.7%
   #   H:   4.0%

atomic - Atomic Data Lookup
----------------------------

Look up atomic scattering factor data and element information.

**Syntax:**

.. code-block:: bash

   xraylabtool atomic ELEMENT [OPTIONS]

**Arguments:**

.. option:: ELEMENT

   Element symbol or comma-separated list of elements.

**Options:**

.. option:: --energy FLOAT

   Energy in eV for scattering factor lookup.

.. option:: --info

   Show detailed element information.

.. option:: --range START STOP STEP

   Energy range for tabulated scattering factors.

**Examples:**

.. code-block:: bash

   # Basic element info
   xraylabtool atomic Si
   # Output:
   # Element: Silicon (Si)
   # Atomic number: 14
   # Atomic weight: 28.0855 g/mol

   # Scattering factors at specific energy
   xraylabtool atomic Si --energy 8000
   # Output:
   # Element: Si at 8000.0 eV
   # f1 (real part): 12.234
   # f2 (imaginary part): 0.456

   # Multiple elements
   xraylabtool atomic Si,O,Al --energy 8000 --info

   # Energy range
   xraylabtool atomic Si --range 5000 15000 1000

bragg - Bragg Diffraction
-------------------------

Calculate Bragg diffraction angles for crystallographic analysis.

**Syntax:**

.. code-block:: bash

   xraylabtool bragg [OPTIONS]

**Options:**

.. option:: --d-spacing FLOAT_LIST

   d-spacing value(s) in Angstroms.

.. option:: --energy FLOAT

   X-ray energy in eV.

.. option:: --wavelength FLOAT

   X-ray wavelength in Angstroms (alternative to energy).

.. option:: --order INTEGER

   Diffraction order (default: 1).

**Examples:**

.. code-block:: bash

   # Single reflection
   xraylabtool bragg --d-spacing 3.14 --energy 8000
   # Output:
   # d-spacing: 3.14 Å
   # Energy: 8000.0 eV (λ = 1.550 Å)
   # Bragg angle (2θ): 29.4°

   # Multiple reflections
   xraylabtool bragg --d-spacing 3.14,1.92,1.64 --energy 8000

   # Using wavelength instead of energy
   xraylabtool bragg --d-spacing 3.14 --wavelength 1.55

   # Higher order reflection
   xraylabtool bragg --d-spacing 3.14 --energy 8000 --order 2

list - Reference Information
----------------------------

Display reference information, constants, and examples.

**Syntax:**

.. code-block:: bash

   xraylabtool list CATEGORY

**Categories:**

.. option:: elements

   List all supported chemical elements.

.. option:: constants

   Show physical constants used in calculations.

.. option:: examples

   Display example materials with typical densities.

.. option:: units

   Show supported units and conversions.

**Examples:**

.. code-block:: bash

   # List supported elements
   xraylabtool list elements

   # Show physical constants
   xraylabtool list constants

   # Example materials
   xraylabtool list examples

   # Unit information
   xraylabtool list units

install-completion - Shell Completion
-------------------------------------

Install Bash shell completion for improved command-line experience.

**Syntax:**

.. code-block:: bash

   xraylabtool install-completion [OPTIONS]

**Options:**

.. option:: --system

   Install system-wide (requires sudo).

.. option:: --path PATH

   Custom installation path.

.. option:: --shell {bash}

   Shell type (bash, zsh, fish, powershell). Auto-detected if not specified.

**Examples:**

.. code-block:: bash

   # Install for current user
   xraylabtool install-completion

   # Install system-wide
   sudo xraylabtool install-completion --system

   # Custom path
   xraylabtool install-completion --path ~/.local/share/bash-completion

**After installation**, restart your shell or source your bash profile:

.. code-block:: bash

   source ~/.bashrc  # or ~/.bash_profile

uninstall-completion - Remove Completion
----------------------------------------

Remove previously installed shell completion.

**Syntax:**

.. code-block:: bash

   xraylabtool uninstall-completion [OPTIONS]

**Options:**

.. option:: --system

   Remove system-wide installation.

.. option:: --shell {bash}

   Shell type (bash, zsh, fish, powershell). Auto-detected if not specified.

**Examples:**

.. code-block:: bash

   # Remove user installation
   xraylabtool uninstall-completion

   # Remove system-wide installation
   sudo xraylabtool uninstall-completion --system

Output Formats
--------------

All commands support multiple output formats where applicable:

**Table Format (default):**

Human-readable tabular output with aligned columns and units.

**CSV Format:**

Comma-separated values suitable for spreadsheet applications:

.. code-block:: bash

   xraylabtool calc Si --density 2.33 --energy 8000 --output csv

**JSON Format:**

Structured JSON for programmatic processing:

.. code-block:: bash

   xraylabtool calc Si --density 2.33 --energy 8000 --output json

Error Handling
--------------

XRayLabTool provides clear error messages with suggestions:

.. code-block:: bash

   $ xraylabtool calc XYZ --density 1.0 --energy 8000
   Error: Unknown element 'XYZ' in formula
   Suggestion: Check element symbols - case-sensitive Si, not si

   $ xraylabtool calc Si --energy 8000
   Error: --density is required
   Usage: xraylabtool calc FORMULA --density FLOAT --energy ENERGY_SPEC

   $ xraylabtool calc Si --density 2.33 --energy -1000
   Error: Energy must be positive
   Supported range: 10 eV to 100,000 eV

Integration Examples
--------------------

**Shell Scripts:**

.. code-block:: bash

   #!/bin/bash

   # Process multiple materials
   for material in Si Al Cu; do
       echo "Processing $material..."
       xraylabtool calc $material --density 2.33 --energy 8000 --output csv >> results.csv
   done

**Python Integration:**

.. code-block:: python

   import subprocess
   import json

   # Call CLI from Python
   result = subprocess.run([
       "xraylabtool", "calc", "Si",
       "--density", "2.33",
       "--energy", "8000",
       "--output", "json"
   ], capture_output=True, text=True)

   if result.returncode == 0:
       data = json.loads(result.stdout)
       print(f"Critical angle: {data[0]['critical_angle_degrees']}")
   else:
       print(f"Error: {result.stderr}")

**Makefiles:**

.. code-block:: makefile

   # Calculate properties for common materials
   results.csv: materials.csv
   	xraylabtool batch materials.csv --output results.csv --show-progress

   clean:
   	rm -f results.csv

Performance Tips
----------------

1. **Use batch processing** for multiple materials
2. **Enable progress bars** for long calculations: ``--show-progress``
3. **Adjust chunk size** for memory optimization: ``--chunk-size 500``
4. **Use CSV output** for faster processing than JSON
5. **Cache results** by saving to files when reprocessing
