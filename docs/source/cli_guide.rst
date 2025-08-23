Command-Line Interface Guide
=============================

XRayLabTool provides a comprehensive command-line interface for quick calculations, batch processing, and integration into workflows.

Installation & Verification
----------------------------

Install XRayLabTool with CLI support:

.. code-block:: bash

   pip install xraylabtool

Verify CLI installation:

.. code-block:: bash

   xraylabtool --version
   # Output: XRayLabTool 0.1.5

Global Options
--------------

These options are available for the main ``xraylabtool`` command:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--help``, ``-h``
     - Show help message and exit
   * - ``--version``
     - Show program version and exit
   * - ``--verbose``, ``-v``
     - Enable verbose output (global)

Available Commands
------------------

XRayLabTool CLI provides 7 subcommands:

.. list-table::
   :widths: 15 35 50
   :header-rows: 1

   * - Command
     - Purpose
     - Example
   * - ``calc``
     - Single material calculations
     - ``xraylabtool calc SiO2 -e 10.0 -d 2.2``
   * - ``batch``
     - Process multiple materials
     - ``xraylabtool batch materials.csv -o results.csv``
   * - ``convert``
     - Energy/wavelength conversion
     - ``xraylabtool convert energy 10.0 --to wavelength``
   * - ``formula``
     - Chemical formula analysis
     - ``xraylabtool formula Al2O3``
   * - ``atomic``
     - Atomic data lookup
     - ``xraylabtool atomic Si,Al,Fe``
   * - ``bragg``
     - Diffraction angle calculations
     - ``xraylabtool bragg -d 3.14 -e 8.0``
   * - ``list``
     - Show constants/fields/examples
     - ``xraylabtool list constants``

calc - Single Material Calculations
------------------------------------

Calculate X-ray optical properties for a single material composition.

**Syntax:**

.. code-block:: bash

   xraylabtool calc FORMULA -e ENERGY -d DENSITY [OPTIONS]

**Required Parameters:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``FORMULA``
     - str
     - Chemical formula (case-sensitive)
   * - ``-e, --energy``
     - str
     - X-ray energy in keV (see energy formats below)
   * - ``-d, --density``
     - float
     - Material density in g/cm³

**Optional Parameters:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``-o, --output``
     - Console
     - Output filename (CSV or JSON based on extension)
   * - ``--format``
     - table
     - Output format: table, csv, json
   * - ``--fields``
     - All fields
     - Comma-separated list of fields to output
   * - ``--precision``
     - 6
     - Number of decimal places

**Energy Input Formats:**

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Format
     - Description
   * - ``10.0``
     - Single energy point
   * - ``5.0,10.0,15.0``
     - Multiple discrete energies
   * - ``5-15:11``
     - 11 linearly spaced points from 5 to 15 keV
   * - ``1-30:100:log``
     - 100 log-spaced points from 1 to 30 keV

**Examples:**

.. code-block:: bash

   # Basic calculation
   xraylabtool calc SiO2 -e 10.0 -d 2.2

   # Multiple energies
   xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33

   # Energy range (linear)
   xraylabtool calc Al2O3 -e 5-15:11 -d 3.95

   # Energy range (logarithmic)
   xraylabtool calc C -e 1-30:100:log -d 3.52

   # CSV output with selected fields
   xraylabtool calc SiO2 -e 8.0,10.0,12.0 -d 2.2 \
     --fields formula,energy_kev,dispersion_delta,critical_angle_degrees \
     -o results.csv

batch - Batch Processing
------------------------

Process multiple materials from a CSV input file with support for parallel processing.

**Syntax:**

.. code-block:: bash

   xraylabtool batch INPUT_FILE -o OUTPUT_FILE [OPTIONS]

**Input CSV Format:**

The input CSV file must have the following columns:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Column
     - Description
   * - ``formula``
     - Chemical formula (required)
   * - ``density``
     - Mass density in g/cm³ (required)
   * - ``energy``
     - Energy specification (required)

**Example Input CSV:**

.. code-block:: text

   formula,density,energy
   SiO2,2.2,10.0
   Al2O3,3.95,"5.0,10.0,15.0"
   Si,2.33,8.0
   Fe2O3,5.24,"8.0,12.0"

**Optional Parameters:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``--format``
     - Auto-detect
     - Output format: csv, json
   * - ``--workers``
     - Auto
     - Number of parallel workers
   * - ``--fields``
     - All fields
     - Output field selection

**Examples:**

.. code-block:: bash

   # Basic batch processing
   xraylabtool batch materials.csv -o results.csv

   # JSON output with parallel processing
   xraylabtool batch materials.csv -o results.json --workers 4

convert - Unit Conversions
--------------------------

Convert between X-ray energy (keV) and wavelength (Å) units.

**Syntax:**

.. code-block:: bash

   xraylabtool convert INPUT_UNIT VALUES --to OUTPUT_UNIT [OPTIONS]

**Examples:**

.. code-block:: bash

   # Energy to wavelength
   xraylabtool convert energy 10.0 --to wavelength

   # Multiple values
   xraylabtool convert energy 5.0,10.0,15.0,20.0 --to wavelength

   # Save to file
   xraylabtool convert energy 8.048,10.0,12.4 --to wavelength -o conversions.csv

formula - Chemical Formula Analysis
-----------------------------------

Parse and analyze chemical formulas to show elemental composition.

**Examples:**

.. code-block:: bash

   # Single formula
   xraylabtool formula SiO2

   # Multiple formulas
   xraylabtool formula SiO2,Al2O3,Fe2O3

   # Complex formula
   xraylabtool formula Ca10P6O26H2

atomic - Atomic Data Lookup
---------------------------

Look up atomic numbers, weights, and other properties for chemical elements.

**Examples:**

.. code-block:: bash

   # Single element
   xraylabtool atomic Si

   # Multiple elements
   xraylabtool atomic H,C,N,O,Si

   # Save to file
   xraylabtool atomic Si,Al,Fe -o atomic_data.csv

bragg - Diffraction Angle Calculations
--------------------------------------

Calculate Bragg diffraction angles using Bragg's law: nλ = 2d sin(θ).

**Syntax:**

.. code-block:: bash

   xraylabtool bragg -d DSPACING (-w WAVELENGTH | -e ENERGY) [OPTIONS]

**Examples:**

.. code-block:: bash

   # Single calculation with wavelength
   xraylabtool bragg -d 3.14 -w 1.54

   # Single calculation with energy
   xraylabtool bragg -d 3.14 -e 8.0

   # Multiple d-spacings
   xraylabtool bragg -d 3.14,2.45,1.92 -e 8.048

   # Higher order diffraction
   xraylabtool bragg -d 3.14 -w 1.54 --order 2

list - Reference Information
----------------------------

Display reference information including physical constants, available output fields, and usage examples.

**Examples:**

.. code-block:: bash

   # Physical constants
   xraylabtool list constants

   # Available output fields
   xraylabtool list fields

   # Usage examples
   xraylabtool list examples

Output Formats
--------------

XRayLabTool CLI supports three output formats:

**Table Format (Default):**
   Human-readable console output with aligned columns and clear headers.

**CSV Format:**
   Comma-separated values suitable for spreadsheets and data analysis.
   
   - Use ``--format csv`` or output file with ``.csv`` extension
   - Headers in first row, one row per energy point

**JSON Format:**
   Structured data format ideal for programmatic processing.
   
   - Use ``--format json`` or output file with ``.json`` extension
   - Nested structure with arrays for energy-dependent properties

Common Use Cases
----------------

**Single Material Analysis:**

.. code-block:: bash

   # Silicon at Cu Kα energy
   xraylabtool calc Si -e 8.048 -d 2.33

   # Quartz across energy range
   xraylabtool calc SiO2 -e 5-20:50 -d 2.2 -o quartz_sweep.csv

**Material Comparison:**

.. code-block:: bash

   # Create batch file for comparison
   cat > comparison.csv << EOF
   formula,density,energy
   SiO2,2.2,10.0
   Si,2.33,10.0
   Al2O3,3.95,10.0
   Fe2O3,5.24,10.0
   EOF

   xraylabtool batch comparison.csv -o material_comparison.csv

**Energy Scan Analysis:**

.. code-block:: bash

   # Log-spaced energy sweep for absorption edge analysis
   xraylabtool calc Fe -e 7-9:100:log -d 7.87 -o iron_edge.csv

   # Linear sweep around specific energy
   xraylabtool calc Si -e 8-8.1:101 -d 2.33 -o silicon_fine.csv

Performance Tips
----------------

1. **Energy Range Selection**
   - Use logarithmic spacing for wide energy ranges
   - Use linear spacing for fine scans around specific features
   - Limit points to what you actually need for analysis

2. **Batch Processing**
   - Use ``--workers`` parameter for large datasets
   - Process similar materials together for cache efficiency

3. **File Formats**
   - Use CSV for spreadsheet analysis
   - Use JSON for programmatic processing
   - Use table format for quick visual inspection

Getting Help
------------

For command-specific help, use:

.. code-block:: bash

   xraylabtool <command> --help

For comprehensive CLI documentation with detailed examples and use cases, see the main CLI reference guide.
