Interfaces Module
=================

The interfaces module provides command-line interface and shell completion functionality.

.. currentmodule:: xraylabtool.interfaces

Command Line Interface
----------------------

.. automodule:: xraylabtool.interfaces.cli
   :members:
   :undoc-members:
   :show-inheritance:

CLI Commands Overview
~~~~~~~~~~~~~~~~~~~~~

XRayLabTool provides 9 comprehensive CLI commands:

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

Command Examples
~~~~~~~~~~~~~~~~

**Single Material Calculation:**

.. code-block:: bash

   # Basic calculation
   xraylabtool calc Si --density 2.33 --energy 8000

   # Multiple energies
   xraylabtool calc SiO2 --density 2.20 --energy 5000,8000,10000

   # Energy range
   xraylabtool calc Al --density 2.70 --energy 1000-20000:100

   # Different output formats
   xraylabtool calc Si --density 2.33 --energy 8000 --output csv
   xraylabtool calc Si --density 2.33 --energy 8000 --output json

**Batch Processing:**

.. code-block:: bash

   # Process CSV file
   xraylabtool batch materials.csv --output results.csv

   # Specify energy column
   xraylabtool batch data.csv --energy-column "Energy (eV)"

   # JSON output
   xraylabtool batch materials.csv --output results.json --format json

**Unit Conversions:**

.. code-block:: bash

   # Energy to wavelength
   xraylabtool convert --energy 8000 --to wavelength

   # Wavelength to energy
   xraylabtool convert --wavelength 1.55 --to energy

   # Multiple values
   xraylabtool convert --energy 5000,8000,10000 --to wavelength

**Formula Analysis:**

.. code-block:: bash

   # Parse chemical formula
   xraylabtool formula SiO2

   # Complex formula
   xraylabtool formula "Ca5(PO4)3F"

   # With molecular weight
   xraylabtool formula SiO2 --molecular-weight

**Atomic Data Lookup:**

.. code-block:: bash

   # Element information
   xraylabtool atomic Si

   # Scattering factors at specific energy
   xraylabtool atomic Si --energy 8000

   # Multiple elements
   xraylabtool atomic Si,Al,O --energy 8000

**Bragg Diffraction:**

.. code-block:: bash

   # Calculate Bragg angle
   xraylabtool bragg --d-spacing 3.14 --energy 8000

   # Multiple reflections
   xraylabtool bragg --d-spacing 3.14,1.92,1.64 --energy 8000

**Reference Information:**

.. code-block:: bash

   # List supported elements
   xraylabtool list elements

   # Physical constants
   xraylabtool list constants

   # Example formulas
   xraylabtool list examples

Output Formats
~~~~~~~~~~~~~~

All commands support multiple output formats:

- **table** (default): Human-readable tabular format
- **csv**: Comma-separated values for spreadsheet import
- **json**: JSON format for programmatic processing

Shell Completion
----------------

.. automodule:: xraylabtool.interfaces.completion
   :members:
   :undoc-members:
   :show-inheritance:

Bash Completion Features
~~~~~~~~~~~~~~~~~~~~~~~~

The shell completion system provides:

- **Command completion**: All 9 CLI commands
- **Option completion**: Command flags and parameters
- **File completion**: Input/output file paths
- **Element completion**: Chemical element symbols
- **Unit completion**: Energy and wavelength units

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install completion for current user
   xraylabtool install-completion

   # Install system-wide (requires sudo)
   sudo xraylabtool install-completion --system

   # Custom completion script location
   xraylabtool install-completion --path /custom/path

Usage
~~~~~

After installation, completion is available by pressing Tab:

.. code-block:: bash

   xraylabtool [TAB]          # Shows all commands
   xraylabtool calc [TAB]     # Shows calc command options
   xraylabtool atomic S[TAB]  # Completes to supported elements starting with 'S'

Uninstallation
~~~~~~~~~~~~~~

.. code-block:: bash

   # Remove completion
   xraylabtool uninstall-completion

   # Remove system-wide completion
   sudo xraylabtool uninstall-completion --system

Platform Support
~~~~~~~~~~~~~~~~

Shell completion is supported for:

- **Bash**: Full support on Linux and macOS (requires bash-completion)
- **Zsh**: Full support on Linux and macOS (requires zsh-completions)
- **Fish**: Native support with built-in completion system
- **PowerShell**: Full support on Windows, macOS, and Linux

Error Handling
--------------

The CLI provides comprehensive error handling with helpful messages:

.. code-block:: bash

   # Invalid formula
   $ xraylabtool calc XYZ --density 1.0 --energy 8000
   Error: Unknown element 'XYZ' in formula

   # Missing required parameter
   $ xraylabtool calc Si --energy 8000
   Error: --density is required

   # Invalid energy range
   $ xraylabtool calc Si --density 2.33 --energy 0
   Error: Energy must be positive

Integration Examples
--------------------

**With Python Scripts:**

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

   data = json.loads(result.stdout)
   print(f"Critical angle: {data[0]['critical_angle_degrees']}")

**With Shell Scripts:**

.. code-block:: bash

   #!/bin/bash

   # Process multiple materials
   for material in Si Al Cu; do
       echo "Processing $material..."
       xraylabtool calc $material --density 2.33 --energy 8000 --output csv >> results.csv
   done
