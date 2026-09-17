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

XRayLabTool provides 11 CLI commands (see :doc:`../guides/cli_reference` for every option):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``calc``
     - Calculate X-ray properties for a single material
   * - ``batch``
     - Process multiple materials from CSV file
   * - ``compare``
     - Compare X-ray properties between multiple materials
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
   * - ``completion``
     - Virtual environment-centric shell completion (install/uninstall/status/list/info)
   * - ``install-completion``
     - Install shell completion (bash, zsh, fish, PowerShell)
   * - ``uninstall-completion``
     - Remove shell completion

Command Examples
~~~~~~~~~~~~~~~~

All energies are in keV.

**Single Material Calculation:**

.. code-block:: bash

   xraylabtool calc Si -e 8.0 -d 2.33
   xraylabtool calc SiO2 -e 5.0,8.0,10.0 -d 2.2
   xraylabtool calc Al -e 1-20:100 -d 2.70
   xraylabtool calc Si -e 8.0 -d 2.33 --format csv
   xraylabtool calc Si -e 8.0 -d 2.33 -o si.json

**Batch Processing:**

.. code-block:: bash

   xraylabtool batch materials.csv -o results.csv
   xraylabtool batch materials.csv -o results.json --workers 8 --progress

**Material Comparison:**

.. code-block:: bash

   xraylabtool compare SiO2,2.2 Si3N4,3.2 -e 8.0,10.0 --properties dispersion_delta,absorption_beta

**Unit Conversions:**

.. code-block:: bash

   xraylabtool convert energy 8.0 --to wavelength
   xraylabtool convert wavelength 1.55 --to energy
   xraylabtool convert energy 5.0,8.0,10.0 --to wavelength

**Formula Analysis:**

.. code-block:: bash

   xraylabtool formula SiO2
   xraylabtool formula "Ca5(PO4)3F"

**Atomic Data Lookup:**

.. code-block:: bash

   xraylabtool atomic Si
   xraylabtool atomic Si,Al,O -o elements.csv

**Bragg Diffraction:**

.. code-block:: bash

   xraylabtool bragg -d 3.14 -e 8.0
   xraylabtool bragg -d 3.14,1.92,1.64 -e 8.0
   xraylabtool bragg -d 3.14 -w 1.55 --order 2

**Reference Information:**

.. code-block:: bash

   xraylabtool list constants
   xraylabtool list fields
   xraylabtool list examples

Output Formats
~~~~~~~~~~~~~~

``calc`` and ``compare`` accept ``--format {table,csv,json}`` for stdout; ``-o/--output FILE``
writes CSV or JSON by file extension.

Shell Completion v2
-------------------

The ``completion_v2`` sub-package is the current virtual-environment-centric
completion system (the legacy ``completion.py`` bridge was removed).

.. automodule:: xraylabtool.interfaces.completion_v2.installer
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xraylabtool.interfaces.completion_v2.shells
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xraylabtool.interfaces.completion_v2.environment
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xraylabtool.interfaces.completion_v2.integration
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xraylabtool.interfaces.completion_v2.cli
   :members:
   :undoc-members:
   :show-inheritance:

Bash Completion Features
~~~~~~~~~~~~~~~~~~~~~~~~

The shell completion system provides:

- **Command completion**: All 11 CLI commands
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

   # Per-virtual-environment install (preferred)
   xraylabtool completion install --shell zsh

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

The CLI provides error handling with helpful messages:

.. code-block:: bash

   # Invalid formula
   $ xraylabtool calc XYZ -e 8.0 -d 1.0
   Error: Invalid chemical formula 'XYZ': Unknown element symbol: X: 'XYZ'

   # Missing required parameter
   $ xraylabtool calc Si -e 8.0
   error: the following arguments are required: -d/--density

   # Invalid energy range
   $ xraylabtool calc Si -e 0 -d 2.33
   Error: All energies must be positive

Integration Examples
--------------------

**With Python Scripts:**

.. code-block:: python

   import subprocess
   import json

   # Call CLI from Python
   result = subprocess.run([
       "xraylabtool", "calc", "Si",
       "-e", "8.0",
       "-d", "2.33",
       "--format", "json"
   ], capture_output=True, text=True, check=True)

   data = json.loads(result.stdout)
   print(f"Critical angle: {data['critical_angle_degrees'][0]}")

**With Shell Scripts:**

.. code-block:: bash

   #!/bin/bash

   # Process multiple materials
   for material in Si Al Cu; do
       echo "Processing $material..."
       xraylabtool calc $material -e 8.0 -d 2.33 --format csv >> results.csv
   done
