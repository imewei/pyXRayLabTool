CLI Reference
=============

Command-line interface with 10+ commands for X-ray property calculations and completion management.

**Usage:** ``xraylabtool [COMMAND] [OPTIONS]``

**Commands:** calc, batch, convert, formula, atomic, bragg, list, completion, install-completion, uninstall-completion

**Global Options:** --version, --help, --verbose/-v, --quiet/-q

calc - Single Material Calculation
-----------------------------------

**Usage:** ``xraylabtool calc FORMULA --density FLOAT --energy ENERGY_SPEC``

**Energy formats:**
- Single: ``8000``
- Multiple: ``5000,8000,10000``
- Range: ``1000-20000:1000``

**Options:** --output {table,csv,json}, --save FILENAME, --precision INTEGER

.. code-block:: bash

   xraylabtool calc Si --density 2.33 --energy 8000
   xraylabtool calc SiO2 --density 2.20 --energy 5000,8000,10000 --output csv

batch - Batch Processing
-------------------------

**Usage:** ``xraylabtool batch INPUT_FILE [OPTIONS]``

**CSV Format:** Formula, Density, Energy columns

**Options:** --output FILENAME, --format {csv,json}, --show-progress, --chunk-size INTEGER

.. code-block:: bash

   xraylabtool batch materials.csv --output results.csv
   xraylabtool batch large_dataset.csv --format json --show-progress

convert - Unit Conversion
-------------------------

**Usage:** ``xraylabtool convert --energy VALUES --to wavelength`` or ``--wavelength VALUES --to energy``

.. code-block:: bash

   xraylabtool convert --energy 8000 --to wavelength
   xraylabtool convert --wavelength 1.55 --to energy
   xraylabtool convert --energy 5000,8000,10000 --to wavelength

formula - Formula Analysis
--------------------------

**Usage:** ``xraylabtool formula FORMULA [OPTIONS]``

**Options:** --molecular-weight, --composition, --normalize

.. code-block:: bash

   xraylabtool formula SiO2
   xraylabtool formula "Ca5(PO4)3F" --molecular-weight
   xraylabtool formula "CuSO4·5H2O" --composition

atomic - Atomic Data Lookup
----------------------------

**Usage:** ``xraylabtool atomic ELEMENT [OPTIONS]``

**Options:** --energy FLOAT, --info, --range START STOP STEP

.. code-block:: bash

   xraylabtool atomic Si
   xraylabtool atomic Si --energy 8000
   xraylabtool atomic Si,O,Al --energy 8000 --info

bragg - Bragg Diffraction
-------------------------

**Usage:** ``xraylabtool bragg --d-spacing VALUES --energy FLOAT`` or ``--wavelength FLOAT``

**Options:** --order INTEGER (default: 1)

.. code-block:: bash

   xraylabtool bragg --d-spacing 3.14 --energy 8000
   xraylabtool bragg --d-spacing 3.14,1.92,1.64 --energy 8000
   xraylabtool bragg --d-spacing 3.14 --wavelength 1.55 --order 2

list - Reference Information
----------------------------

**Usage:** ``xraylabtool list CATEGORY``

**Categories:** elements, constants, examples, units

.. code-block:: bash

   xraylabtool list elements
   xraylabtool list constants
   xraylabtool list examples

completion - Virtual Environment-Centric Shell Completion
---------------------------------------------------------

**Usage:** ``xraylabtool completion [SUBCOMMAND] [OPTIONS]``

The new completion system installs per virtual environment and automatically
activates/deactivates with environment changes.

**Subcommands:**

.. code-block:: bash

   xraylabtool completion install              # Install in current environment
   xraylabtool completion install --shell zsh  # Install for specific shell
   xraylabtool completion list                 # List all environments
   xraylabtool completion status               # Show current environment status
   xraylabtool completion uninstall            # Remove from current environment
   xraylabtool completion uninstall --all      # Remove from all environments
   xraylabtool completion info                 # Show system information

**Supported Environments:**
- venv / virtualenv
- conda / mamba
- Poetry
- Pipenv

**Supported Shells:**
- bash (native completion)
- zsh (native completion)
- fish (native completion)
- PowerShell (native completion)

**Example Workflow:**

.. code-block:: bash

   # Activate your environment
   conda activate myproject

   # Install completion in the environment
   xraylabtool completion install

   # Completion is now available
   xraylabtool <TAB>  # Shows available commands

   # Deactivate environment - completion automatically unavailable
   conda deactivate

install-completion / uninstall-completion - Legacy Compatibility
----------------------------------------------------------------

**Install:** ``xraylabtool install-completion [--shell SHELL]``

**Uninstall:** ``xraylabtool uninstall-completion``

These legacy commands use the new completion system backend for backward compatibility:

.. code-block:: bash

   xraylabtool install-completion              # Install in current environment
   xraylabtool uninstall-completion            # Remove from current environment

Output Formats
--------------

**Formats:** table (default), csv, json

.. code-block:: bash

   xraylabtool calc Si --density 2.33 --energy 8000 --output csv
   xraylabtool calc Si --density 2.33 --energy 8000 --output json

Error Handling
--------------

Clear error messages with suggestions provided for invalid inputs, missing arguments, and unsupported values.

Integration Examples
--------------------

**Shell Script:**

.. code-block:: bash

   for material in Si Al Cu; do
       xraylabtool calc $material --density 2.33 --energy 8000 --output csv >> results.csv
   done

**Python:**

.. code-block:: python

   import subprocess, json
   result = subprocess.run(["xraylabtool", "calc", "Si", "--density", "2.33",
                           "--energy", "8000", "--output", "json"],
                          capture_output=True, text=True)
   data = json.loads(result.stdout)

**Performance Tips:** Use batch processing, enable --show-progress, adjust --chunk-size
