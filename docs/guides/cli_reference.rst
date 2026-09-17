CLI Reference
=============

Command-line interface for X-ray property calculations and shell-completion management.
Every option below is taken from ``xraylabtool <command> --help``; run that for the
authoritative list on your installed version.

**Usage:** ``xraylabtool [GLOBAL OPTIONS] COMMAND [OPTIONS]``

**Alias:** ``xtool`` is a short alias for ``xraylabtool`` and accepts the same commands and options
(e.g. ``xtool calc Si -e 8.0 -d 2.33``). The desktop GUI is launched with
``xraylabtool-gui`` or its alias ``xtool-gui``.

**Commands:** calc, batch, compare, convert, formula, atomic, bragg, list, completion,
install-completion, uninstall-completion

**Global Options:** ``--version``, ``--help``, ``-v/--verbose``, ``--debug``

.. note::

   All energies are in **keV** (tabulated range 0.03–30 keV). ``8000`` triggers an out-of-range warning;
   use ``8.0`` for Cu Kα-like energies.

Energy specification
--------------------

``-e/--energy`` accepts:

- Single value: ``10.0``
- Comma-separated: ``5.0,10.0,15.0``
- Linear range with count: ``5-15:11`` (11 points from 5 to 15 keV)
- Log range with count: ``1-30:100:log`` (100 log-spaced points)

calc - Single Material Calculation
-----------------------------------

**Usage:** ``xraylabtool calc FORMULA -e ENERGY -d DENSITY [OPTIONS]``

**Options:**

- ``-e/--energy ENERGY`` (required) — keV, see *Energy specification*
- ``-d/--density DENSITY`` (required) — g/cm³
- ``-o/--output FILE`` — save to CSV or JSON (format inferred from extension)
- ``--format {table,csv,json}`` — stdout format (default: table)
- ``--fields FIELDS`` — comma-separated subset of result fields
- ``--precision N`` — decimal places
- ``--debug``

.. code-block:: bash

   xraylabtool calc Si -e 8.0 -d 2.33
   xraylabtool calc SiO2 -e 5.0,8.0,10.0 -d 2.2 --format csv
   xraylabtool calc Al -e 1-20:100 -d 2.70 -o al_sweep.csv

batch - Batch Processing
-------------------------

**Usage:** ``xraylabtool batch INPUT_FILE -o OUTPUT [OPTIONS]``

**CSV Format:** ``formula,density,energy`` columns

**Options:**

- ``-o/--output FILE`` (required) — CSV or JSON by extension
- ``--format {csv,json}`` — override auto-detected format
- ``--workers N`` — parallel workers (default: auto)
- ``--fields FIELDS`` — comma-separated subset of result fields
- ``--progress`` / ``--no-progress`` — progress bar control
- ``--debug``

.. code-block:: bash

   xraylabtool batch materials.csv -o results.csv
   xraylabtool batch large_dataset.csv -o results.json --workers 8 --progress

compare - Multi-Material Comparison
-----------------------------------

**Usage:** ``xraylabtool compare MATERIAL [MATERIAL ...] -e ENERGY [OPTIONS]``

Each ``MATERIAL`` is ``formula,density`` (e.g. ``SiO2,2.2``).

**Options:**

- ``-e/--energy ENERGY`` (required)
- ``--properties LIST`` — comma-separated properties (default: all standard)
- ``-o/--output FILE``, ``--format {table,csv,json}``, ``--precision N``
- ``--report`` — detailed comparison report

.. code-block:: bash

   xraylabtool compare SiO2,2.2 Si3N4,3.2 -e 8.0,10.0,12.0 --properties dispersion_delta,absorption_beta
   xraylabtool compare SiO2,2.2 Al2O3,3.95 -e 10.0 -o comparison.csv

convert - Unit Conversion
-------------------------

**Usage:** ``xraylabtool convert {energy,wavelength} VALUES --to {energy,wavelength} [-o FILE]``

Energies in keV, wavelengths in Å. ``VALUES`` is a single number or a comma-separated list.

.. code-block:: bash

   xraylabtool convert energy 8.0 --to wavelength
   xraylabtool convert wavelength 1.55 --to energy
   xraylabtool convert energy 5.0,8.0,10.0 --to wavelength

formula - Formula Analysis
--------------------------

**Usage:** ``xraylabtool formula FORMULAS [-o FILE] [--debug]``

Prints element composition, atom counts, and molecular weight. ``FORMULAS`` may be
comma-separated.

.. code-block:: bash

   xraylabtool formula SiO2
   xraylabtool formula "Ca5(PO4)3F"
   xraylabtool formula SiO2,Al2O3 -o formulas.json

atomic - Atomic Data Lookup
----------------------------

**Usage:** ``xraylabtool atomic ELEMENTS [-o FILE]``

.. code-block:: bash

   xraylabtool atomic Si
   xraylabtool atomic Si,O,Al -o elements.csv

bragg - Bragg Diffraction
-------------------------

**Usage:** ``xraylabtool bragg -d DSPACING (-e ENERGY | -w WAVELENGTH) [--order N] [-o FILE]``

``DSPACING`` in Å, single or comma-separated.

.. code-block:: bash

   xraylabtool bragg -d 3.14 -e 8.0
   xraylabtool bragg -d 3.14,1.92,1.64 -e 8.0
   xraylabtool bragg -d 3.14 -w 1.55 --order 2

list - Reference Information
----------------------------

**Usage:** ``xraylabtool list {constants,fields,examples}``

.. code-block:: bash

   xraylabtool list constants   # physical constants used in calculations
   xraylabtool list fields      # XRayResult field names for --fields
   xraylabtool list examples    # example invocations

completion - Virtual Environment-Centric Shell Completion
---------------------------------------------------------

**Usage:** ``xraylabtool completion ACTION [OPTIONS]``

Installs per virtual environment and activates/deactivates with it.

.. code-block:: bash

   xraylabtool completion install                  # current environment, auto-detected shell
   xraylabtool completion install --shell zsh      # bash | zsh | fish | powershell
   xraylabtool completion install --env myenv --force
   xraylabtool completion list                     # environments with completion status
   xraylabtool completion status                   # current environment
   xraylabtool completion uninstall                # current environment
   xraylabtool completion uninstall --all          # every environment
   xraylabtool completion info                     # system information

**Supported Environments:** venv / virtualenv, conda / mamba, Poetry, Pipenv

**Supported Shells:** bash, zsh, fish, PowerShell

install-completion / uninstall-completion - Legacy Compatibility
----------------------------------------------------------------

**Install:** ``xraylabtool install-completion [--user | --system] [--test]``

**Uninstall:** ``xraylabtool uninstall-completion``

These legacy commands delegate to the ``completion`` backend:

.. code-block:: bash

   xraylabtool install-completion           # current user (default)
   xraylabtool install-completion --test    # verify completion works
   xraylabtool uninstall-completion

Output Formats
--------------

``--format`` selects the stdout format for ``calc`` and ``compare``: ``table`` (default),
``csv``, ``json``. ``-o/--output FILE`` writes to disk with the format inferred from the
extension.

.. code-block:: bash

   xraylabtool calc Si -e 8.0 -d 2.33 --format csv
   xraylabtool calc Si -e 8.0 -d 2.33 --format json
   xraylabtool calc Si -e 8.0 -d 2.33 -o si.json

Error Handling
--------------

Invalid formulas and missing required arguments exit non-zero with a one-line message;
energies outside 0.03–30 keV print a warning. Add ``--debug`` for a traceback.

Integration Examples
--------------------

**Shell Script:**

.. code-block:: bash

   for material in Si Al Cu; do
       xraylabtool calc $material -e 8.0 -d 2.33 --format csv >> results.csv
   done

**Python:**

.. code-block:: python

   import json
   import subprocess

   result = subprocess.run(
       ["xraylabtool", "calc", "Si", "-e", "8.0", "-d", "2.33", "--format", "json"],
       capture_output=True,
       text=True,
       check=True,
   )
   data = json.loads(result.stdout)

**Performance Tips:** Use ``batch`` for many materials; tune ``--workers``.
