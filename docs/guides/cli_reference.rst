CLI Reference
=============

Command-line interface for X-ray property calculations and shell-completion management.
Every option below is taken from ``xraylabtool <command> --help``; run that for the
authoritative list on your installed version.

**Usage:** ``xraylabtool [GLOBAL OPTIONS] COMMAND [OPTIONS]``

**Alias:** ``xtool`` and ``xlt`` are short aliases for ``xraylabtool`` and accepts the same commands and options
(e.g. ``xtool calc Si -e 8.0 -d 2.33``). The desktop GUI is launched with
``xraylabtool-gui`` or its aliases ``xtool-gui`` and ``xlt-gui``.

**Commands:** calc, batch, compare, convert, kk, formula, atomic, bragg, list, completion,
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
- ``--fields FIELDS`` — comma-separated subset of result fields; an unknown name prints
  ``Unknown field(s): ...`` and exits 1 for every output format (``xraylabtool list fields``
  shows the valid names)
- ``--precision N`` — decimal places
- ``--debug``

A formula whose element counts are all zero (for example ``Si0``) has no molecular
weight and exits 1 with ``formula 'Si0' has zero molecular weight``.

.. code-block:: bash

   xraylabtool calc Si -e 8.0 -d 2.33
   xraylabtool calc SiO2 -e 5.0,8.0,10.0 -d 2.2 --format csv
   xraylabtool calc Al -e 1-20:100 -d 2.70 -o al_sweep.csv

batch - Batch Processing
-------------------------

**Usage:** ``xraylabtool batch INPUT_FILE -o OUTPUT [OPTIONS]``

**CSV Format:** ``formula,density,energy`` columns (a UTF-8 BOM is accepted; duplicate
column names are rejected). Give several energies for one material as a quoted,
comma-separated cell, e.g. ``SiO2,2.2,"8,10"``. A blank energy cell, a truncated row, an unquoted extra energy cell or an unparsable
value skips that row with a warning on stderr. A formula the calculator rejects, or a
non-finite or out-of-range energy (outside 0.03-30 keV), is logged as a failed material
and left out. Valid rows are still written unless ``--strict`` is set.

**Options:**

- ``-o/--output FILE`` (required) — CSV or JSON by extension
- ``--format {csv,json}`` — override auto-detected format
- ``--workers N`` — accepted for compatibility and ignored (batches run as blocks on the JAX device, not in worker pools)
- ``--strict`` — exit with status 2 and write no output if any input row is skipped or fails while at least one row succeeds; with no successful row it exits 1 (default: write the valid rows and exit 0)
- ``--fields FIELDS`` — comma-separated subset of result fields; an unknown name exits 1
  before any material is computed and writes nothing
- ``--progress`` / ``--no-progress`` — progress bar control
- ``--debug``

.. code-block:: bash

   xraylabtool batch materials.csv -o results.csv
   xraylabtool batch large_dataset.csv -o results.json --progress

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

kk - Kramers-Kronig f1
----------------------

**Usage:** ``xraylabtool kk FORMULA [--nearedge FILE --input-type {nexafs,beta,asf}] [--energy-unit {eV,keV}] [--merge LOW,HIGH] [-d G_CM3] [--refine TOL] [--energies SPEC] [-o FILE] [--format {csv,json}]``

Calculates the real scattering factor f1 from the tabulated f2 by the
Kramers-Kronig transform and prints ``energy_keV,f1,f2`` (CSV, or JSON with the
same three keys). Without ``--nearedge`` this is the Kramers-Kronig-consistent f1 of
the tabulated spectrum. With ``--nearedge`` the measured absorption data replace the
tabulated f2 between the merge points after normalisation to it.

- ``--input-type`` is required with ``--nearedge``: ``nexafs`` (photoabsorption
  cross-section values), ``beta`` (imaginary part of the refractive index) or ``asf``
  (already f2).
- ``--energy-unit`` (default ``eV``) is the unit of the near-edge file's first column
  and of ``--merge``. ``--energies`` is always in keV.
- ``--merge LOW,HIGH`` must lie inside both the near-edge data and the tabulated
  range (10 eV to 500 keV); the default is their whole overlap.
- ``--energies`` takes the formats of ``calc -e`` (keV, 0.01 to 500). Without it the
  output grid is the spectrum's own knots within 0.03-30 keV.
- ``--refine TOL`` inserts midpoints until linear interpolation of f1 and f2 is within
  ``TOL`` electrons per formula unit; the output is then sorted and unique, and the
  number of points added is reported on stderr.
- ``--density`` is validated (finite, positive) and accepted for ``beta`` input; it
  cancels in the near-edge normalisation and does not change the result.
- Errors (bad file, missing ``--input-type``, no overlap with the tabulated range,
  energies out of range) print to stderr and exit with code 1.

.. code-block:: bash

   xraylabtool kk Si
   xraylabtool kk SiO2 --energies 1.0,2.0,5.0
   xraylabtool kk C --nearedge c_edge.txt --input-type nexafs
   xraylabtool kk C --nearedge c_edge.txt --input-type nexafs --merge 285,315 --refine 0.05 -o c_edge.csv

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

**Performance Tips:** Use ``batch`` for many materials; it runs them as blocks through one compiled program.
