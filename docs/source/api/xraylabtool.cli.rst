xraylabtool.cli module
======================

Command-line interface module providing comprehensive CLI functionality for XRayLabTool.

This module includes all CLI commands, argument parsing, result formatting, and the main entry point
for the ``xraylabtool`` command-line application.

.. automodule:: xraylabtool.cli
   :members:
   :undoc-members:
   :show-inheritance:

CLI Commands
------------

The CLI provides the following commands:

* **calc** - Single material X-ray property calculations
* **batch** - Batch processing from CSV files
* **convert** - Energy/wavelength unit conversions
* **formula** - Chemical formula parsing and analysis
* **atomic** - Atomic data lookup for elements
* **bragg** - Bragg diffraction angle calculations
* **list** - List constants, fields, and examples
* **install-completion** - Install shell completion functionality

Command Functions
-----------------

.. autofunction:: xraylabtool.cli.cmd_calc
.. autofunction:: xraylabtool.cli.cmd_batch
.. autofunction:: xraylabtool.cli.cmd_convert
.. autofunction:: xraylabtool.cli.cmd_formula
.. autofunction:: xraylabtool.cli.cmd_atomic
.. autofunction:: xraylabtool.cli.cmd_bragg
.. autofunction:: xraylabtool.cli.cmd_list
.. autofunction:: xraylabtool.cli.cmd_install_completion

Utility Functions
-----------------

.. autofunction:: xraylabtool.cli.parse_energy_string
.. autofunction:: xraylabtool.cli.format_xray_result
.. autofunction:: xraylabtool.cli.main
