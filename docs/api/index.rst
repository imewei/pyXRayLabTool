API Reference
=============

This section contains the API reference for XRayLabTool's modular architecture.

XRayLabTool is organized into focused sub-packages:

- **calculators**: Core X-ray physics calculations
- **data_handling**: Atomic data caching and batch processing
- **interfaces**: CLI and completion systems
- **validation**: Input validation and error handling
- **io_operations**: File operations and export functionality
- **analysis**: Material comparison and absorption edge detection
- **gui**: PySide6 desktop application
- **backend**: NumPy/JAX array backend abstraction
- **utils**: Utility functions and constants

.. toctree::
   :maxdepth: 2
   :caption: Core API

   calculators
   data_handling
   interfaces
   validation
   io_operations
   utils
   constants

.. toctree::
   :maxdepth: 2
   :caption: Extended API

   analysis
   gui
   backend

High-Level Interface
--------------------

The primary entry points for most users:

.. currentmodule:: xraylabtool

Main Calculation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: calculate_single_material_properties
   :no-index:

.. autofunction:: calculate_xray_properties
   :no-index:


Quick Reference
---------------

**Single Material Calculation:**

.. code-block:: python

   import xraylabtool as xrt

   result = xrt.calculate_single_material_properties("Si", 8.0, 2.33)  # formula, keV, g/cm³

**Batch Calculation:**

.. code-block:: python

   results = xrt.calculate_xray_properties(
       ["Si", "Al"], [5.0, 8.0, 10.0], [2.33, 2.70]
   )  # dict[formula, XRayResult]

**Formula Parsing:**

.. code-block:: python

   from xraylabtool.utils import parse_formula

   composition = parse_formula("SiO2")
   # Returns: {"Si": 1, "O": 2}

**Unit Conversions:**

.. code-block:: python

   from xraylabtool.utils import energy_to_wavelength, wavelength_to_energy

   wavelength = energy_to_wavelength(8.0)  # keV -> Å (1.5498)
   energy = wavelength_to_energy(1.55)     # Å -> keV (7.999)

Exception Hierarchy
-------------------

.. autoclass:: xraylabtool.exceptions.XRayLabToolError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: xraylabtool.exceptions.ValidationError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: xraylabtool.exceptions.FormulaError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: xraylabtool.exceptions.EnergyError
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: xraylabtool.exceptions.CalculationError
   :members:
   :show-inheritance:
   :no-index:
