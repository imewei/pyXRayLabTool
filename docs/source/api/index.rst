API Reference
=============

This section contains the complete API documentation for XRayLabTool.

.. grid:: 2

    .. grid-item-card:: 🧮 Core Calculators
        :link: xraylabtool.calculators
        :link-type: doc

        Main calculation engines for X-ray optical properties

    .. grid-item-card:: 📊 Data Handling
        :link: xraylabtool.data_handling
        :link-type: doc

        Batch processing and memory management utilities

    .. grid-item-card:: 🖥️ Interfaces
        :link: xraylabtool.interfaces
        :link-type: doc

        Command-line interface and shell completion

    .. grid-item-card:: 💾 Input/Output
        :link: xraylabtool.io
        :link-type: doc

        File operations and data export functionality

    .. grid-item-card:: ✅ Validation
        :link: xraylabtool.validation
        :link-type: doc

        Input validation and error handling

Quick Access
------------

**Most commonly used functions:**

.. currentmodule:: xraylabtool

.. autofunction:: calculate_single_material_properties
   :no-index:
.. autofunction:: calculate_xray_properties
   :no-index:
.. autofunction:: energy_to_wavelength
   :no-index:
.. autofunction:: wavelength_to_energy
   :no-index:
.. autofunction:: parse_formula
   :no-index:

**Main data structure:**

.. currentmodule:: xraylabtool.calculators.core

The main data structure is documented in detail below.

Package Structure
-----------------

.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   xraylabtool.calculators
   xraylabtool.data_handling
   xraylabtool.interfaces
   xraylabtool.io
   xraylabtool.validation

Legacy Modules
--------------

.. note::

   The following modules are deprecated and maintained for backward compatibility.
   New code should use the modular structure above.

.. toctree::
   :maxdepth: 2
   :caption: Legacy API (Deprecated)

   xraylabtool.core
   xraylabtool.atomic_data_cache
   xraylabtool.cli
   xraylabtool.completion_installer
   xraylabtool.constants
   xraylabtool.exceptions
   xraylabtool.utils

Main Package
------------

.. toctree::
   :maxdepth: 1

   xraylabtool

Usage Patterns
--------------

Basic Calculations
~~~~~~~~~~~~~~~~~~

The most common usage patterns for XRayLabTool:

.. code-block:: python

   import xraylabtool as xlt

   # Single material, single energy
   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

   # Single material, multiple energies
   import numpy as np
   energies = np.linspace(5, 15, 11)
   result = xlt.calculate_single_material_properties("Si", energies, 2.33)

   # Multiple materials
   materials = ["SiO2", "Al2O3", "Si"]
   densities = [2.2, 3.95, 2.33]
   results = xlt.calculate_xray_properties(materials, 10.0, densities)

Batch Processing
~~~~~~~~~~~~~~~~

For large-scale calculations:

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import (
       calculate_batch_properties, BatchConfig
   )

   # Configure for optimal performance
   config = BatchConfig(
       chunk_size=100,
       max_workers=8,
       memory_limit_gb=4.0
   )

   # Process many materials efficiently
   results = calculate_batch_properties(formulas, energies, densities, config)

Data Structures
---------------

The main data structure returned by calculations:

.. currentmodule:: xraylabtool.calculators.core

XRayResult
~~~~~~~~~~

.. autoclass:: XRayResult
   :members:
   :show-inheritance:
   :no-index:

   **Material Properties:**

   .. autoattribute:: formula
      :no-index:
   .. autoattribute:: molecular_weight_g_mol
      :no-index:
   .. autoattribute:: density_g_cm3
      :no-index:
   .. autoattribute:: total_electrons
      :no-index:
   .. autoattribute:: electron_density_per_ang3
      :no-index:

   **X-ray Properties (Arrays):**

   .. autoattribute:: energy_kev
      :no-index:
   .. autoattribute:: wavelength_angstrom
      :no-index:
   .. autoattribute:: dispersion_delta
      :no-index:
   .. autoattribute:: absorption_beta
      :no-index:
   .. autoattribute:: scattering_factor_f1
      :no-index:
   .. autoattribute:: scattering_factor_f2
      :no-index:

   **Derived Quantities (Arrays):**

   .. autoattribute:: critical_angle_degrees
      :no-index:
   .. autoattribute:: attenuation_length_cm
      :no-index:
   .. autoattribute:: real_sld_per_ang2
      :no-index:
   .. autoattribute:: imaginary_sld_per_ang2
      :no-index:

Exception Hierarchy
-------------------

XRayLabTool provides a comprehensive exception hierarchy for error handling:

.. currentmodule:: xraylabtool.validation.exceptions

.. autoexception:: XRayLabToolError
   :no-index:
.. autoexception:: CalculationError
   :no-index:
.. autoexception:: FormulaError
   :no-index:
.. autoexception:: EnergyError
   :no-index:
.. autoexception:: ValidationError
   :no-index:
.. autoexception:: AtomicDataError
   :no-index:
.. autoexception:: UnknownElementError
   :no-index:
.. autoexception:: BatchProcessingError
   :no-index:
.. autoexception:: DataFileError
   :no-index:
.. autoexception:: ConfigurationError
   :no-index:

Migration Guide
---------------

.. admonition:: Migrating from Legacy API

   If you're upgrading from an older version of XRayLabTool, note these changes:

   **Import Changes:**

   .. code-block:: python

      # Old (still works but deprecated)
      from xraylabtool.core import calculate_single_material_properties
      from xraylabtool.batch_processor import BatchProcessor

      # New (recommended)
      import xraylabtool as xlt  # Main functions available at package level
      from xraylabtool.data_handling.batch_processing import calculate_batch_properties

   **Field Name Changes:**

   .. code-block:: python

      # Old field names (deprecated but still work)
      result.Critical_Angle  # ⚠️ DeprecationWarning
      result.MW              # ⚠️ DeprecationWarning

      # New field names (recommended)
      result.critical_angle_degrees  # ✅
      result.molecular_weight_g_mol  # ✅

For complete migration information, see the :doc:`../migration_guide`.

Performance Notes
-----------------

**Optimized Functions:**

The following functions are highly optimized for performance:

- :func:`~xraylabtool.calculate_single_material_properties`: Optimized for both single and multiple energies
- :func:`~xraylabtool.calculate_xray_properties`: Parallel processing for multiple materials
- :func:`~xraylabtool.data_handling.batch_processing.calculate_batch_properties`: Memory-efficient chunked processing

**Performance Tips:**

1. **Use common elements**: Si, O, Al, Fe, C are preloaded for 10-50x speed improvement
2. **Vectorize energy calculations**: Pass arrays instead of looping
3. **Use batch processing**: For multiple materials, use parallel processing
4. **Monitor memory**: Use :class:`~xraylabtool.data_handling.batch_processing.MemoryMonitor` for large datasets

Type Information
----------------

XRayLabTool includes comprehensive type hints. For IDE support and static analysis:

.. code-block:: python

   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from xraylabtool.calculators.core import XRayResult

   def process_result(result: 'XRayResult') -> None:
       # Full type checking support
       angle: float = result.critical_angle_degrees[0]

Extension Points
----------------

For advanced users who want to extend XRayLabTool:

**Custom Atomic Data:**

.. code-block:: python

   from xraylabtool.data_handling.atomic_cache import add_custom_element

   # Add custom scattering factor data
   add_custom_element("Xx", atomic_number=119, scattering_data=custom_data)

**Custom Validators:**

.. code-block:: python

   from xraylabtool.validation.validators import register_custom_validator

   def custom_density_validator(density: float) -> bool:
       return 0.1 <= density <= 25.0  # Custom range

   register_custom_validator("density", custom_density_validator)

See the advanced tutorials for more information on extending XRayLabTool.

Index and Module Search
-----------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
