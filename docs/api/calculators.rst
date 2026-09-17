Calculators Module
==================

The calculators module contains the core X-ray physics calculations and the main result data structure.

.. currentmodule:: xraylabtool.calculators

Core Calculations
-----------------

.. automodule:: xraylabtool.calculators.core
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Formula, MW, Number_Of_Electrons, Density, Electron_Density, Energy, Wavelength, Dispersion, Absorption, f1, f2, Critical_Angle, Attenuation_Length, reSLD, imSLD
   :no-index:

   **XRayResult fields** (all per-energy fields are NumPy arrays aligned with ``energy_kev``):

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Field Name
        - Unit
        - Description
      * - ``formula``
        - -
        - Chemical formula of the material
      * - ``molecular_weight_g_mol``
        - g/mol
        - Molecular weight
      * - ``total_electrons``
        - -
        - Electrons per formula unit
      * - ``density_g_cm3``
        - g/cm³
        - Material density
      * - ``electron_density_per_ang3``
        - Å⁻³
        - Electron number density
      * - ``energy_kev``
        - keV
        - X-ray photon energies
      * - ``wavelength_angstrom``
        - Å
        - X-ray wavelengths
      * - ``dispersion_delta``
        - -
        - Real part of refractive index decrement (δ)
      * - ``absorption_beta``
        - -
        - Imaginary part of refractive index decrement (β)
      * - ``scattering_factor_f1``
        - electrons
        - Real atomic scattering factor summed over formula
      * - ``scattering_factor_f2``
        - electrons
        - Imaginary atomic scattering factor summed over formula
      * - ``critical_angle_degrees``
        - degrees
        - Critical angle for total external reflection
      * - ``attenuation_length_cm``
        - cm
        - 1/e attenuation length
      * - ``real_sld_per_ang2``
        - Å⁻²
        - Real scattering length density
      * - ``imaginary_sld_per_ang2``
        - Å⁻²
        - Imaginary scattering length density

   Legacy CamelCase accessors (``Formula``, ``MW``, ``Energy``, ``Dispersion``, ...) still
   resolve with a ``DeprecationWarning``.

Derived Quantities
------------------

.. automodule:: xraylabtool.calculators.derived_quantities
   :members:
   :undoc-members:
   :show-inheritance:

XRay Result
-----------

.. automodule:: xraylabtool.calculators.xray_result
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Calculation Cache
-----------------

.. automodule:: xraylabtool.calculators.cache
   :members:
   :undoc-members:
   :show-inheritance:

Scattering Data
---------------

.. automodule:: xraylabtool.calculators.scattering_data
   :members:
   :undoc-members:
   :show-inheritance:

Kernels
-------

.. automodule:: xraylabtool.calculators.kernels
   :members:
   :undoc-members:
   :show-inheritance:

Physics Background
------------------

The calculations are based on the complex refractive index for X-rays:

.. math::

   n = 1 - \\delta - i\\beta

Where:

- **δ (delta)**: Real part of the refractive index decrement, related to phase shifts
- **β (beta)**: Imaginary part, related to absorption

The critical angle for total external reflection is:

.. math::

   \\theta_c = \\sqrt{2\\delta}

The linear absorption coefficient is:

.. math::

   \\mu = \\frac{4\\pi \\beta}{\\lambda}

Usage Examples
--------------

**Single Energy Calculation:**

.. code-block:: python

   from xraylabtool.calculators.core import calculate_single_material_properties

   result = calculate_single_material_properties("Si", 8.0, 2.33)  # formula, keV, g/cm³

   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm[0]:.4f} cm")

**Energy Array Calculation:**

.. code-block:: python

   import numpy as np
   from xraylabtool.calculators.core import calculate_single_material_properties

   energies = np.logspace(0, np.log10(30), 100)  # 1 keV to 30 keV

   # One call; every per-energy field comes back as an array of length 100
   result = calculate_single_material_properties("Si", energies, 2.33)
   print(result.critical_angle_degrees.shape)  # (100,)

**Multiple Materials:**

.. code-block:: python

   from xraylabtool.calculators.core import calculate_xray_properties

   results = calculate_xray_properties(
       ["Si", "SiO2", "Al"], 8.0, [2.33, 2.20, 2.70]
   )  # dict[str, XRayResult] keyed by formula
