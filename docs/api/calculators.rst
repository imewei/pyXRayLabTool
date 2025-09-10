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

   **Field Descriptions:**

   .. list-table::
      :header-rows: 1
      :widths: 30 20 50

      * - Field Name
        - Unit
        - Description
      * - ``formula``
        - -
        - Chemical formula of the material
      * - ``density_g_cm3``
        - g/cm³
        - Material density
      * - ``energy_ev``
        - eV
        - X-ray photon energy
      * - ``wavelength_angstrom``
        - Å
        - X-ray wavelength
      * - ``delta``
        - -
        - Real part of refractive index decrement
      * - ``beta``
        - -
        - Imaginary part of refractive index decrement
      * - ``attenuation_length_cm``
        - cm
        - 1/e attenuation length
      * - ``critical_angle_mrad``
        - mrad
        - Critical angle for total external reflection
      * - ``critical_angle_degrees``
        - degrees
        - Critical angle in degrees
      * - ``linear_absorption_coefficient``
        - cm⁻¹
        - Linear absorption coefficient (μ)
      * - ``mass_absorption_coefficient``
        - cm²/g
        - Mass absorption coefficient (μ/ρ)

Derived Quantities
------------------

.. automodule:: xraylabtool.calculators.derived_quantities
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

   result = calculate_single_material_properties(
       formula="Si",
       density=2.33,
       energy=8000
   )

   print(f"Critical angle: {result.critical_angle_degrees:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm:.2f} cm")

**Energy Array Calculation:**

.. code-block:: python

   import numpy as np
   from xraylabtool.calculators.core import calculate_single_material_properties

   energies = np.logspace(3, 5, 100)  # 1 keV to 100 keV

   results = []
   for energy in energies:
       result = calculate_single_material_properties(
           formula="Si", density=2.33, energy=energy
       )
       results.append(result)

**Multiple Materials:**

.. code-block:: python

   from xraylabtool.calculators.core import calculate_xray_properties

   materials = [
       {"formula": "Si", "density": 2.33},
       {"formula": "SiO2", "density": 2.20},
       {"formula": "Al", "density": 2.70}
   ]

   results = calculate_xray_properties(materials, energy=8000)
