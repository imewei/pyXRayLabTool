Constants Module
================

The constants module provides physical constants and conversion factors used throughout XRayLabTool.

.. automodule:: xraylabtool.constants
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**Energy-Wavelength Conversion:**

.. code-block:: python

   from xraylabtool.constants import HC_EV_ANGSTROM

   def energy_to_wavelength(energy_ev):
       return HC_EV_ANGSTROM / energy_ev

   wavelength = energy_to_wavelength(8000)  # 1.55 Å

**Critical Angle Calculation:**

.. code-block:: python

   from xraylabtool.constants import MRAD_TO_DEGREE
   import numpy as np

   def critical_angle_degrees(delta):
       theta_mrad = 1000 * np.sqrt(2 * delta)  # Convert to mrad
       return theta_mrad * MRAD_TO_DEGREE

**Unit Conversions:**

.. code-block:: python

   from xraylabtool.constants import G_CM3_TO_KG_M3, ANGSTROM_TO_METER

   # Convert density
   density_si = 2.33 * G_CM3_TO_KG_M3  # g/cm³ to kg/m³

   # Convert wavelength
   wavelength_m = 1.55 * ANGSTROM_TO_METER  # Å to m

**Material Properties:**

.. code-block:: python

   from xraylabtool.constants import COMMON_MATERIAL_DENSITIES, SILICON_DENSITY

   # Get standard material densities
   materials = ['Si', 'Al', 'Cu', 'Fe']
   for material in materials:
       if material in COMMON_MATERIAL_DENSITIES:
           density = COMMON_MATERIAL_DENSITIES[material]
           print(f"{material}: {density} g/cm³")

**Energy Range Validation:**

.. code-block:: python

   from xraylabtool.constants import MIN_ENERGY_EV, MAX_ENERGY_EV

   def validate_energy(energy):
       if energy < MIN_ENERGY_EV:
           raise ValueError(f"Energy {energy} eV below minimum {MIN_ENERGY_EV} eV")
       if energy > MAX_ENERGY_EV:
           raise ValueError(f"Energy {energy} eV above maximum {MAX_ENERGY_EV} eV")
       return True

Constants Reference Table
-------------------------

.. list-table:: Key Physical Constants
   :header-rows: 1
   :widths: 30 30 25 15

   * - Constant
     - Symbol
     - Value
     - Unit
   * - Planck constant
     - h
     - 4.136e-15
     - eV·s
   * - Speed of light
     - c
     - 2.998e8
     - m/s
   * - Classical electron radius
     - r₀
     - 2.818e-15
     - m
   * - Electron rest energy
     - mₑc²
     - 510,999
     - eV
   * - hc product
     - hc
     - 12,398.4
     - eV·Å

.. list-table:: Conversion Factors
   :header-rows: 1
   :widths: 40 35 25

   * - Conversion
     - Factor
     - Usage
   * - Å → m
     - 1.0e-10
     - Length units
   * - eV → J
     - 1.602e-19
     - Energy units
   * - mrad → deg
     - 0.0573
     - Angular units
   * - g/cm³ → kg/m³
     - 1000
     - Density units

CODATA Standards
----------------

All physical constants are based on the 2018 CODATA internationally recommended values, ensuring compatibility with modern scientific standards and other physics software packages.
