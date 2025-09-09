Constants Module
================

The constants module provides physical constants and conversion factors used throughout XRayLabTool.

.. currentmodule:: xraylabtool.constants

Physical Constants
------------------

.. autodata:: xraylabtool.constants.PLANCK_CONSTANT
   
   Planck's constant in eV·s.
   
   **Value:** 4.135667696e-15 eV·s

.. autodata:: xraylabtool.constants.SPEED_OF_LIGHT
   
   Speed of light in vacuum in m/s.
   
   **Value:** 299,792,458 m/s

.. autodata:: xraylabtool.constants.ELECTRON_CHARGE
   
   Elementary charge in Coulombs.
   
   **Value:** 1.602176634e-19 C

.. autodata:: xraylabtool.constants.AVOGADRO_NUMBER
   
   Avogadro's number in mol⁻¹.
   
   **Value:** 6.02214076e23 mol⁻¹

.. autodata:: xraylabtool.constants.CLASSICAL_ELECTRON_RADIUS
   
   Classical electron radius in meters.
   
   **Value:** 2.8179403262e-15 m

.. autodata:: xraylabtool.constants.ATOMIC_MASS_UNIT
   
   Atomic mass unit in kg.
   
   **Value:** 1.66053906660e-27 kg

X-ray Specific Constants
------------------------

.. autodata:: xraylabtool.constants.HC_EV_ANGSTROM
   
   Product of Planck's constant and speed of light in eV·Å.
   
   **Value:** 12,398.4198 eV·Å
   
   **Usage:** Energy-wavelength conversions
   
   .. math::
   
      \\lambda(Å) = \\frac{12398.4198}{E(eV)}

.. autodata:: xraylabtool.constants.THOMSON_SCATTERING_LENGTH
   
   Thomson scattering length in meters.
   
   **Value:** 2.8179403262e-15 m
   
   **Usage:** Classical electron scattering cross-section calculations

.. autodata:: xraylabtool.constants.ELECTRON_REST_ENERGY
   
   Electron rest mass energy in eV.
   
   **Value:** 510,998.95 eV

Unit Conversion Factors
-----------------------

.. autodata:: xraylabtool.constants.ANGSTROM_TO_METER
   
   Conversion factor from Angstroms to meters.
   
   **Value:** 1.0e-10

.. autodata:: xraylabtool.constants.EV_TO_JOULE
   
   Conversion factor from electron volts to Joules.
   
   **Value:** 1.602176634e-19

.. autodata:: xraylabtool.constants.MRAD_TO_DEGREE
   
   Conversion factor from milliradians to degrees.
   
   **Value:** 0.057295779513

.. autodata:: xraylabtool.constants.CM_TO_METER
   
   Conversion factor from centimeters to meters.
   
   **Value:** 0.01

.. autodata:: xraylabtool.constants.G_CM3_TO_KG_M3
   
   Conversion factor from g/cm³ to kg/m³.
   
   **Value:** 1000.0

Energy Range Constants
----------------------

.. autodata:: xraylabtool.constants.MIN_ENERGY_EV
   
   Minimum supported X-ray energy in eV.
   
   **Value:** 10 eV
   
   **Note:** Below this energy, atomic data may be unreliable

.. autodata:: xraylabtool.constants.MAX_ENERGY_EV
   
   Maximum supported X-ray energy in eV.
   
   **Value:** 100,000 eV (100 keV)
   
   **Note:** Above this energy, relativistic effects become important

.. autodata:: xraylabtool.constants.TYPICAL_SYNCHROTRON_ENERGIES
   
   List of typical synchrotron X-ray energies in eV.
   
   **Values:** [3000, 5000, 8000, 10000, 12000, 15000, 20000]
   
   **Usage:** Default energy points for energy-dependent calculations

Material Property Constants
---------------------------

.. autodata:: xraylabtool.constants.VACUUM_DENSITY
   
   Density of vacuum (used for normalization).
   
   **Value:** 0.0 g/cm³

.. autodata:: xraylabtool.constants.WATER_DENSITY
   
   Standard density of water at room temperature.
   
   **Value:** 1.0 g/cm³

.. autodata:: xraylabtool.constants.SILICON_DENSITY
   
   Standard density of crystalline silicon.
   
   **Value:** 2.33 g/cm³

.. autodata:: xraylabtool.constants.COMMON_MATERIAL_DENSITIES
   
   Dictionary of densities for commonly used materials.
   
   **Example materials:**
   
   .. code-block:: python
   
      from xraylabtool.constants import COMMON_MATERIAL_DENSITIES
      
      print(COMMON_MATERIAL_DENSITIES['Si'])    # 2.33
      print(COMMON_MATERIAL_DENSITIES['Al'])    # 2.70
      print(COMMON_MATERIAL_DENSITIES['Cu'])    # 8.96

Numerical Constants
-------------------

.. autodata:: xraylabtool.constants.MACHINE_EPSILON
   
   Machine epsilon for numerical precision.
   
   **Value:** 2.220446049250313e-16
   
   **Usage:** Avoiding division by zero and numerical instabilities

.. autodata:: xraylabtool.constants.DEFAULT_TOLERANCE
   
   Default tolerance for numerical comparisons.
   
   **Value:** 1.0e-12

.. autodata:: xraylabtool.constants.MAX_ITERATIONS
   
   Maximum iterations for iterative calculations.
   
   **Value:** 1000

Mathematical Constants
----------------------

.. autodata:: xraylabtool.constants.PI
   
   Pi constant with high precision.
   
   **Value:** 3.141592653589793

.. autodata:: xraylabtool.constants.E
   
   Euler's number with high precision.
   
   **Value:** 2.718281828459045

.. autodata:: xraylabtool.constants.SQRT_2
   
   Square root of 2 with high precision.
   
   **Value:** 1.4142135623730951

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