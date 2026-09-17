Constants Module
================

The constants module provides physical constants and conversion factors used throughout XRayLabTool.

.. automodule:: xraylabtool.constants
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Prefer the helpers in :mod:`xraylabtool.utils` (``energy_to_wavelength``,
``wavelength_to_energy``) over recomputing from raw constants. The constants are exposed for
custom derivations:

**Energy-Wavelength Conversion:**

.. code-block:: python

   from xraylabtool.constants import ENERGY_TO_WAVELENGTH_FACTOR, METER_TO_ANGSTROM

   def energy_to_wavelength_angstrom(energy_kev):
       # ENERGY_TO_WAVELENGTH_FACTOR = h*c/e / 1000, in m·keV
       return ENERGY_TO_WAVELENGTH_FACTOR / energy_kev * METER_TO_ANGSTROM

   energy_to_wavelength_angstrom(8.0)  # 1.5498 Å

**Critical Angle in Degrees:**

.. code-block:: python

   import numpy as np
   from xraylabtool.constants import RADIANS_TO_DEGREES

   def critical_angle_degrees(delta):
       return np.sqrt(2 * delta) * RADIANS_TO_DEGREES

**Unit Conversions:**

.. code-block:: python

   from xraylabtool.constants import ANGSTROM_TO_METER, CM_TO_METER, EV_TO_KEV

   wavelength_m = 1.55 * ANGSTROM_TO_METER  # Å -> m
   length_m = 9.84 * CM_TO_METER            # cm -> m
   energy_kev = 8048 * EV_TO_KEV            # eV -> keV

**Scattering Length Density Prefactor:**

.. code-block:: python

   from xraylabtool.constants import AVOGADRO, THOMPSON

   # THOMPSON is the classical electron radius r_e in metres; the δ/β kernels use
   # SCATTERING_FACTOR = THOMPSON * AVOGADRO * 1e6 / (2π)  (g/cm³ -> kg/m³ folded in).

Constants Reference Table
-------------------------

.. list-table:: Physical Constants (module attribute, value)
   :header-rows: 1
   :widths: 35 25 25 15

   * - Constant
     - Attribute
     - Value
     - Unit
   * - Planck constant
     - ``PLANCK``
     - 6.626068e-34
     - J·s
   * - Speed of light
     - ``SPEED_OF_LIGHT``
     - 2.99792458e8
     - m/s
   * - Elementary charge
     - ``ELEMENT_CHARGE``
     - 1.60217646e-19
     - C
   * - Avogadro constant
     - ``AVOGADRO``
     - 6.02214199e23
     - mol⁻¹
   * - Classical electron radius
     - ``THOMPSON``
     - 2.8179403227e-15
     - m
   * - hc/e ÷ 1000
     - ``ENERGY_TO_WAVELENGTH_FACTOR``
     - 1.23984e-9
     - m·keV

.. list-table:: Conversion Factors
   :header-rows: 1
   :widths: 40 35 25

   * - Conversion
     - Attribute
     - Factor
   * - Å → m / m → Å
     - ``ANGSTROM_TO_METER`` / ``METER_TO_ANGSTROM``
     - 1e-10 / 1e10
   * - cm → m / m → cm
     - ``CM_TO_METER`` / ``METER_TO_CM``
     - 1e-2 / 1e2
   * - keV → eV / eV → keV
     - ``KEV_TO_EV`` / ``EV_TO_KEV``
     - 1e3 / 1e-3
   * - rad → deg / deg → rad
     - ``RADIANS_TO_DEGREES`` / ``DEGREES_TO_RADIANS``
     - 57.2958 / 0.0174533

CODATA Standards
----------------

All physical constants are based on the 2018 CODATA internationally recommended values, ensuring compatibility with modern scientific standards and other physics software packages.
