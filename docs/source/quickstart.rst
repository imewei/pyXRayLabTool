Quick Start Guide
=================

This guide will get you up and running with XRayLabTool in just a few minutes.

Your First Calculation
-----------------------

Let's start with a simple X-ray property calculation for silicon dioxide (quartz):

.. code-block:: python

   import xraylabtool as xlt
   import numpy as np

   # Calculate properties for SiO2 at 10 keV
   result = xlt.calculate_single_material_properties(
       "SiO2", 10.0, 2.2)

   # Display key results
   print(f"Material: {result.formula}")
   print(
       f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(
       f"Attenuation length: {result.attenuation_length_cm[0]:.2f} cm")

**Expected Output:**

.. code-block:: text

   Material: SiO2
   Critical angle: 0.152°
   Attenuation length: 45.23 cm

Basic Python API
----------------

Single Material Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Calculate X-ray properties for a single material:

.. code-block:: python

   import xraylabtool as xlt

   # Single energy calculation
   result = xlt.calculate_single_material_properties(
       "Si", 10.0, 2.33)

   print(f"Silicon at 10 keV:")
   print(f"  Dispersion (δ): {result.dispersion_delta[0]:.2e}")
   print(f"  Absorption (β): {result.absorption_beta[0]:.2e}")
   print(f"  Critical angle: {result.critical_angle_degrees[0]:.3f}°")

Multiple Energy Points
~~~~~~~~~~~~~~~~~~~~~~

Analyze material properties across an energy range:

.. code-block:: python

   import numpy as np

   # Energy range from 8-12 keV (5 points)
   energies = np.linspace(8.0, 12.0, 5)
   result = xlt.calculate_single_material_properties("Al2O3", energies, 3.95)

   print("Sapphire (Al2O3) energy scan:")
   for i, energy in enumerate(result.energy_kev):
       print(f"  {energy:.1f} keV: θc = {result.critical_angle_degrees[i]:.3f}°")

Comparing Multiple Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare X-ray properties of different materials:

.. code-block:: python

   # Common X-ray optics materials
   materials = {
       "SiO2": 2.2,      # Fused silica
       "Si": 2.33,       # Silicon
       "Al2O3": 3.95,    # Sapphire
       "C": 3.52,        # Diamond
   }

   energy = 10.0  # keV
   formulas = list(materials.keys())
   densities = list(materials.values())

   results = xlt.calculate_xray_properties(formulas, energy, densities)

   print(f"Material comparison at {energy} keV:")
   for formula, result in results.items():
       print(f"  {formula:6}: θc = {result.critical_angle_degrees[0]:.3f}°, "
             f"μ = {result.attenuation_length_cm[0]:.1f} cm")

Command-Line Interface
----------------------

Quick CLI Examples
~~~~~~~~~~~~~~~~~~

XRayLabTool provides a powerful command-line interface for rapid calculations:

.. tabs::

   .. group-tab:: Basic Calculation

      .. code-block:: bash

         # Single material at one energy
         xraylabtool calc SiO2 -e 10.0 -d 2.2

   .. group-tab:: Energy Range

      .. code-block:: bash

         # Energy sweep from 5-15 keV (11 points)
         xraylabtool calc Si -e 5-15:11 -d 2.33

   .. group-tab:: Multiple Energies

      .. code-block:: bash

         # Specific energy points
         xraylabtool calc Al2O3 -e 8.0,10.0,12.0 -d 3.95

   .. group-tab:: Save Results

      .. code-block:: bash

         # Save to CSV file
         xraylabtool calc SiO2 -e 8-12:5 -d 2.2 -o results.csv

**Sample CLI Output:**

.. code-block:: text

   Material: SiO2 (density: 2.2 g/cm³)
   ┌─────────┬──────────┬─────────────┬─────────────┬─────────────┬──────────────┐
   │ Energy  │ Critical │ Dispersion  │ Absorption  │ Attenuation │ Scattering   │
   │ (keV)   │ Angle(°) │ Delta       │ Beta        │ Length(cm)  │ Length(Å⁻²)  │
   ├─────────┼──────────┼─────────────┼─────────────┼─────────────┼──────────────┤
   │  10.000 │    0.152 │   1.15e-06  │   2.55e-09  │      45.23  │    7.96e-06  │
   └─────────┴──────────┴─────────────┴─────────────┴─────────────┴──────────────┘

Understanding the Results
-------------------------

XRayResult Data Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

All calculations return an ``XRayResult`` object with descriptive field names:

.. code-block:: python

   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

   # Material properties
   print(result.formula)                        # "SiO2"
   print(result.molecular_weight_g_mol)         # 60.08 g/mol
   print(result.density_g_cm3)                  # 2.2 g/cm³
   print(result.electron_density_per_ang3)      # electrons/Å³

   # X-ray properties (arrays for multiple energies)
   print(result.energy_kev)                     # [10.0] keV
   print(result.wavelength_angstrom)            # [1.24] Å
   print(result.dispersion_delta)               # [δ] values
   print(result.absorption_beta)                # [β] values

   # Derived quantities
   print(result.critical_angle_degrees)         # [θc] in degrees
   print(result.attenuation_length_cm)          # [μ] in cm
   print(result.real_sld_per_ang2)             # Real SLD in Å⁻²

Key Physical Quantities
~~~~~~~~~~~~~~~~~~~~~~~

**Dispersion (δ)**: Determines refraction and critical angles

.. code-block:: python

   # Larger δ means larger critical angle
   δ = result.dispersion_delta[0]
   critical_angle = result.critical_angle_degrees[0]
   print(f"δ = {δ:.2e} → θc = {critical_angle:.3f}°")

**Absorption (β)**: Determines attenuation through materials

.. code-block:: python

   # Smaller β means less absorption (more transparent)
   β = result.absorption_beta[0]
   attenuation = result.attenuation_length_cm[0]
   print(f"β = {β:.2e} → μ = {attenuation:.1f} cm")

Common Use Cases
----------------

Mirror Reflectivity
~~~~~~~~~~~~~~~~~~~

Calculate critical angle for X-ray mirrors:

.. code-block:: python

   # Platinum-coated mirror at 10 keV
   result = xlt.calculate_single_material_properties("Pt", 10.0, 21.45)

   critical_angle = result.critical_angle_degrees[0]
   print(f"Pt mirror critical angle: {critical_angle:.3f}°")
   print(f"For total external reflection, use θ < {critical_angle:.3f}°")

Absorption Analysis
~~~~~~~~~~~~~~~~~~~

Determine optimal thickness for X-ray windows:

.. code-block:: python

   # Beryllium window at 8 keV
   result = xlt.calculate_single_material_properties("Be", 8.0, 1.85)

   attenuation_length = result.attenuation_length_cm[0]
   thickness_10percent = attenuation_length * 0.1  # 90% transmission

   print(f"Be window at 8 keV:")
   print(f"  Attenuation length: {attenuation_length:.1f} cm")
   print(f"  For 90% transmission: {thickness_10percent*10:.1f} mm thick")

Energy-Dependent Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how properties change with energy:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Energy range for copper
   energies = np.logspace(np.log10(1), np.log10(20), 50)  # 1-20 keV
   result = xlt.calculate_single_material_properties("Cu", energies, 8.96)

   # Find energy where absorption edge occurs (if any in range)
   max_beta_idx = np.argmax(result.absorption_beta)
   edge_energy = result.energy_kev[max_beta_idx]

   print(f"Copper analysis (1-20 keV):")
   print(f"  Peak absorption at: {edge_energy:.1f} keV")
   print(f"  Critical angle range: {result.critical_angle_degrees.min():.3f}° - "
         f"{result.critical_angle_degrees.max():.3f}°")

Shell Completion
----------------

Enable intelligent tab completion for faster CLI usage:

.. code-block:: bash

   # Install completion for your shell
   xraylabtool install-completion

   # Now try typing and pressing Tab:
   xraylabtool calc Si<TAB>        # Suggests SiO2, Si, etc.
   xraylabtool calc SiO2 -e <TAB>  # Suggests common energies
   xraylabtool calc SiO2 -e 10.0 -d <TAB>  # Suggests densities

Next Steps
----------

Now that you've mastered the basics:

.. grid:: 2

    .. grid-item-card:: 📚 Tutorials
        :link: tutorials/index
        :link-type: doc

        Detailed tutorials for specific use cases

    .. grid-item-card:: 🖥️ CLI Guide
        :link: cli_guide
        :link-type: doc

        Complete command-line reference

    .. grid-item-card:: 📊 Examples
        :link: examples
        :link-type: doc

        Real-world calculation examples

    .. grid-item-card:: ⚡ Performance
        :link: performance_guide
        :link-type: doc

        Optimize for high-performance computing

Troubleshooting
---------------

**Common Issues:**

.. dropdown:: ImportError: No module named 'xraylabtool'

   Make sure you've installed XRayLabTool and activated the correct environment:

   .. code-block:: bash

      pip install xraylabtool
      python -c "import xraylabtool; print('Success!')"

.. dropdown:: Slow calculations

   For better performance:

   - Use common elements (Si, O, Al, Fe, C) which are preloaded in cache
   - Consider batch processing for multiple materials
   - Check the :doc:`performance_guide` for optimization tips

.. dropdown:: Unexpected results

   Verify your inputs:

   .. code-block:: python

      # Check if formula is parsed correctly
      from xraylabtool.utils import parse_formula
      elements, counts = parse_formula("Al2O3")
      print(f"Elements: {elements}, Counts: {counts}")

Need help? Check the :doc:`faq` or open an issue on `GitHub <https://github.com/imewei/pyXRayLabTool/issues>`_.
