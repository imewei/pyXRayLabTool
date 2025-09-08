Basic Usage Tutorial
====================

This tutorial covers the fundamental concepts and usage patterns of XRayLabTool. After completing this guide, you'll be able to perform basic X-ray optical property calculations and understand the key concepts.

Learning Objectives
-------------------

By the end of this tutorial, you will be able to:

- Calculate X-ray optical properties for single materials
- Work with energy ranges and multiple energy points
- Compare properties of different materials
- Understand and interpret the results
- Export data for further analysis

Prerequisites
~~~~~~~~~~~~~

- Python 3.12+ with XRayLabTool installed
- Basic familiarity with Python
- Understanding of NumPy arrays (helpful but not required)

Getting Started
---------------

Let's start by importing XRayLabTool and calculating properties for a simple material:

.. code-block:: python

   import xraylabtool as xlt
   import numpy as np

   # Our first calculation: Silicon dioxide at 10 keV
   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

   print(f"Material: {result.formula}")
   print(f"Density: {result.density_g_cm3} g/cm³")
   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm[0]:.2f} cm")

Understanding the Results
-------------------------

The ``XRayResult`` Object
~~~~~~~~~~~~~~~~~~~~~~~~~

All calculations return an ``XRayResult`` object containing:

.. code-block:: python

   # Material properties (scalars)
   print("Material Properties:")
   print(f"  Formula: {result.formula}")
   print(f"  Molecular weight: {result.molecular_weight_g_mol:.2f} g/mol")
   print(f"  Total electrons: {result.total_electrons:.0f}")
   print(f"  Electron density: {result.electron_density_per_ang3:.2f} e⁻/Å³")

.. code-block:: python

   # X-ray properties (arrays, even for single energy)
   print("\nX-ray Properties:")
   print(f"  Energy: {result.energy_kev[0]:.1f} keV")
   print(f"  Wavelength: {result.wavelength_angstrom[0]:.3f} Å")
   print(f"  Dispersion (δ): {result.dispersion_delta[0]:.2e}")
   print(f"  Absorption (β): {result.absorption_beta[0]:.2e}")

.. code-block:: python

   # Derived quantities (arrays)
   print("\nDerived Quantities:")
   print(f"  Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"  Attenuation length: {result.attenuation_length_cm[0]:.1f} cm")
   print(f"  Real SLD: {result.real_sld_per_ang2[0]:.2e} Å⁻²")

Key Physical Quantities
~~~~~~~~~~~~~~~~~~~~~~~

**Dispersion (δ)**: Determines refraction and critical angles

- Larger δ → larger critical angle
- Related to electron density and energy
- Critical for X-ray mirrors and grazing incidence optics

**Absorption (β)**: Determines attenuation through materials

- Larger β → more absorption (less transparent)
- Determines the penetration depth of X-rays
- Important for window materials and filters

**Critical Angle**: Angle for total external reflection

- Grazing incidence angle where reflection becomes total
- Essential for X-ray mirror design
- θc ≈ √(2δ) for small angles

Multiple Energy Calculations
----------------------------

Working with Energy Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~

XRayLabTool efficiently handles multiple energies:

.. code-block:: python

   # Method 1: NumPy array
   energies = np.linspace(8, 12, 5)  # 8-12 keV, 5 points
   print(f"Energies: {energies}")

.. code-block:: python

   # Calculate properties across energy range
   result = xlt.calculate_single_material_properties("Al2O3", energies, 3.95)

   print("Sapphire (Al2O3) energy scan:")
   for i, energy in enumerate(result.energy_kev):
       print(f"  {energy:.1f} keV: θc = {result.critical_angle_degrees[i]:.3f}°, "
             f"μ = {result.attenuation_length_cm[i]:.1f} cm")

.. code-block:: python

   # Method 2: Logarithmic spacing (common for X-ray analysis)
   log_energies = np.logspace(np.log10(5), np.log10(20), 8)  # 5-20 keV, 8 points
   result_log = xlt.calculate_single_material_properties("Si", log_energies, 2.33)

   print("Silicon logarithmic energy scan:")
   for i, energy in enumerate(result_log.energy_kev):
       print(f"  {energy:.1f} keV: δ = {result_log.dispersion_delta[i]:.2e}")

Energy-Dependent Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~

X-ray properties change significantly with energy:

.. code-block:: python

   # Wide energy range to see trends
   wide_energies = np.logspace(np.log10(1), np.log10(30), 50)
   copper_result = xlt.calculate_single_material_properties("Cu", wide_energies, 8.96)

   # Find minimum and maximum critical angles
   min_idx = np.argmin(copper_result.critical_angle_degrees)
   max_idx = np.argmax(copper_result.critical_angle_degrees)

   print(f"Copper critical angle range:")
   print(f"  Minimum: {copper_result.critical_angle_degrees[min_idx]:.3f}° at {copper_result.energy_kev[min_idx]:.1f} keV")
   print(f"  Maximum: {copper_result.critical_angle_degrees[max_idx]:.3f}° at {copper_result.energy_kev[max_idx]:.1f} keV")

Comparing Multiple Materials
----------------------------

Batch Processing
~~~~~~~~~~~~~~~~

Compare multiple materials at the same energy:

.. code-block:: python

   # Common X-ray optics materials
   materials = {
       "SiO2": 2.2,      # Fused silica
       "Si": 2.33,       # Silicon
       "Al2O3": 3.95,    # Sapphire
       "C": 3.52,        # Diamond
       "Pt": 21.45,      # Platinum
   }

   energy = 10.0  # keV (Cu Kα)
   formulas = list(materials.keys())
   densities = list(materials.values())

.. code-block:: python

   # Calculate properties for all materials
   results = xlt.calculate_xray_properties(formulas, energy, densities)

   print(f"Material comparison at {energy} keV:")
   print(f"{'Material':<8} {'θc (°)':<8} {'δ':<12} {'β':<12} {'μ (cm)':<8}")
   print("-" * 60)

   for formula, result in results.items():
       print(f"{formula:<8} {result.critical_angle_degrees[0]:<8.3f} "
             f"{result.dispersion_delta[0]:<12.2e} {result.absorption_beta[0]:<12.2e} "
             f"{result.attenuation_length_cm[0]:<8.1f}")

Material Selection Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different applications require different material properties:

.. code-block:: python

   # Mirror materials (want high critical angle)
   print("Mirror materials (ranked by critical angle):")
   mirror_data = [(formula, result.critical_angle_degrees[0])
                  for formula, result in results.items()]
   mirror_data.sort(key=lambda x: x[1], reverse=True)

   for formula, angle in mirror_data:
       print(f"  {formula}: {angle:.3f}°")

.. code-block:: python

   # Window materials (want low absorption)
   print("\nWindow materials (ranked by transparency):")
   window_data = [(formula, result.attenuation_length_cm[0])
                  for formula, result in results.items()]
   window_data.sort(key=lambda x: x[1], reverse=True)

   for formula, length in window_data:
       print(f"  {formula}: {length:.1f} cm attenuation length")

Practical Applications
----------------------

X-ray Mirror Design
~~~~~~~~~~~~~~~~~~~

Calculate the optimal geometry for an X-ray mirror:

.. code-block:: python

   # Platinum-coated mirror
   energy = 8.048  # keV (Cu Kα)
   result = xlt.calculate_single_material_properties("Pt", energy, 21.45)

   critical_angle = result.critical_angle_degrees[0]
   print(f"Platinum mirror at {energy} keV:")
   print(f"  Critical angle: {critical_angle:.3f}°")
   print(f"  For 90% reflectivity, use θ < {critical_angle * 0.8:.3f}°")
   print(f"  Mirror length for 1 cm beam height: {1.0 / np.sin(np.radians(critical_angle * 0.8)):.1f} cm")

X-ray Window Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Determine the optimal thickness for X-ray windows:

.. code-block:: python

   # Beryllium window for low energy X-rays
   energy = 6.0  # keV
   result = xlt.calculate_single_material_properties("Be", energy, 1.85)

   attenuation_length = result.attenuation_length_cm[0]

   # Calculate thickness for different transmission levels
   print(f"Beryllium window at {energy} keV:")
   print(f"  Attenuation length: {attenuation_length:.2f} cm")

   for transmission in [0.9, 0.8, 0.5]:
       thickness = -attenuation_length * np.log(transmission)
       print(f"  For {transmission*100:.0f}% transmission: {thickness*10:.2f} mm thick")

Absorption Edge Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Investigate how properties change near absorption edges:

.. code-block:: python

   # Energy range around copper K-edge (8.979 keV)
   cu_edge_energies = np.linspace(8.5, 9.5, 21)
   cu_result = xlt.calculate_single_material_properties("Cu", cu_edge_energies, 8.96)

   # Find the energy with maximum absorption
   max_absorption_idx = np.argmax(cu_result.absorption_beta)
   edge_energy = cu_result.energy_kev[max_absorption_idx]

   print(f"Copper absorption analysis:")
   print(f"  Peak absorption at: {edge_energy:.2f} keV")
   print(f"  Peak β value: {cu_result.absorption_beta[max_absorption_idx]:.2e}")

Data Export and Visualization
------------------------------

Saving Results
~~~~~~~~~~~~~~

Export your calculations for further analysis:

.. code-block:: python

   import pandas as pd

   # Convert results to DataFrame
   data = {
       'Energy_keV': cu_result.energy_kev,
       'Wavelength_A': cu_result.wavelength_angstrom,
       'Critical_Angle_deg': cu_result.critical_angle_degrees,
       'Dispersion_delta': cu_result.dispersion_delta,
       'Absorption_beta': cu_result.absorption_beta,
       'Attenuation_Length_cm': cu_result.attenuation_length_cm
   }

   df = pd.DataFrame(data)
   print("First few rows of exported data:")
   print(df.head())

.. code-block:: python

   # Save to CSV file
   # df.to_csv('copper_properties.csv', index=False)
   print("Data exported to CSV file (commented out for tutorial)")

Basic Plotting
~~~~~~~~~~~~~~

Create simple plots to visualize your results:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Set up the plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Plot dispersion and absorption
   ax1.loglog(cu_result.energy_kev, cu_result.dispersion_delta, 'b-', linewidth=2, label='δ (dispersion)')
   ax1.loglog(cu_result.energy_kev, cu_result.absorption_beta, 'r-', linewidth=2, label='β (absorption)')
   ax1.set_xlabel('Energy (keV)')
   ax1.set_ylabel('Optical constants')
   ax1.set_title('Copper: Dispersion & Absorption')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Plot critical angle
   ax2.semilogx(cu_result.energy_kev, cu_result.critical_angle_degrees, 'g-', linewidth=2)
   ax2.set_xlabel('Energy (keV)')
   ax2.set_ylabel('Critical angle (°)')
   ax2.set_title('Copper: Critical Angle')
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Best Practices
--------------

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use vectorized calculations** for multiple energies:

.. code-block:: python

   # Good: Calculate all energies at once
   energies = np.linspace(5, 15, 100)
   result = xlt.calculate_single_material_properties("Si", energies, 2.33)

   # Avoid: Looping over individual energies (much slower)
   # for energy in energies:
   #     result = xlt.calculate_single_material_properties("Si", energy, 2.33)

2. **Use common elements** for best performance:

.. code-block:: python

   # Fast: Common elements are preloaded
   fast_materials = ["SiO2", "Al2O3", "Fe2O3", "Si", "C", "Cu", "Au"]

   # Slower: Exotic elements require database lookup
   # slow_materials = ["Uuo", "Fl", "Mc"]  # Uncomment to test

   print("Common elements calculate fastest due to preloaded atomic data")

3. **Use batch processing** for multiple materials:

.. code-block:: python

   # Efficient batch processing
   materials = ["SiO2", "Al2O3", "Si"] * 10  # 30 materials
   densities = [2.2, 3.95, 2.33] * 10
   energy = 10.0

   # This is optimized and runs in parallel
   results = xlt.calculate_xray_properties(materials, energy, densities)
   print(f"Calculated properties for {len(results)} materials efficiently")

Error Handling
~~~~~~~~~~~~~~

Handle common errors gracefully:

.. code-block:: python

   from xraylabtool.validation.exceptions import FormulaError, EnergyError

   # Handle invalid formulas
   try:
       result = xlt.calculate_single_material_properties("InvalidFormula", 10.0, 2.0)
   except FormulaError as e:
       print(f"Formula error: {e}")

   # Handle invalid energies
   try:
       result = xlt.calculate_single_material_properties("SiO2", -5.0, 2.2)  # Negative energy
   except EnergyError as e:
       print(f"Energy error: {e}")

Input Validation
~~~~~~~~~~~~~~~~

Always validate your inputs:

.. code-block:: python

   # Check if a formula is valid
   from xraylabtool.utils import parse_formula

   formula = "Ca10P6O26H2"  # Hydroxyapatite
   try:
       elements, counts = parse_formula(formula)
       print(f"Valid formula: {formula}")
       print(f"Elements: {dict(zip(elements, counts))}")
   except FormulaError:
       print(f"Invalid formula: {formula}")

Summary
-------

In this tutorial, you learned:

✅ **Basic calculations** for single materials and energy points
✅ **Multiple energy calculations** using NumPy arrays
✅ **Material comparisons** using batch processing
✅ **Result interpretation** and understanding physical quantities
✅ **Practical applications** for X-ray optics design
✅ **Data export** and basic visualization
✅ **Best practices** for performance and error handling

Next Steps
----------

Now that you understand the basics, explore these advanced topics:

.. grid:: 2

    .. grid-item-card:: 📊 Advanced Examples
        :link: advanced_examples
        :link-type: doc

        Complex calculations for real-world scenarios

    .. grid-item-card:: 🚀 Performance Optimization
        :link: ../performance_guide
        :link-type: doc

        High-performance computing techniques

    .. grid-item-card:: 🖥️ Command-Line Interface
        :link: ../cli_guide
        :link-type: doc

        Master the CLI for rapid calculations

    .. grid-item-card:: 📈 Data Analysis
        :link: ../howto/data_processing
        :link-type: doc

        Advanced data processing and visualization

Exercises
---------

Try these exercises to reinforce your learning:

1. **Material Database Analysis**: Create a database of 20 materials and calculate their critical angles at 8 keV. Which would be best for X-ray mirrors?

2. **Energy Optimization**: For a beryllium window, find the optimal energy range where transmission is >80% for 1 mm thickness.

3. **Absorption Edge Mapping**: Choose an element and map its absorption properties across a wide energy range to locate absorption edges.

4. **Mirror Design**: Design a platinum-coated mirror system for 10 keV X-rays, determining the optimal grazing angle and mirror length.

Solutions to these exercises can be found in the :doc:`advanced_examples` tutorial.
