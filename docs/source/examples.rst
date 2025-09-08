Usage Examples & Interactive Tutorials
=======================================

**Comprehensive Examples Showcasing XRayLabTool's High-Performance
Capabilities**

This page provides practical examples for both the Python API and CLI usage of
XRayLabTool, including performance optimization examples that demonstrate the
package's advanced capabilities.

.. grid:: 2

    .. grid-item-card:: 🚀 Interactive Jupyter Notebook
        :link: https://mybinder.org/v2/gh/imewei/pyXRayLabTool/HEAD?labpath=notebooks%2Fgetting_started.ipynb
        :link-type: url
        :class-title: text-center

        **Try XRayLabTool in your browser!**

        Complete hands-on tutorial with live code execution,
        interactive plots, and performance demonstrations.

        +++

        .. button-link:: https://mybinder.org/v2/gh/imewei/pyXRayLabTool/HEAD?labpath=notebooks%2Fgetting_started.ipynb
            :color: primary
            :outline:
            :expand:

            Launch Interactive Notebook

    .. grid-item-card:: 📊 Google Colab Version
        :link: https://colab.research.google.com/github/imewei/pyXRayLabTool/blob/main/notebooks/getting_started.ipynb
        :link-type: url
        :class-title: text-center

        **Run on Google Colab with GPU acceleration**

        Same interactive tutorial optimized for Google Colab
        environment with enhanced performance.

        +++

        .. button-link:: https://colab.research.google.com/github/imewei/pyXRayLabTool/blob/main/notebooks/getting_started.ipynb
            :color: info
            :outline:
            :expand:

            Open in Colab

.. attention::
   **🎯 New to XRayLabTool?** Start with the interactive notebook above for a
   hands-on introduction with live code execution, visualizations, and
   real-world examples!

Python API Examples
-------------------

Basic Single Material Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Python Code

      .. code-block:: python
         :linenos:

         import xraylabtool as xlt
         import numpy as np

         # Calculate properties for quartz at 10 keV
         result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)

         # Using new descriptive field names (recommended)
         print(f"Formula: {result.formula}")
         print(f"Molecular Weight: {result.molecular_weight_g_mol:.2f} g/mol")
         print(f"Critical Angle: {result.critical_angle_degrees[0]:.3f}°")
         print(f"Attenuation Length: {result.attenuation_length_cm[0]:.2f} cm")

   .. tab:: Expected Output

      .. code-block:: text

         Formula: SiO2
         Molecular Weight: 60.08 g/mol
         Critical Angle: 0.223°
         Attenuation Length: 12.5 cm

   .. tab:: CLI Equivalent

      .. code-block:: bash

         xraylabtool calc SiO2 -e 10.0 -d 2.2 --fields formula,molecular_weight_g_mol,critical_angle_degrees,attenuation_length_cm

Multiple Materials Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare common X-ray optics materials
   materials = {
       "SiO2": 2.2,      # Fused silica
       "Si": 2.33,       # Silicon
       "Al2O3": 3.95,    # Sapphire
       "C": 3.52,        # Diamond
   }

   formulas = list(materials.keys())
   densities = list(materials.values())
   energy = 10.0  # keV (Cu Kα)

   results = xlt.calculate_xray_properties(formulas, energy, densities)

   # Compare using new field names
   for formula, result in results.items():
       print(f"{formula:8}: θc = {result.critical_angle_degrees[0]:.3f}°, "
             f"δ = {result.dispersion_delta[0]:.2e}, "
             f"μ = {result.attenuation_length_cm[0]:.1f} cm")

Energy Range Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Energy sweep for material characterization
   energies = np.logspace(np.log10(1), np.log10(30), 100)  # 1-30 keV
   result = xlt.calculate_single_material_properties("Si", energies, 2.33)

   print(f"Energy range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV")
   print(f"Data points: {len(result.energy_kev)}")
   print(f"Dispersion range: {result.dispersion_delta.min():.2e} to {result.dispersion_delta.max():.2e}")

Plotting with Matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Energy-dependent properties with new field names
   energies = np.logspace(np.log10(1), np.log10(20), 100)
   result = xlt.calculate_single_material_properties("Si", energies, 2.33)

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Plot using new descriptive field names
   ax1.loglog(result.energy_kev, result.dispersion_delta, 'b-',
              label='δ (dispersion)', linewidth=2)
   ax1.loglog(result.energy_kev, result.absorption_beta, 'r-',
              label='β (absorption)', linewidth=2)
   ax1.set_xlabel('Energy (keV)')
   ax1.set_ylabel('Optical constants')
   ax1.set_title('Silicon: Dispersion & Absorption')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Plot critical angle with new field name
   ax2.semilogx(result.energy_kev, result.critical_angle_degrees, 'g-', linewidth=2)
   ax2.set_xlabel('Energy (keV)')
   ax2.set_ylabel('Critical angle (°)')
   ax2.set_title('Silicon: Critical Angle')
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Performance and Cache Examples
------------------------------

Cache Status Interactive Demo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Check Cache Status

      .. code-block:: python
         :linenos:

         from xraylabtool.atomic_data_cache import get_cache_stats, is_element_preloaded

         # Check overall cache statistics
         stats = get_cache_stats()
         print(f"Cache Statistics:")
         print(f"  Preloaded elements: {stats['preloaded_elements']}")
         print(f"  Runtime cached: {stats['runtime_cached_elements']}")
         print(f"  Total cached: {stats['total_cached_elements']}")

         # Check specific elements
         elements = ["H", "Si", "O", "Al", "Fe", "Au", "U"]
         print(f"\nElement Cache Status:")
         for element in elements:
             status = "✓ Preloaded" if is_element_preloaded(element) else "○ Runtime"
             print(f"  {element:2}: {status}")

   .. tab:: Expected Output

      .. code-block:: text

         Cache Statistics:
           Preloaded elements: 92
           Runtime cached: 0
           Total cached: 92

         Element Cache Status:
           H : ✓ Preloaded
           Si: ✓ Preloaded
           O : ✓ Preloaded
           Al: ✓ Preloaded
           Fe: ✓ Preloaded
           Au: ✓ Preloaded
           U : ✓ Preloaded

Material Properties Comparison Table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Common X-ray Optics Materials at 10 keV
   :widths: 20 15 20 25 20
   :header-rows: 1
   :class: material-properties

   * - Formula
     - Density (g/cm³)
     - Critical Angle (°)
     - Atten. Length (cm)
     - Cache Status
   * - :class:`chemical-formula` SiO₂
     - 2.20
     - 0.223
     - 12.5
     - :class:`cache-status preloaded` ✓ Preloaded
   * - :class:`chemical-formula` Si
     - 2.33
     - 0.158
     - 15.2
     - :class:`cache-status preloaded` ✓ Preloaded
   * - :class:`chemical-formula` Al₂O₃
     - 3.95
     - 0.182
     - 8.9
     - :class:`cache-status preloaded` ✓ Preloaded
   * - :class:`chemical-formula` C
     - 3.52
     - 0.112
     - 18.7
     - :class:`cache-status preloaded` ✓ Preloaded
   * - :class:`chemical-formula` Au
     - 19.32
     - 0.418
     - 0.23
     - :class:`cache-status preloaded` ✓ Preloaded

Interactive Energy Range Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tabs::

   .. tab:: Single Energy

      .. code-block:: python

         import xraylabtool as xlt

         # Single energy calculation
         result = xlt.calculate_single_material_properties("Si", 10.0, 2.33)
         print(f"Single energy result:")
         print(f"  Energy: {result.energy_kev[0]:.1f} keV")
         print(f"  Critical angle: {result.critical_angle_degrees[0]:.3f}°")

   .. tab:: Energy List

      .. code-block:: python

         import xraylabtool as xlt

         # Multiple discrete energies
         energies = [5.0, 10.0, 15.0, 20.0]
         result = xlt.calculate_single_material_properties("Si", energies, 2.33)

         print(f"Multiple energies result:")
         for i, e in enumerate(result.energy_kev):
             print(f"  {e:.1f} keV: θc = {result.critical_angle_degrees[i]:.3f}°")

   .. tab:: Energy Range

      .. code-block:: python

         import xraylabtool as xlt
         import numpy as np

         # Energy range (logarithmic spacing)
         energies = np.logspace(np.log10(1), np.log10(30), 50)
         result = xlt.calculate_single_material_properties("Si", energies, 2.33)

         print(f"Energy range result:")
         print(f"  Range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV")
         print(f"  Points: {len(result.energy_kev)}")
         print(f"  Min θc: {result.critical_angle_degrees.min():.3f}°")
         print(f"  Max θc: {result.critical_angle_degrees.max():.3f}°")

Utility Functions
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Energy and wavelength conversions
   energy = 10.0  # keV
   wavelength = xlt.energy_to_wavelength(energy)
   print(f"{energy} keV = {wavelength:.4f} Å")

   # Parse chemical formulas
   elements, counts = xlt.parse_formula("Ca10P6O26H2")
   print(f"Elements: {elements}")
   print(f"Counts: {counts}")

   # Atomic data lookup
   atomic_number = xlt.get_atomic_number("Si")
   atomic_weight = xlt.get_atomic_weight("Si")
   print(f"Silicon: Z = {atomic_number}, MW = {atomic_weight:.3f} u")

CLI Examples
------------

Single Material Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic calculation for quartz at 10 keV
   xraylabtool calc SiO2 -e 10.0 -d 2.2

   # Multiple discrete energies
   xraylabtool calc Si -e 5.0,10.0,15.0,20.0 -d 2.33

   # Energy range (11 points from 5 to 15 keV)
   xraylabtool calc Al2O3 -e 5-15:11 -d 3.95

   # Logarithmic energy range (100 points from 1 to 30 keV)
   xraylabtool calc C -e 1-30:100:log -d 3.52

Energy Range Scans
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Silicon energy scan for X-ray optics design
   xraylabtool calc Si -e 5-20:50 -d 2.33 -o silicon_scan.csv

   # Fine scan around Cu Kα line
   xraylabtool calc SiO2 -e 7.5-8.5:21 -d 2.2 -o sio2_cu_ka_scan.csv

   # Full spectrum logarithmic scan
   xraylabtool calc Al2O3 -e 0.1-30:200:log -d 3.95 -o al2o3_full_spectrum.csv

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create materials comparison file
   cat > materials.csv << EOF
   formula,density,energy
   SiO2,2.2,10.0
   Si,2.33,10.0
   Al2O3,3.95,10.0
   C,3.52,10.0
   Fe,7.87,10.0
   EOF

   # Process batch
   xraylabtool batch materials.csv -o comparison_results.csv

   # Batch with energy ranges
   cat > energy_scans.csv << EOF
   formula,density,energy
   SiO2,2.2,"5-15:11"
   Si,2.33,"5-15:11"
   Al2O3,3.95,"5-15:11"
   EOF

   xraylabtool batch energy_scans.csv -o energy_scan_results.csv --workers 4

Unit Conversions
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Common X-ray line energies to wavelengths
   xraylabtool convert energy 8.048,17.478,59.318 --to wavelength

   # Wavelengths to energies
   xraylabtool convert wavelength 1.54,0.71,0.21 --to energy

   # Save conversions to file
   xraylabtool convert energy 5,10,15,20,25,30 --to wavelength -o energy_wavelength_table.csv

Chemical Formula Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Simple oxides
   xraylabtool formula SiO2,Al2O3,TiO2,Fe2O3

   # Complex biological molecules
   xraylabtool formula C6H12O6  # Glucose
   xraylabtool formula C8H18N2O4S  # Amino acid

   # Minerals and ceramics
   xraylabtool formula Ca10P6O26H2  # Hydroxyapatite
   xraylabtool formula Al2SiO5  # Andalusite

Atomic Data Lookup
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Common elements in X-ray optics
   xraylabtool atomic Si,Al,Fe,Au,Pt

   # Light elements
   xraylabtool atomic H,C,N,O

   # Save atomic data to file
   xraylabtool atomic Si,Al,Ti,Fe,Ni,Cu,Zn,Mo,W,Au -o elements.csv

Bragg Diffraction Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Silicon crystal d-spacings at Cu Kα
   xraylabtool bragg -d 3.135,1.920,1.637,1.358 -e 8.048

   # Diamond d-spacings at different energies
   xraylabtool bragg -d 2.06,1.26,1.08,0.89 -e 10.0

   # Higher order reflections
   xraylabtool bragg -d 3.14 -e 8.048 --order 1
   xraylabtool bragg -d 3.14 -e 8.048 --order 2

Output Formatting
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # JSON output with selected fields
   xraylabtool calc SiO2 -e 10.0 -d 2.2 \
     --format json \
     --fields formula,energy_kev,dispersion_delta,critical_angle_degrees \
     -o sio2_properties.json

   # CSV output with high precision
   xraylabtool calc Si -e 8.048 -d 2.33 \
     --format csv \
     --precision 10 \
     -o silicon_high_precision.csv

   # Table output with selected fields
   xraylabtool calc Al2O3 -e 5,10,15,20 -d 3.95 \
     --fields formula,energy_kev,critical_angle_degrees,attenuation_length_cm

Reference Information
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # View physical constants
   xraylabtool list constants

   # See all available output fields
   xraylabtool list fields

   # Get usage examples
   xraylabtool list examples

Workflow Integration Examples
-----------------------------

Synchrotron Beamline Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate mirror reflectivity for different coatings
   import xraylabtool as xlt
   import numpy as np

   energies = np.linspace(5, 20, 100)
   coatings = {"Si": 2.33, "SiO2": 2.2, "Au": 19.3, "Pt": 21.45}

   for coating, density in coatings.items():
       result = xlt.calculate_single_material_properties(coating, energies, density)
       # Calculate reflectivity curves, optimize mirror angles, etc.

Materials Research Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Characterize new thin film materials
   # 1. Analyze composition
   xraylabtool formula MyMaterial

   # 2. Calculate properties across energy range
   xraylabtool calc MyMaterial -e 1-30:200:log -d 5.0 -o material_properties.csv

   # 3. Compare with reference materials
   cat > comparison.csv << EOF
   formula,density,energy
   MyMaterial,5.0,10.0
   Si,2.33,10.0
   SiO2,2.2,10.0
   EOF

   xraylabtool batch comparison.csv -o material_comparison.csv

X-ray Optics Design
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Design multilayer mirror
   energies = np.linspace(8, 12, 100)

   # Layer materials
   heavy_layer = xlt.calculate_single_material_properties("W", energies, 19.3)
   light_layer = xlt.calculate_single_material_properties("B4C", energies, 2.52)

   # Calculate optical constants for multilayer design
   # Use dispersion and absorption for reflectivity calculations

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   import xraylabtool as xlt

   # Benchmark single material calculations
   start_time = time.time()
   for _ in range(1000):
       result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
   single_time = time.time() - start_time

   print(f"1000 single material calculations: {single_time:.3f} s")
   print(f"Average per calculation: {single_time/1000*1000:.3f} ms")

Integration with Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   import xraylabtool as xlt

   # Load experimental data
   exp_data = pd.read_csv("experimental_results.csv")

   # Calculate theoretical properties
   theoretical_results = []
   for _, row in exp_data.iterrows():
       result = xlt.calculate_single_material_properties(
           row['formula'], row['energy'], row['density']
       )
       theoretical_results.append({
           'formula': result.formula,
           'energy': result.energy_kev[0],
           'theoretical_critical_angle': result.critical_angle_degrees[0],
           'theoretical_attenuation': result.attenuation_length_cm[0]
       })

   # Merge with experimental data
   theory_df = pd.DataFrame(theoretical_results)
   combined_data = exp_data.merge(theory_df, on=['formula', 'energy'])

   # Analyze differences
   combined_data['angle_difference'] = (
       combined_data['measured_critical_angle'] -
       combined_data['theoretical_critical_angle']
   )

These examples demonstrate the versatility and power of XRayLabTool for various X-ray science applications, from quick calculations to comprehensive materials analysis workflows.
