How-To Guides
=============

These guides provide practical solutions to common problems and specific tasks you might encounter when using XRayLabTool.

.. grid:: 2

    .. grid-item-card:: 🔧 Installation & Setup
        :link: installation_troubleshooting
        :link-type: doc

        Solve installation issues and configure your environment.

    .. grid-item-card:: ⚡ Performance Optimization
        :link: performance_tips
        :link-type: doc

        Speed up calculations and handle large datasets efficiently.

    .. grid-item-card:: 📊 Data Processing
        :link: data_processing
        :link-type: doc

        Process, analyze, and visualize your calculation results.

    .. grid-item-card:: 🔗 Integration
        :link: ../installation
        :link-type: doc

        Setup and installation troubleshooting guides.

Quick Solutions
---------------

.. toctree::
   :maxdepth: 2
   :caption: Problem-Solving Guides

   installation_troubleshooting
   performance_tips
   data_processing

Common Tasks
------------

Installation & Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Fix import errors after installation

    **Problem**: ``ImportError: No module named 'xraylabtool'``

    **Solutions**:

    1. Verify installation in correct environment:

       .. code-block:: bash

          # Check if installed
          pip list | grep xraylabtool

          # Reinstall if missing
          pip install --force-reinstall xraylabtool

    2. Check Python path:

       .. code-block:: python

          import sys
          print(sys.path)

          # Should include your environment's site-packages

.. dropdown:: Enable shell completion

    **Task**: Set up intelligent tab completion for the CLI

    **Solution**:

    .. code-block:: bash

       # Auto-detect shell and install
       xraylabtool install-completion

       # Install for specific shell
       xraylabtool install-completion bash

       # Test completion works
       xraylabtool install-completion --test

.. dropdown:: Configure for high-performance computing

    **Task**: Optimize XRayLabTool for HPC environments

    **Solution**:

    .. code-block:: python

       from xraylabtool.data_handling.batch_processing import BatchConfig
       import os

       # Configure for HPC
       config = BatchConfig(
           max_workers=os.cpu_count(),  # Use all cores
           chunk_size=1000,            # Larger chunks for HPC
           memory_limit_gb=16.0,       # More memory available
           enable_progress=False       # Disable for batch jobs
       )

Calculations & Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Handle very large energy ranges

    **Problem**: Memory issues with thousands of energy points

    **Solution**:

    .. code-block:: python

       # Use chunked processing for large energy arrays
       import numpy as np
       from xraylabtool.data_handling.batch_processing import calculate_batch_properties

       # Split large energy range into chunks
       energies = np.logspace(0, 2, 10000)  # 10,000 points
       chunks = np.array_split(energies, 10)  # 10 chunks

       results = []
       for chunk in chunks:
           result = xlt.calculate_single_material_properties("Si", chunk, 2.33)
           results.append(result)

.. dropdown:: Compare many materials efficiently

    **Task**: Analyze 100+ materials at multiple energies

    **Solution**:

    .. code-block:: python

       # Use batch processing for many materials
       formulas = ["SiO2", "Al2O3", "Fe2O3"] * 50  # 150 materials
       densities = [2.2, 3.95, 5.24] * 50
       energies = np.linspace(5, 15, 50)

       from xraylabtool.data_handling.batch_processing import calculate_batch_properties, BatchConfig

       config = BatchConfig(chunk_size=50, max_workers=8)
       results = calculate_batch_properties(formulas, energies, densities, config)

.. dropdown:: Create publication-quality plots

    **Task**: Generate professional figures for papers

    **Solution**:

    .. code-block:: python

       import matplotlib.pyplot as plt
       import numpy as np

       # Set publication style
       plt.style.use('seaborn-v0_8-paper')
       plt.rcParams['font.size'] = 12
       plt.rcParams['figure.figsize'] = (8, 6)

       # Create your plot
       energies = np.linspace(5, 15, 100)
       result = xlt.calculate_single_material_properties("Si", energies, 2.33)

       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

       # Plot dispersion and absorption
       ax1.loglog(result.energy_kev, result.dispersion_delta, 'b-', linewidth=2)
       ax1.set_xlabel('Energy (keV)')
       ax1.set_ylabel('Dispersion δ')

       ax2.loglog(result.energy_kev, result.critical_angle_degrees, 'r-', linewidth=2)
       ax2.set_xlabel('Energy (keV)')
       ax2.set_ylabel('Critical Angle (°)')

       plt.tight_layout()
       plt.savefig('silicon_properties.pdf', dpi=300, bbox_inches='tight')

Data Management
~~~~~~~~~~~~~~~

.. dropdown:: Export results to different formats

    **Task**: Save calculation results for use in other programs

    **Solution**:

    .. code-block:: python

       import pandas as pd
       import json

       # Calculate properties
       result = xlt.calculate_single_material_properties("SiO2", energies, 2.2)

       # Export to CSV
       data = {
           'Energy_keV': result.energy_kev,
           'Critical_Angle_deg': result.critical_angle_degrees,
           'Dispersion_delta': result.dispersion_delta,
           'Absorption_beta': result.absorption_beta,
           'Attenuation_Length_cm': result.attenuation_length_cm
       }
       df = pd.DataFrame(data)
       df.to_csv('sio2_properties.csv', index=False)

       # Export to JSON
       json_data = {
           'material': result.formula,
           'density_g_cm3': result.density_g_cm3,
           'molecular_weight_g_mol': result.molecular_weight_g_mol,
           'properties': data
       }
       with open('sio2_properties.json', 'w') as f:
           json.dump(json_data, f, indent=2, default=str)

.. dropdown:: Process batch calculation results

    **Task**: Analyze results from many materials efficiently

    **Solution**:

    .. code-block:: python

       # Process batch results into summary statistics
       import pandas as pd

       # Assume you have results from batch processing
       summary_data = []
       for formula, result in batch_results.items():
           summary_data.append({
               'Formula': formula,
               'Density': result.density_g_cm3,
               'MW': result.molecular_weight_g_mol,
               'Critical_Angle_10keV': result.critical_angle_degrees[10],  # 10 keV point
               'Attenuation_10keV': result.attenuation_length_cm[10]
           })

       summary_df = pd.DataFrame(summary_data)
       print(summary_df.describe())  # Statistical summary

Integration & Automation
~~~~~~~~~~~~~~~~~~~~~~~~

.. dropdown:: Integrate with measurement data

    **Task**: Combine XRayLabTool with experimental measurements

    **Solution**:

    .. code-block:: python

       # Load experimental reflectivity data
       experimental_data = pd.read_csv('reflectivity_measurement.csv')

       # Calculate theoretical values
       energies = experimental_data['Energy_keV'].values
       result = xlt.calculate_single_material_properties("Pt", energies, 21.45)

       # Compare with theory
       theoretical_critical = result.critical_angle_degrees
       measured_angles = experimental_data['Angle_deg'].values

       # Calculate agreement
       difference = np.abs(theoretical_critical - measured_angles)
       print(f"Average difference: {difference.mean():.3f}°")

.. dropdown:: Create automated analysis pipeline

    **Task**: Set up automated processing of material databases

    **Solution**:

    .. code-block:: python

       def analyze_material_database(database_file, output_dir):
           """Automated analysis pipeline for material databases."""

           # Load material database
           materials_df = pd.read_csv(database_file)

           results = {}
           for _, row in materials_df.iterrows():
               formula = row['Formula']
               density = row['Density_g_cm3']

               # Calculate properties at standard energies
               standard_energies = [5, 8, 10, 15, 20]  # keV
               result = xlt.calculate_single_material_properties(
                   formula, standard_energies, density
               )
               results[formula] = result

               # Save individual result
               output_file = f"{output_dir}/{formula.replace('/', '_')}.csv"
               # ... save to file

           return results

Troubleshooting
~~~~~~~~~~~~~~~

.. dropdown:: Debug unexpected calculation results

    **Problem**: Results don't match expected values

    **Solution**:

    .. code-block:: python

       # Debug calculation step by step
       from xraylabtool.utils import parse_formula
       from xraylabtool.calculators.core import calculate_scattering_factors

       formula = "SiO2"
       energy = 10.0

       # 1. Check formula parsing
       elements, counts = parse_formula(formula)
       print(f"Parsed: {dict(zip(elements, counts))}")

       # 2. Check atomic data
       f1, f2 = calculate_scattering_factors(elements, [energy])
       print(f"Scattering factors: f1={f1}, f2={f2}")

       # 3. Check intermediate calculations
       result = xlt.calculate_single_material_properties(formula, energy, 2.2)
       print(f"Dispersion: {result.dispersion_delta[0]:.2e}")
       print(f"Absorption: {result.absorption_beta[0]:.2e}")

.. dropdown:: Handle memory errors with large calculations

    **Problem**: ``MemoryError`` with large datasets

    **Solution**:

    .. code-block:: python

       # Use memory monitoring and chunked processing
       from xraylabtool.data_handling.batch_processing import MemoryMonitor, BatchConfig

       monitor = MemoryMonitor(limit_gb=4.0)  # Set memory limit

       if not monitor.check_memory():
           print("Warning: High memory usage detected")
           # Use smaller chunks or fewer workers
           config = BatchConfig(
               chunk_size=50,      # Smaller chunks
               max_workers=2,      # Fewer workers
               memory_limit_gb=4.0
           )

Advanced Topics
---------------

.. note::
   Advanced guides are coming soon. Check back for content on extending XRayLabTool,
   custom atomic data, parallel processing, web integration, and database integration.

Quick Reference
---------------

.. list-table:: Common Command Patterns
   :widths: 40 60
   :header-rows: 1

   * - Task
     - Command/Code
   * - Single calculation
     - ``xraylabtool calc SiO2 -e 10.0 -d 2.2``
   * - Energy range
     - ``xraylabtool calc Si -e 5-15:11 -d 2.33``
   * - Batch processing
     - ``xraylabtool batch materials.csv -o results.csv``
   * - Install completion
     - ``xraylabtool install-completion``
   * - Check version
     - ``xraylabtool --version``

Need More Help?
---------------

If these guides don't solve your problem:

1. Check the :doc:`../faq` for frequently asked questions
2. Search the `GitHub Issues <https://github.com/imewei/pyXRayLabTool/issues>`_
3. Ask on `GitHub Discussions <https://github.com/imewei/pyXRayLabTool/discussions>`_
4. Review the :doc:`../api/index` for detailed API documentation

Contributing
------------

Found a problem not covered here? Help improve these guides:

- Submit fixes via pull requests
- Suggest new how-to topics via GitHub issues
- Share your solutions in GitHub discussions

See our :doc:`../contributing` guide for details on how to contribute.
