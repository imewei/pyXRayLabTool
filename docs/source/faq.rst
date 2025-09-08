Frequently Asked Questions
===========================

This page answers the most common questions about XRayLabTool.

Installation & Setup
--------------------

**Q: How do I install XRayLabTool?**

A: The simplest method is using pip:

.. code-block:: bash

   pip install xraylabtool

For detailed instructions, see the :doc:`installation` guide.

**Q: What Python version is required?**

A: XRayLabTool requires Python 3.12 or higher. We recommend using the latest stable Python version for best performance.

**Q: Can I use XRayLabTool with conda?**

A: Yes! While XRayLabTool isn't on conda-forge yet, you can install it in a conda environment:

.. code-block:: bash

   conda create -n xraylabtool python=3.12
   conda activate xraylabtool
   pip install xraylabtool

**Q: How do I enable shell completion?**

A: Run the completion installer:

.. code-block:: bash

   xraylabtool install-completion
   # Restart your shell, then try:
   xraylabtool calc Si<TAB>

Basic Usage
-----------

**Q: How do I calculate X-ray properties for a material?**

A: Use the Python API or command-line interface:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import xraylabtool as xlt
         result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
         print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")

   .. tab:: Command Line

      .. code-block:: bash

         xraylabtool calc SiO2 -e 10.0 -d 2.2

**Q: What energy units does XRayLabTool use?**

A: XRayLabTool uses **keV (kiloelectron volts)** for energy. Common X-ray energies:

- Cu Kα: 8.048 keV
- Mo Kα: 17.479 keV
- Synchrotron: 5-30 keV range

**Q: How do I specify chemical formulas?**

A: Use standard chemical notation:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Correct
     - Examples
   * - ``"SiO2"``
     - Silicon dioxide (quartz)
   * - ``"Al2O3"``
     - Aluminum oxide (sapphire)
   * - ``"Ca10P6O26H2"``
     - Hydroxyapatite
   * - ``"Fe2O3"``
     - Iron oxide (rust)

.. warning::

   Chemical formulas are **case-sensitive**:

   - ``"CO"`` = carbon monoxide
   - ``"Co"`` = cobalt metal

**Q: Where do I find material densities?**

A: Common sources:

- NIST Chemistry WebBook: https://webbook.nist.gov/
- CRC Handbook of Chemistry and Physics
- Materials databases (ICDD, ICSD)
- Published literature

Common densities (g/cm³):

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Material
     - Formula
     - Density (g/cm³)
   * - Silicon
     - Si
     - 2.33
   * - Fused silica
     - SiO2
     - 2.2
   * - Sapphire
     - Al2O3
     - 3.95
   * - Diamond
     - C
     - 3.52
   * - Platinum
     - Pt
     - 21.45

Advanced Usage
--------------

**Q: How do I calculate properties for multiple energies?**

A: Pass an array of energies:

.. code-block:: python

   import numpy as np

   # Method 1: Numpy array
   energies = np.linspace(5, 15, 11)  # 5-15 keV, 11 points

   # Method 2: Python list
   energies = [5.0, 8.0, 10.0, 12.0, 15.0]

   result = xlt.calculate_single_material_properties("Si", energies, 2.33)

**Q: How do I compare multiple materials?**

A: Use the batch processing function:

.. code-block:: python

   materials = ["SiO2", "Al2O3", "Si"]
   densities = [2.2, 3.95, 2.33]
   energy = 10.0

   results = xlt.calculate_xray_properties(materials, energy, densities)

   for formula, result in results.items():
       print(f"{formula}: {result.critical_angle_degrees[0]:.3f}°")

**Q: How do I process large datasets efficiently?**

A: Use the optimized batch processor:

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import calculate_batch_properties, BatchConfig

   config = BatchConfig(
       chunk_size=100,
       max_workers=8,
       memory_limit_gb=4.0
   )

   results = calculate_batch_properties(formulas, energies, densities, config)

**Q: What's the difference between dispersion (δ) and absorption (β)?**

A: Both are parts of the complex refractive index n = 1 - δ - iβ:

- **δ (dispersion)**: Causes refraction and determines critical angles
- **β (absorption)**: Causes attenuation and determines penetration depth

Relationship to measurable quantities:

- Critical angle: θc ≈ √(2δ)
- Attenuation length: μ = λ/(4πβ)

Performance & Optimization
--------------------------

**Q: Why are my calculations slow?**

A: Several factors affect performance:

1. **Use common elements** (cached for speed):

   .. code-block:: python

      # Fast (preloaded in cache)
      fast_materials = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]

      # Slower (requires database lookup)
      slow_materials = ["Uuo", "Fl", "Mc"]  # Exotic elements

2. **Use vectorized calculations**:

   .. code-block:: python

      # Good: Calculate all energies at once
      energies = np.linspace(5, 15, 100)
      result = xlt.calculate_single_material_properties("Si", energies, 2.33)

      # Bad: Calculate one energy at a time
      # results = []
      # for energy in energies:
      #     result = xlt.calculate_single_material_properties("Si", energy, 2.33)
      #     results.append(result)

3. **Use batch processing for multiple materials**:

   .. code-block:: python

      # Good: Batch processing (parallel)
      results = xlt.calculate_xray_properties(formulas, energies, densities)

      # Bad: Sequential processing
      # results = {}
      # for formula, density in zip(formulas, densities):
      #     results[formula] = xlt.calculate_single_material_properties(formula, energies, density)

**Q: How much memory do large calculations use?**

A: Memory usage scales with:

- Number of materials × number of energy points × ~8 bytes per result
- Example: 100 materials × 1000 energies = ~800 KB

For very large calculations, use chunked processing:

.. code-block:: python

   from xraylabtool.data_handling.batch_processing import MemoryMonitor

   monitor = MemoryMonitor(limit_gb=4.0)
   if monitor.check_memory():
       print("Memory usage OK")
   else:
       print("Consider using smaller chunks")

**Q: Can I use XRayLabTool on high-performance computing (HPC) systems?**

A: Yes! XRayLabTool is designed for HPC environments:

.. code-block:: python

   import os
   from xraylabtool.data_handling.batch_processing import BatchConfig

   # Configure for HPC
   config = BatchConfig(
       max_workers=os.cpu_count(),    # Use all available cores
       chunk_size=1000,              # Larger chunks for HPC
       memory_limit_gb=32.0,         # More memory available
       enable_progress=False         # Disable progress bars for batch jobs
   )

Error Handling & Troubleshooting
--------------------------------

**Q: I get "ImportError: No module named 'xraylabtool'"**

A: This usually means:

1. XRayLabTool isn't installed in the current environment
2. You're using the wrong Python/environment

**Fix**:

.. code-block:: bash

   # Check if installed
   pip list | grep xraylabtool

   # If not installed
   pip install xraylabtool

   # Check Python version
   python --version  # Should be 3.12+

**Q: I get "FormulaError: Invalid chemical formula"**

A: Check your formula syntax:

.. code-block:: python

   # Common mistakes:
   # "sio2"     -> "SiO2"     (capitalization)
   # "Al2 O3"   -> "Al2O3"    (no spaces)
   # "Fe(OH)3"  -> "FeO3H3"   (expand parentheses)

   # Test formula parsing:
   from xraylabtool.utils import parse_formula
   elements, counts = parse_formula("Al2O3")
   print(f"Elements: {elements}, Counts: {counts}")

**Q: My results seem wrong compared to literature values**

A: Check these common issues:

1. **Verify material density**:

   .. code-block:: python

      # Different phases have different densities
      # SiO2: crystalline (2.65) vs. fused silica (2.2) vs. aerogel (0.1)

2. **Check energy units** (keV not eV):

   .. code-block:: python

      # Wrong: 10000 eV
      # Right: 10.0 keV

3. **Verify formula stoichiometry**:

   .. code-block:: python

      # Some materials have non-obvious formulas
      # Hydroxyapatite: Ca10P6O26H2 (not Ca5(PO4)3(OH))

**Q: How accurate are XRayLabTool calculations?**

A: XRayLabTool uses CXRO/NIST atomic scattering factor databases, which are accurate to:

- δ (dispersion): ~1-5% for most elements and energies
- β (absorption): ~2-10% depending on proximity to absorption edges
- Critical angles: ~1-3% for typical X-ray optics materials

Accuracy is highest for:

- Common elements (Z < 92)
- Energies away from absorption edges
- Well-characterized materials

**Q: Can I use XRayLabTool for neutrons or electrons?**

A: No, XRayLabTool is specifically designed for X-rays. For neutrons, consider:

- SasView (small-angle neutron scattering)
- NIST neutron activation calculator

For electrons, consider:

- NIST electron inelastic mean free path database
- Penn algorithm implementations

Output & Data Format
--------------------

**Q: What fields are available in XRayResult?**

A: The ``XRayResult`` dataclass contains:

.. dropdown:: Material Properties

   - ``formula``: Chemical formula string
   - ``molecular_weight_g_mol``: Molecular weight (g/mol)
   - ``density_g_cm3``: Mass density (g/cm³)
   - ``total_electrons``: Total electrons per molecule
   - ``electron_density_per_ang3``: Electron density (electrons/Å³)

.. dropdown:: X-ray Properties (Arrays)

   - ``energy_kev``: X-ray energies (keV)
   - ``wavelength_angstrom``: X-ray wavelengths (Å)
   - ``dispersion_delta``: Dispersion coefficient δ
   - ``absorption_beta``: Absorption coefficient β
   - ``scattering_factor_f1``: Real scattering factor
   - ``scattering_factor_f2``: Imaginary scattering factor

.. dropdown:: Derived Quantities (Arrays)

   - ``critical_angle_degrees``: Critical angles (°)
   - ``attenuation_length_cm``: Attenuation lengths (cm)
   - ``real_sld_per_ang2``: Real scattering length density (Å⁻²)
   - ``imaginary_sld_per_ang2``: Imaginary scattering length density (Å⁻²)

**Q: How do I export results to Excel/CSV?**

A: Use pandas for data export:

.. code-block:: python

   import pandas as pd

   result = xlt.calculate_single_material_properties("SiO2", energies, 2.2)

   # Create DataFrame
   data = {
       'Energy_keV': result.energy_kev,
       'Critical_Angle_deg': result.critical_angle_degrees,
       'Attenuation_Length_cm': result.attenuation_length_cm
   }
   df = pd.DataFrame(data)

   # Export
   df.to_csv('results.csv', index=False)
   df.to_excel('results.xlsx', index=False)

**Q: Can I save results in HDF5 format?**

A: Yes, for large datasets:

.. code-block:: python

   import h5py
   import numpy as np

   with h5py.File('results.h5', 'w') as f:
       # Create groups
       material = f.create_group(result.formula)

       # Save data
       material.create_dataset('energy_kev', data=result.energy_kev)
       material.create_dataset('critical_angle_deg', data=result.critical_angle_degrees)

       # Save metadata
       material.attrs['density'] = result.density_g_cm3
       material.attrs['molecular_weight'] = result.molecular_weight_g_mol

Integration & Development
-------------------------

**Q: How do I integrate XRayLabTool with my existing analysis pipeline?**

A: XRayLabTool is designed to work well with the scientific Python ecosystem:

.. code-block:: python

   # With matplotlib for plotting
   import matplotlib.pyplot as plt

   # With pandas for data analysis
   import pandas as pd

   # With scipy for fitting
   from scipy.optimize import curve_fit

   # With xarray for multi-dimensional data
   import xarray as xr

**Q: Can I extend XRayLabTool with custom materials?**

A: Yes! You can define custom atomic scattering factors:

.. code-block:: python

   # For most use cases, use standard chemical formulas
   # For exotic cases, you may need to extend the atomic data
   # See the advanced tutorials for details

**Q: Is XRayLabTool suitable for real-time calculations?**

A: Yes, for small to medium datasets:

- Single calculation: ~0.03 ms
- 100 energy points: ~0.3 ms
- Batch processing: 150,000+ calculations/second

For real-time applications, preload common calculations and use the cache effectively.

**Q: How do I cite XRayLabTool in publications?**

A: Use this BibTeX entry:

.. code-block:: bibtex

   @software{xraylabtool,
     title = {XRayLabTool: High-Performance X-ray Optical Properties Calculator},
     author = {Wei Chen},
     url = {https://github.com/imewei/pyXRayLabTool},
     year = {2024},
     version = {0.1.10}
   }

Still Have Questions?
---------------------

If your question isn't answered here:

1. **Search the documentation**: Use the search box in the top navigation
2. **Check GitHub Issues**: `<https://github.com/imewei/pyXRayLabTool/issues>`_
3. **Ask on GitHub Discussions**: `<https://github.com/imewei/pyXRayLabTool/discussions>`_
4. **Read the tutorials**: :doc:`tutorials/index` for detailed examples

When asking for help, please include:

- XRayLabTool version: ``xraylabtool --version``
- Python version: ``python --version``
- Operating system
- Complete error message (if any)
- Minimal code example that reproduces the issue
