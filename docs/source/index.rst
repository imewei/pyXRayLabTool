XRayLabTool Documentation
=========================

**High-Performance X-ray Optical Properties Calculator for Materials Science**

XRayLabTool is a comprehensive Python package and command-line tool for calculating X-ray optical 
properties of materials based on their chemical formulas and densities. Designed for synchrotron 
scientists, materials researchers, and X-ray optics developers, it provides fast, accurate 
calculations using CXRO/NIST atomic scattering factor data.

Key Features
------------

🐍 **Python API**
   - Complete programmatic access with descriptive field names
   - Easy-to-use dataclass-based output
   - Vectorized calculations using NumPy for high performance
   - Built-in caching system for atomic scattering factor data

⚡ **Command-Line Interface**
   - Powerful CLI for batch processing and quick calculations
   - 7 subcommands: calc, batch, convert, formula, atomic, bragg, list
   - Multiple output formats: Table, CSV, and JSON
   - Flexible energy input formats and parallel processing

🔬 **Scientific Accuracy**
   - Based on CXRO/NIST atomic scattering factor databases
   - PCHIP interpolation for accurate scattering factor calculations
   - Compute optical constants (δ, β), scattering factors (f1, f2)
   - Enhanced robustness with complex number handling and type safety

🚀 **Ultra-High Performance & Reliability**
   - **150,000+ calculations/second** with advanced optimizations
   - **Preloaded cache** for 92 common elements (10-50x speed boost)
   - **Vectorized matrix operations** for multi-element materials
   - **Memory-efficient batch processing** with intelligent memory management
   - **Parallel execution** with optimal worker count auto-detection
   - Support for both single and multiple material calculations
   - Comprehensive testing with robust error handling

Quick Start
-----------

**Python API:**

.. code-block:: python

   import xraylabtool as xlt
   
   # Calculate properties for quartz at 10 keV
   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
   print(f"Formula: {result.formula}")
   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm[0]:.2f} cm")
   
   # Multiple materials comparison
   formulas = ["SiO2", "Al2O3", "Fe2O3"]
   densities = [2.2, 3.95, 5.24]
   results = xlt.calculate_xray_properties(formulas, 10.0, densities)
   
   for formula, result in results.items():
       print(f"{formula}: θc = {result.critical_angle_degrees[0]:.3f}°")

**Command-Line Interface:**

.. code-block:: bash

   # Single material calculation
   xraylabtool calc SiO2 -e 10.0 -d 2.2
   
   # Energy range scan
   xraylabtool calc Si -e 5-15:11 -d 2.33 -o silicon_scan.csv
   
   # Unit conversion
   xraylabtool convert energy 10.0 --to wavelength
   
   # Batch processing
   xraylabtool batch materials.csv -o results.csv


Installation
------------

Install via pip:

.. code-block:: bash

   pip install xraylabtool

Or for development:

.. code-block:: bash

   git clone https://github.com/imewei/pyXRayLabTool.git
   cd pyXRayLabTool
   pip install -e .


Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guides:
   
   examples
   cli_guide
   performance_guide
   migration_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   license


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

