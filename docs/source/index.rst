XRayLabTool Documentation
=========================

**Material Property Calculations for X-ray Interactions**

XRayLabTool is a Python package that provides functions to calculate X-ray optical properties 
of materials based on their chemical formulas and densities. It is particularly useful for 
synchrotron scientists, materials researchers, and X-ray optics developers.

Features
--------

- Compute optical constants (δ, β), scattering factors (f1, f2), and other X-ray interaction parameters
- Support for both single and multiple material calculations
- Easy-to-use dataclass-based output
- Based on CXRO/NIST data tables
- Vectorized calculations using NumPy for high performance
- Built-in caching system for atomic scattering factor data
- **Enhanced robustness** with complex number handling and type safety
- **Modern compatibility** with updated pandas and numpy methods
- **PCHIP interpolation** for accurate scattering factor calculations
- **Comprehensive testing** with 100% coverage and robust error handling

Quick Start
-----------

.. code-block:: python

   import xraylabtool as xlt
   
   # Calculate properties for quartz at 10 keV
   result = xlt.SubRefrac("SiO2", 10.0, 2.2)
   print(f"Critical angle: {result.Critical_Angle[0]:.3f}°")
   
   # Multiple materials comparison
   formulas = ["SiO2", "Al2O3", "Fe2O3"]
   densities = [2.2, 3.95, 5.24]
   results = xlt.Refrac(formulas, 10.0, densities)


Installation
------------

Install via pip:

.. code-block:: bash

   pip install xraylabtool

Or for development:

.. code-block:: bash

   git clone https://github.com/yourusername/xraylabtool.git
   cd xraylabtool
   pip install -e .


Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   changelog
   license


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

