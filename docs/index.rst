.. XRayLabTool documentation master file

XRayLabTool Documentation
=========================

.. image:: https://img.shields.io/pypi/v/xraylabtool.svg
   :target: https://pypi.org/project/xraylabtool/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/xraylabtool.svg
   :target: https://pypi.org/project/xraylabtool/
   :alt: Python versions

**XRayLabTool** is a Python package and command-line tool for calculating X-ray optical properties of materials. It provides both a Python API and CLI for synchrotron scientists, materials researchers, and X-ray optics developers.

Key Features
------------

**Performance**
   - 150,000+ calculations per second
   - Preloaded atomic data cache for 10-50x speed improvement
   - Vectorized calculations and smart memory management

**X-ray Physics**
   - Complex refractive index calculations
   - Attenuation coefficients and penetration depths
   - Critical angles for total external reflection
   - Transmission and reflection coefficients

**Architecture**
   - Modular design with 5 focused sub-packages:

     - **calculators**: Core X-ray physics calculations
     - **data_handling**: Atomic data caching and batch processing
     - **interfaces**: CLI and completion systems
     - **io**: File operations and export functionality
     - **validation**: Input validation and error handling

   - Type-safe with complete type hints
   - Extensive test coverage (>95%)
   - Cross-platform compatibility

**CLI Interface**
   - 10+ commands for common tasks and completion management
   - Multiple output formats (table, CSV, JSON)
   - Virtual environment-centric shell completion (bash, zsh, fish, PowerShell)
   - Batch processing from CSV files
   - Environment isolation for completion

**Scientific Data Handling**
   - Built on CXRO/NIST atomic scattering databases
   - Support for energy ranges and arrays
   - Chemical formula parsing and validation
   - Export capabilities for downstream analysis

Quick Start
-----------

Interactive Notebooks
~~~~~~~~~~~~~~~~~~~~~

Open the notebook gallery (Colab / nbviewer links) on the :doc:`Examples page <examples/index>`.

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install xraylabtool

Basic Usage
~~~~~~~~~~~

**Python API:**

.. code-block:: python

   import xraylabtool as xrt

   # Calculate X-ray properties for silicon at 8 keV
   result = xrt.calculate_single_material_properties(
       formula="Si",
       density=2.33,
       energy=8000
   )

   print(f"Critical angle: {result.critical_angle_degrees:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm:.2f} cm")

**Command Line:**

.. code-block:: bash

   # Single material calculation
   xraylabtool calc Si --density 2.33 --energy 8000

   # Batch processing
   xraylabtool batch materials.csv --output results.csv

Navigation
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   tutorials/index
   cli_reference
   completion_guide
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Scientific Background

   physics/xray_optics
   physics/atomic_data
   physics/calculations

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   performance
   testing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Links

   GitHub Repository <https://github.com/b80985/pyXRayLabTool>
   PyPI Package <https://pypi.org/project/xraylabtool/>
   Issue Tracker <https://github.com/b80985/pyXRayLabTool/issues>

Performance Highlights
----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operation
     - Speed
     - Use Case
   * - Single calculation
     - < 0.1 ms
     - Interactive analysis
   * - Batch processing (1000 materials)
     - < 10 ms
     - High-throughput screening
   * - Energy array (100 points)
     - < 1 ms
     - Spectroscopy analysis

Target Audience
---------------

- **Synchrotron Scientists**: Beamline optimization and experimental planning
- **Materials Researchers**: X-ray characterization and property prediction
- **X-ray Optics Developers**: Mirror and multilayer design
- **Students & Educators**: Learning X-ray physics and optics

Citation
--------

If you use XRayLabTool in your research, please cite:

.. code-block:: bibtex

   @software{xraylabtool,
     title={XRayLabTool: X-ray Optical Properties Calculator},
     author={XRayLabTool Contributors},
     url={https://github.com/b80985/pyXRayLabTool},
     version={0.2.3},
     year={2025}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
