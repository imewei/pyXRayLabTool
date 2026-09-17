Getting Started
===============

This guide shows XRayLabTool installation and basic usage.

Interactive Notebook
--------------------

Open the Getting Started notebook directly (no embedded render in the docs):

- `Open in Colab <https://colab.research.google.com/github/imewei/pyXRayLabTool/blob/main/docs/examples/getting_started.ipynb>`__
- `View on nbviewer <https://nbviewer.org/github/imewei/pyXRayLabTool/blob/main/docs/examples/getting_started.ipynb>`__
- `Download (.ipynb) <https://raw.githubusercontent.com/imewei/pyXRayLabTool/main/docs/examples/getting_started.ipynb>`__

Installation
------------

System Requirements
~~~~~~~~~~~~~~~~~~~

XRayLabTool requires:

- **Python 3.12 or higher**
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 512 MB RAM (recommended 2 GB for large calculations)
- **Storage**: 50 MB for basic installation

Install from PyPI
~~~~~~~~~~~~~~~~~

Install XRayLabTool using pip:

.. code-block:: bash

   # Core package (NumPy + JAX CPU backends, CLI, GUI, matplotlib)
   pip install xraylabtool

   # NVIDIA GPU acceleration: pick the CUDA major matching your driver
   pip install "xraylabtool[gpu_cuda13]"
   pip install "xraylabtool[gpu_cuda12]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features (uses ``uv``):

.. code-block:: bash

   git clone https://github.com/imewei/pyXRayLabTool.git
   cd pyXRayLabTool
   uv sync
   uv run pytest  # verify installation

Verify Installation
~~~~~~~~~~~~~~~~~~~

Verify installation:

.. code-block:: bash

   # Test CLI (xtool is a short alias for xraylabtool)
   xraylabtool --version
   xtool --version
   xraylabtool --help

   # Test Python API
   python -c "import xraylabtool; print('Installation successful!')"

Shell Completion (Virtual Environment-Centric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install completion in your current virtual environment:

.. code-block:: bash

   # Install completion in current environment
   xraylabtool completion install

   # Verify installation
   xraylabtool completion status

   # List all environments with completion status
   xraylabtool completion list

The new completion system provides:

- **Environment Isolation**: Completion only available when environment is active
- **Multi-Shell Support**: Native completion for bash, zsh, fish, PowerShell
- **No System Changes**: No sudo required, installs per environment
- **Auto-Activation**: Completion activates/deactivates with environment
- **Alias Support**: Completion works for both ``xraylabtool`` and its short alias ``xtool``

For legacy compatibility, the old commands still work:

.. code-block:: bash

   xraylabtool install-completion    # Uses new system backend

First Steps
-----------

Simple Calculation
~~~~~~~~~~~~~~~~~~

Let's calculate X-ray properties for silicon at 8 keV:

**Using the CLI:**

.. code-block:: bash

   xraylabtool calc Si -e 8.0 -d 2.33

**Using Python:**

.. code-block:: python

   import xraylabtool as xrt

   # Positional order is (formula, energy_keV, density)
   result = xrt.calculate_single_material_properties("Si", 8.0, 2.33)

   print(f"Formula: {result.formula}")
   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm[0] * 1e4:.1f} µm")

Expected output::

   Formula: Si
   Critical angle: 0.225°
   Attenuation length: 69.7 µm

All per-energy fields on :class:`~xraylabtool.XRayResult` are NumPy arrays, even for a single
energy, so index ``[0]`` for a scalar.

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

The main properties calculated are:

- **Critical angle**: Angle for total external reflection
- **Attenuation length**: Distance for 1/e intensity reduction
- **Delta (δ)**: Real part of refractive index decrement
- **Beta (β)**: Imaginary part related to absorption

Multiple Energies
~~~~~~~~~~~~~~~~~

Calculate properties across an energy range:

**CLI:**

.. code-block:: bash

   xraylabtool calc Si -e 5.0,8.0,10.0 -d 2.33

**Python:**

.. code-block:: python

   result = xrt.calculate_single_material_properties("Si", [5.0, 8.0, 10.0], 2.33)

   for e, theta_c in zip(result.energy_kev, result.critical_angle_degrees):
       print(f"{e:.1f} keV: θc = {theta_c:.3f}°")

Different Materials
~~~~~~~~~~~~~~~~~~~

Compare several materials in one call:

.. code-block:: python

   formulas = ["Si", "SiO2", "Al", "Cu"]
   densities = [2.33, 2.20, 2.70, 8.96]

   results = xrt.calculate_xray_properties(formulas, 8.0, densities)

   for formula, r in results.items():
       print(f"{formula:5}: θc = {r.critical_angle_degrees[0]:.3f}°, "
             f"attenuation length = {r.attenuation_length_cm[0] * 1e4:.1f} µm")

Batch Processing
----------------

For many materials, use the ``batch`` command. Create ``materials.csv`` with lowercase
``formula,density,energy`` columns (energy in keV; comma-separate several energies in one cell):

.. code-block:: text

   formula,density,energy
   Si,2.33,8.0
   SiO2,2.20,8.0
   Al,2.70,"5.0,8.0,10.0"
   Cu,8.96,8.0

Process the batch:

.. code-block:: bash

   xraylabtool batch materials.csv -o results.csv

Or in Python, :func:`~xraylabtool.calculate_xray_properties` (above) handles material lists
directly.

Graphical User Interface (GUI)
------------------------------

XRayLabTool includes a modern desktop application for interactive analysis.

Launch via:

.. code-block:: bash

   python -m xraylabtool.gui

**Key Features:**

- **Interactive Analysis**: Single and multiple material comparisons.
- **Modern Interface**: Clean design with Light and Dark data-optimized themes.
- **Theme Toggle**: Switch between Light and Dark modes via the status bar toggle.
- **Persisted Settings**: Your theme preference is saved automatically.

Common Use Cases
----------------

Mirror Design
~~~~~~~~~~~~~

For X-ray mirror applications:

.. code-block:: python

   # Compare substrate materials
   substrates = ["Si", "SiO2", "SiC"]
   densities = [2.33, 2.20, 3.21]
   energy = 8.0  # keV

   print("Mirror substrate comparison at 8 keV:")
   print("Material | Critical Angle | Attenuation Length")
   print("---------|----------------|-------------------")

   for formula, density in zip(substrates, densities):
       result = xrt.calculate_single_material_properties(formula, energy, density)
       print(f"{formula:8} | {result.critical_angle_degrees[0]:13.3f}° | "
             f"{result.attenuation_length_cm[0] * 1e4:15.1f} µm")

Beamline Planning
~~~~~~~~~~~~~~~~~

For synchrotron beamline design:

.. code-block:: python

   # Energy scan for beamline components
   energies = np.logspace(0, np.log10(30), 50)  # 1 keV to 30 keV

   result = xrt.calculate_single_material_properties("Si", energies, 2.33)
   critical_angles = np.radians(result.critical_angle_degrees) * 1e3  # mrad
   attenuation_lengths = result.attenuation_length_cm

   # Plot or analyze the energy dependence
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   ax1.loglog(energies, critical_angles)
   ax1.set_xlabel('Energy (keV)')
   ax1.set_ylabel('Critical Angle (mrad)')
   ax1.set_title('Critical Angle vs Energy')

   ax2.loglog(energies, attenuation_lengths)
   ax2.set_xlabel('Energy (keV)')
   ax2.set_ylabel('Attenuation Length (cm)')
   ax2.set_title('Attenuation Length vs Energy')

   plt.tight_layout()
   plt.show()

Understanding the Architecture
------------------------------

XRayLabTool uses a modular architecture:

**backend/**
   NumPy/JAX array backend abstraction. Switch at runtime with ``set_backend("jax")`` for JIT-compiled GPU acceleration.

**calculators/**
   Core X-ray physics calculations and algorithms. Contains the main calculation functions and ``XRayResult`` data structure.

**data_handling/**
   Atomic data caching and batch processing. Provides high-performance data management with preloaded elements.

**interfaces/**
   User interfaces including the complete CLI and shell completion systems.

**io/**
   File operations and data export functionality. Handles CSV, JSON, and other format conversions.

**validation/**
   Input validation and error handling. Ensures data quality and provides detailed error messages.

**gui/**
   PySide6 desktop application with PyQtGraph interactive plots.

This modular design allows you to import only what you need:

.. code-block:: python

   # Import main calculation functions
   from xraylabtool.calculators import calculate_single_material_properties

   # Import specific utilities
   from xraylabtool.utils import parse_formula, energy_to_wavelength

   # Switch to JAX backend for GPU acceleration
   from xraylabtool.backend import set_backend
   set_backend("jax")

Next Steps
----------

Next steps:

1. **Explore the CLI**: Try all 9 commands with ``xraylabtool --help``
2. **Read the Tutorials**: Learn techniques and workflows
3. **Study Examples**: See applications
4. **Check the API Reference**: View available functions
5. **Learn the Physics**: Understand the X-ray optics background

Key Documentation Sections:

- `CLI Reference <cli_reference.rst>`_ - Complete command-line interface documentation
- `Examples <../examples/index.rst>`_ - Real-world usage examples
- `API Reference <../api/index.rst>`_ - Complete API reference
- `X-ray Physics <../physics/xray_optics.rst>`_ - X-ray physics background

Getting Help
------------

If you encounter issues:

1. **Check the FAQ**: Common questions and solutions
2. **Read Error Messages**: XRayLabTool provides detailed error descriptions
3. **Use Help Commands**: ``xraylabtool --help`` and ``xraylabtool <command> --help``
4. **Check Documentation**: This documentation covers most use cases
5. **Report Issues**: Use the GitHub issue tracker for bugs

**Command-line help:**

.. code-block:: bash

   xraylabtool --help                    # General help
   xraylabtool calc --help               # Help for calc command
   xraylabtool list examples             # Show example materials

**Python help:**

.. code-block:: python

   import xraylabtool as xrt
   help(xrt.calculate_single_material_properties)

.. code-block:: text

   # Or in IPython/Jupyter for interactive help
   In [1]: xrt.calculate_single_material_properties?

Performance Tips
----------------

For better performance:

1. **Use preloaded elements**: Si, O, Al, Fe, C, etc. are cached for speed
2. **Batch processing**: Process multiple materials together when possible
3. **Energy arrays**: Use NumPy arrays for energy ranges
4. **Avoid repeated parsing**: Cache formula parsing results

.. code-block:: python

   # Good - one call, energy array, all materials
   results = xrt.calculate_xray_properties(formulas, energies, densities)

   # Less efficient - one call per (material, energy) pair
   for formula, density in zip(formulas, densities):
       for energy in energies:
           result = xrt.calculate_single_material_properties(formula, energy, density)
