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

   pip install xraylabtool[all]

This installs the package with required dependencies.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or to get the latest features:

.. code-block:: bash

   git clone https://github.com/b80985/pyXRayLabTool.git
   cd pyXRayLabTool
   pip install -e .[dev]

Verify Installation
~~~~~~~~~~~~~~~~~~~

Verify installation:

.. code-block:: bash

   # Test CLI
   xraylabtool --version
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

   xraylabtool calc Si --density 2.33 --energy 8000

**Using Python:**

.. code-block:: python

   import xraylabtool as xrt

   result = xrt.calculate_single_material_properties(
       formula="Si",
       density=2.33,  # g/cm³
       energy=8000    # eV
   )

   print(f"Formula: {result.formula}")
   print(f"Critical angle: {result.critical_angle_degrees:.3f}°")
   print(f"Attenuation length: {result.attenuation_length_cm:.2f} cm")

Expected output::

   Formula: Si
   Critical angle: 0.158°
   Attenuation length: 9.84 cm

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

   xraylabtool calc Si --density 2.33 --energy 5000,8000,10000

**Python:**

.. code-block:: python

   import numpy as np

   energies = [5000, 8000, 10000]  # eV
   results = []

   for energy in energies:
       result = xrt.calculate_single_material_properties("Si", 2.33, energy)
       results.append(result)

   for result in results:
       print(f"{result.energy_ev} eV: θc = {result.critical_angle_degrees:.3f}°")

Different Materials
~~~~~~~~~~~~~~~~~~~

Try other materials:

.. code-block:: python

   # Silicon dioxide (quartz)
   sio2 = xrt.calculate_single_material_properties("SiO2", 2.20, 8000)

   # Aluminum
   al = xrt.calculate_single_material_properties("Al", 2.70, 8000)

   # Copper
   cu = xrt.calculate_single_material_properties("Cu", 8.96, 8000)

   materials = [("Si", sio2), ("SiO2", sio2), ("Al", al), ("Cu", cu)]
   for name, result in materials:
       print(f"{name}: Critical angle = {result.critical_angle_degrees:.3f}°")

Batch Processing
----------------

For multiple materials, use batch processing:

Create a CSV file ``materials.csv``:

.. code-block:: text

   Formula,Density,Energy
   Si,2.33,8000
   SiO2,2.20,8000
   Al,2.70,8000
   Cu,8.96,8000

Process the batch:

.. code-block:: bash

   xraylabtool batch materials.csv --output results.csv

Or in Python:

.. code-block:: python

   # Define materials
   materials = [
       {"formula": "Si", "density": 2.33},
       {"formula": "SiO2", "density": 2.20},
       {"formula": "Al", "density": 2.70},
       {"formula": "Cu", "density": 8.96}
   ]

   # Calculate for all materials at 8 keV
   results = xrt.calculate_xray_properties(materials, energy=8000)

   # Display results
      print(f"{result.formula}: "
             f"θc = {result.critical_angle_degrees:.3f}°, "
             f"μ⁻¹ = {result.attenuation_length_cm:.2f} cm")

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
   substrates = ["Si", "SiO2", "Zerodur"]  # Zerodur is a glass-ceramic
   densities = [2.33, 2.20, 2.53]
   energy = 8000  # eV

   print("Mirror substrate comparison at 8 keV:")
   print("Material | Critical Angle | Attenuation Length")
   print("---------|----------------|-------------------")

   for formula, density in zip(substrates, densities):
       result = xrt.calculate_single_material_properties(formula, density, energy)
       print(f"{formula:8} | {result.critical_angle_degrees:13.3f}° | "
             f"{result.attenuation_length_cm:15.2f} cm")

Beamline Planning
~~~~~~~~~~~~~~~~~

For synchrotron beamline design:

.. code-block:: python

   # Energy scan for beamline components
   energies = np.logspace(3, 4.5, 50)  # 1 keV to ~32 keV
   material = "Si"
   density = 2.33

   critical_angles = []
   attenuation_lengths = []

   for energy in energies:
       result = xrt.calculate_single_material_properties(material, density, energy)
       critical_angles.append(result.critical_angle_mrad)
       attenuation_lengths.append(result.attenuation_length_cm)

   # Plot or analyze the energy dependence
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

   ax1.loglog(energies, critical_angles)
   ax1.set_xlabel('Energy (eV)')
   ax1.set_ylabel('Critical Angle (mrad)')
   ax1.set_title('Critical Angle vs Energy')

   ax2.loglog(energies, attenuation_lengths)
   ax2.set_xlabel('Energy (eV)')
   ax2.set_ylabel('Attenuation Length (cm)')
   ax2.set_title('Attenuation Length vs Energy')

   plt.tight_layout()
   plt.show()

Understanding the Architecture
------------------------------

XRayLabTool uses a modular architecture with 5 focused sub-packages:

**calculators/**
   Core X-ray physics calculations and algorithms. Contains the main calculation functions and XRayResult data structure.

**data_handling/**
   Atomic data caching and batch processing. Provides high-performance data management with preloaded elements.

**interfaces/**
   User interfaces including the complete CLI and shell completion systems.

**io/**
   File operations and data export functionality. Handles CSV, JSON, and other format conversions.

**validation/**
   Input validation and error handling. Ensures data quality and provides detailed error messages.

This modular design allows you to import only what you need:

.. code-block:: python

   # Import main calculation functions
   from xraylabtool.calculators import calculate_single_material_properties

   # Import specific utilities
   from xraylabtool.utils import parse_formula, energy_to_wavelength

   # Import data structures
   from xraylabtool.calculators.core import XRayResult

Next Steps
----------

Next steps:

1. **Explore the CLI**: Try all 9 commands with ``xraylabtool --help``
2. **Read the Tutorials**: Learn techniques and workflows
3. **Study Examples**: See applications
4. **Check the API Reference**: View available functions
5. **Learn the Physics**: Understand the X-ray optics background

Key Documentation Sections:

- `Tutorials <tutorials/index.rst>`_ - Step-by-step guides for common tasks
- `CLI Reference <cli_reference.rst>`_ - Complete command-line interface documentation
- `Examples <examples/index.rst>`_ - Real-world usage examples
- `API Reference <api/index.rst>`_ - Complete API reference
- `X-ray Physics <physics/xray_optics.rst>`_ - X-ray physics background

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

   # Good - batch processing
   results = xrt.calculate_xray_properties(materials, energies)

   # Less efficient - individual calculations
   for material in materials:
       for energy in energies:
           result = xrt.calculate_single_material_properties(
               material['formula'], material['density'], energy
           )
