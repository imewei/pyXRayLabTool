Installation Guide
==================

This guide covers various methods to install XRayLabTool on different platforms.

Prerequisites
-------------

**System Requirements:**

- Python ≥ 3.12
- 2 GB RAM minimum (4 GB recommended for large calculations)
- 100 MB disk space

**Platform Support:**

- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)

Basic Installation
------------------

PyPI Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install XRayLabTool is from PyPI:

.. tabs::

   .. group-tab:: Linux/macOS

      .. code-block:: bash

         pip install xraylabtool

   .. group-tab:: Windows

      .. code-block:: batch

         pip install xraylabtool

   .. group-tab:: Conda

      .. code-block:: bash

         # Note: Not yet available on conda-forge, use pip in conda environment
         conda create -n xraylabtool python=3.12
         conda activate xraylabtool
         pip install xraylabtool

Virtual Environment Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We strongly recommend using a virtual environment:

.. tabs::

   .. group-tab:: venv (Python built-in)

      .. code-block:: bash

         # Create virtual environment
         python -m venv xraylab_env

         # Activate (Linux/macOS)
         source xraylab_env/bin/activate

         # Activate (Windows)
         xraylab_env\\Scripts\\activate

         # Install XRayLabTool
         pip install xraylabtool

   .. group-tab:: conda

      .. code-block:: bash

         # Create conda environment
         conda create -n xraylabtool python=3.12
         conda activate xraylabtool

         # Install XRayLabTool
         pip install xraylabtool

   .. group-tab:: pipenv

      .. code-block:: bash

         # Create pipenv environment
         pipenv install xraylabtool

         # Activate environment
         pipenv shell

Development Installation
------------------------

For Contributing or Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute to XRayLabTool or install from source:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/imewei/pyXRayLabTool.git
   cd pyXRayLabTool

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in development mode with all dependencies
   pip install -e .[dev]

Optional Dependencies
---------------------

Performance Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

For enhanced performance monitoring and profiling:

.. code-block:: bash

   pip install xraylabtool[perf]

Documentation Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

To build documentation locally:

.. code-block:: bash

   pip install xraylabtool[docs]

   # Build documentation
   cd docs
   make html

All Dependencies
~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

   pip install xraylabtool[all]

Verification
------------

Test Installation
~~~~~~~~~~~~~~~~~

After installation, verify that XRayLabTool is working correctly:

.. code-block:: bash

   # Test command-line interface
   xraylabtool --version

   # Test basic functionality
   xraylabtool calc SiO2 -e 10.0 -d 2.2

.. code-block:: python

   # Test Python API
   import xraylabtool as xlt

   # Quick test calculation
   result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
   print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")

Shell Completion Setup
~~~~~~~~~~~~~~~~~~~~~~

Enable intelligent tab completion for enhanced productivity:

.. code-block:: bash

   # Auto-detect shell and install completion
   xraylabtool install-completion

   # Test completion is working
   xraylabtool install-completion --test

   # Restart your shell or source your config file
   source ~/.bashrc  # or ~/.zshrc, ~/.config/fish/config.fish

Platform-Specific Notes
------------------------

Windows
~~~~~~~

**PowerShell Users:**

.. code-block:: powershell

   # PowerShell 5.1+ or PowerShell Core 7+ required
   # Check version
   $PSVersionTable.PSVersion

   # Install completion
   xraylabtool install-completion powershell

**Common Issues:**

- If you encounter permission errors, run PowerShell as Administrator
- For long path issues, enable long path support in Windows settings

macOS
~~~~~

**Homebrew Users:**

.. code-block:: bash

   # Install Python 3.12 via Homebrew if needed
   brew install python@3.12

   # Install bash-completion for full shell completion functionality
   brew install bash-completion@2

**M1/M2 Mac Users:**

XRayLabTool works natively on Apple Silicon. No special configuration needed.

Linux
~~~~~

**Ubuntu/Debian:**

.. code-block:: bash

   # Install Python 3.12 if not available
   sudo apt update
   sudo apt install python3.12 python3.12-pip python3.12-venv

   # Install bash-completion
   sudo apt install bash-completion

**RHEL/CentOS/Fedora:**

.. code-block:: bash

   # Install Python 3.12
   sudo dnf install python3.12 python3.12-pip

   # Install bash-completion
   sudo dnf install bash-completion

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'xraylabtool'**

- Ensure you've activated the correct virtual environment
- Verify installation with ``pip list | grep xraylabtool``

**Command not found: xraylabtool**

- Check if the CLI is in your PATH: ``which xraylabtool``
- Try reinstalling with ``pip install --force-reinstall xraylabtool``

**Permission denied errors**

- Use virtual environments instead of system-wide installation
- On Windows, run as Administrator if necessary

**Slow calculations**

- Ensure NumPy is using optimized BLAS libraries
- Check available memory with ``xraylabtool list constants``

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `troubleshooting guide <howto/troubleshooting.html>`_
2. Search existing `GitHub issues <https://github.com/imewei/pyXRayLabTool/issues>`_
3. Create a new issue with your system information:

.. code-block:: bash

   # Gather system information for bug reports
   python -c "
   import platform
   import sys
   import xraylabtool as xlt

   print(f'Python: {sys.version}')
   print(f'Platform: {platform.platform()}')
   print(f'XRayLabTool: {xlt.__version__}')
   "

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade xraylabtool

To upgrade all dependencies:

.. code-block:: bash

   pip install --upgrade xraylabtool[all]

Uninstalling
------------

To completely remove XRayLabTool:

.. code-block:: bash

   # Remove shell completion first
   xraylabtool install-completion --uninstall

   # Uninstall the package
   pip uninstall xraylabtool

Next Steps
----------

After successful installation:

- Read the :doc:`quickstart` guide for your first calculations
- Explore :doc:`tutorials/index` for detailed examples
- Check out the :doc:`cli_guide` for command-line usage
- Review :doc:`examples` for common use cases
