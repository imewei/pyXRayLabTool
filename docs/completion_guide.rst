Shell Completion Guide
======================

This comprehensive guide covers the new virtual environment-centric shell completion system for XRayLabTool.

Overview
--------

XRayLabTool features a modern completion system that provides intelligent shell completion while respecting virtual environment boundaries. Unlike traditional system-wide completion, this system automatically activates and deactivates with your Python environments.

Key Features
~~~~~~~~~~~~

**Virtual Environment Isolation**
   Completion is only available when the relevant virtual environment is active, preventing conflicts between projects.

**Multi-Shell Support**
   Native completion support for bash, zsh, fish, and PowerShell with shell-specific optimizations.

**No System Modifications**
   Installation requires no sudo privileges and makes no system-wide changes.

**Automatic Management**
   Completion automatically activates when you enter an environment and deactivates when you leave.

**Environment Detection**
   Supports venv, virtualenv, conda, mamba, Poetry, and Pipenv environments.

Quick Start
-----------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Activate your virtual environment
   conda activate myproject
   # or: source venv/bin/activate
   # or: poetry shell

   # 2. Install completion in the environment
   xraylabtool completion install

   # 3. Verify installation
   xraylabtool completion status

   # 4. Test completion (start new shell if needed)
   xraylabtool <TAB>

Environment Management
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # List all environments with completion status
   xraylabtool completion list

   # Show detailed status for current environment
   xraylabtool completion status

   # Show system information
   xraylabtool completion info

   # Remove completion from current environment
   xraylabtool completion uninstall

   # Remove completion from all environments
   xraylabtool completion uninstall --all

Shell-Specific Installation
----------------------------

The completion system auto-detects your shell, but you can specify explicitly:

Bash
~~~~

.. code-block:: bash

   # Install bash completion
   xraylabtool completion install --shell bash

   # Prerequisites (if not already installed)
   # macOS (Homebrew)
   brew install bash-completion@2

   # Linux (Ubuntu/Debian)
   sudo apt install bash-completion

Zsh
~~~

.. code-block:: bash

   # Install native zsh completion
   xraylabtool completion install --shell zsh

   # Prerequisites (if not already installed)
   # macOS (Homebrew)
   brew install zsh-completions

   # Add to ~/.zshrc if not present:
   if type brew &>/dev/null; then
     FPATH="$(brew --prefix)/share/zsh-completions:${FPATH}"
     autoload -U compinit
     compinit
   fi

Fish
~~~~

.. code-block:: bash

   # Install fish completion (no prerequisites required)
   xraylabtool completion install --shell fish

PowerShell
~~~~~~~~~~

.. code-block:: bash

   # Install PowerShell completion (Windows, macOS, Linux)
   xraylabtool completion install --shell powershell

Environment-Specific Workflows
-------------------------------

Conda/Mamba Environments
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create and activate environment
   conda create -n xraywork python=3.12
   conda activate xraywork

   # Install XRayLabTool and completion
   pip install xraylabtool
   xraylabtool completion install

   # Completion is now active
   xraylabtool c<TAB>  # Completes to 'calc', 'completion', 'convert'

   # Deactivate - completion unavailable
   conda deactivate
   xraylabtool c<TAB>  # No completion

   # Reactivate - completion returns
   conda activate xraywork
   xraylabtool c<TAB>  # Completion works again

Venv/Virtualenv
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create and activate venv
   python -m venv myproject
   source myproject/bin/activate

   # Install and set up completion
   pip install xraylabtool
   xraylabtool completion install

   # Test completion
   xraylabtool <TAB>

Poetry Projects
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create Poetry project
   poetry new xray-analysis
   cd xray-analysis

   # Add XRayLabTool dependency
   poetry add xraylabtool

   # Activate Poetry shell
   poetry shell

   # Install completion in Poetry environment
   xraylabtool completion install

   # Completion works within Poetry shell
   xraylabtool atomic <TAB>

Pipenv Projects
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create Pipenv environment
   mkdir xray-project && cd xray-project
   pipenv install xraylabtool

   # Activate Pipenv shell
   pipenv shell

   # Install completion
   xraylabtool completion install

   # Use completion
   xraylabtool batch <TAB>

Advanced Configuration
----------------------

Multiple Shells
~~~~~~~~~~~~~~~

You can install completion for multiple shells in the same environment:

.. code-block:: bash

   # Install for multiple shells
   xraylabtool completion install --shell bash
   xraylabtool completion install --shell zsh
   xraylabtool completion install --shell fish

   # Check what's installed
   xraylabtool completion status

Force Reinstallation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Force reinstall (useful for updates)
   xraylabtool completion install --force

   # Or uninstall and reinstall
   xraylabtool completion uninstall
   xraylabtool completion install

Completion Features
-------------------

The completion system provides intelligent suggestions for:

Command Completion
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xraylabtool <TAB>
   # Suggests: calc, batch, convert, formula, atomic, bragg, list, completion

Option Completion
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xraylabtool calc --<TAB>
   # Suggests: --material, --energy, --density, --output, --format, --help

File Path Completion
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   xraylabtool batch <TAB>
   # Completes file paths in current directory

   xraylabtool calc Si --output <TAB>
   # Completes file paths for output files

Value Completion
~~~~~~~~~~~~~~~~

.. code-block:: bash

   xraylabtool convert --energy <TAB>
   # Suggests common energies: 8.048, 10.0, 12.4

   xraylabtool calc <TAB>
   # Suggests common formulas: Si, SiO2, Al2O3, etc.

Migration from Legacy System
-----------------------------

If you previously used the old system-wide completion:

Removal of Old System
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Remove old system-wide completion (if installed)
   # Check common locations:
   ls /usr/local/share/bash-completion/completions/xraylabtool
   ls ~/.bash_completion.d/xraylabtool

   # Remove if found
   sudo rm /usr/local/share/bash-completion/completions/xraylabtool
   rm ~/.bash_completion.d/xraylabtool

Migration Steps
~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Update XRayLabTool to latest version
   pip install --upgrade xraylabtool

   # 2. For each environment where you want completion:
   conda activate myenv
   xraylabtool completion install

   # 3. Legacy commands still work (using new backend)
   xraylabtool install-completion    # Same as 'completion install'

Compatibility
~~~~~~~~~~~~~

The new system maintains backward compatibility:

.. code-block:: bash

   # These legacy commands still work:
   xraylabtool install-completion      # Uses new system
   xraylabtool uninstall-completion    # Uses new system
   xraylabtool --install-completion    # Uses new system

   # But these new commands are recommended:
   xraylabtool completion install      # More features
   xraylabtool completion uninstall    # Better management

Troubleshooting
---------------

Completion Not Working
~~~~~~~~~~~~~~~~~~~~~~

**Check environment status:**

.. code-block:: bash

   xraylabtool completion status

**Common issues:**

1. **Environment not active**: Ensure your virtual environment is activated
2. **Need new shell**: Start a new shell session after installation
3. **Shell prerequisites**: Install bash-completion, zsh-completions, etc.

**Debug steps:**

.. code-block:: bash

   # 1. Verify installation
   xraylabtool completion status

   # 2. Check if completion files exist
   ls $VIRTUAL_ENV/share/xraylabtool/completion/

   # 3. Force reinstall
   xraylabtool completion install --force

Shell Prerequisites Missing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Bash:**

.. code-block:: bash

   # Test if bash-completion is available
   type complete >/dev/null 2>&1 && echo "OK" || echo "Install bash-completion"

   # Install if missing
   # macOS: brew install bash-completion@2
   # Linux: sudo apt install bash-completion

**Zsh:**

.. code-block:: bash

   # Check if compinit is available
   which compinit >/dev/null && echo "OK" || echo "Configure zsh completions"

Environment Detection Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check environment detection
   xraylabtool completion list

   # If your environment isn't detected, check:
   echo $VIRTUAL_ENV      # For venv/virtualenv
   echo $CONDA_PREFIX     # For conda/mamba
   echo $POETRY_ACTIVE    # For Poetry

Permission Errors
~~~~~~~~~~~~~~~~~

The new system should never require sudo. If you see permission errors:

.. code-block:: bash

   # Check that you're not trying to install system-wide
   xraylabtool completion install  # Should work without sudo

   # If you get permission errors, check environment activation
   which python  # Should point to environment, not system

Performance and Caching
-----------------------

The completion system includes performance optimizations:

Caching System
~~~~~~~~~~~~~~

- **Command caching**: Available commands are cached for faster access
- **Environment caching**: Detected environments are cached with timeout
- **Completion script caching**: Generated scripts are cached per shell

Cache Management
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Cache is automatically managed, but you can clear it if needed
   # Cache location: ~/.xraylabtool/cache/

   # Force cache refresh (completion will detect and update)
   xraylabtool completion install --force

Best Practices
--------------

Development Workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Set up project environment
   git clone myproject
   cd myproject
   python -m venv venv
   source venv/bin/activate

   # 2. Install dependencies including XRayLabTool
   pip install -r requirements.txt
   # (assuming xraylabtool is in requirements.txt)

   # 3. Install completion for development
   xraylabtool completion install

   # 4. Now completion works during development
   xraylabtool calc Si --energy <TAB>

Multiple Projects
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Each project can have its own completion environment
   cd project1
   conda activate project1-env
   xraylabtool completion install

   cd ../project2
   conda activate project2-env
   xraylabtool completion install

   # Completion automatically switches between projects

Team Development
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Add completion setup to project documentation:
   # In README.md or CONTRIBUTING.md:

   # Setup Instructions:
   # 1. Install dependencies: pip install -r requirements.txt
   # 2. Install completion: xraylabtool completion install
   # 3. Start new shell or source shell config

CI/CD Considerations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # In CI environments, you typically don't need completion
   # But if testing CLI features:

   pip install xraylabtool
   # Skip: xraylabtool completion install (not needed in CI)
   xraylabtool calc Si --energy 10.0  # Test CLI functionality

API Reference
-------------

The completion system provides a programmatic API for advanced users:

Python API
~~~~~~~~~~

.. code-block:: python

   from xraylabtool.interfaces.completion_v2 import (
       EnvironmentDetector,
       CompletionInstaller
   )

   # Detect environments
   detector = EnvironmentDetector()
   current = detector.get_current_environment()
   all_envs = detector.discover_all_environments()

   # Manage completion
   installer = CompletionInstaller()
   installer.install(shell='bash')
   installer.list_environments()

Environment Information
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get detailed environment information
   from xraylabtool.interfaces.completion_v2 import EnvironmentDetector

   detector = EnvironmentDetector()
   env = detector.get_current_environment()

   if env:
       print(f"Type: {env.env_type}")
       print(f"Path: {env.path}")
       print(f"Name: {env.name}")
       print(f"Has completion: {env.has_completion}")
       print(f"Python version: {env.python_version}")

See Also
--------

- :doc:`cli_reference` - Complete CLI command reference
- :doc:`getting_started` - Basic installation and setup
- :doc:`index` - Main documentation index

For more help, use:

.. code-block:: bash

   xraylabtool completion --help
   xraylabtool completion install --help
   xraylabtool completion list --help
