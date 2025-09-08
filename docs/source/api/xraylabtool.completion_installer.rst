xraylabtool.completion_installer module
========================================

Shell completion installation and management module.

This module provides functionality for installing, managing, and testing bash shell
completion for the XRayLabTool CLI. It includes the completion script, installer logic,
and integration with the main CLI through the ``install-completion`` command.

.. automodule:: xraylabtool.completion_installer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Overview
--------

The completion installer module provides:

* **Embedded Bash Completion Script**: Full bash completion functionality built into
  the package
* **Installation Management**: User and system-wide installation options
* **Testing Utilities**: Test completion installation and functionality
* **Cross-Platform Support**: Handles different bash completion directory
  structures

Key Components
--------------

Completion Script
~~~~~~~~~~~~~~~~~

The module includes a complete bash completion script (``BASH_COMPLETION_SCRIPT``)
that provides:

* Command completion for all 8 xraylabtool commands
* Parameter and option completion
* Chemical formula and element suggestions
* Energy and density value hints
* File completion for input/output operations
* Format and field option completion

CompletionInstaller Class
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: xraylabtool.completion_installer.CompletionInstaller
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Main Functions
--------------

.. autofunction:: xraylabtool.completion_installer.install_completion_main
   :no-index:

Constants
---------

.. autodata:: xraylabtool.completion_installer.BASH_COMPLETION_SCRIPT
   :annotation: str

   The complete bash completion script embedded in the module. This script
   provides intelligent tab completion for all xraylabtool commands, options,
   and common values.

Usage Example
-------------

.. code-block:: python

   from xraylabtool.completion_installer import CompletionInstaller

   # Create installer instance
   installer = CompletionInstaller()

   # Install completion for current user
   success = installer.install_bash_completion(system_wide=False)

   # Test completion installation
   working = installer.test_completion()

Shell Integration
-----------------

The completion system integrates with the main CLI through the
``install-completion`` command:

.. code-block:: bash

   # Install completion (recommended after package installation)
   xraylabtool install-completion

   # Test installation
   xraylabtool install-completion --test

   # System-wide installation (requires sudo)
   xraylabtool install-completion --system

   # Uninstall completion
   xraylabtool install-completion --uninstall
