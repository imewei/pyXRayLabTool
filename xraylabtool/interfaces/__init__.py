"""
XRayLabTool Interfaces Module.

This module contains user interfaces including CLI and completion utilities.
"""

from xraylabtool.interfaces.cli import *
from xraylabtool.interfaces.completion import *

__all__ = [
    # CLI interface
    "main",
    "parse_energy_string",
    "cmd_calc",
    "cmd_batch",
    "cmd_convert",
    "cmd_formula",
    "cmd_atomic",
    "cmd_bragg",
    "cmd_list",
    "cmd_install_completion",
    # Completion system
    "CompletionInstaller",
    "install_completion_main",
]
