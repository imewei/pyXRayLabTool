"""
XRayLabTool Interfaces Module.

This module contains user interfaces including CLI and completion utilities.
"""

from xraylabtool.interfaces.cli import (
    cmd_atomic,
    cmd_batch,
    cmd_bragg,
    cmd_calc,
    cmd_convert,
    cmd_formula,
    cmd_install_completion,
    cmd_list,
    main,
    parse_energy_string,
)
from xraylabtool.interfaces.completion_v2.installer import CompletionInstaller
from xraylabtool.interfaces.completion_v2.integration import install_completion_main

__all__ = [
    # Completion system
    "CompletionInstaller",
    "cmd_atomic",
    "cmd_batch",
    "cmd_bragg",
    "cmd_calc",
    "cmd_convert",
    "cmd_formula",
    "cmd_install_completion",
    "cmd_list",
    "install_completion_main",
    # CLI interface
    "main",
    "parse_energy_string",
]
