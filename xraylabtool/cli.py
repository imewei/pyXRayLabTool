"""
Command-line interface compatibility module.

This module provides backward compatibility by re-exporting CLI functionality
from xraylabtool.interfaces.cli. New code should import directly from
xraylabtool.interfaces.cli for better organization.

All CLI functions and classes are imported from xraylabtool.interfaces.cli.
"""

import warnings

# Re-export everything from interfaces.cli for backward compatibility
from xraylabtool.interfaces.cli import (
    cmd_atomic,
    cmd_batch,
    cmd_bragg,
    cmd_calc,
    cmd_convert,
    cmd_formula,
    cmd_install_completion,
    cmd_list,
    cmd_uninstall_completion,
    create_parser,
    format_xray_result,
    main,
    parse_energy_string,
)

# Emit a deprecation warning when this module is imported
warnings.warn(
    "Importing from xraylabtool.cli is deprecated. "
    "Please import from xraylabtool.interfaces.cli instead. "
    "Support for xraylabtool.cli imports will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "main",
    "create_parser",
    "cmd_calc",
    "cmd_batch",
    "cmd_convert",
    "cmd_formula",
    "cmd_atomic",
    "cmd_bragg",
    "cmd_list",
    "cmd_install_completion",
    "cmd_uninstall_completion",
    "format_xray_result",
    "parse_energy_string",
]
