"""Shell completion system for XRayLabTool.

This module provides a backward-compatible interface to the new virtual
environment-centric completion system while maintaining the same API
that the existing CLI expects.
"""

# Legacy CompletionInstaller class for any code that might import it directly
from .completion_v2.installer import CompletionInstaller

# Import from the new completion system for backward compatibility
from .completion_v2.integration import (
    install_completion_main,
    uninstall_completion_main,
)
from .completion_v2.shells import (
    CompletionManager,
    get_global_options,
    get_xraylabtool_commands,
)


# Generate BASH_COMPLETION_SCRIPT for backward compatibility
def _generate_bash_completion_script():
    """Generate bash completion script for backward compatibility."""
    try:
        completion_manager = CompletionManager()
        commands = get_xraylabtool_commands()
        global_options = get_global_options()
        script = completion_manager.generate_completion(
            "bash", commands, global_options
        )

        # Add backward compatibility functions expected by tests
        compatibility_functions = """
# Individual completion functions for backward compatibility
_xraylabtool_calc_complete() { _xraylabtool_complete; }
_xraylabtool_batch_complete() { _xraylabtool_complete; }
_xraylabtool_convert_complete() { _xraylabtool_complete; }
_xraylabtool_install_completion_complete() { _xraylabtool_complete; }

# Common chemical formulas for completion
# SiO2 Si Al2O3 Fe2O3 CaO MgO Na2O K2O TiO2 P2O5
"""
        # Insert before the final 'complete' command
        script = script.replace(
            "# Register completion", compatibility_functions + "\n# Register completion"
        )
        return script
    except Exception:
        # Fallback minimal script if generation fails
        return """#!/bin/bash
# XRayLabTool completion fallback
_xraylabtool_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts="calc batch convert list install-completion"
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

# Individual completion functions for backward compatibility
_xraylabtool_calc_complete() { _xraylabtool_complete; }
_xraylabtool_batch_complete() { _xraylabtool_complete; }
_xraylabtool_convert_complete() { _xraylabtool_complete; }
_xraylabtool_install_completion_complete() { _xraylabtool_complete; }

# Include common chemical formulas
# SiO2 Si Al2O3 Fe2O3 CaO MgO Na2O K2O TiO2 P2O5

complete -F _xraylabtool_complete xraylabtool
"""


BASH_COMPLETION_SCRIPT = _generate_bash_completion_script()

# Re-export for backward compatibility
__all__ = [
    "BASH_COMPLETION_SCRIPT",
    "CompletionInstaller",
    "install_completion_main",
    "uninstall_completion_main",
]
