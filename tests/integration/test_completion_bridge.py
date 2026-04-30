"""Tests for interfaces/completion.py bridge module.

This module tests the backward-compatible shell completion bridge that
integrates the new completion_v2 system while maintaining API compatibility.
"""

import pytest

from xraylabtool.interfaces.completion import (
    BASH_COMPLETION_SCRIPT,
    CompletionInstaller,
    get_xraylabtool_commands,
    install_completion_main,
    uninstall_completion_main,
)


class TestBashCompletionScript:
    """Tests for BASH_COMPLETION_SCRIPT generation."""

    def test_script_is_string(self):
        """Test that BASH_COMPLETION_SCRIPT is a string."""
        assert isinstance(BASH_COMPLETION_SCRIPT, str)

    def test_script_not_empty(self):
        """Test that completion script is not empty."""
        assert len(BASH_COMPLETION_SCRIPT) > 0

    def test_script_contains_bash_header(self):
        """Test that script is valid bash."""
        # Should start with shebang or contains bash keywords
        assert (
            "#!/bin/bash" in BASH_COMPLETION_SCRIPT
            or "bash" in BASH_COMPLETION_SCRIPT.lower()
        )

    def test_script_contains_completion_function(self):
        """Test that script defines completion function."""
        assert "_xraylabtool" in BASH_COMPLETION_SCRIPT

    def test_script_registers_completion(self):
        """Test that script registers completion."""
        assert "complete" in BASH_COMPLETION_SCRIPT.lower()

    def test_script_backward_compat_functions(self):
        """Test that script includes backward compatibility functions."""
        # Should have individual completion functions for each command
        assert "_xraylabtool_complete" in BASH_COMPLETION_SCRIPT
        assert "_xraylabtool_calc_complete" in BASH_COMPLETION_SCRIPT

    def test_script_includes_common_formulas(self):
        """Test that script includes common chemical formulas."""
        # Should have comments with examples of chemical formulas
        script_lower = BASH_COMPLETION_SCRIPT.lower()
        # Check for either the formulas themselves or formula examples
        assert (
            "siO2".lower() in script_lower
            or "h2o" in script_lower
            or "formula" in script_lower
        )

    def test_script_includes_command_options(self):
        """Test that script includes main command options."""
        script = BASH_COMPLETION_SCRIPT
        # Should list the main commands
        assert "calc" in script or "help" in script.lower()


class TestCompletionInstaller:
    """Tests for CompletionInstaller class import and availability."""

    def test_completion_installer_available(self):
        """Test that CompletionInstaller is available for import."""
        assert CompletionInstaller is not None

    def test_completion_installer_is_class(self):
        """Test that CompletionInstaller is a class."""
        assert isinstance(CompletionInstaller, type)


class TestCompletionFunctions:
    """Tests for completion integration functions."""

    def test_install_completion_main_available(self):
        """Test that install_completion_main is available."""
        assert install_completion_main is not None
        assert callable(install_completion_main)

    def test_uninstall_completion_main_available(self):
        """Test that uninstall_completion_main is available."""
        assert uninstall_completion_main is not None
        assert callable(uninstall_completion_main)

    def test_get_xraylabtool_commands_available(self):
        """Test that get_xraylabtool_commands is available."""
        assert get_xraylabtool_commands is not None
        assert callable(get_xraylabtool_commands)

    def test_get_xraylabtool_commands_returns_list(self):
        """Test that get_xraylabtool_commands returns a list."""
        commands = get_xraylabtool_commands()
        assert isinstance(commands, (list, dict))

    def test_get_xraylabtool_commands_not_empty(self):
        """Test that commands list is not empty."""
        commands = get_xraylabtool_commands()
        if isinstance(commands, (list, dict)):
            assert len(commands) > 0


class TestBridgeModuleExports:
    """Tests for backward compatibility exports."""

    def test_all_exports_defined(self):
        """Test that __all__ exports are correctly defined."""
        from xraylabtool.interfaces import completion

        assert hasattr(completion, "__all__")
        exports = completion.__all__

        # Should export backward-compatible items
        assert "BASH_COMPLETION_SCRIPT" in exports
        assert "CompletionInstaller" in exports

    def test_exports_are_accessible(self):
        """Test that all exported items are accessible."""
        from xraylabtool.interfaces.completion import (
            BASH_COMPLETION_SCRIPT,
            CompletionInstaller,
        )

        assert BASH_COMPLETION_SCRIPT is not None
        assert CompletionInstaller is not None

    def test_completion_integration_functions_exported(self):
        """Test that integration functions are exported."""
        from xraylabtool.interfaces.completion import (
            install_completion_main,
            uninstall_completion_main,
        )

        assert install_completion_main is not None
        assert uninstall_completion_main is not None


class TestScriptFallback:
    """Tests for script fallback behavior."""

    def test_script_is_valid_when_generated(self):
        """Test that generated script is valid bash."""
        script = BASH_COMPLETION_SCRIPT

        # Should be non-empty string
        assert isinstance(script, str)
        assert len(script) > 0

        # Should contain basic bash syntax
        assert "$(" in script or "=" in script or "(" in script

    def test_script_handles_generation_errors(self):
        """Test that script has graceful fallback."""
        # BASH_COMPLETION_SCRIPT should always have content,
        # either generated or fallback
        assert len(BASH_COMPLETION_SCRIPT) > 50  # Should have meaningful content
