#!/usr/bin/env python3
"""
Tests for the completion installer module of XRayLabTool.

This module contains comprehensive tests for the shell completion installation
functionality, testing both the installer logic and the bash completion script
content.
"""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

try:
    from xraylabtool.completion_installer import (
        BASH_COMPLETION_SCRIPT,
        CompletionInstaller,
        install_completion_main,
    )
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.completion_installer import (
        BASH_COMPLETION_SCRIPT,
        CompletionInstaller,
        install_completion_main,
    )


class TestBashCompletionScript:
    """Test the embedded bash completion script content."""

    def test_script_is_valid_bash(self):
        """Test that the completion script has valid bash syntax."""
        assert isinstance(BASH_COMPLETION_SCRIPT, str)
        assert len(BASH_COMPLETION_SCRIPT) > 1000  # Should be substantial

        # Check for bash shebang or completion structure
        assert "#!/bin/bash" in BASH_COMPLETION_SCRIPT
        assert "_xraylabtool_complete" in BASH_COMPLETION_SCRIPT

    def test_script_contains_all_commands(self):
        """Test that completion script includes all xraylabtool commands."""
        expected_commands = [
            "calc",
            "batch",
            "convert",
            "formula",
            "atomic",
            "bragg",
            "list",
            "install-completion",
        ]

        for command in expected_commands:
            assert (
                command in BASH_COMPLETION_SCRIPT
            ), f"Command '{command}' not found in completion script"

    def test_script_contains_completion_functions(self):
        """Test that all major completion functions are present."""
        expected_functions = [
            "_xraylabtool_complete",
            "_xraylabtool_calc_complete",
            "_xraylabtool_batch_complete",
            "_xraylabtool_convert_complete",
            "_xraylabtool_formula_complete",
            "_xraylabtool_atomic_complete",
            "_xraylabtool_bragg_complete",
            "_xraylabtool_list_complete",
            "_xraylabtool_install_completion_complete",
        ]

        for func in expected_functions:
            assert (
                func in BASH_COMPLETION_SCRIPT
            ), f"Function '{func}' not found in completion script"

    def test_script_contains_chemical_suggestions(self):
        """Test that common chemical formulas are suggested."""
        expected_formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]
        expected_elements = ["H", "C", "N", "O", "Si", "Al", "Fe"]

        for formula in expected_formulas:
            assert formula in BASH_COMPLETION_SCRIPT

        for element in expected_elements:
            assert element in BASH_COMPLETION_SCRIPT

    def test_script_contains_energy_suggestions(self):
        """Test that common energy values are suggested."""
        expected_energies = ["10.0", "8.048", "5.0,10.0,15.0"]

        for energy in expected_energies:
            assert energy in BASH_COMPLETION_SCRIPT

    def test_script_contains_density_suggestions(self):
        """Test that common density values are suggested."""
        expected_densities = ["2.2", "2.33", "3.95", "5.24"]

        for density in expected_densities:
            assert density in BASH_COMPLETION_SCRIPT

    def test_script_completion_registration(self):
        """Test that completion is properly registered."""
        assert "complete -F _xraylabtool_complete xraylabtool" in BASH_COMPLETION_SCRIPT


class TestCompletionInstaller:
    """Test the CompletionInstaller class functionality."""

    def test_installer_initialization(self):
        """Test CompletionInstaller can be initialized."""
        installer = CompletionInstaller()
        assert installer is not None
        assert hasattr(installer, "install_bash_completion")
        assert hasattr(installer, "uninstall_bash_completion")
        assert hasattr(installer, "test_completion")

    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.exists")
    def test_get_bash_completion_dir(self, mock_exists, mock_is_dir):
        """Test finding system bash completion directories."""
        installer = CompletionInstaller()

        # Test when directories exist and are accessible
        mock_exists.return_value = True
        mock_is_dir.return_value = True
        result = installer.get_bash_completion_dir()
        assert result is not None
        assert isinstance(result, Path)

        # Test when directories exist but are not accessible
        mock_exists.return_value = True
        mock_is_dir.side_effect = PermissionError("Permission denied")
        result = installer.get_bash_completion_dir()
        # Should continue to next candidate or return None

        # Test when no directories exist
        mock_exists.return_value = False
        mock_is_dir.side_effect = None
        mock_is_dir.return_value = False
        result = installer.get_bash_completion_dir()
        assert result is None

    def test_get_user_bash_completion_dir(self):
        """Test getting user bash completion directory."""
        installer = CompletionInstaller()

        with patch("pathlib.Path.mkdir") as mock_mkdir:
            result = installer.get_user_bash_completion_dir()
            assert isinstance(result, Path)
            assert ".bash_completion.d" in str(result)
            mock_mkdir.assert_called_once_with(exist_ok=True)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_test_completion_success(self, mock_exists, mock_subprocess):
        """Test completion testing when everything works."""
        installer = CompletionInstaller()

        # Mock successful which command and completion check
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "complete -F _xraylabtool_complete xraylabtool"
        )

        result = installer.test_completion()
        assert isinstance(result, bool)

    @patch("subprocess.run")
    def test_test_completion_command_not_found(self, mock_subprocess):
        """Test completion testing when xraylabtool command not found."""
        installer = CompletionInstaller()

        # Mock command not found with subprocess.CalledProcessError
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            1, "which", "command not found"
        )

        result = installer.test_completion()
        assert isinstance(result, bool)
        assert result is False  # Should return False when command not found

    @patch("pathlib.Path.write_text")
    @patch("pathlib.Path.mkdir")
    def test_install_bash_completion_user(self, mock_mkdir, mock_write):
        """Test user installation of bash completion."""
        installer = CompletionInstaller()

        with patch.object(installer, "_add_bash_completion_sourcing"):
            result = installer.install_bash_completion(system_wide=False)
            assert isinstance(result, bool)
            mock_write.assert_called_once_with(BASH_COMPLETION_SCRIPT)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    @patch("os.unlink")
    def test_install_bash_completion_system(
        self, mock_unlink, mock_temp, mock_subprocess
    ):
        """Test system-wide installation of bash completion."""
        installer = CompletionInstaller()

        # Mock successful system installation
        mock_subprocess.return_value.returncode = 0
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_completion"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        with patch.object(installer, "get_bash_completion_dir") as mock_get_dir:
            mock_get_dir.return_value = Path("/usr/share/bash-completion/completions")
            result = installer.install_bash_completion(system_wide=True)
            assert isinstance(result, bool)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.unlink")
    def test_uninstall_bash_completion_user(self, mock_unlink, mock_exists):
        """Test user uninstallation of bash completion."""
        installer = CompletionInstaller()

        mock_exists.return_value = True
        result = installer.uninstall_bash_completion(system_wide=False)
        assert isinstance(result, bool)
        mock_unlink.assert_called_once()

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_uninstall_bash_completion_system(self, mock_exists, mock_subprocess):
        """Test system-wide uninstallation of bash completion."""
        installer = CompletionInstaller()

        mock_exists.return_value = True
        mock_subprocess.return_value.returncode = 0

        with patch.object(installer, "get_bash_completion_dir") as mock_get_dir:
            mock_get_dir.return_value = Path("/usr/share/bash-completion/completions")
            result = installer.uninstall_bash_completion(system_wide=True)
            assert isinstance(result, bool)

    @patch("pathlib.Path.exists")
    def test_uninstall_completion_not_installed(self, mock_exists):
        """Test uninstalling when completion is not installed."""
        installer = CompletionInstaller()

        mock_exists.return_value = False
        result = installer.uninstall_bash_completion(system_wide=False)
        assert result is True  # Should succeed if already not installed

    def test_add_bash_completion_sourcing_new(self):
        """Test adding completion sourcing when not already present."""
        installer = CompletionInstaller()

        # Test that the method doesn't crash when called
        # We'll mock the file operations to avoid system changes
        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.read_text") as mock_read_text,
            patch("builtins.open", mock_open()) as mock_file,
            patch("builtins.print"),
        ):

            mock_exists.return_value = True
            mock_read_text.return_value = "existing content without completion"

            # Call the method - it should complete without error
            installer._add_bash_completion_sourcing()

            # The method should have tried to read the file
            mock_read_text.assert_called_once()
            # And should have opened the file for writing (append mode)
            mock_file.assert_called()

    @patch("os.environ.get")
    @patch(
        "builtins.open",
        new_callable=mock_open,
    )
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.exists")
    def test_add_bash_completion_sourcing_existing(
        self, mock_exists, mock_read_text, mock_file, mock_environ
    ):
        """Test adding completion sourcing when already present."""
        installer = CompletionInstaller()

        mock_exists.return_value = True
        # Mock read_text to return content that contains the sourcing line
        mock_read_text.return_value = "source ~/.bash_completion.d/xraylabtool"
        # Mock shell environment to return bash instead of zsh
        mock_environ.side_effect = lambda key, default="": (
            "/bin/bash" if key == "SHELL" else default
        )
        installer._add_bash_completion_sourcing()

        # Should not write if already present
        mock_file().write.assert_not_called()


class TestInstallCompletionMain:
    """Test the main install_completion_main function."""

    def test_install_completion_main_install(self):
        """Test main function in install mode."""

        class MockArgs:
            uninstall = False
            test = False
            system = False
            shell = None

        args = MockArgs()

        with patch.object(CompletionInstaller, "install_completion") as mock_install:
            mock_install.return_value = True
            result = install_completion_main(args)
            assert result == 0
            mock_install.assert_called_once_with(shell_type=None, system_wide=False)

    def test_install_completion_main_uninstall(self):
        """Test main function in uninstall mode."""

        class MockArgs:
            uninstall = True
            test = False
            system = False
            shell = None

        args = MockArgs()

        with patch.object(
            CompletionInstaller, "uninstall_completion"
        ) as mock_uninstall:
            mock_uninstall.return_value = True
            result = install_completion_main(args)
            assert result == 0
            mock_uninstall.assert_called_once_with(shell_type=None, system_wide=False)

    def test_install_completion_main_test(self):
        """Test main function in test mode."""

        class MockArgs:
            uninstall = False
            test = True
            system = False
            shell = None

        args = MockArgs()

        with patch.object(CompletionInstaller, "test_completion") as mock_test:
            result = install_completion_main(args)
            assert result == 0
            mock_test.assert_called_once()

    def test_install_completion_main_system_wide(self):
        """Test main function with system-wide installation."""

        class MockArgs:
            uninstall = False
            test = False
            system = True
            shell = None

        args = MockArgs()

        with patch.object(CompletionInstaller, "install_completion") as mock_install:
            mock_install.return_value = True
            result = install_completion_main(args)
            assert result == 0
            mock_install.assert_called_once_with(shell_type=None, system_wide=True)

    def test_install_completion_main_failure(self):
        """Test main function when installation fails."""

        class MockArgs:
            uninstall = False
            test = False
            system = False
            shell = None

        args = MockArgs()

        with patch.object(CompletionInstaller, "install_completion") as mock_install:
            mock_install.return_value = False
            result = install_completion_main(args)
            assert result == 1


class TestCompletionScriptEdgeCases:
    """Test edge cases and specific patterns in completion script."""

    def test_completion_handles_energy_formats(self):
        """Test that completion script handles various energy formats."""
        # Check for energy format patterns
        energy_patterns = ["5-15:11", "1-30:100:log", "5.0,10.0,15.0"]

        for pattern in energy_patterns:
            assert pattern in BASH_COMPLETION_SCRIPT

    def test_completion_handles_file_extensions(self):
        """Test that completion script handles file extensions."""
        file_patterns = [".csv", "*.csv"]

        for pattern in file_patterns:
            assert pattern in BASH_COMPLETION_SCRIPT

    def test_completion_script_structure(self):
        """Test overall structure of completion script."""
        # Should have proper function definitions
        assert "() {" in BASH_COMPLETION_SCRIPT  # Function definitions
        assert "COMPREPLY=(" in BASH_COMPLETION_SCRIPT  # Completion array
        assert "compgen" in BASH_COMPLETION_SCRIPT  # Completion generator
        assert "case" in BASH_COMPLETION_SCRIPT  # Case statements for commands

    def test_completion_error_handling(self):
        """Test that completion script has error handling."""
        # Should handle array bounds safely
        assert "COMP_CWORD" in BASH_COMPLETION_SCRIPT
        assert "COMP_WORDS" in BASH_COMPLETION_SCRIPT

        # Should have safety checks for array access
        assert (
            "#COMP_WORDS" in BASH_COMPLETION_SCRIPT
            or "COMP_WORDS[@]" in BASH_COMPLETION_SCRIPT
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
