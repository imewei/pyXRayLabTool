#!/usr/bin/env python3
"""
Integration tests for XRayLabTool shell completion functionality.

This module contains integration tests that verify the shell completion works
end-to-end, including CLI integration, completion installation, and actual
bash completion behavior.
"""

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from xraylabtool.cli import main
    from xraylabtool.completion_installer import BASH_COMPLETION_SCRIPT
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.cli import main
    from xraylabtool.completion_installer import BASH_COMPLETION_SCRIPT


class TestCLICompletionIntegration:
    """Test integration between CLI and completion functionality."""

    def test_help_includes_install_completion(self):
        """Test that main CLI help includes install-completion command."""
        with patch("sys.argv", ["xraylabtool", "--help"]):
            with patch("sys.stdout") as mock_stdout:
                with pytest.raises(SystemExit):
                    main()

                # Check help output contains install-completion
                stdout_calls = [
                    call.args[0]
                    for call in mock_stdout.write.call_args_list
                    if call.args
                ]
                help_text = "".join(stdout_calls)
                assert "install-completion" in help_text

    def test_install_completion_command_exists(self):
        """Test that install-completion command is accessible."""
        with patch("sys.argv", ["xraylabtool", "install-completion", "--help"]):
            with patch("sys.stdout") as mock_stdout:
                with pytest.raises(SystemExit):
                    main()

                # Should show install-completion specific help
                stdout_calls = [
                    call.args[0]
                    for call in mock_stdout.write.call_args_list
                    if call.args
                ]
                help_text = "".join(stdout_calls)
                assert "Install or manage shell completion" in help_text

    @patch("builtins.print")
    def test_install_completion_test_integration(self, mock_print):
        """Test install-completion --test command integration."""
        with patch("sys.argv", ["xraylabtool", "install-completion", "--test"]):
            result = main()

            # Should complete without error
            assert result in [0, 1]  # Both success and "not installed" are valid

            # Should print some output about testing
            print_calls = [str(call) for call in mock_print.call_args_list]
            # Should contain some indication of testing completion
            output_text = "".join(print_calls)
            assert "xraylabtool" in output_text

    def test_list_examples_includes_install_completion(self):
        """Test that 'list examples' includes install-completion."""
        with patch("sys.argv", ["xraylabtool", "list", "examples"]):
            with patch("builtins.print") as mock_print:
                result = main()
                assert result == 0

                # Check that install-completion is in examples
                print_calls = [str(call) for call in mock_print.call_args_list]
                examples_text = "".join(print_calls)
                assert "install-completion" in examples_text

    def test_main_command_routing(self):
        """Test that install-completion routes to correct handler."""
        # Test that the command exists in the command handlers
        with patch("sys.argv", ["xraylabtool", "install-completion", "--test"]):
            # Mock the completion installer to avoid actual system interaction
            with patch("xraylabtool.cli.cmd_install_completion") as mock_cmd:
                mock_cmd.return_value = 0

                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestCompletionScriptIntegration:
    """Test integration of the bash completion script."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Bash not available on Windows CI")
    def test_completion_script_syntax(self):
        """Test that completion script has valid bash syntax."""
        # Create a temporary script file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bash", delete=False
        ) as temp_file:
            temp_file.write(BASH_COMPLETION_SCRIPT)
            temp_file.flush()

            try:
                # Test bash syntax with 'bash -n'
                result = subprocess.run(
                    ["bash", "-n", temp_file.name], capture_output=True, text=True
                )

                # Should have no syntax errors
                assert result.returncode == 0, f"Bash syntax errors: {result.stderr}"

            finally:
                os.unlink(temp_file.name)

    @pytest.mark.skipif(platform.system() == "Windows", reason="Bash not available on Windows CI")
    def test_completion_script_function_loading(self):
        """Test that completion script functions can be loaded."""
        bash_command = """
        source /dev/stdin
        declare -f _xraylabtool_complete
        """

        try:
            result = subprocess.run(
                ["bash", "-c", bash_command],
                input=BASH_COMPLETION_SCRIPT,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should be able to load and declare the function
            assert result.returncode == 0
            assert "_xraylabtool_complete" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Bash not available or timeout occurred")

    @pytest.mark.skipif(platform.system() == "Windows", reason="Bash not available on Windows CI")
    def test_completion_script_registration(self):
        """Test that completion registration works."""
        bash_command = """
        source /dev/stdin
        complete -p xraylabtool
        """

        try:
            result = subprocess.run(
                ["bash", "-c", bash_command],
                input=BASH_COMPLETION_SCRIPT,
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Should register completion for xraylabtool
            assert result.returncode == 0
            assert "xraylabtool" in result.stdout
            assert "_xraylabtool_complete" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Bash not available or timeout occurred")


class TestCompletionInstallationIntegration:
    """Test end-to-end completion installation."""

    def test_completion_installation_flow(self):
        """Test complete installation workflow without system changes."""
        # Create a temporary directory to simulate installation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_completion_dir = Path(temp_dir) / ".bash_completion.d"
            temp_completion_dir.mkdir()  # Create the directory
            temp_completion_file = temp_completion_dir / "xraylabtool"

            # Mock the completion installer to use our temp directory
            with patch(
                "xraylabtool.completion_installer.CompletionInstaller.get_user_bash_completion_dir"
            ) as mock_dir:
                mock_dir.return_value = temp_completion_dir

                with patch(
                    "xraylabtool.completion_installer.CompletionInstaller._add_bash_completion_sourcing"
                ):
                    # Test installation
                    with patch("sys.argv", ["xraylabtool", "install-completion"]):
                        with patch("builtins.print"):
                            result = main()

                            # Should succeed
                            assert result == 0

                            # Should create completion file
                            assert temp_completion_file.exists()

                            # File should contain completion script
                            content = temp_completion_file.read_text()
                            assert "_xraylabtool_complete" in content
                            assert "xraylabtool" in content

    def test_completion_uninstallation_flow(self):
        """Test complete uninstallation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_completion_dir = Path(temp_dir) / ".bash_completion.d"
            temp_completion_dir.mkdir()
            temp_completion_file = temp_completion_dir / "xraylabtool"

            # Create a fake completion file
            temp_completion_file.write_text("fake completion content")
            assert temp_completion_file.exists()

            # Mock the completion installer
            with patch(
                "xraylabtool.completion_installer.CompletionInstaller.get_user_bash_completion_dir"
            ) as mock_dir:
                mock_dir.return_value = temp_completion_dir

                # Test uninstallation
                with patch(
                    "sys.argv", ["xraylabtool", "install-completion", "--uninstall"]
                ):
                    with patch("builtins.print"):
                        result = main()

                        # Should succeed
                        assert result == 0

                        # Should remove completion file
                        assert not temp_completion_file.exists()

    @patch("subprocess.run")
    def test_completion_test_functionality(self, mock_subprocess):
        """Test completion testing functionality."""
        # Mock successful testing
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "complete -F _xraylabtool_complete xraylabtool"
        )

        with patch("sys.argv", ["xraylabtool", "install-completion", "--test"]):
            with patch("builtins.print"):
                result = main()

                # Should succeed
                assert result == 0


class TestCompletionContentValidation:
    """Test that completion content matches CLI capabilities."""

    def test_completion_commands_match_cli(self):
        """Test that completion script includes all CLI commands."""
        # Get commands from completion script
        completion_commands = []
        for line in BASH_COMPLETION_SCRIPT.split("\n"):
            if "commands=" in line and '"' in line:
                # Extract commands from line like: local commands="calc batch convert..."
                start = line.find('"') + 1
                end = line.rfind('"')
                if start > 0 and end > start:
                    completion_commands = line[start:end].split()
                    break

        # Expected commands from CLI
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

        for cmd in expected_commands:
            assert (
                cmd in completion_commands
            ), f"Command '{cmd}' missing from completion"

    def test_completion_options_coverage(self):
        """Test that completion covers major command options."""
        # Test calc command options
        calc_options = [
            "--energy",
            "--density",
            "--output",
            "--format",
            "--fields",
            "--precision",
        ]
        for option in calc_options:
            assert (
                option in BASH_COMPLETION_SCRIPT
            ), f"Calc option '{option}' missing from completion"

        # Test batch command options
        batch_options = ["--output", "--format", "--workers", "--fields"]
        for option in batch_options:
            assert (
                option in BASH_COMPLETION_SCRIPT
            ), f"Batch option '{option}' missing from completion"

        # Test install-completion command options
        install_options = ["--user", "--system", "--test", "--uninstall"]
        for option in install_options:
            assert (
                option in BASH_COMPLETION_SCRIPT
            ), f"Install-completion option '{option}' missing from completion"

    def test_completion_value_suggestions(self):
        """Test that completion suggests appropriate values."""
        # Energy values
        energy_values = ["10.0", "8.048", "5.0,10.0,15.0", "5-15:11", "1-30:100:log"]
        for value in energy_values:
            assert (
                value in BASH_COMPLETION_SCRIPT
            ), f"Energy value '{value}' missing from completion"

        # Format values
        format_values = ["table", "csv", "json"]
        for value in format_values:
            assert (
                value in BASH_COMPLETION_SCRIPT
            ), f"Format value '{value}' missing from completion"

        # Common chemical formulas
        formulas = ["SiO2", "Al2O3", "Fe2O3", "Si", "C"]
        for formula in formulas:
            assert (
                formula in BASH_COMPLETION_SCRIPT
            ), f"Formula '{formula}' missing from completion"


class TestCompletionRobustness:
    """Test completion system robustness and edge cases."""

    def test_completion_handles_missing_bash(self):
        """Test that installation handles missing bash gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("bash not found")

            with patch("sys.argv", ["xraylabtool", "install-completion", "--test"]):
                with patch("builtins.print"):
                    result = main()
                    # Should handle gracefully without crashing
                    assert isinstance(result, int)

    def test_completion_handles_permission_errors(self):
        """Test that installation handles permission errors gracefully."""
        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.side_effect = PermissionError("Permission denied")

            with patch("sys.argv", ["xraylabtool", "install-completion"]):
                with patch("builtins.print"):
                    result = main()
                    # Should handle gracefully and return error code
                    assert result == 1

    def test_completion_handles_nonexistent_directories(self):
        """Test completion installation with non-existent directories."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            with patch("sys.argv", ["xraylabtool", "install-completion", "--system"]):
                with patch("builtins.print"):
                    result = main()
                    # Should handle gracefully
                    assert isinstance(result, int)

    def test_completion_script_array_safety(self):
        """Test that completion script handles array access safely."""
        # Check for safe array access patterns
        safe_patterns = [
            "${#COMP_WORDS[@]}",  # Array length check
            "${COMP_CWORD} -gt 0",  # Index bounds check (actual pattern used)
        ]

        for pattern in safe_patterns:
            assert (
                pattern in BASH_COMPLETION_SCRIPT
            ), f"Safe array pattern '{pattern}' not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
