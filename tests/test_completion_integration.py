#!/usr/bin/env python3
"""
Integration tests for XRayLabTool shell completion functionality.

This module contains integration tests that verify the shell completion works
end-to-end, including CLI integration, completion installation, and actual
bash completion behavior.
"""

import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
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
                assert "Install shell completion" in help_text

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
            with patch("xraylabtool.interfaces.cli.cmd_install_completion") as mock_cmd:
                mock_cmd.return_value = 0

                result = main()
                assert result == 0
                mock_cmd.assert_called_once()


class TestCompletionScriptIntegration:
    """Test integration of the bash completion script."""

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Bash not available on Windows CI"
    )
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
                    ["bash", "-n", temp_file.name],
                    check=False,
                    capture_output=True,
                    text=True,
                )

                # Should have no syntax errors
                assert result.returncode == 0, f"Bash syntax errors: {result.stderr}"

            finally:
                os.unlink(temp_file.name)

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Bash not available on Windows CI"
    )
    def test_completion_script_function_loading(self):
        """Test that completion script functions can be loaded."""
        bash_command = """
        source /dev/stdin
        declare -f _xraylabtool_complete
        """

        try:
            result = subprocess.run(
                ["bash", "-c", bash_command],
                check=False,
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

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Bash not available on Windows CI"
    )
    def test_completion_script_registration(self):
        """Test that completion registration works."""
        bash_command = """
        source /dev/stdin
        complete -p xraylabtool
        """

        try:
            result = subprocess.run(
                ["bash", "-c", bash_command],
                check=False,
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
        """Test complete installation workflow in virtual environment."""
        # Create a temporary directory to simulate virtual environment
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_venv = Path(temp_dir) / "test_venv"
            temp_venv.mkdir()

            # Create bin directory with activate script (simulate VE)
            bin_dir = temp_venv / "bin"
            bin_dir.mkdir()
            activate_script = bin_dir / "activate"
            activate_script.write_text(
                "#!/bin/bash\n# Virtual environment activate script\n"
            )

            # Mock the virtual environment detection
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(temp_venv)}):
                with patch(
                    "xraylabtool.completion_installer.CompletionInstaller._modify_venv_activation_scripts"
                ) as mock_modify:
                    mock_modify.return_value = True

                    # Test installation
                    with patch("sys.argv", ["xraylabtool", "install-completion"]):
                        with patch("builtins.print"):
                            result = main()

                            # Should succeed
                            assert result == 0

                            # Should create virtual environment completion structure
                            completion_dir = temp_venv / "xraylabtool-completion"
                            assert completion_dir.exists()

                            # Should create bash completion file
                            bash_completion_file = (
                                completion_dir / "bash" / "xraylabtool.bash"
                            )
                            assert bash_completion_file.exists()

                            # File should contain completion script
                            content = bash_completion_file.read_text()
                            assert "_xraylabtool_complete" in content
                            assert "xraylabtool" in content

                            # Should create activation script
                            activation_script = (
                                completion_dir / "activate-completion.sh"
                            )
                            assert activation_script.exists()

    def test_completion_uninstallation_flow(self):
        """Test complete uninstallation workflow from virtual environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_venv = Path(temp_dir) / "test_venv"
            temp_venv.mkdir()

            # Create bin directory with activate script (simulate VE)
            bin_dir = temp_venv / "bin"
            bin_dir.mkdir()
            activate_script = bin_dir / "activate"
            activate_script.write_text(
                "#!/bin/bash\n# Virtual environment activate script\n"
            )

            # Create fake completion structure
            completion_dir = temp_venv / "xraylabtool-completion"
            completion_dir.mkdir()
            bash_dir = completion_dir / "bash"
            bash_dir.mkdir()
            bash_completion_file = bash_dir / "xraylabtool.bash"
            bash_completion_file.write_text("fake completion content")
            activation_script = completion_dir / "activate-completion.sh"
            activation_script.write_text("fake activation content")

            # Mock the virtual environment detection
            with patch.dict(os.environ, {"VIRTUAL_ENV": str(temp_venv)}):
                # Test uninstallation using new separate command
                with patch("sys.argv", ["xraylabtool", "uninstall-completion"]):
                    with patch("builtins.print"):
                        result = main()

                        # Should succeed
                        assert result == 0

                        # Should remove completion directory
                        assert not completion_dir.exists()

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

        # Test install-completion command options (simplified system)
        install_options = ["--test", "--help"]
        for option in install_options:
            assert (
                option in BASH_COMPLETION_SCRIPT
            ), f"Install-completion option '{option}' missing from completion"

        # Test that obsolete options are NOT present
        obsolete_options = ["--user", "--system", "--uninstall"]
        for option in obsolete_options:
            assert (
                option not in BASH_COMPLETION_SCRIPT
            ), f"Obsolete install-completion option '{option}' still in completion"

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


class TestSimplifiedVirtualEnvironmentIntegration:
    """Test simplified virtual environment integration with shell completion."""

    def test_uninstall_completion_command_integration(self):
        """Test new uninstall-completion command."""
        with (
            patch("sys.argv", ["xraylabtool", "uninstall-completion", "bash"]),
            patch("xraylabtool.interfaces.cli.cmd_uninstall_completion") as mock_cmd,
        ):
            mock_cmd.return_value = 0

            result = main()
            assert result == 0
            mock_cmd.assert_called_once()

    def test_install_completion_simplified_command(self):
        """Test simplified install-completion command."""
        with (
            patch("sys.argv", ["xraylabtool", "install-completion", "bash"]),
            patch("xraylabtool.interfaces.cli.cmd_install_completion") as mock_cmd,
        ):
            mock_cmd.return_value = 0

            result = main()
            assert result == 0
            mock_cmd.assert_called_once()

    def test_completion_commands_without_shell_type(self):
        """Test completion commands without explicit shell type (auto-detection)."""
        test_commands = ["install-completion", "uninstall-completion"]

        for command in test_commands:
            with (
                patch("sys.argv", ["xraylabtool", command]),
                patch(
                    f"xraylabtool.interfaces.cli.cmd_{command.replace('-', '_')}"
                ) as mock_cmd,
            ):
                mock_cmd.return_value = 0

                result = main()
                assert result == 0
                mock_cmd.assert_called_once()

    def test_deprecated_options_are_rejected(self):
        """Test that deprecated VE options are properly rejected with error codes."""
        # These commands should return error codes since the options don't exist
        deprecated_commands = [
            ["install-completion", "--uninstall", "--venv"],
            ["install-completion", "--uninstall", "--conda"],
            ["install-completion", "--uninstall", "--all-environments"],
            ["install-completion", "--system"],
            ["install-completion", "--user"],
        ]

        for cmd_args in deprecated_commands:
            with patch("sys.argv", ["xraylabtool"] + cmd_args):
                result = main()
                # Should return non-zero exit code for unrecognized options
                assert result != 0

    @patch("builtins.print")
    def test_uninstall_completion_integration_flow(self, mock_print):
        """Test complete uninstall completion workflow integration."""
        from xraylabtool.completion_installer import uninstall_completion_main

        class MockArgs:
            shell = "bash"

        args = MockArgs()

        with patch(
            "xraylabtool.completion_installer.CompletionInstaller.uninstall_completion"
        ) as mock_uninstall:
            mock_uninstall.return_value = True

            result = uninstall_completion_main(args)

            assert result == 0
            mock_uninstall.assert_called_once_with(
                shell_type="bash",
                cleanup_session=True,
            )


class TestCompletionScriptSimplifiedOptions:
    """Test that completion scripts include simplified completion options."""

    def test_bash_completion_excludes_obsolete_options(self):
        """Test that bash completion script excludes obsolete VE options."""
        from xraylabtool.completion_installer import BASH_COMPLETION_SCRIPT

        # Should contain new commands
        assert "install-completion" in BASH_COMPLETION_SCRIPT
        assert "uninstall-completion" in BASH_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "--venv",
            "--conda",
            "--all-environments",
            "--no-cleanup-session",
        ]
        for option in obsolete_options:
            assert (
                option not in BASH_COMPLETION_SCRIPT
            ), f"Obsolete option '{option}' still in bash completion"

    def test_fish_completion_excludes_obsolete_options(self):
        """Test that Fish completion script excludes obsolete VE options."""
        from xraylabtool.completion_installer import FISH_COMPLETION_SCRIPT

        # Should contain new commands
        assert "install-completion" in FISH_COMPLETION_SCRIPT
        assert "uninstall-completion" in FISH_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "-l venv",
            "-l conda",
            "-l all-environments",
            "-l no-cleanup-session",
        ]
        for option in obsolete_options:
            assert (
                option not in FISH_COMPLETION_SCRIPT
            ), f"Obsolete option '{option}' still in Fish completion"

    def test_powershell_completion_excludes_obsolete_options(self):
        """Test that PowerShell completion script excludes obsolete VE options."""
        from xraylabtool.completion_installer import POWERSHELL_COMPLETION_SCRIPT

        # Should contain new commands
        assert "install-completion" in POWERSHELL_COMPLETION_SCRIPT
        assert "uninstall-completion" in POWERSHELL_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "--venv",
            "--conda",
            "--all-environments",
            "--no-cleanup-session",
        ]
        for option in obsolete_options:
            assert (
                option not in POWERSHELL_COMPLETION_SCRIPT
            ), f"Obsolete option '{option}' still in PowerShell completion"

    def test_completion_script_simplified_logic(self):
        """Test that completion scripts have simplified logic without obsolete VE options."""
        from xraylabtool.completion_installer import BASH_COMPLETION_SCRIPT

        # Should contain both completion commands and their functions
        assert "_xraylabtool_install_completion_complete" in BASH_COMPLETION_SCRIPT
        assert "_xraylabtool_uninstall_completion_complete" in BASH_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE option variables
        assert "ve_opts" not in BASH_COMPLETION_SCRIPT


class TestSimplifiedCompletionArguments:
    """Test simplified completion command arguments."""

    def test_uninstall_completion_main_basic_args(self):
        """Test uninstall_completion_main handles basic arguments correctly."""
        from xraylabtool.completion_installer import uninstall_completion_main

        class MockArgs:
            shell = "bash"

        args = MockArgs()

        with patch(
            "xraylabtool.completion_installer.CompletionInstaller.uninstall_completion"
        ) as mock_uninstall:
            mock_uninstall.return_value = True

            result = uninstall_completion_main(args)

            assert result == 0
            mock_uninstall.assert_called_once_with(
                shell_type="bash",
                cleanup_session=True,
            )

    def test_install_completion_main_simplified_args(self):
        """Test install_completion_main handles simplified arguments correctly."""
        from xraylabtool.completion_installer import install_completion_main

        class MockArgs:
            test = False
            shell = None

        args = MockArgs()

        with patch(
            "xraylabtool.completion_installer.CompletionInstaller.install_completion"
        ) as mock_install:
            mock_install.return_value = True

            result = install_completion_main(args)

            assert result == 0
            mock_install.assert_called_once_with(shell_type=None)

    def test_uninstall_completion_main_default_args(self):
        """Test uninstall_completion_main handles default arguments correctly."""
        from xraylabtool.completion_installer import uninstall_completion_main

        class MockArgs:
            shell = None

        args = MockArgs()

        with patch(
            "xraylabtool.completion_installer.CompletionInstaller.uninstall_completion"
        ) as mock_uninstall:
            mock_uninstall.return_value = True

            result = uninstall_completion_main(args)

            assert result == 0
            mock_uninstall.assert_called_once_with(
                shell_type=None,
                cleanup_session=True,
            )


class TestSimplifiedCompletionHelpIntegration:
    """Test simplified completion help text."""

    def test_install_completion_help_simplified(self):
        """Test that install-completion help shows simplified options."""
        with patch("sys.argv", ["xraylabtool", "install-completion", "--help"]):
            with patch("sys.stdout") as mock_stdout:
                with pytest.raises(SystemExit):
                    main()

                # Check help output contains simplified options
                stdout_calls = [
                    call.args[0]
                    for call in mock_stdout.write.call_args_list
                    if call.args
                ]
                help_text = "".join(stdout_calls)

                # Should contain basic options
                assert "--test" in help_text
                assert "virtual environment" in help_text

                # Should NOT contain obsolete options
                obsolete_options = [
                    "--venv",
                    "--conda",
                    "--all-environments",
                    "--no-cleanup-session",
                    "--system",
                    "--user",
                    "--uninstall",
                ]
                for option in obsolete_options:
                    assert (
                        option not in help_text
                    ), f"Obsolete option '{option}' found in help text"

    def test_uninstall_completion_help_exists(self):
        """Test that uninstall-completion help is available."""
        with patch("sys.argv", ["xraylabtool", "uninstall-completion", "--help"]):
            with patch("sys.stdout") as mock_stdout:
                with pytest.raises(SystemExit):
                    main()

                stdout_calls = [
                    call.args[0]
                    for call in mock_stdout.write.call_args_list
                    if call.args
                ]
                help_text = "".join(stdout_calls)

                # Should contain uninstall-specific content
                assert (
                    "Remove shell completion" in help_text
                    or "uninstall" in help_text.lower()
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
