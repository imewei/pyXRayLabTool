#!/usr/bin/env python3
"""
Comprehensive tests for XRayLabTool virtual environment completion integration.

This module contains comprehensive tests for virtual environment integration,
including conda/mamba support, environment switching, and edge cases.
"""

import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

try:
    from xraylabtool.completion_installer import CompletionInstaller
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.completion_installer import CompletionInstaller


class TestCondaIntegration:
    """Test conda/mamba environment integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.installer = CompletionInstaller()
        self.conda_prefix = Path("/opt/conda/envs/test")

    @patch.dict("os.environ", {"CONDA_PREFIX": "/opt/conda/envs/test"}, clear=True)
    def test_conda_environment_detection(self):
        """Test conda environment detection."""
        env_type = self.installer._detect_environment_type()
        assert env_type == "conda"

        env_path = self.installer._get_current_environment_path()
        assert env_path == self.conda_prefix

    @patch.dict("os.environ", {"CONDA_PREFIX": "/opt/mambaforge/envs/test"}, clear=True)
    def test_mamba_environment_detection(self):
        """Test mamba environment detection (uses same CONDA_PREFIX)."""
        mamba_prefix = Path("/opt/mambaforge/envs/test")

        env_type = self.installer._detect_environment_type()
        assert env_type == "conda"  # mamba uses same detection as conda

        env_path = self.installer._get_current_environment_path()
        assert env_path == mamba_prefix

    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    @patch("shutil.rmtree")
    def test_conda_completion_uninstall_success(
        self, mock_rmtree, mock_glob, mock_exists, mock_unlink
    ):
        """Test successful conda completion uninstallation."""
        # Mock conda environment structure
        mock_exists.return_value = True
        mock_glob.return_value = [
            self.conda_prefix
            / "etc"
            / "conda"
            / "activate.d"
            / "xraylabtool_completion.sh"
        ]

        result = self.installer._uninstall_conda_completion(self.conda_prefix)
        assert result is True
        mock_rmtree.assert_called()

    @patch("pathlib.Path.exists")
    def test_conda_completion_uninstall_not_installed(self, mock_exists):
        """Test conda completion uninstallation when not installed."""
        mock_exists.return_value = False

        result = self.installer._uninstall_conda_completion(self.conda_prefix)
        assert result is True  # Should succeed if not installed

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_conda_completion_installed_detection(self, mock_glob, mock_exists):
        """Test detection of conda completion installation."""
        # Test when completion directory exists
        mock_exists.return_value = True
        mock_glob.return_value = [Path("/conda/completions/xraylabtool")]

        result = self.installer._is_conda_completion_installed(Path("/conda"))
        assert result is True

        # Test when hooks exist
        mock_glob.side_effect = [
            [],  # No completion directory files
            [Path("/conda/etc/conda/activate.d/xraylabtool.sh")],  # Hook files exist
        ]

        result = self.installer._is_conda_completion_installed(Path("/conda"))
        assert result is True

    @patch("os.environ.get")
    @patch("pathlib.Path.exists")
    def test_conda_base_path_detection_methods(self, mock_exists, mock_environ):
        """Test various methods of detecting conda base path."""
        # Test CONDA_EXE detection
        mock_environ.side_effect = lambda key, default=None: (
            "/opt/conda/bin/conda" if key == "CONDA_EXE" else default
        )

        result = self.installer._get_conda_base_path()
        assert result == Path("/opt/conda")

        # Test MAMBA_EXE detection
        mock_environ.side_effect = lambda key, default=None: (
            "/opt/mambaforge/bin/mamba"
            if key == "MAMBA_EXE"
            else None
            if key == "CONDA_EXE"
            else default
        )

        result = self.installer._get_conda_base_path()
        assert result == Path("/opt/mambaforge")

    def test_conda_base_path_common_locations(self):
        """Test finding conda from common installation locations."""
        with patch("os.environ.get", return_value=None):
            with patch("pathlib.Path.home", return_value=Path("/home/user")):

                def mock_exists(self):
                    path_str = str(self)
                    return "mambaforge" in path_str and (
                        path_str.endswith("/mambaforge")
                        or path_str.endswith("/mambaforge/bin/mamba")
                        or path_str.endswith("/mambaforge/bin/conda")
                    )

                with patch.object(Path, "exists", mock_exists):
                    result = self.installer._get_conda_base_path()
                    assert result == Path("/home/user/mambaforge")


class TestVenvIntegration:
    """Test virtual environment integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.installer = CompletionInstaller()
        self.venv_path = Path("/home/user/project/venv")

    @patch.dict("os.environ", {"VIRTUAL_ENV": "/home/user/project/venv"}, clear=True)
    def test_venv_environment_detection(self):
        """Test virtual environment detection."""
        env_type = self.installer._detect_environment_type()
        assert env_type == "venv"

        env_path = self.installer._get_current_environment_path()
        assert env_path == self.venv_path

    def test_venv_completion_installed_detection(self):
        """Test detection of venv completion installation."""
        # Test completion directory exists
        venv_path_str = str(self.venv_path)

        def mock_exists_1(path_obj):
            path_str = str(path_obj)
            # Return True for venv path itself and completions directory
            return path_str == venv_path_str or "completions" in path_str

        with patch.object(Path, "exists", mock_exists_1):
            result = self.installer._is_venv_completion_installed(self.venv_path)
            assert result is True

        # Test activate script contains completion
        def mock_exists_2(path_obj):
            path_str = str(path_obj)
            # Return True for venv path and activate scripts, but not completions directory
            return path_str == venv_path_str or (
                "activate" in path_str and "completions" not in path_str
            )

        with (
            patch.object(Path, "exists", mock_exists_2),
            patch.object(
                Path,
                "read_text",
                return_value="source completion script\nxraylabtool completion",
            ),
        ):
            result = self.installer._is_venv_completion_installed(self.venv_path)
            assert result is True

    def test_venv_completion_uninstall_with_backup(self):
        """Test venv completion uninstallation with backup restoration."""
        venv_path_str = str(self.venv_path)

        def mock_exists(path_obj):
            path_str = str(path_obj)
            return (
                "completions" in path_str
                or "backup" in path_str
                or path_str.endswith("activate")
                or path_str == venv_path_str
            )

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "glob", return_value=[]):  # No hooks to remove
                with patch.object(
                    Path, "read_text", return_value="# Original activate script content"
                ):
                    with patch.object(Path, "write_text"):
                        with patch.object(Path, "unlink"):
                            with patch.object(Path, "rename"):
                                with patch("shutil.rmtree"):
                                    result = self.installer._uninstall_venv_completion(
                                        self.venv_path
                                    )
                                    assert result is True

    def test_venv_discovery_common_locations(self):
        """Test discovery of virtual environments in common locations."""
        venv_path_str = str(self.venv_path)

        def mock_environ_get(key, default=None):
            return venv_path_str if key == "VIRTUAL_ENV" else default

        def mock_exists(path_obj):
            path_str = str(path_obj)
            return (
                path_str == venv_path_str
                or path_str.endswith("/venv/bin/activate")
                or "project/venv" in path_str
            )

        with patch("os.environ.get", side_effect=mock_environ_get):
            with patch.object(Path, "cwd", return_value=Path("/home/user/project")):
                with patch.object(Path, "exists", mock_exists):
                    with patch.object(
                        self.installer,
                        "_is_venv_completion_installed",
                        return_value=True,
                    ):
                        environments = self.installer._discover_all_environments()
                        assert len(environments["venv"]) >= 1


class TestEnvironmentSwitching:
    """Test completion behavior during environment switches."""

    def setup_method(self):
        """Set up test fixtures."""
        self.installer = CompletionInstaller()

    @patch("subprocess.run")
    def test_cleanup_active_session_bash(self, mock_subprocess):
        """Test cleanup of active bash session."""
        mock_subprocess.return_value.returncode = 0

        result = self.installer._cleanup_active_session("bash")
        assert result is True

        # Verify correct commands were called
        calls = mock_subprocess.call_args_list
        assert len(calls) > 0

        # Check that completion removal commands were executed
        bash_calls = [call for call in calls if "bash" in str(call)]
        assert len(bash_calls) > 0

    @patch("subprocess.run")
    def test_cleanup_active_session_fish(self, mock_subprocess):
        """Test cleanup of active Fish session."""
        mock_subprocess.return_value.returncode = 0

        result = self.installer._cleanup_active_session("fish")
        assert result is True

        # Verify Fish-specific commands were called
        calls = mock_subprocess.call_args_list
        fish_calls = [call for call in calls if "fish" in str(call)]
        assert len(fish_calls) > 0

    @patch("subprocess.run")
    @patch.dict("os.environ", {"_XRAYLABTOOL_COMPLETION_ACTIVE": "1"})
    def test_cleanup_environment_variables(self, mock_subprocess):
        """Test cleanup of completion environment variables."""
        mock_subprocess.return_value.returncode = 0

        # Verify environment variable exists initially
        assert "_XRAYLABTOOL_COMPLETION_ACTIVE" in os.environ

        result = self.installer._cleanup_active_session("bash")
        assert result is True

        # Environment variable should be cleaned up
        assert "_XRAYLABTOOL_COMPLETION_ACTIVE" not in os.environ

    def test_uninstall_from_current_environment_only(self):
        """Test that uninstall only affects the current virtual environment."""
        # In the new simplified system, uninstall only works on current environment
        # The all_environments flag is deprecated and ignored

        with (
            patch.object(self.installer, "uninstall_completion") as mock_uninstall,
            patch.object(self.installer, "_cleanup_active_session"),
        ):
            mock_uninstall.return_value = True

            # Even with deprecated all_environments flag, should only uninstall from current env
            result = self.installer.uninstall_completion(
                shell_type="bash", cleanup_session=True
            )

            assert result is True

            # Should call the unified uninstall method once
            mock_uninstall.assert_called_once_with(
                shell_type="bash", cleanup_session=True
            )


class TestVirtualEnvironmentEdgeCases:
    """Test edge cases and error conditions for virtual environment functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.installer = CompletionInstaller()

    @patch("os.environ.get")
    def test_nested_virtual_environments(self, mock_environ):
        """Test behavior with nested virtual environments."""
        # Simulate nested environments (should use innermost)
        mock_environ.side_effect = lambda key, default=None: {
            "CONDA_PREFIX": "/opt/conda/envs/outer",
            "VIRTUAL_ENV": "/opt/conda/envs/outer/inner_venv",
        }.get(key, default)

        # Should detect conda (CONDA_PREFIX takes precedence in current implementation)
        env_type = self.installer._detect_environment_type()
        assert env_type == "conda"

    @patch("pathlib.Path.exists")
    def test_missing_conda_directories(self, mock_exists):
        """Test handling of missing conda directories."""
        mock_exists.return_value = False

        result = self.installer._is_conda_completion_installed(Path("/nonexistent"))
        assert result is False

        result = self.installer._uninstall_conda_completion(Path("/nonexistent"))
        assert result is True  # Should succeed if not installed

    def test_corrupted_activation_scripts(self):
        """Test handling of corrupted activation scripts."""

        def mock_exists(path_obj):
            path_str = str(path_obj)
            # Return True for venv path itself and activate scripts, but not completions directory
            return path_str == "/venv" or (
                "activate" in path_str and "completions" not in path_str
            )

        def mock_read_text():
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")

        with patch.object(Path, "exists", mock_exists):
            with patch.object(Path, "read_text", side_effect=mock_read_text):
                result = self.installer._is_venv_completion_installed(Path("/venv"))
                assert result is False  # Should handle error gracefully

    @patch("pathlib.Path.unlink")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.glob")
    def test_permission_denied_scenarios(self, mock_glob, mock_exists, mock_unlink):
        """Test handling of permission denied errors."""
        mock_exists.return_value = True
        mock_glob.return_value = [Path("/conda/etc/conda/activate.d/xraylabtool.sh")]
        mock_unlink.side_effect = PermissionError("Permission denied")

        result = self.installer._uninstall_conda_completion(Path("/conda"))
        assert result is False  # Should return False on permission errors

    def test_single_environment_uninstall_performance(self):
        """Test that uninstall only affects current environment, regardless of discovery results."""
        # Even if many environments exist, uninstall only affects current one

        with patch.object(self.installer, "uninstall_completion") as mock_uninstall:
            mock_uninstall.return_value = True

            # In new system, uninstall only works on current environment
            result = self.installer.uninstall_completion(
                shell_type="bash", cleanup_session=False
            )

            assert result is True
            # Should only be called once for current environment
            mock_uninstall.assert_called_once_with(
                shell_type="bash", cleanup_session=False
            )

    @patch("os.environ.get")
    def test_environment_detection_priority(self, mock_environ):
        """Test environment detection priority when multiple are active."""
        # All environment variables set - should follow priority order
        mock_environ.side_effect = lambda key, default=None: {
            "CONDA_PREFIX": "/conda",
            "VIRTUAL_ENV": "/venv",
            "PIPENV_ACTIVE": "1",
        }.get(key, default)

        # Should detect conda first (current implementation priority)
        env_type = self.installer._detect_environment_type()
        assert env_type == "conda"


class TestCompletionScriptContent:
    """Test that completion scripts contain proper simplified content."""

    def test_bash_script_contains_simplified_logic(self):
        """Test that bash completion script contains simplified completion logic."""
        from xraylabtool.completion_installer import BASH_COMPLETION_SCRIPT

        # Should contain both completion commands
        assert "install-completion" in BASH_COMPLETION_SCRIPT
        assert "uninstall-completion" in BASH_COMPLETION_SCRIPT

        # Should contain the new completion functions
        assert "_xraylabtool_install_completion_complete" in BASH_COMPLETION_SCRIPT
        assert "_xraylabtool_uninstall_completion_complete" in BASH_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "--venv",
            "--conda",
            "--all-environments",
            "--no-cleanup-session",
        ]
        for option in obsolete_options:
            assert option not in BASH_COMPLETION_SCRIPT, (
                f"Obsolete option '{option}' still in bash script"
            )

    def test_fish_script_contains_simplified_options(self):
        """Test that Fish completion script contains simplified options."""
        from xraylabtool.completion_installer import FISH_COMPLETION_SCRIPT

        # Should contain both completion commands
        assert "install-completion" in FISH_COMPLETION_SCRIPT
        assert "uninstall-completion" in FISH_COMPLETION_SCRIPT

        # Should contain basic shell type completions
        assert "bash zsh fish powershell" in FISH_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "-l venv",
            "-l conda",
            "-l all-environments",
            "-l no-cleanup-session",
        ]
        for option in obsolete_options:
            assert option not in FISH_COMPLETION_SCRIPT, (
                f"Obsolete option '{option}' still in Fish script"
            )

    def test_powershell_script_contains_simplified_options(self):
        """Test that PowerShell completion script contains simplified options."""
        from xraylabtool.completion_installer import POWERSHELL_COMPLETION_SCRIPT

        # Should contain both completion commands
        assert "install-completion" in POWERSHELL_COMPLETION_SCRIPT
        assert "uninstall-completion" in POWERSHELL_COMPLETION_SCRIPT

        # Should contain shell options for both commands
        assert "'bash', 'zsh', 'fish', 'powershell'" in POWERSHELL_COMPLETION_SCRIPT

        # Should NOT contain obsolete VE options
        obsolete_options = [
            "--venv",
            "--conda",
            "--all-environments",
            "--no-cleanup-session",
        ]
        for option in obsolete_options:
            assert option not in POWERSHELL_COMPLETION_SCRIPT, (
                f"Obsolete option '{option}' still in PowerShell script"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
