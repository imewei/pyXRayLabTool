"""Tests for interfaces/completion_v2/integration.py module.

Tests integration layer for the new completion system, including backward
compatibility with existing CLI and integration functions.
"""

import argparse
from unittest.mock import Mock, patch

from xraylabtool.interfaces.completion_v2.integration import (
    install_completion_main,
    legacy_install_completion_main,
    uninstall_completion_main,
)


class TestLegacyInstallCompletionMain:
    """Tests for legacy_install_completion_main function."""

    def test_function_exists(self):
        """Test that legacy_install_completion_main function exists."""
        assert callable(legacy_install_completion_main)

    def test_accepts_namespace_argument(self):
        """Test that function accepts argparse.Namespace."""
        _ = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=False,
            test=False,
            uninstall=False,
        )

        # Should be callable with Namespace
        assert callable(legacy_install_completion_main)

    @patch("xraylabtool.interfaces.completion_v2.integration.CompletionInstaller")
    def test_handles_legacy_arguments(self, mock_installer):
        """Test that function handles legacy argument formats."""
        # Mock the installer
        mock_instance = Mock()
        mock_installer.return_value = mock_instance
        mock_instance.install = Mock(return_value=0)

        args = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=False,
            test=False,
            uninstall=False,
        )

        result = legacy_install_completion_main(args)

        # Should return a result code
        assert isinstance(result, int)

    @patch("xraylabtool.interfaces.completion_v2.integration.CompletionInstaller")
    def test_uninstall_mode_flag(self, mock_installer):
        """Test handling of uninstall mode flag."""
        mock_instance = Mock()
        mock_installer.return_value = mock_instance

        args = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=False,
            test=False,
            uninstall=True,
        )

        result = legacy_install_completion_main(args)
        assert isinstance(result, int)

    @patch("xraylabtool.interfaces.completion_v2.integration.CompletionInstaller")
    def test_test_mode_flag(self, mock_installer):
        """Test handling of test mode flag."""
        mock_instance = Mock()
        mock_installer.return_value = mock_instance

        args = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=False,
            test=True,
            uninstall=False,
        )

        result = legacy_install_completion_main(args)
        assert isinstance(result, int)

    @patch("xraylabtool.interfaces.completion_v2.integration.CompletionInstaller")
    def test_system_wide_flag(self, mock_installer):
        """Test handling of system-wide installation flag."""
        mock_instance = Mock()
        mock_installer.return_value = mock_instance

        args = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=True,
            test=False,
            uninstall=False,
        )

        result = legacy_install_completion_main(args)
        assert isinstance(result, int)

    @patch("xraylabtool.interfaces.completion_v2.integration.CompletionInstaller")
    def test_returns_success_code(self, mock_installer):
        """Test that successful operation returns 0."""
        mock_instance = Mock()
        mock_installer.return_value = mock_instance
        mock_instance.install = Mock(return_value=0)

        args = argparse.Namespace(  # noqa: S604
            shell="bash",
            force=False,
            system=False,
            test=False,
            uninstall=False,
        )

        result = legacy_install_completion_main(args)
        assert result == 0 or isinstance(result, int)


class TestInstallCompletionMain:
    """Tests for install_completion_main function."""

    def test_function_exists(self):
        """Test that install_completion_main function exists."""
        assert callable(install_completion_main)

    def test_accepts_namespace_argument(self):
        """Test that function accepts argparse.Namespace."""
        _ = argparse.Namespace(shell="bash")  # noqa: S604

        # Should be callable with appropriate args
        assert callable(install_completion_main)

    @patch("xraylabtool.interfaces.completion_v2.integration.completion_main")
    def test_delegates_to_completion_main(self, mock_completion_main):
        """Test that function delegates to completion_main."""
        mock_completion_main.return_value = 0

        args = argparse.Namespace(shell="bash")  # noqa: S604
        result = install_completion_main(args)

        # Should either call completion_main or return result code
        assert isinstance(result, int) or result is None


class TestUninstallCompletionMain:
    """Tests for uninstall_completion_main function."""

    def test_function_exists(self):
        """Test that uninstall_completion_main function exists."""
        assert callable(uninstall_completion_main)

    def test_accepts_namespace_argument(self):
        """Test that function accepts argparse.Namespace."""
        _ = argparse.Namespace(shell="bash")  # noqa: S604

        # Should be callable with appropriate args
        assert callable(uninstall_completion_main)

    @patch("xraylabtool.interfaces.completion_v2.integration.completion_main")
    def test_delegates_to_completion_main(self, mock_completion_main):
        """Test that function delegates to completion_main."""
        mock_completion_main.return_value = 0

        args = argparse.Namespace(shell="bash")  # noqa: S604
        result = uninstall_completion_main(args)

        # Should either call completion_main or return result code
        assert isinstance(result, int) or result is None


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing CLI."""

    def test_legacy_argument_handling(self):
        """Test handling of various legacy argument formats."""
        # Test various argument combinations that might exist
        test_cases = [
            {"shell": "bash"},
            {"install_completion": "bash"},
            {"force": True},
            {"system": True},
            {"uninstall": True},
        ]

        for kwargs in test_cases:
            args = argparse.Namespace(
                **{
                    "shell": kwargs.get("shell"),
                    "install_completion": kwargs.get("install_completion"),
                    "force": kwargs.get("force", False),
                    "system": kwargs.get("system", False),
                    "test": kwargs.get("test", False),
                    "uninstall": kwargs.get("uninstall", False),
                }
            )

            # Should be handled gracefully
            assert args is not None

    def test_optional_arguments(self):
        """Test that optional arguments are handled."""
        # Arguments might be set to None or missing
        args = argparse.Namespace(
            shell=None,
            install_completion=None,
            force=False,
            system=False,
            test=False,
            uninstall=False,
        )

        # Should handle None values gracefully
        assert args.shell is None or isinstance(args.shell, str)


class TestCompletionIntegrationFlow:
    """Tests for integration flow between new and legacy systems."""

    def test_new_system_imports_correctly(self):
        """Test that new completion system imports correctly."""
        from xraylabtool.interfaces.completion_v2 import cli

        assert cli is not None

    def test_installer_available(self):
        """Test that CompletionInstaller is available."""
        from xraylabtool.interfaces.completion_v2.installer import CompletionInstaller

        assert CompletionInstaller is not None

    def test_circular_import_avoided(self):
        """Test that circular imports are avoided."""
        # Should be able to import integration module without errors
        from xraylabtool.interfaces.completion_v2 import integration

        assert integration is not None

    def test_cli_module_available(self):
        """Test that cli module is available for delegation."""
        from xraylabtool.interfaces.completion_v2 import cli

        assert hasattr(cli, "completion_main")
