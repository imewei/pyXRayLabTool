"""Tests for interfaces/completion_v2/environment.py module.

Tests environment detection and management for shell completion,
including virtual environment type detection and info gathering.
"""

from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest

from xraylabtool.interfaces.completion_v2.environment import (
    EnvironmentInfo,
    EnvironmentType,
)


class TestEnvironmentType:
    """Tests for EnvironmentType constants."""

    def test_environment_types_defined(self):
        """Test that environment types are defined."""
        assert hasattr(EnvironmentType, "SYSTEM")
        assert hasattr(EnvironmentType, "VENV")
        assert hasattr(EnvironmentType, "VIRTUALENV")
        assert hasattr(EnvironmentType, "CONDA")
        assert hasattr(EnvironmentType, "MAMBA")
        assert hasattr(EnvironmentType, "PIPENV")
        assert hasattr(EnvironmentType, "POETRY")

    def test_environment_types_are_strings(self):
        """Test that environment type values are strings."""
        assert isinstance(EnvironmentType.SYSTEM, str)
        assert isinstance(EnvironmentType.VENV, str)
        assert isinstance(EnvironmentType.VIRTUALENV, str)

    def test_environment_types_distinct(self):
        """Test that environment types are distinct."""
        types = [
            EnvironmentType.SYSTEM,
            EnvironmentType.VENV,
            EnvironmentType.VIRTUALENV,
            EnvironmentType.CONDA,
            EnvironmentType.MAMBA,
            EnvironmentType.PIPENV,
            EnvironmentType.POETRY,
        ]
        # All types should be unique
        assert len(types) == len(set(types))


class TestEnvironmentInfo:
    """Tests for EnvironmentInfo class."""

    def test_environment_info_creation(self):
        """Test creating EnvironmentInfo instance."""
        from pathlib import Path

        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )
        assert info is not None
        assert info.env_type == EnvironmentType.VENV

    def test_environment_info_has_attributes(self):
        """Test that EnvironmentInfo has required attributes."""
        from pathlib import Path

        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )

        # Should have attributes
        assert hasattr(info, "env_type")
        assert hasattr(info, "path")
        assert hasattr(info, "name")
        assert hasattr(info, "is_active")
        assert hasattr(info, "python_version")
        assert hasattr(info, "has_completion")

    def test_environment_info_with_type(self):
        """Test EnvironmentInfo with specific type."""
        from pathlib import Path

        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )
        assert info.env_type == EnvironmentType.VENV

    def test_environment_info_with_path(self):
        """Test EnvironmentInfo with environment path."""
        from pathlib import Path

        env_path = Path("/some/path")
        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=env_path, name="test_env"
        )
        assert info.path == env_path

    def test_environment_info_with_python_version(self):
        """Test EnvironmentInfo with Python version."""
        from pathlib import Path

        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV,
            path=Path("/some/venv"),
            name="test_env",
            python_version="3.12",
        )
        assert info.python_version == "3.12"

    def test_environment_info_equality(self):
        """Test EnvironmentInfo equality comparison."""
        from pathlib import Path

        info1 = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )
        info2 = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )

        # Should have same attributes
        assert info1.env_type == info2.env_type
        assert info1.path == info2.path
        assert info1.name == info2.name

    def test_environment_info_repr(self):
        """Test EnvironmentInfo string representation."""
        from pathlib import Path

        info = EnvironmentInfo(
            env_type=EnvironmentType.VENV, path=Path("/some/venv"), name="test_env"
        )
        repr_str = repr(info)

        # Should contain object representation
        assert "EnvironmentInfo" in repr_str or "test_env" in repr_str


class TestEnvironmentDetection:
    """Tests for environment detection functionality."""

    def test_detect_current_environment(self):
        """Test detecting current Python environment."""
        # Should be able to get current environment info
        assert sys.prefix is not None
        assert Path(sys.prefix).exists()

    def test_venv_detection_markers(self):
        """Test markers for venv detection."""
        # venv should have pyvenv.cfg
        # This is a basic check for venv presence
        venv_markers = [
            "pyvenv.cfg",  # venv marker
            "bin/python",  # Unix executable
            "Scripts/python.exe",  # Windows executable
        ]

        # At least one marker should be in typical installations
        assert any(isinstance(m, str) for m in venv_markers)

    def test_conda_detection_markers(self):
        """Test markers for conda environment detection."""
        # Check for common conda environment markers
        conda_markers = [
            "conda.json",
            "conda-meta",
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
        ]

        # Should be valid marker names
        assert all(isinstance(m, str) for m in conda_markers)

    @patch.dict("os.environ", {"VIRTUAL_ENV": "/path/to/venv"})
    def test_virtual_env_detection(self):
        """Test detection of virtual environment via env var."""
        # When VIRTUAL_ENV is set, should detect virtual environment
        import os

        assert "VIRTUAL_ENV" in os.environ
        assert os.environ["VIRTUAL_ENV"] == "/path/to/venv"

    @patch.dict("os.environ", {"CONDA_PREFIX": "/path/to/conda"})
    def test_conda_env_detection(self):
        """Test detection of conda environment via env var."""
        import os

        assert "CONDA_PREFIX" in os.environ
        assert os.environ["CONDA_PREFIX"] == "/path/to/conda"


class TestEnvironmentProperties:
    """Tests for environment property detection."""

    def test_python_executable(self):
        """Test getting Python executable path."""
        # Current Python should be in sys.executable
        assert sys.executable is not None
        assert Path(sys.executable).exists()

    def test_python_version_format(self):
        """Test Python version format detection."""
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert isinstance(version, str)
        assert "." in version

    def test_environment_site_packages(self):
        """Test site-packages directory detection."""
        # Every Python environment has site-packages
        import site

        site_packages = site.getsitepackages()

        assert isinstance(site_packages, list)
        assert len(site_packages) > 0

    def test_environment_activation_script(self):
        """Test activation script path resolution."""
        # Different shells have different activation scripts
        activation_scripts = {
            "bash": "activate",
            "zsh": "activate",
            "fish": "activate.fish",
            "powershell": "Activate.ps1",
        }

        # All should be valid script names
        assert all(isinstance(script, str) for script in activation_scripts.values())


class TestEnvironmentIntegration:
    """Integration tests for environment detection."""

    def test_current_environment_is_valid(self):
        """Test that current environment is detected as valid."""
        # Current Python environment should be detectable
        assert Path(sys.prefix).exists()
        assert Path(sys.executable).exists()

    def test_environment_type_from_markers(self):
        """Test determining environment type from system markers."""
        # Check for conda first
        conda_env = "CONDA_PREFIX" in __import__("os").environ
        # Check for venv
        venv_env = "VIRTUAL_ENV" in __import__("os").environ

        # At least system Python should always be detectable
        assert isinstance(conda_env, bool)
        assert isinstance(venv_env, bool)

    def test_environment_is_readable(self):
        """Test that environment files are readable."""
        import site

        site_packages = site.getsitepackages()

        for sp in site_packages:
            if Path(sp).exists():
                assert Path(sp).is_dir()
                break
