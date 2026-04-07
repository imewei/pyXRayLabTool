"""Tests for GPU/device detection in xraylabtool/device.py."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# get_gpu_info
# ---------------------------------------------------------------------------


class TestGetGpuInfo:
    def test_returns_none_when_nvidia_smi_not_found(self):
        """Returns (None, None) gracefully when nvidia-smi is not on PATH."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            name, sm = device.get_gpu_info()
        assert name is None
        assert sm is None

    def test_returns_gpu_name_and_sm_version_on_success(self):
        """Parses GPU name and compute capability when nvidia-smi succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4090, 8.9\n"

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            from importlib import reload

            from xraylabtool import device as dev

            name, sm = dev.get_gpu_info()

        assert name == "NVIDIA GeForce RTX 4090"
        assert sm == pytest.approx(8.9)

    def test_returns_none_on_nonzero_returncode(self):
        """Returns (None, None) when nvidia-smi returns non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
        ):
            from xraylabtool import device

            name, sm = device.get_gpu_info()

        assert name is None
        assert sm is None

    def test_handles_timeout_gracefully(self):
        """Returns (None, None) if nvidia-smi times out."""
        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)
            ),
        ):
            from xraylabtool import device

            name, sm = device.get_gpu_info()

        assert name is None
        assert sm is None

    def test_returns_valid_type_regardless_of_hardware(self):
        """get_gpu_info always returns a 2-tuple of (str|None, float|None)."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            result = device.get_gpu_info()

        assert isinstance(result, tuple)
        assert len(result) == 2
        name, sm = result
        assert name is None or isinstance(name, str)
        assert sm is None or isinstance(sm, float)


# ---------------------------------------------------------------------------
# get_system_cuda_version
# ---------------------------------------------------------------------------


class TestGetSystemCudaVersion:
    def test_returns_none_when_nvcc_not_found(self):
        """Returns (None, None) gracefully when nvcc is not on PATH."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            version, major = device.get_system_cuda_version()
        assert version is None
        assert major is None

    def test_parses_cuda_version_on_success(self):
        """Parses CUDA version string from nvcc --version output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "nvcc: NVIDIA (R) Cuda compiler driver\n"
            "Copyright (c) 2005-2023 NVIDIA Corporation\n"
            "Built on Mon_Apr_3_17:16:06_PDT_2023\n"
            "Cuda compilation tools, release 12.1, V12.1.105\n"
        )

        with (
            patch("shutil.which", return_value="/usr/local/cuda/bin/nvcc"),
            patch("subprocess.run", return_value=mock_result),
        ):
            from xraylabtool import device

            version, major = device.get_system_cuda_version()

        assert version == "12.1"
        assert major == 12

    def test_returns_none_on_parse_failure(self):
        """Returns (None, None) when stdout contains no recognizable release line."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Unexpected output format\n"

        with (
            patch("shutil.which", return_value="/usr/bin/nvcc"),
            patch("subprocess.run", return_value=mock_result),
        ):
            from xraylabtool import device

            version, major = device.get_system_cuda_version()

        assert version is None
        assert major is None


# ---------------------------------------------------------------------------
# check_gpu_availability
# ---------------------------------------------------------------------------


class TestCheckGpuAvailability:
    def test_returns_false_when_no_gpu_hardware(self):
        """Returns False gracefully when no GPU hardware is detected."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            result = device.check_gpu_availability(warn=False)

        assert result is False

    def test_returns_false_when_jax_not_installed(self):
        """Returns False (not an exception) when JAX is not importable."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3080, 8.6\n"

        import sys

        with (
            patch("shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("subprocess.run", return_value=mock_result),
            patch.dict(sys.modules, {"jax": None}),
        ):
            from xraylabtool import device

            result = device.check_gpu_availability(warn=False)

        assert result is False

    def test_returns_bool(self):
        """check_gpu_availability always returns a bool."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            result = device.check_gpu_availability(warn=False)

        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# get_device_info
# ---------------------------------------------------------------------------


class TestGetDeviceInfo:
    def test_returns_dict_with_expected_keys(self):
        """get_device_info returns a dict containing all required keys."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            info = device.get_device_info()

        required_keys = {
            "jax_version",
            "jax_backend",
            "devices",
            "gpu_count",
            "using_gpu",
            "gpu_hardware",
            "gpu_sm_version",
            "system_cuda_version",
            "system_cuda_major",
            "recommended_package",
            "plugin_issues",
        }
        assert required_keys.issubset(info.keys())

    def test_no_gpu_produces_sensible_defaults(self):
        """When no GPU hardware is present, gpu_hardware and sm_version are None."""
        with patch("shutil.which", return_value=None):
            from xraylabtool import device

            info = device.get_device_info()

        assert info["gpu_hardware"] is None
        assert info["gpu_sm_version"] is None
        assert info["system_cuda_version"] is None
        assert info["system_cuda_major"] is None
        assert isinstance(info["plugin_issues"], list)
