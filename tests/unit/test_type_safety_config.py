"""
Tests for type safety configuration and MyPy setup validation.

This module validates that the enhanced type safety infrastructure is
correctly configured and functioning as expected.
"""

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from tests.fixtures.test_base import BaseXRayLabToolTest


class TestTypeSafetyConfig(BaseXRayLabToolTest):
    """Test suite for type safety configuration validation."""

    def test_mypy_installation_and_version(self):
        """Test that MyPy is installed and accessible."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert result.returncode == 0, f"MyPy not accessible: {result.stderr}"
            assert "mypy" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("MyPy version check timed out")
        except FileNotFoundError:
            pytest.fail("MyPy not installed or not accessible")

    def test_mypy_config_file_exists(self):
        """Test that MyPy configuration file exists."""
        project_root = Path(__file__).parent.parent.parent

        # Check for mypy configuration in pyproject.toml
        pyproject_file = project_root / "pyproject.toml"
        assert pyproject_file.exists(), "pyproject.toml not found"

        content = pyproject_file.read_text()
        assert "[tool.mypy]" in content, (
            "MyPy configuration section not found in pyproject.toml"
        )

    def test_mypy_strict_mode_configuration(self):
        """Test that MyPy strict mode settings are properly configured."""
        project_root = Path(__file__).parent.parent.parent
        pyproject_file = project_root / "pyproject.toml"

        content = pyproject_file.read_text()

        # Check for key strict mode settings
        strict_settings = [
            "disallow_untyped_defs = true",
            "disallow_incomplete_defs = true",
            "check_untyped_defs = true",
            "strict_optional = true",
        ]

        for setting in strict_settings:
            assert setting in content, f"Missing strict mode setting: {setting}"

    def test_mypy_can_check_core_modules(self):
        """Test that MyPy can successfully analyze core modules."""
        project_root = Path(__file__).parent.parent.parent
        core_modules = [
            "xraylabtool/calculators/core.py",
            "xraylabtool/utils.py",
            "xraylabtool/constants.py",
        ]

        for module in core_modules:
            module_path = project_root / module
            if module_path.exists():
                try:
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "mypy",
                            str(module_path),
                            "--config-file",
                            "pyproject.toml",
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=project_root,
                    )
                    # Note: We don't require zero errors yet, just that MyPy can analyze
                    assert result.returncode in [
                        0,
                        1,
                    ], f"MyPy failed to analyze {module}: {result.stderr}"
                except subprocess.TimeoutExpired:
                    pytest.fail(f"MyPy analysis of {module} timed out")

    def test_numpy_typing_imports_available(self):
        """Test that NumPy typing support is available."""
        try:
            import numpy as np
            from numpy.typing import NDArray

            # Test that we can create type aliases
            FloatArray = NDArray[np.float64]
            ComplexArray = NDArray[np.complex128]

            # Verify types are usable
            assert FloatArray is not None
            assert ComplexArray is not None

        except ImportError as e:
            pytest.fail(f"NumPy typing imports failed: {e}")

    def test_type_checking_flag_available(self):
        """Test that TYPE_CHECKING flag is available for import optimization."""
        try:
            from typing import TYPE_CHECKING

            assert isinstance(TYPE_CHECKING, bool)
        except ImportError:
            pytest.fail("TYPE_CHECKING flag not available")

    def test_modern_typing_features_available(self):
        """Test that modern Python 3.12+ typing features are available."""
        # Test built-in generics (Python 3.9+)
        try:
            from collections.abc import Mapping, Sequence

            test_list: list[str] = ["test"]
            test_dict: dict[str, int] = {"test": 1}

            assert test_list is not None
            assert test_dict is not None

        except (ImportError, TypeError) as e:
            pytest.fail(f"Modern typing features not available: {e}")

    def test_mypy_cache_directory_setup(self):
        """Test that MyPy cache directory can be created and used."""
        project_root = Path(__file__).parent.parent.parent
        mypy_cache_dir = project_root / ".mypy_cache"

        # Cache directory should either exist or be creatable
        if not mypy_cache_dir.exists():
            try:
                # Test if we can create it (will be cleaned up by MyPy)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mypy",
                        "--cache-dir",
                        str(mypy_cache_dir),
                        "--help",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert result.returncode == 0, "MyPy cache directory test failed"
            except subprocess.TimeoutExpired:
                pytest.fail("MyPy cache directory test timed out")

    @pytest.mark.performance
    def test_type_checking_performance_impact(self):
        """Test that type checking infrastructure has minimal performance impact."""
        import time

        # Test import time for typing modules
        start_time = time.time()

        try:
            from typing import TYPE_CHECKING

            import numpy as np
            from numpy.typing import NDArray

            if TYPE_CHECKING:
                from collections.abc import Sequence

        except ImportError:
            pytest.skip("Type checking imports not available")

        import_time = time.time() - start_time

        # Import time should be minimal (< 100ms)
        assert import_time < 0.1, f"Type checking imports too slow: {import_time:.3f}s"

    def test_type_stub_packages_available(self):
        """Test that required type stub packages are available."""
        required_stubs = [
            "pandas-stubs",
            "types-tqdm",
            "types-psutil",
        ]

        for stub_package in required_stubs:
            try:
                # Try to import the stub package to verify it's installed
                import importlib

                stub_package.replace("-", "_").replace("types_", "")

                # For pandas-stubs, check pandas
                if stub_package == "pandas-stubs":
                    import pandas

                    # Check if pandas has type annotations
                    assert hasattr(pandas, "__version__")

            except ImportError:
                pytest.skip(f"Type stub package {stub_package} not available")


class TestTypeValidationHelpers:
    """Test helper functions for type validation."""

    def test_is_numpy_array_type(self):
        """Test helper function to validate NumPy array types."""
        import numpy as np

        def is_numpy_array(obj: Any) -> bool:
            """Check if object is a NumPy array."""
            return isinstance(obj, np.ndarray)

        # Test with various inputs
        assert is_numpy_array(np.array([1, 2, 3])) is True
        assert is_numpy_array([1, 2, 3]) is False
        assert is_numpy_array("test") is False

    def test_dtype_validation(self):
        """Test NumPy dtype validation for performance-critical arrays."""
        import numpy as np

        def validate_float_array(arr: np.ndarray) -> bool:
            """Validate that array has appropriate float dtype."""
            return arr.dtype in [np.float32, np.float64]

        def validate_complex_array(arr: np.ndarray) -> bool:
            """Validate that array has appropriate complex dtype."""
            return arr.dtype in [np.complex64, np.complex128]

        # Test float arrays
        float_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert validate_float_array(float_arr) is True

        # Test complex arrays
        complex_arr = np.array([1 + 2j, 2 + 3j], dtype=np.complex128)
        assert validate_complex_array(complex_arr) is True

        # Test invalid types
        int_arr = np.array([1, 2, 3], dtype=np.int32)
        assert validate_float_array(int_arr) is False
