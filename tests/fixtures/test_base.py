"""
Base test classes and utilities for xraylabtool tests.

This module provides base test classes, common test utilities, and
standardized test patterns for consistent testing across the codebase.
"""

import time

# Removed ABC import as base class doesn't need to be abstract
from typing import Any

import numpy as np
import pytest

from tests.fixtures.test_config import (
    NUMERICAL_TOLERANCES,
    PERFORMANCE_THRESHOLDS,
    REFERENCE_VALUES,
)


class BaseXRayLabToolTest:
    """
    Base class for all XRayLabTool tests.

    Provides common test utilities, assertion methods, and standardized
    test patterns for consistent behavior across the test suite.
    """

    @staticmethod
    def assert_arrays_close(
        actual: np.ndarray,
        expected: np.ndarray,
        tolerance: str = "default",
        rtol: float | None = None,
        atol: float | None = None,
    ) -> None:
        """
        Assert that two numpy arrays are close within specified tolerance.

        Args:
            actual: Actual array values
            expected: Expected array values
            tolerance: Tolerance key from NUMERICAL_TOLERANCES
            rtol: Relative tolerance (overrides tolerance if provided)
            atol: Absolute tolerance (overrides tolerance if provided)
        """
        if rtol is None and atol is None:
            tol = NUMERICAL_TOLERANCES[tolerance]
            rtol = tol
            atol = tol

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=f"Arrays not close within {tolerance} tolerance",
        )

    @staticmethod
    def assert_performance(
        execution_time_ms: float, threshold_key: str, operation_name: str = "operation"
    ) -> None:
        """
        Assert that operation performance meets threshold requirements.

        Args:
            execution_time_ms: Actual execution time in milliseconds
            threshold_key: Key from PERFORMANCE_THRESHOLDS
            operation_name: Name of operation being tested
        """
        threshold = PERFORMANCE_THRESHOLDS[threshold_key]
        assert execution_time_ms <= threshold, (
            f"{operation_name} took {execution_time_ms: .2f}ms, "
            f"exceeds threshold of {threshold}ms"
        )

    @staticmethod
    def time_operation(func, *args, **kwargs) -> tuple[Any, float]:
        """
        Time an operation and return result and execution time.

        Args:
            func: Function to time
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Tuple of (result, execution_time_ms)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        return result, execution_time_ms

    @staticmethod
    def validate_xray_result_structure(result: Any) -> None:
        """
        Validate that result has expected XRayResult structure.

        Args:
            result: Result object to validate
        """
        required_fields = [
            "formula",
            "molecular_weight_g_mol",
            "total_electrons",
            "density_g_cm3",
            "electron_density_per_ang3",
            "energy_kev",
            "wavelength_angstrom",
            "dispersion_delta",
            "absorption_beta",
            "scattering_factor_f1",
            "scattering_factor_f2",
            "critical_angle_degrees",
            "attenuation_length_cm",
            "real_sld_per_ang2",
            "imaginary_sld_per_ang2",
        ]

        for field in required_fields:
            assert hasattr(result, field), f"Missing required field: {field}"

        # Validate array fields are numpy arrays
        array_fields = [
            "energy_kev",
            "wavelength_angstrom",
            "dispersion_delta",
            "absorption_beta",
            "scattering_factor_f1",
            "scattering_factor_f2",
            "critical_angle_degrees",
            "attenuation_length_cm",
            "real_sld_per_ang2",
            "imaginary_sld_per_ang2",
        ]

        for field in array_fields:
            field_value = getattr(result, field)
            assert isinstance(field_value, np.ndarray), (
                f"Field {field} should be numpy array, got {type(field_value)}"
            )

    @staticmethod
    def validate_reference_values(
        result: Any, reference_key: str, tolerance: str = "default"
    ) -> None:
        """
        Validate result against reference values from Julia implementation.

        Args:
            result: XRayResult object to validate
            reference_key: Key from REFERENCE_VALUES
            tolerance: Tolerance for comparisons
        """
        if reference_key not in REFERENCE_VALUES:
            pytest.skip(f"No reference values for {reference_key}")

        ref_values = REFERENCE_VALUES[reference_key]
        tol = NUMERICAL_TOLERANCES[tolerance]

        for field_name, expected_value in ref_values.items():
            if hasattr(result, field_name):
                actual_value = getattr(result, field_name)

                if isinstance(actual_value, np.ndarray) and len(actual_value) > 0:
                    actual_value = actual_value[0]

                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=tol,
                    atol=tol,
                    err_msg=f"Reference validation failed for {field_name}",
                )


class BaseUnitTest(BaseXRayLabToolTest):
    """Base class for unit tests."""

    def setup_method(self) -> None:
        """Setup for each test method."""
        pass

    def teardown_method(self) -> None:
        """Teardown for each test method."""
        pass


class BaseIntegrationTest(BaseXRayLabToolTest):
    """Base class for integration tests."""

    def setup_method(self) -> None:
        """Setup for each integration test."""
        pass

    def teardown_method(self) -> None:
        """Teardown for each integration test."""
        pass


class BasePerformanceTest(BaseXRayLabToolTest):
    """Base class for performance tests."""

    def setup_method(self) -> None:
        """Setup for performance tests."""
        # Clear caches to ensure consistent performance measurements
        import gc

        from xraylabtool.calculators.core import clear_scattering_factor_cache

        clear_scattering_factor_cache()
        gc.collect()

    def teardown_method(self) -> None:
        """Teardown for performance tests."""
        pass


class ParametrizedTestMixin:
    """Mixin providing utilities for parametrized tests."""

    @staticmethod
    def create_material_params(materials: list[tuple]) -> list[pytest.param]:
        """Create pytest parameters for material testing."""
        return [
            pytest.param(formula, density, id=f"{formula}_{density}")
            for formula, density in materials
        ]

    @staticmethod
    def create_energy_params(energies: dict[str, np.ndarray]) -> list[pytest.param]:
        """Create pytest parameters for energy testing."""
        return [
            pytest.param(energy_array, id=energy_name)
            for energy_name, energy_array in energies.items()
        ]


# Utility functions for test data generation
def generate_test_materials(count: int = 5) -> list[tuple]:
    """Generate random test materials for stress testing."""
    import random

    from tests.test_config import TEST_MATERIALS

    return random.sample(TEST_MATERIALS, min(count, len(TEST_MATERIALS)))


def generate_test_energies(
    energy_range: tuple = (1.0, 30.0), num_points: int = 50, spacing: str = "linear"
) -> np.ndarray:
    """Generate test energy arrays."""
    if spacing == "linear":
        return np.linspace(energy_range[0], energy_range[1], num_points)
    elif spacing == "log":
        return np.logspace(
            np.log10(energy_range[0]), np.log10(energy_range[1]), num_points
        )
    else:
        raise ValueError(f"Unknown spacing: {spacing}")
