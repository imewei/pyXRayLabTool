"""
Test utilities and helper functions for xraylabtool tests.

This module provides utility functions, decorators, and classes
to support testing across the xraylabtool test suite.
"""

from collections.abc import Callable
from contextlib import contextmanager
import functools
import gc
import time
from typing import Any
import warnings

import numpy as np
import psutil
import pytest

from xraylabtool.calculators.core import clear_scattering_factor_cache


# Performance testing utilities
class PerformanceBenchmark:
    """Utility class for performance benchmarking."""

    def __init__(self, name: str):
        self.name = name
        self.measurements: list[float] = []
        self.start_time: float | None = None

    def start(self) -> None:
        """Start timing measurement."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop timing and record measurement."""
        if self.start_time is None:
            raise RuntimeError("Benchmark not started")

        elapsed = time.perf_counter() - self.start_time
        self.measurements.append(elapsed)
        self.start_time = None
        return elapsed

    @contextmanager
    def measure(self):
        """Context manager for measuring execution time."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    @property
    def average_ms(self) -> float:
        """Get average execution time in milliseconds."""
        if not self.measurements:
            return 0.0
        return (sum(self.measurements) / len(self.measurements)) * 1000

    @property
    def min_ms(self) -> float:
        """Get minimum execution time in milliseconds."""
        if not self.measurements:
            return 0.0
        return min(self.measurements) * 1000

    @property
    def max_ms(self) -> float:
        """Get maximum execution time in milliseconds."""
        if not self.measurements:
            return 0.0
        return max(self.measurements) * 1000

    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.start_time = None

    def assert_performance(self, threshold_ms: float, percentile: int = 95) -> None:
        """Assert that performance meets threshold."""
        if not self.measurements:
            raise RuntimeError("No measurements available")

        # Use percentile to handle outliers
        times_ms = [t * 1000 for t in self.measurements]
        p_value = np.percentile(times_ms, percentile)

        assert p_value <= threshold_ms, (
            f"{self.name} {percentile}th percentile: {p_value: .3f}ms "
            f"exceeds threshold: {threshold_ms: .3f}ms"
        )


class MemoryProfiler:
    """Utility class for memory profiling."""

    def __init__(self, name: str):
        self.name = name
        self.start_memory: int | None = None
        self.measurements: list[dict[str, float]] = []

    def start(self) -> None:
        """Start memory profiling."""
        gc.collect()  # Clean up before measurement
        self.start_memory = psutil.Process().memory_info().rss

    def stop(self) -> dict[str, float]:
        """Stop profiling and record measurement."""
        if self.start_memory is None:
            raise RuntimeError("Memory profiler not started")

        gc.collect()  # Clean up before final measurement
        end_memory = psutil.Process().memory_info().rss

        measurement = {
            "start_mb": self.start_memory / (1024 * 1024),
            "end_mb": end_memory / (1024 * 1024),
            "increase_mb": (end_memory - self.start_memory) / (1024 * 1024),
        }

        self.measurements.append(measurement)
        self.start_memory = None
        return measurement

    @contextmanager
    def profile(self):
        """Context manager for memory profiling."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    @property
    def average_increase_mb(self) -> float:
        """Get average memory increase in MB."""
        if not self.measurements:
            return 0.0
        increases = [m["increase_mb"] for m in self.measurements]
        return sum(increases) / len(increases)

    @property
    def max_increase_mb(self) -> float:
        """Get maximum memory increase in MB."""
        if not self.measurements:
            return 0.0
        increases = [m["increase_mb"] for m in self.measurements]
        return max(increases)

    def assert_memory_usage(self, threshold_mb: float) -> None:
        """Assert that memory usage is within threshold."""
        if not self.measurements:
            raise RuntimeError("No measurements available")

        max_increase = self.max_increase_mb
        assert max_increase <= threshold_mb, (
            f"{self.name} memory increase: {max_increase: .2f}MB "
            f"exceeds threshold: {threshold_mb: .2f}MB"
        )


# Test decorators
def performance_test(threshold_ms: float = 1000.0):
    """Decorator to mark and validate performance tests."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            benchmark = PerformanceBenchmark(func.__name__)
            with benchmark.measure():
                result = func(*args, **kwargs)
            benchmark.assert_performance(threshold_ms)
            return result

        # Add performance marker
        wrapper = pytest.mark.performance(wrapper)
        return wrapper

    return decorator


def memory_test(threshold_mb: float = 50.0):
    """Decorator to mark and validate memory tests."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = MemoryProfiler(func.__name__)
            with profiler.profile():
                result = func(*args, **kwargs)
            profiler.assert_memory_usage(threshold_mb)
            return result

        # Add memory marker
        wrapper = pytest.mark.memory(wrapper)
        return wrapper

    return decorator


def stability_test(func: Callable) -> Callable:
    """Decorator to mark numerical stability tests."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = func(*args, **kwargs)

            # Check for numerical warnings
            numerical_warnings = [
                warning
                for warning in w
                if any(
                    keyword in str(warning.message).lower()
                    for keyword in ["overflow", "underflow", "invalid", "divide"]
                )
            ]

            if numerical_warnings:
                pytest.fail(f"Numerical stability warnings: {numerical_warnings}")

            return result

    # Add stability marker
    wrapper = pytest.mark.stability(wrapper)
    return wrapper


def requires_memory(min_gb: float):
    """Decorator to skip tests if insufficient memory is available."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            available_gb = psutil.virtual_memory().available / (1024**3)
            if available_gb < min_gb:
                pytest.skip(
                    f"Requires {min_gb}GB memory, {available_gb: .1f}GB available"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def clean_cache_around(func: Callable) -> Callable:
    """Decorator to clean cache before and after test execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        clear_scattering_factor_cache()
        gc.collect()
        try:
            result = func(*args, **kwargs)
        finally:
            clear_scattering_factor_cache()
            gc.collect()
        return result

    return wrapper


# Data generation utilities
class TestDataFactory:
    """Factory for generating test data."""

    COMMON_MATERIALS = [
        ("SiO2", 2.2),
        ("Al2O3", 3.95),
        ("Fe2O3", 5.24),
        ("Si", 2.33),
        ("Au", 19.32),
        ("C", 2.267),
        ("TiO2", 4.23),
        ("ZnO", 5.61),
        ("Cu", 8.96),
        ("Pt", 21.45),
        ("W", 19.25),
        ("Mo", 10.28),
    ]

    @classmethod
    def get_materials(cls, count: int = None) -> list[tuple[str, float]]:
        """Get list of test materials."""
        if count is None:
            return cls.COMMON_MATERIALS.copy()
        return cls.COMMON_MATERIALS[:count]

    @classmethod
    def get_material_formulas(cls, count: int = None) -> list[str]:
        """Get list of material formulas."""
        materials = cls.get_materials(count)
        return [material[0] for material in materials]

    @classmethod
    def get_material_densities(cls, count: int = None) -> list[float]:
        """Get list of material densities."""
        materials = cls.get_materials(count)
        return [material[1] for material in materials]

    @staticmethod
    def linear_energies(
        start: float = 1.0, stop: float = 20.0, count: int = 50
    ) -> np.ndarray:
        """Generate linear energy array."""
        return np.linspace(start, stop, count)

    @staticmethod
    def logarithmic_energies(
        start: float = 0.1, stop: float = 30.0, count: int = 100
    ) -> np.ndarray:
        """Generate logarithmic energy array."""
        return np.logspace(np.log10(start), np.log10(stop), count)

    @staticmethod
    def random_energies(
        count: int, min_energy: float = 1.0, max_energy: float = 20.0
    ) -> np.ndarray:
        """Generate random energy array."""
        return np.random.uniform(min_energy, max_energy, count)

    @staticmethod
    def edge_case_energies() -> np.ndarray:
        """Generate energy values that test edge cases."""
        return np.array([0.03, 0.1, 1.0, 10.0, 20.0, 29.9, 30.0])


# Validation utilities
class ResultValidator:
    """Utility class for validating calculation results."""

    @staticmethod
    def validate_xray_result(
        result, formula: str, energies: np.ndarray, density: float
    ) -> None:
        """Validate XRayResult object."""
        assert result is not None, "Result should not be None"
        assert hasattr(result, "formula"), "Result should have formula attribute"
        assert result.formula == formula, (
            f"Formula mismatch: {result.formula} != {formula}"
        )

        # Validate arrays
        assert hasattr(result, "energy_kev"), "Result should have energy_kev"
        assert len(result.energy_kev) == len(energies), "Energy array length mismatch"

        # Validate critical angle
        assert hasattr(result, "critical_angle_degrees"), (
            "Result should have critical_angle_degrees"
        )
        assert len(result.critical_angle_degrees) == len(energies), (
            "Critical angle array length mismatch"
        )
        assert np.all(np.isfinite(result.critical_angle_degrees)), (
            "Critical angles should be finite"
        )
        assert np.all(result.critical_angle_degrees >= 0), (
            "Critical angles should be non-negative"
        )

        # Validate transmission
        assert hasattr(result, "transmission"), "Result should have transmission"
        assert len(result.transmission) == len(energies), (
            "Transmission array length mismatch"
        )
        assert np.all(np.isfinite(result.transmission)), (
            "Transmission values should be finite"
        )
        assert np.all(result.transmission >= 0), "Transmission should be non-negative"
        assert np.all(result.transmission <= 1), "Transmission should not exceed 1"

    @staticmethod
    def validate_batch_results(
        results: dict[str, Any], expected_formulas: list[str]
    ) -> None:
        """Validate batch calculation results."""
        assert isinstance(results, dict), "Results should be a dictionary"

        # Check that we have results for expected formulas
        for formula in expected_formulas:
            assert formula in results, f"Missing result for formula: {formula}"

        # Validate each result
        for formula, result in results.items():
            if result is not None:  # Some results might be None due to errors
                assert hasattr(result, "formula"), (
                    f"Result for {formula} should have formula"
                )
                assert result.formula == formula, f"Formula mismatch for {formula}"

    @staticmethod
    def validate_numerical_stability(
        values: np.ndarray, operation: str = "operation"
    ) -> None:
        """Validate numerical stability of calculated values."""
        # Check for NaN values
        nan_count = np.sum(np.isnan(values))
        assert nan_count == 0, f"{operation} produced {nan_count} NaN values"

        # Check for infinite values
        inf_count = np.sum(np.isinf(values))
        assert inf_count == 0, f"{operation} produced {inf_count} infinite values"

        # Check for unreasonably large values (could indicate overflow)
        large_count = np.sum(np.abs(values) > 1e10)
        if large_count > 0:
            warnings.warn(
                f"{operation} produced {large_count} very large values (>1e10)",
                stacklevel=2,
            )


# Comparison utilities
def arrays_close(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Check if arrays are close within tolerance."""
    return np.allclose(a, b, rtol=rtol, atol=atol)


def values_close(a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Check if values are close within tolerance."""
    return np.isclose(a, b, rtol=rtol, atol=atol)


def assert_arrays_close(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8, msg: str = ""
) -> None:
    """Assert that arrays are close within tolerance."""
    if not arrays_close(a, b, rtol, atol):
        max_diff = np.max(np.abs(a - b))
        pytest.fail(
            f"Arrays not close{': ' + msg if msg else ''}. Max difference: {max_diff}"
        )


def assert_values_close(
    a: float, b: float, rtol: float = 1e-5, atol: float = 1e-8, msg: str = ""
) -> None:
    """Assert that values are close within tolerance."""
    if not values_close(a, b, rtol, atol):
        diff = abs(a - b)
        pytest.fail(
            f"Values not close{': ' + msg if msg else ''}. "
            f"a={a}, b={b}, difference={diff}"
        )


# Performance regression testing
class RegressionTracker:
    """Track performance regressions across test runs."""

    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: dict[str, float] = {}
        self.load_baselines()

    def load_baselines(self) -> None:
        """Load performance baselines from file."""
        try:
            import json
            from pathlib import Path

            baseline_path = Path(self.baseline_file)
            if baseline_path.exists():
                with open(baseline_path) as f:
                    self.baselines = json.load(f)
        except Exception:
            # If loading fails, start with empty baselines
            self.baselines = {}

    def save_baselines(self) -> None:
        """Save performance baselines to file."""
        try:
            import json
            from pathlib import Path

            baseline_path = Path(self.baseline_file)
            baseline_path.parent.mkdir(parents=True, exist_ok=True)

            with open(baseline_path, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except Exception:
            # If saving fails, continue silently
            pass

    def check_regression(
        self, test_name: str, current_time: float, tolerance_percent: float = 10.0
    ) -> None:
        """Check for performance regression."""
        if test_name not in self.baselines:
            # First run - establish baseline
            self.baselines[test_name] = current_time
            self.save_baselines()
            return

        baseline = self.baselines[test_name]
        regression_threshold = baseline * (1 + tolerance_percent / 100)

        if current_time > regression_threshold:
            improvement_threshold = baseline * (1 - tolerance_percent / 100)
            if current_time < improvement_threshold:
                # Significant improvement - update baseline
                self.baselines[test_name] = current_time
                self.save_baselines()
            else:
                pytest.fail(
                    f"Performance regression in {test_name}: "
                    f"{current_time: .3f}s > {regression_threshold: .3f}s "
                    f"(baseline: {baseline: .3f}s, tolerance: {tolerance_percent}%)"
                )
        elif current_time < baseline * 0.8:  # 20% improvement
            # Significant improvement - update baseline
            self.baselines[test_name] = current_time
            self.save_baselines()


# Global regression tracker instance
regression_tracker = RegressionTracker()


# Batch testing utilities
def create_batch_test_data(
    material_count: int, energy_count: int
) -> tuple[list[str], np.ndarray, list[float]]:
    """Create batch test data with specified counts."""
    materials = TestDataFactory.get_materials(material_count)
    formulas = [m[0] for m in materials]
    densities = [m[1] for m in materials]
    energies = TestDataFactory.linear_energies(count=energy_count)
    return formulas, energies, densities


def validate_batch_performance(
    batch_size: int, total_time: float, max_time_per_item: float = 1.0
) -> None:
    """Validate batch calculation performance."""
    time_per_item = total_time / batch_size
    assert time_per_item <= max_time_per_item, (
        f"Batch performance too slow: {time_per_item: .3f}s per item "
        f"(threshold: {max_time_per_item: .3f}s)"
    )


# Error handling utilities
@contextmanager
def expect_no_warnings():
    """Context manager to ensure no warnings are raised."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list

        if warning_list:
            warning_messages = [str(w.message) for w in warning_list]
            pytest.fail(f"Unexpected warnings: {warning_messages}")


@contextmanager
def expect_warnings(expected_count: int = None, warning_type: type = None):
    """Context manager to validate expected warnings."""
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list

        if expected_count is not None:
            assert len(warning_list) == expected_count, (
                f"Expected {expected_count} warnings, got {len(warning_list)}"
            )

        if warning_type is not None:
            for w in warning_list:
                assert issubclass(w.category, warning_type), (
                    f"Expected warning type {warning_type}, got {w.category}"
                )


# System resource utilities
def get_system_info() -> dict[str, Any]:
    """Get system information for test context."""
    import platform

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }


def log_system_info():
    """Log system information for debugging."""
    info = get_system_info()
    print("\nSystem Info : ")
    for key, value in info.items():
        print(f"  {key}: {value}")


# Test skip conditions
def skip_if_insufficient_memory(required_gb: float):
    """Skip test if insufficient memory is available."""
    available_gb = psutil.virtual_memory().available / (1024**3)
    return pytest.mark.skipif(
        available_gb < required_gb,
        reason=f"Requires {required_gb}GB memory, {available_gb: .1f}GB available",
    )


def skip_if_insufficient_cpu(required_cores: int):
    """Skip test if insufficient CPU cores are available."""
    available_cores = psutil.cpu_count()
    return pytest.mark.skipif(
        available_cores < required_cores,
        reason=f"Requires {required_cores} CPU cores, {available_cores} available",
    )
