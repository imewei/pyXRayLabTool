"""
Shared test configuration and fixtures for xraylabtool tests.

This module provides common fixtures, test data, and configuration
for all test modules in the xraylabtool test suite.
"""

import gc
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import psutil
import pytest

# Import centralized test configuration
from tests.fixtures.test_config import (
    PERFORMANCE_THRESHOLDS,
    TEST_ENERGIES,
    TEST_MATERIALS,
)
import xraylabtool as xlt
from xraylabtool.calculators.core import clear_scattering_factor_cache
from xraylabtool.data_handling.batch_processing import BatchConfig, MemoryMonitor

# Extract specific energy arrays for fixture use
TEST_ENERGIES_SMALL = TEST_ENERGIES["small"]
TEST_ENERGIES_MEDIUM = TEST_ENERGIES["medium"]
TEST_ENERGIES_LARGE = TEST_ENERGIES["large"]


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment once per test session."""
    # Ensure clean start
    clear_scattering_factor_cache()
    gc.collect()

    yield

    # Clean up after all tests
    clear_scattering_factor_cache()
    gc.collect()


@pytest.fixture(scope="function")
def clean_cache():
    """Clean cache before and after each test."""
    clear_scattering_factor_cache()
    gc.collect()
    yield
    clear_scattering_factor_cache()
    gc.collect()


@pytest.fixture
def test_materials():
    """Provide standard test materials with their densities."""
    return TEST_MATERIALS.copy()


@pytest.fixture
def test_energies_small():
    """Provide small energy array for quick tests."""
    return TEST_ENERGIES_SMALL.copy()


@pytest.fixture
def test_energies_medium():
    """Provide medium energy array for standard tests."""
    return TEST_ENERGIES_MEDIUM.copy()


@pytest.fixture
def test_energies_large():
    """Provide large energy array for performance tests."""
    return TEST_ENERGIES_LARGE.copy()


@pytest.fixture
def performance_thresholds():
    """Provide performance threshold values."""
    return PERFORMANCE_THRESHOLDS.copy()


@pytest.fixture
def memory_monitor():
    """Provide a memory monitor for testing."""
    return MemoryMonitor(memory_limit_gb=4.0)


@pytest.fixture
def batch_config_default():
    """Provide default batch configuration."""
    return BatchConfig()


@pytest.fixture
def batch_config_small():
    """Provide batch configuration optimized for small tests."""
    return BatchConfig(
        max_workers=2,
        chunk_size=10,
        memory_limit_gb=1.0,
        enable_progress=False,
        cache_results=False,
    )


@pytest.fixture
def batch_config_performance():
    """Provide batch configuration optimized for performance tests."""
    return BatchConfig(
        max_workers=1,  # Single worker for consistent timing
        chunk_size=50,
        memory_limit_gb=2.0,
        enable_progress=False,
        cache_results=False,
    )


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_calculation_result():
    """Provide a sample calculation result for testing."""
    return xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)


@pytest.fixture
def sample_batch_data():
    """Provide sample data for batch calculations."""
    formulas = ["SiO2", "Al2O3", "Si"]
    densities = [2.2, 3.95, 2.33]
    energies = np.array([8.0, 10.0, 12.0])
    return formulas, energies, densities


@pytest.fixture
def large_batch_data():
    """Provide large dataset for performance testing."""
    formulas = [material[0] for material in TEST_MATERIALS] * 10  # 80 materials
    densities = [material[1] for material in TEST_MATERIALS] * 10
    energies = np.linspace(5.0, 15.0, 100)
    return formulas, energies, densities


@pytest.fixture
def mock_low_memory():
    """Mock low memory condition for testing memory management."""

    def mock_virtual_memory():
        class MockMemory:
            total = 1024**3  # 1GB
            available = 100 * 1024**2  # 100MB

        return MockMemory()

    with patch("psutil.virtual_memory", side_effect=mock_virtual_memory):
        yield


@pytest.fixture
def benchmark_baseline():
    """Provide benchmark baseline values for performance regression testing."""
    return {
        "single_calculation_time": 0.001,  # 1ms baseline
        "batch_calculation_time_per_item": 0.0005,  # 0.5ms per item
        "memory_usage_mb": 10.0,  # 10MB baseline
        "cache_hit_rate": 0.9,  # 90% cache hit rate
    }


@pytest.fixture(
    params=TEST_MATERIALS[:3]
)  # Use first 3 materials for parameterized tests
def material_and_density(request):
    """Parameterized fixture providing different materials and densities."""
    return request.param


@pytest.fixture(
    params=[np.array([10.0]), np.array([5.0, 10.0, 15.0]), np.linspace(8.0, 12.0, 20)]
)
def energy_arrays(request):
    """Parameterized fixture providing different energy arrays."""
    return request.param


class PerformanceTimer:
    """Context manager for measuring execution time."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


@pytest.fixture
def performance_timer():
    """Provide a performance timer for measuring execution time."""
    return PerformanceTimer


class MemoryTracker:
    """Context manager for tracking memory usage."""

    def __init__(self):
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.memory_increase_mb = None

    def __enter__(self):
        import tracemalloc

        tracemalloc.start()
        self.start_memory = psutil.Process().memory_info().rss
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import tracemalloc

        self.end_memory = psutil.Process().memory_info().rss
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory = peak
        tracemalloc.stop()

        self.memory_increase_mb = (self.end_memory - self.start_memory) / 1024 / 1024


@pytest.fixture
def memory_tracker():
    """Provide a memory tracker for measuring memory usage."""
    return MemoryTracker


@pytest.fixture(autouse=True)
def enforce_test_isolation():
    """Ensure test isolation by cleaning up between tests."""
    # Setup: ensure clean state
    gc.collect()

    yield

    # Teardown: clean up after test
    gc.collect()


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (can be slow)"
    )
    config.addinivalue_line(
        "markers", "memory: marks tests that check memory usage and management"
    )
    config.addinivalue_line(
        "markers", "stability: marks tests that check numerical stability"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests that check for performance regressions"
    )
    config.addinivalue_line(
        "markers",
        "large_data: marks tests that use large datasets (may require more memory)",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names and locations."""
    for item in items:
        # Add performance marker to performance-related tests
        if any(
            keyword in item.name.lower()
            for keyword in ["performance", "benchmark", "timing"]
        ):
            item.add_marker(pytest.mark.performance)

        # Add memory marker to memory-related tests
        if any(keyword in item.name.lower() for keyword in ["memory", "leak", "usage"]):
            item.add_marker(pytest.mark.memory)

        # Add stability marker to stability tests
        if any(
            keyword in item.name.lower()
            for keyword in ["stability", "numerical", "precision"]
        ):
            item.add_marker(pytest.mark.stability)

        # Add large_data marker to tests with large datasets
        if any(keyword in item.name.lower() for keyword in ["large", "batch", "bulk"]):
            item.add_marker(pytest.mark.large_data)

        # Add regression marker to regression tests
        if "regression" in item.name.lower():
            item.add_marker(pytest.mark.regression)


@pytest.fixture
def skip_if_low_memory():
    """Skip test if system has insufficient memory."""

    def _skip_if_low_memory(required_gb=4):
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < required_gb:
            pytest.skip(
                f"Insufficient memory: {available_gb:.1f}GB available, {required_gb}GB required"
            )

    return _skip_if_low_memory


@pytest.fixture
def cpu_count():
    """Provide the number of available CPU cores."""
    return os.cpu_count() or 1


@pytest.fixture
def test_data_generator():
    """Provide utilities for generating test data."""

    class TestDataGenerator:
        @staticmethod
        def random_materials(count: int) -> list[tuple[str, float]]:
            """Generate random materials from the test set."""
            import random

            return random.choices(TEST_MATERIALS, k=count)

        @staticmethod
        def random_energies(
            count: int, min_energy: float = 1.0, max_energy: float = 20.0
        ) -> np.ndarray:
            """Generate random energy values."""
            return np.random.uniform(min_energy, max_energy, count)

        @staticmethod
        def linear_energies(
            count: int, min_energy: float = 1.0, max_energy: float = 20.0
        ) -> np.ndarray:
            """Generate linear energy values."""
            return np.linspace(min_energy, max_energy, count)

    return TestDataGenerator


# Utility functions for common test operations
def assert_calculation_result_valid(result, formula: str, energies: np.ndarray):
    """Assert that a calculation result is valid."""
    assert result is not None
    assert result.formula == formula
    assert len(result.energy_kev) == len(energies)
    assert np.all(np.isfinite(result.critical_angle_degrees))
    assert np.all(result.critical_angle_degrees >= 0)
    assert np.all(np.isfinite(result.transmission))
    assert np.all(result.transmission >= 0)
    assert np.all(result.transmission <= 1)


def assert_performance_within_threshold(
    elapsed_ms: float, threshold_ms: float, operation: str = "operation"
):
    """Assert that performance is within acceptable threshold."""
    assert elapsed_ms <= threshold_ms, (
        f"{operation} took {elapsed_ms:.3f}ms, "
        f"exceeding threshold of {threshold_ms:.3f}ms"
    )


def assert_memory_usage_reasonable(
    memory_increase_mb: float, threshold_mb: float, operation: str = "operation"
):
    """Assert that memory usage is reasonable."""
    assert memory_increase_mb <= threshold_mb, (
        f"{operation} used {memory_increase_mb:.2f}MB, "
        f"exceeding threshold of {threshold_mb:.2f}MB"
    )
