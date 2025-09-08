"""
Centralized test configuration and constants.

This module provides centralized configuration, constants, and utilities
for all test modules in the xraylabtool test suite.
"""

from typing import Any

import numpy as np

# =============================================================================
# TEST DATA CONSTANTS
# =============================================================================

# Standard test materials with known properties
TEST_MATERIALS: list[tuple[str, float]] = [
    ("SiO2", 2.2),  # Fused silica
    ("Al2O3", 3.95),  # Sapphire
    ("Fe2O3", 5.24),  # Iron oxide
    ("Si", 2.33),  # Silicon
    ("Au", 19.32),  # Gold
    ("C", 2.267),  # Diamond
    ("TiO2", 4.23),  # Titanium dioxide
    ("ZnO", 5.61),  # Zinc oxide
    ("CaCO3", 2.71),  # Calcite
    ("BaF2", 4.89),  # Barium fluoride
]

# Test energy arrays for different scenarios
TEST_ENERGIES: dict[str, np.ndarray] = {
    "single": np.array([10.0]),
    "small": np.array([5.0, 10.0, 15.0]),
    "medium": np.linspace(1.0, 20.0, 50),
    "large": np.linspace(0.1, 30.0, 1000),
    "cu_ka": np.array([8.048]),  # Cu Kα energy
    "synchrotron": np.logspace(np.log10(1), np.log10(30), 100),
    "edge_scan": np.linspace(7.0, 7.2, 201),  # Around Fe K-edge
}

# Performance test thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS: dict[str, float] = {
    "single_calculation": 1.0,
    "batch_calculation": 5.0,
    "energy_sweep": 10.0,
    "large_batch": 50.0,
    "cache_lookup": 0.1,
}

# Memory usage thresholds (in MB)
MEMORY_THRESHOLDS: dict[str, float] = {
    "single_calculation": 10.0,
    "small_batch": 50.0,
    "medium_batch": 200.0,
    "large_batch": 500.0,
}

# Expected numerical tolerances
NUMERICAL_TOLERANCES: dict[str, float] = {
    "default": 1e-6,
    "high_precision": 1e-10,
    "interpolation": 1e-8,
    "integration": 1e-5,
    "physics": 1e-6,
}

# Test chemical formulas for parsing tests
TEST_FORMULAS: dict[str, dict[str, Any]] = {
    "simple": {
        "SiO2": {"Si": 1, "O": 2},
        "Al2O3": {"Al": 2, "O": 3},
        "Fe2O3": {"Fe": 2, "O": 3},
    },
    "complex": {
        "Ca10P6O26H2": {"Ca": 10, "P": 6, "O": 26, "H": 2},
        "Na2Ca3Al2F14": {"Na": 2, "Ca": 3, "Al": 2, "F": 14},
        "Ba2TiSi2O8": {"Ba": 2, "Ti": 1, "Si": 2, "O": 8},
    },
    "fractional": {
        "Ca0.5Sr0.5TiO3": {"Ca": 0.5, "Sr": 0.5, "Ti": 1, "O": 3},
        "La0.67Ca0.33MnO3": {"La": 0.67, "Ca": 0.33, "Mn": 1, "O": 3},
    },
}

# Reference values from Julia implementation (for validation)
REFERENCE_VALUES: dict[str, dict[str, Any]] = {
    "SiO2_10keV": {
        "dispersion_delta": 9.451484792575434e-6,
        "critical_angle_degrees": 0.223,
        "attenuation_length_cm": 12.5,
        "molecular_weight_g_mol": 60.08,
    },
    "Si_8.048keV": {
        "dispersion_delta": 1.20966554922812e-6,
        "critical_angle_degrees": 0.158,
        "f1": 14.048053047106292,
        "f2": 0.053331074920700626,
    },
    "H2O_10keV": {
        "dispersion_delta": 4.734311949237782e-6,
        "critical_angle_degrees": 0.153,
        "molecular_weight_g_mol": 18.015,
    },
}

# Test file paths and data
TEST_DATA_PATHS: dict[str, str] = {
    "scattering_factors": "data/AtomicScatteringFactor/",
    "test_materials": "tests/data/sample_materials.json",
    "test_energies": "tests/data/test_energies.json",
    "performance_baselines": "tests/data/performance_baselines.json",
}

# CLI test configurations
CLI_TEST_CONFIGS: dict[str, dict[str, Any]] = {
    "basic_calc": {
        "command": "calc",
        "args": ["SiO2", "-e", "10.0", "-d", "2.2"],
        "expected_fields": ["formula", "energy_kev", "critical_angle_degrees"],
    },
    "energy_range": {
        "command": "calc",
        "args": ["Si", "-e", "5-15:11", "-d", "2.33"],
        "expected_points": 11,
    },
    "batch_processing": {
        "command": "batch",
        "input_file": "test_materials.csv",
        "output_file": "test_results.csv",
    },
}

# Error test cases
ERROR_TEST_CASES: dict[str, dict[str, Any]] = {
    "invalid_formula": {
        "formulas": ["XYZ123", "Si2O", ""],
        "expected_error": "FormulaError",
    },
    "invalid_energy": {
        "energies": [-1.0, 0.0, float("inf"), float("nan")],
        "expected_error": "EnergyError",
    },
    "invalid_density": {
        "densities": [-1.0, 0.0, float("inf")],
        "expected_error": "ValidationError",
    },
}

# Platform-specific configurations
PLATFORM_CONFIG: dict[str, Any] = {
    "windows": {
        "shell_completion": "powershell",
        "path_separator": "\\",
    },
    "unix": {
        "shell_completion": "bash",
        "path_separator": "/",
    },
}

# Test markers and categories
TEST_CATEGORIES: dict[str, list[str]] = {
    "unit": ["test_core", "test_utils", "test_exceptions"],
    "integration": ["test_integration", "test_cli"],
    "performance": ["test_performance_benchmarks", "test_memory_management"],
    "io": ["test_batch_processor", "test_scattering_factors"],
    "numerical": ["test_numerical_stability", "test_edge_cases"],
}

# Pytest markers
PYTEST_MARKERS: list[str] = [
    "unit: unit tests",
    "integration: integration tests",
    "performance: performance tests",
    "slow: slow tests (> 1 second)",
    "memory: memory usage tests",
    "benchmark: benchmark tests",
    "cli: CLI tests",
    "physics: physics validation tests",
    "regression: regression tests",
]
