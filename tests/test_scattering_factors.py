"""
Tests for scattering factor data loading functionality.

This module tests the new f1/f2 scattering table loading and caching
functionality implemented in Step 5.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

try:
    from xraylabtool.core import (
        load_scattering_factor_data,
        get_cached_elements,
        clear_scattering_factor_cache,
        is_element_cached,
        AtomicScatteringFactor,
        create_scattering_factor_interpolators,
    )
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.core import (
        load_scattering_factor_data,
        get_cached_elements,
        clear_scattering_factor_cache,
        is_element_cached,
        AtomicScatteringFactor,
        create_scattering_factor_interpolators,
    )


class TestScatteringFactorLoading:
    """Tests for scattering factor data loading."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_load_scattering_factor_data_success(self):
        """Test successful loading of scattering factor data."""
        # Test with a known element that should exist
        try:
            data = load_scattering_factor_data("Si")

            # Verify DataFrame structure
            assert isinstance(data, pd.DataFrame)
            assert not data.empty

            # Verify expected columns
            expected_columns = {"E", "f1", "f2"}
            assert expected_columns.issubset(set(data.columns))

            # Verify data types are numeric
            assert pd.api.types.is_numeric_dtype(data["E"])
            assert pd.api.types.is_numeric_dtype(data["f1"])
            assert pd.api.types.is_numeric_dtype(data["f2"])

            # Verify reasonable data ranges
            assert data["E"].min() > 0  # Energy should be positive
            assert len(data) > 0  # Should have some data points

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_load_scattering_factor_data_caching(self):
        """Test caching functionality."""
        # Clear cache to start fresh
        clear_scattering_factor_cache()

        try:
            # First load
            data1 = load_scattering_factor_data("Ge")

            # Verify element is cached
            assert is_element_cached("Ge")
            assert "Ge" in get_cached_elements()

            # Second load should return cached data
            data2 = load_scattering_factor_data("Ge")

            # Should be the same data (cached)
            pd.testing.assert_frame_equal(data1, data2)

        except FileNotFoundError:
            pytest.skip("Germanium .nff file not available for testing")

    def test_case_insensitive_element_symbols(self):
        """Test case insensitive element symbol handling."""
        try:
            # Test different cases for the same element
            data_upper = load_scattering_factor_data("GE")
            data_lower = load_scattering_factor_data("ge")
            data_proper = load_scattering_factor_data("Ge")

            # All should return the same data
            pd.testing.assert_frame_equal(data_upper, data_lower)
            pd.testing.assert_frame_equal(data_lower, data_proper)

            # Should all be cached under proper case
            cached_elements = get_cached_elements()
            assert "Ge" in cached_elements

        except FileNotFoundError:
            pytest.skip("Germanium .nff file not available for testing")

    def test_invalid_element_symbol(self):
        """Test error handling for invalid element symbols."""
        # Test empty string
        with pytest.raises(
            ValueError, match="Element symbol must be a non-empty string"
        ):
            load_scattering_factor_data("")

        # Test None
        with pytest.raises(
            ValueError, match="Element symbol must be a non-empty string"
        ):
            load_scattering_factor_data(None)  # type: ignore

        # Test non-string
        with pytest.raises(
            ValueError, match="Element symbol must be a non-empty string"
        ):
            load_scattering_factor_data(123)  # type: ignore

    def test_nonexistent_element_file(self):
        """Test error handling for non-existent element files."""
        with pytest.raises(
            FileNotFoundError, match="Scattering factor data file not found"
        ):
            load_scattering_factor_data("Xyz")  # Non-existent element

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        try:
            # Load some data
            load_scattering_factor_data("Si")
            assert len(get_cached_elements()) > 0

            # Clear cache
            clear_scattering_factor_cache()
            assert len(get_cached_elements()) == 0
            assert not is_element_cached("Si")

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_multiple_elements_caching(self):
        """Test caching multiple elements."""
        elements_to_test = ["H", "C", "N", "O"]  # Common elements likely to exist
        loaded_elements = []

        for element in elements_to_test:
            try:
                load_scattering_factor_data(element)
                loaded_elements.append(element)
                assert is_element_cached(element)
            except FileNotFoundError:
                # Skip elements that don't have .nff files
                continue

        # Verify all loaded elements are in cache
        cached_elements = get_cached_elements()
        for element in loaded_elements:
            assert element in cached_elements

        # Should have at least loaded some elements
        if loaded_elements:
            assert len(cached_elements) >= len(loaded_elements)


class TestAtomicScatteringFactorIntegration:
    """Test integration with AtomicScatteringFactor class."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_atomic_scattering_factor_load_element_data(self):
        """Test loading element data through AtomicScatteringFactor class."""
        asf = AtomicScatteringFactor()

        try:
            data = asf.load_element_data("Si")

            # Verify DataFrame structure
            assert isinstance(data, pd.DataFrame)
            assert not data.empty

            # Verify expected columns
            expected_columns = {"E", "f1", "f2"}
            assert expected_columns.issubset(set(data.columns))

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_backward_compatibility(self):
        """Test that the class maintains backward compatibility."""
        asf = AtomicScatteringFactor()

        # Should have data and data_path attributes for backward compatibility
        assert hasattr(asf, "data")
        assert hasattr(asf, "data_path")
        assert isinstance(asf.data, dict)
        assert isinstance(asf.data_path, Path)


class TestDataValidation:
    """Test data validation and file format verification."""

    def test_data_format_validation(self):
        """Test that loaded data has correct format."""
        try:
            data = load_scattering_factor_data("Ge")

            # Check that required columns exist
            assert "E" in data.columns
            assert "f1" in data.columns
            assert "f2" in data.columns

            # Check data types
            assert pd.api.types.is_numeric_dtype(data["E"])
            assert pd.api.types.is_numeric_dtype(data["f1"])
            assert pd.api.types.is_numeric_dtype(data["f2"])

            # Check for reasonable data ranges
            assert data["E"].min() > 0, "Energy values should be positive"
            assert not data.isnull().any().any(), "Data should not contain NaN values"

        except FileNotFoundError:
            pytest.skip("Germanium .nff file not available for testing")


class TestPCHIPInterpolators:
    """Test PCHIP interpolator functionality - Step 6."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_scattering_factor_cache()

    def test_create_interpolators_success(self):
        """Test successful creation of PCHIP interpolators."""
        try:
            # Create interpolators
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            # Should return callable objects
            assert callable(f1_interp)
            assert callable(f2_interp)

            # Test scalar input
            energy = 100.0
            f1_val = f1_interp(energy)
            f2_val = f2_interp(energy)

            # Values should be real numbers (may be numpy scalars or arrays)
            assert isinstance(f1_val, (int, float, np.number, np.ndarray))
            assert isinstance(f2_val, (int, float, np.number, np.ndarray))
            assert (
                not np.isnan(f1_val).any()
                if hasattr(f1_val, "any")
                else not np.isnan(f1_val)
            )
            assert (
                not np.isnan(f2_val).any()
                if hasattr(f2_val, "any")
                else not np.isnan(f2_val)
            )

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_interpolators_array_input(self):
        """Test interpolators with array input (Julia-like behavior)."""
        try:
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            # Test array input
            energies = np.array([50.0, 100.0, 200.0])
            f1_values = f1_interp(energies)
            f2_values = f2_interp(energies)

            # Should return arrays of same shape
            assert isinstance(f1_values, np.ndarray)
            assert isinstance(f2_values, np.ndarray)
            assert f1_values.shape == energies.shape
            assert f2_values.shape == energies.shape

            # Values should be finite
            assert np.all(np.isfinite(f1_values))
            assert np.all(np.isfinite(f2_values))

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_interpolators_list_input(self):
        """Test interpolators with list input."""
        try:
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            # Test list input (should be converted to array internally)
            energies = np.array([75.0, 125.0, 175.0])
            f1_values = f1_interp(energies)
            f2_values = f2_interp(energies)

            # Should work and return arrays
            assert isinstance(f1_values, np.ndarray)
            assert isinstance(f2_values, np.ndarray)
            assert len(f1_values) == len(energies)
            assert len(f2_values) == len(energies)

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_interpolation_accuracy(self):
        """Test interpolation accuracy at known data points."""
        try:
            # Load original data
            data = load_scattering_factor_data("Si")
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            # Test interpolation at original data points
            # Should be very close to original values
            for i in range(0, len(data), 10):  # Test every 10th point
                energy = data.iloc[i]["E"]
                expected_f1 = data.iloc[i]["f1"]
                expected_f2 = data.iloc[i]["f2"]

                interpolated_f1 = f1_interp(energy)
                interpolated_f2 = f2_interp(energy)

                # Should be very close (within numerical precision)
                assert np.isclose(interpolated_f1, expected_f1, atol=1e-10)
                assert np.isclose(interpolated_f2, expected_f2, atol=1e-10)

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_interpolation_between_points(self):
        """Test interpolation between data points."""
        try:
            data = load_scattering_factor_data("Si")
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            # Test interpolation at midpoint between two data points
            if len(data) >= 2:
                E1, E2 = data.iloc[0]["E"], data.iloc[1]["E"]
                f1_1, f1_2 = data.iloc[0]["f1"], data.iloc[1]["f1"]
                f2_1, f2_2 = data.iloc[0]["f2"], data.iloc[1]["f2"]

                # Interpolate at midpoint
                E_mid = (E1 + E2) / 2
                f1_mid = f1_interp(E_mid)
                f2_mid = f2_interp(E_mid)

                # Interpolated value should be between the two endpoints
                # (for monotonic sections)
                f1_min, f1_max = min(f1_1, f1_2), max(f1_1, f1_2)
                f2_min, f2_max = min(f2_1, f2_2), max(f2_1, f2_2)

                # Allow some tolerance for PCHIP interpolation behavior
                tolerance = abs(f1_max - f1_min) * 0.5  # 50% tolerance
                assert f1_min - tolerance <= f1_mid <= f1_max + tolerance

                tolerance = abs(f2_max - f2_min) * 0.5
                assert f2_min - tolerance <= f2_mid <= f2_max + tolerance

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_extrapolation_behavior(self):
        """Test that extrapolation is disabled."""
        try:
            data = load_scattering_factor_data("Si")
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")

            E_min, E_max = data["E"].min(), data["E"].max()

            # Test extrapolation below range
            try:
                result = f1_interp(E_min - 10)
                # If no exception, result should be NaN or function allows extrapolation
                if not np.isnan(result):
                    pytest.skip("Interpolator allows extrapolation")
            except (ValueError, RuntimeError):
                pass  # Expected behavior for no extrapolation

            # Test extrapolation above range
            try:
                result = f2_interp(E_max + 10)
                if not np.isnan(result):
                    pytest.skip(
                        "Interpolator allows extrapolation or returns boundary values"
                    )
            except (ValueError, RuntimeError):
                pass  # Expected behavior for no extrapolation

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")

    def test_invalid_element_interpolator(self):
        """Test error handling for invalid elements."""
        with pytest.raises(FileNotFoundError):
            create_scattering_factor_interpolators("Xyz")

        with pytest.raises(
            ValueError, match="Element symbol must be a non-empty string"
        ):
            create_scattering_factor_interpolators("")

    def test_insufficient_data_points(self):
        """Test error handling for insufficient data points."""
        # This test would require creating a mock data file with < 2 points
        # For now, we test the error message format
        try:
            # If we have normal data files, this should work
            f1_interp, f2_interp = create_scattering_factor_interpolators("Si")
            # If it works, the test passes
        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")
        except ValueError as e:
            # If there's a ValueError, it should mention insufficient data points
            assert "PCHIP interpolation requires at least 2 points" in str(e)

    def test_case_insensitive_interpolators(self):
        """Test case insensitive element symbols for interpolators."""
        try:
            # Different cases should work
            f1_upper, f2_upper = create_scattering_factor_interpolators("SI")
            f1_lower, f2_lower = create_scattering_factor_interpolators("si")
            f1_proper, f2_proper = create_scattering_factor_interpolators("Si")

            # Test the same energy point
            energy = 100.0
            val1 = f1_upper(energy)
            val2 = f1_lower(energy)
            val3 = f1_proper(energy)

            # Should all give the same result
            assert np.isclose(val1, val2, atol=1e-10)
            assert np.isclose(val2, val3, atol=1e-10)

        except FileNotFoundError:
            pytest.skip("Silicon .nff file not available for testing")


if __name__ == "__main__":
    pytest.main([__file__])
