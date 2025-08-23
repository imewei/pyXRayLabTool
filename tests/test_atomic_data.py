"""
Tests for atomic data lookup functions with LRU caching.
"""

import pytest
import numpy as np
from xraylabtool.utils import (
    get_atomic_number,
    get_atomic_weight,
    get_atomic_data,
    UnknownElementError,
)


class TestAtomicDataLookup:
    """Test atomic data lookup with caching."""

    def test_atomic_number_valid_elements(self):
        """Test atomic number lookup for valid elements."""
        test_cases = [
            ("H", 1),
            ("C", 6),
            ("N", 7),
            ("O", 8),
            ("Si", 14),
            ("Al", 13),
            ("Fe", 26),
            ("Cu", 29),
        ]

        for symbol, expected in test_cases:
            result = get_atomic_number(symbol)
            assert result == expected
            assert isinstance(result, int)

    def test_atomic_weight_valid_elements(self):
        """Test atomic weight lookup for valid elements."""
        test_cases = [
            ("H", 1.008, 0.001),
            ("C", 12.011, 0.001),
            ("O", 15.999, 0.001),
            ("Si", 28.085, 0.001),
            ("Al", 26.982, 0.001),
            ("Fe", 55.845, 0.001),
        ]

        for symbol, expected, tolerance in test_cases:
            result = get_atomic_weight(symbol)
            assert np.isclose(result, expected, atol=tolerance)
            assert isinstance(result, float)

    def test_atomic_data_comprehensive(self):
        """Test comprehensive atomic data lookup."""
        data = get_atomic_data("Si")

        expected_keys = {"symbol", "atomic_number", "atomic_weight", "name", "density"}
        assert set(data.keys()) == expected_keys
        assert data["symbol"] == "Si"
        assert data["atomic_number"] == 14
        assert abs(data["atomic_weight"] - 28.085) < 0.1
        assert data["name"] == "Silicon"
        assert isinstance(data["density"], (float, type(None)))

    def test_unknown_element_exceptions(self):
        """Test proper exceptions for unknown elements."""
        invalid_elements = ["Xx", "Unknown", "123", "ABC", "!@#"]

        for element in invalid_elements:
            with pytest.raises(UnknownElementError):
                get_atomic_number(element)

            with pytest.raises(UnknownElementError):
                get_atomic_weight(element)

            with pytest.raises(UnknownElementError):
                get_atomic_data(element)

    def test_caching_functionality(self):
        """Test that LRU caching works correctly."""
        # Clear cache first
        get_atomic_number.cache_clear()

        # First call should be a cache miss
        result1 = get_atomic_number("Fe")

        # Second call should be a cache hit (same result)
        result2 = get_atomic_number("Fe")

        assert result1 == result2

        if hasattr(get_atomic_number, "cache_info"):
            cache_info = get_atomic_number.cache_info()
            assert cache_info.hits > 0, "Should have cache hits"
            assert cache_info.misses > 0, "Should have cache misses"

    def test_cache_independence(self):
        """Test that different functions have independent caches."""
        # Clear all caches
        get_atomic_number.cache_clear()
        get_atomic_weight.cache_clear()
        get_atomic_data.cache_clear()

        element = "C"

        # Call each function
        atomic_num = get_atomic_number(element)
        atomic_weight = get_atomic_weight(element)
        atomic_data = get_atomic_data(element)

        # Verify results are consistent
        assert atomic_num == 6
        assert abs(atomic_weight - 12.011) < 0.001
        assert atomic_data["atomic_number"] == 6
        assert abs(atomic_data["atomic_weight"] - 12.011) < 0.001

    def test_case_sensitivity(self):
        """Test that element symbols are case-sensitive."""
        # Valid case
        assert get_atomic_number("H") == 1

        # Invalid cases should raise exceptions
        with pytest.raises(UnknownElementError):
            get_atomic_number("h")  # lowercase

        with pytest.raises(UnknownElementError):
            get_atomic_number("HE")  # all uppercase for two-letter symbol
