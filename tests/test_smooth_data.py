"""
Tests for smooth_data function improvements.

These tests verify both the pandas fillna method updates and the new
optimized NumPy convolution implementation in v0.1.2.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import xraylabtool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xraylabtool.utils import smooth_data


class TestSmoothDataUpdates:
    """Test smooth_data function with updated pandas methods."""

    def test_smooth_data_basic(self):
        """Test basic smoothing functionality."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.RandomState(42).randn(100)  # Reproducible noise

        smoothed = smooth_data(x, y, window_size=5)

        assert len(smoothed) == len(y)
        assert isinstance(smoothed, np.ndarray)
        # Smoothed data should have lower variance
        assert np.var(smoothed) <= np.var(y)

    def test_smooth_data_edge_handling(self):
        """Test that edge values are properly filled."""
        # Create data where edge effects would be visible
        x = np.arange(10)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        smoothed = smooth_data(x, y, window_size=3)

        # Check that no NaN values remain
        assert not np.any(np.isnan(smoothed))

        # Check that array length is preserved
        assert len(smoothed) == len(y)

        # Edge values should be filled (not NaN)
        assert not np.isnan(smoothed[0])
        assert not np.isnan(smoothed[-1])

    def test_smooth_data_window_sizes(self):
        """Test different window sizes."""
        x = np.arange(20)
        y = np.random.RandomState(42).randn(20)

        for window_size in [3, 5, 7]:
            smoothed = smooth_data(x, y, window_size=window_size)
            assert len(smoothed) == len(y)
            assert not np.any(np.isnan(smoothed))

    def test_smooth_data_small_array(self):
        """Test smoothing with small arrays."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])

        # Window size equal to array length
        smoothed = smooth_data(x, y, window_size=3)
        assert len(smoothed) == len(y)
        assert not np.any(np.isnan(smoothed))

        # Window size larger than array length
        smoothed_large = smooth_data(x, y, window_size=5)
        expected_mean = np.mean(y)
        assert np.allclose(smoothed_large, expected_mean)

    def test_smooth_data_invalid_window(self):
        """Test error handling for invalid window sizes."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])

        with pytest.raises(ValueError):
            smooth_data(x, y, window_size=0)

        with pytest.raises(ValueError):
            smooth_data(x, y, window_size=-1)


class TestSmoothDataPerformanceOptimizations:
    """Test performance optimizations in smooth_data function (v0.1.2)."""

    def test_smooth_data_performance_numpy_vs_pandas(self):
        """Test that optimized NumPy implementation produces consistent results."""
        # Create test data
        x = np.linspace(0, 10, 1000)
        y = np.sin(x) + 0.05 * np.random.RandomState(42).randn(1000)

        # Test the optimized function
        smoothed = smooth_data(x, y, window_size=10)

        # Verify basic properties
        assert len(smoothed) == len(y)
        assert isinstance(smoothed, np.ndarray)
        assert smoothed.dtype == np.float64  # Should be float64 for precision
        assert not np.any(np.isnan(smoothed))

        # Should have lower variance (smoothed)
        assert np.var(smoothed) < np.var(y)

    def test_smooth_data_edge_padding_consistency(self):
        """Test that edge padding in optimized version works correctly."""
        # Create simple step function to test edge handling
        x = np.arange(20)
        y = np.concatenate([np.ones(5), np.ones(10) * 5, np.ones(5)])

        smoothed = smooth_data(x, y, window_size=5)

        # No NaN values
        assert not np.any(np.isnan(smoothed))

        # Length preserved
        assert len(smoothed) == len(y)

        # Edge values should be reasonable (not the same as middle due to padding)
        assert 0.5 < smoothed[0] < 6  # Should be between extreme values
        assert 0.5 < smoothed[-1] < 6

    def test_smooth_data_mathematical_correctness(self):
        """Test that the optimized convolution produces mathematically correct results."""
        # Simple constant array - should remain unchanged
        x = np.arange(10)
        y = np.ones(10) * 5.0

        smoothed = smooth_data(x, y, window_size=3)

        # Should be approximately constant (within floating point precision)
        np.testing.assert_allclose(smoothed, 5.0, rtol=1e-10)

    def test_smooth_data_linear_trend_preservation(self):
        """Test that linear trends are approximately preserved in smoothed data."""
        # Linear trend
        x = np.arange(50)
        y = 2.0 * x + 1.0  # y = 2x + 1

        smoothed = smooth_data(x, y, window_size=5)

        # Test that the middle portion (avoiding edge effects) maintains linear trend
        # Use middle 60% of data to avoid edge effects
        start_idx = len(x) // 5
        end_idx = 4 * len(x) // 5

        middle_x = x[start_idx:end_idx]
        middle_smoothed = smoothed[start_idx:end_idx]

        # Fit line to middle portion
        coeffs = np.polyfit(middle_x, middle_smoothed, 1)

        # Slope should be approximately 2.0 in the middle portion
        np.testing.assert_allclose(coeffs[0], 2.0, rtol=5e-3)

    def test_smooth_data_dtype_handling(self):
        """Test that the optimized function handles different input dtypes correctly."""
        x = np.arange(10)

        # Test with different input dtypes
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype)
            smoothed = smooth_data(x, y, window_size=3)

            # Output should always be float64 for precision
            assert smoothed.dtype == np.float64
            assert not np.any(np.isnan(smoothed))
            assert len(smoothed) == len(y)


if __name__ == "__main__":
    pytest.main([__file__])
