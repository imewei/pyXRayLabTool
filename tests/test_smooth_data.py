"""
Tests for smooth_data function improvements.

These tests verify that the pandas fillna method updates work correctly.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import xraylabtool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

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


if __name__ == '__main__':
    pytest.main([__file__])