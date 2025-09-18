"""
Tests for basic analysis functions.

This module tests the simplified analysis functionality for material comparison
and absorption edge detection.
"""

import numpy as np
import pytest

from xraylabtool.analysis import compare_materials, find_absorption_edges
from xraylabtool.calculators.core import calculate_single_material_properties


class TestBasicAnalysis:
    """Test the basic analysis functionality."""

    def test_find_absorption_edges_basic(self):
        """Test basic absorption edge detection."""
        # Create test data with a clear edge
        energies = np.array([8.0, 9.0, 9.5, 9.9, 10.0, 10.1, 10.5, 11.0, 12.0])
        f2_values = np.array([0.1, 0.15, 0.2, 0.3, 1.5, 1.4, 1.2, 1.0, 0.8])

        edges = find_absorption_edges(energies, f2_values, threshold=0.5)

        assert len(edges) > 0
        # Should detect the edge around 10.0 keV
        edge_energies = [edge[0] for edge in edges]
        assert any(9.5 < energy < 10.5 for energy in edge_energies)

    def test_find_absorption_edges_empty(self):
        """Test edge detection with insufficient data."""
        energies = np.array([8.0, 9.0])
        f2_values = np.array([0.1, 0.15])

        edges = find_absorption_edges(energies, f2_values)

        assert edges == []

    def test_find_absorption_edges_no_edges(self):
        """Test edge detection with no edges present."""
        energies = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        f2_values = np.array([0.1, 0.12, 0.13, 0.14, 0.15])  # Gradual increase

        edges = find_absorption_edges(energies, f2_values, threshold=0.1)

        assert len(edges) == 0

    def test_compare_materials_basic(self):
        """Test basic material comparison."""
        # Generate test results
        result1 = calculate_single_material_properties("SiO2", 10.0, 2.2)
        result2 = calculate_single_material_properties("Al2O3", 10.0, 3.9)
        results = [result1, result2]

        stats = compare_materials(results, "critical_angle_degrees")

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert stats["count"] == 2

    def test_compare_materials_empty(self):
        """Test material comparison with empty results."""
        stats = compare_materials([])

        assert stats == {}

    def test_compare_materials_different_property(self):
        """Test material comparison with different property."""
        result1 = calculate_single_material_properties("SiO2", 10.0, 2.2)
        result2 = calculate_single_material_properties("Al2O3", 10.0, 3.9)
        results = [result1, result2]

        stats = compare_materials(results, "attenuation_length_cm")

        assert "mean" in stats
        assert stats["count"] == 2
        assert stats["mean"] > 0  # Should be positive

    def test_compare_materials_nonexistent_property(self):
        """Test material comparison with non-existent property."""
        result1 = calculate_single_material_properties("SiO2", 10.0, 2.2)
        results = [result1]

        stats = compare_materials(results, "nonexistent_property")

        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__])
