"""
Tests for the utils module.
"""

import os
import sys
import numpy as np
import pytest

try:
    from xraylabtool.utils import (
        wavelength_to_energy,
        energy_to_wavelength,
        bragg_angle,
        d_spacing_cubic,
        q_from_angle,
        angle_from_q,
        smooth_data,
        normalize_intensity,
    )
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.utils import (
        wavelength_to_energy,
        energy_to_wavelength,
        bragg_angle,
        d_spacing_cubic,
        q_from_angle,
        angle_from_q,
        smooth_data,
        normalize_intensity,
    )


class TestUnitConversions:
    """Tests for unit conversion functions."""

    def test_wavelength_to_energy(self):
        """Test wavelength to energy conversion."""
        # Test with Cu Kα radiation (1.5418 Å)
        wavelength = 1.5418  # Angstroms
        energy = wavelength_to_energy(wavelength)

        # Expected energy is approximately 8.05 keV
        assert abs(energy - 8.05) < 0.1

    def test_energy_to_wavelength(self):
        """Test energy to wavelength conversion."""
        energy = 8.05  # keV
        wavelength = energy_to_wavelength(energy)

        # Expected wavelength is approximately 1.54 Å
        assert abs(wavelength - 1.54) < 0.01

    def test_wavelength_energy_round_trip(self):
        """Test round-trip conversion."""
        original_wavelength = 1.5418
        energy = wavelength_to_energy(original_wavelength)
        final_wavelength = energy_to_wavelength(energy)

        assert abs(original_wavelength - final_wavelength) < 1e-10


class TestCrystallographicCalculations:
    """Tests for crystallographic calculation functions."""

    def test_bragg_angle(self):
        """Test Bragg angle calculation."""
        d_spacing = 3.0  # Angstroms
        wavelength = 1.5418  # Angstroms

        angle = bragg_angle(d_spacing, wavelength)

        # Expected angle for these values
        expected = np.degrees(np.arcsin(wavelength / (2 * d_spacing)))
        assert abs(angle - expected) < 1e-10

    def test_bragg_angle_invalid_parameters(self):
        """Test Bragg angle with invalid parameters."""
        with pytest.raises(ValueError):
            bragg_angle(-1.0, 1.5418)  # Negative d-spacing

        with pytest.raises(ValueError):
            bragg_angle(1.0, 3.0)  # sin(theta) > 1

    def test_d_spacing_cubic(self):
        """Test d-spacing calculation for cubic system."""
        a = 5.0  # Angstroms
        h, k, miller_l = 1, 0, 0

        d = d_spacing_cubic(h, k, miller_l, a)
        expected = a / np.sqrt(h**2 + k**2 + miller_l**2)

        assert abs(d - expected) < 1e-10

    def test_q_from_angle(self):
        """Test momentum transfer calculation from angle."""
        two_theta = 30.0  # degrees
        wavelength = 1.5418  # Angstroms

        q = q_from_angle(two_theta, wavelength)

        # Calculate expected value
        theta_rad = np.radians(two_theta / 2)
        expected = (4 * np.pi * np.sin(theta_rad)) / wavelength

        assert abs(q - expected) < 1e-10

    def test_angle_from_q(self):
        """Test angle calculation from momentum transfer."""
        q = 2.0  # Ų⁻¹
        wavelength = 1.5418  # Angstroms

        angle = angle_from_q(q, wavelength)

        # Calculate expected value
        sin_theta = (q * wavelength) / (4 * np.pi)
        expected = 2 * np.degrees(np.arcsin(sin_theta))

        assert abs(angle - expected) < 1e-10

    def test_q_angle_round_trip(self):
        """Test round-trip q to angle conversion."""
        original_angle = 30.0
        wavelength = 1.5418

        q = q_from_angle(original_angle, wavelength)
        final_angle = angle_from_q(q, wavelength)

        assert abs(original_angle - final_angle) < 1e-10


class TestDataProcessing:
    """Tests for data processing functions."""

    def test_smooth_data(self):
        """Test data smoothing."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + 0.1 * np.random.randn(100)  # Noisy sine wave

        smoothed = smooth_data(x, y, window_size=5)

        assert len(smoothed) == len(y)
        assert isinstance(smoothed, np.ndarray)
        # Smoothed data should have lower variance
        assert np.var(smoothed) <= np.var(y)

    def test_smooth_data_invalid_window(self):
        """Test smoothing with invalid window size."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])

        with pytest.raises(ValueError):
            smooth_data(x, y, window_size=0)

    def test_normalize_intensity_max(self):
        """Test intensity normalization by maximum."""
        y = np.array([1, 2, 3, 4, 5])
        normalized = normalize_intensity(y, method="max")

        assert np.max(normalized) == 1.0
        assert np.allclose(normalized, y / 5.0)

    def test_normalize_intensity_area(self):
        """Test intensity normalization by area."""
        y = np.array([1, 2, 3, 4, 5])
        normalized = normalize_intensity(y, method="area")

        # Area under curve should be 1
        assert abs(np.trapezoid(normalized) - 1.0) < 1e-10

    def test_normalize_intensity_standard(self):
        """Test standard score normalization."""
        y = np.array([1, 2, 3, 4, 5])
        normalized = normalize_intensity(y, method="standard")

        # Mean should be 0, std should be 1
        assert abs(np.mean(normalized)) < 1e-10
        assert abs(np.std(normalized) - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
