"""
Tests for the core module.
"""

import os
import sys
import numpy as np
import pytest

try:
    from xraylabtool.core import (
        AtomicScatteringFactor,
        CrystalStructure,
        load_data_file,
    )
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.core import (
        AtomicScatteringFactor,
        CrystalStructure,
        load_data_file,
    )


class TestAtomicScatteringFactor:
    """Tests for AtomicScatteringFactor class."""

    def test_initialization(self):
        """Test AtomicScatteringFactor initialization."""
        asf = AtomicScatteringFactor()
        assert isinstance(asf.data, dict)
        assert len(asf.data) == 0
        assert asf.data_path.exists()

    def test_get_scattering_factor(self):
        """Test scattering factor calculation."""
        asf = AtomicScatteringFactor()
        q_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        factors = asf.get_scattering_factor("C", q_values)

        assert isinstance(factors, np.ndarray)
        assert len(factors) == len(q_values)


class TestCrystalStructure:
    """Tests for CrystalStructure class."""

    def test_initialization(self):
        """Test CrystalStructure initialization."""
        lattice_params = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)  # cubic
        crystal = CrystalStructure(lattice_params)

        assert crystal.a == 5.0
        assert crystal.b == 5.0
        assert crystal.c == 5.0
        assert crystal.alpha == 90.0
        assert crystal.beta == 90.0
        assert crystal.gamma == 90.0
        assert len(crystal.atoms) == 0

    def test_add_atom(self):
        """Test adding atoms to crystal structure."""
        lattice_params = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        crystal = CrystalStructure(lattice_params)

        crystal.add_atom("C", (0.0, 0.0, 0.0))
        crystal.add_atom("O", (0.5, 0.5, 0.5), occupancy=0.5)

        assert len(crystal.atoms) == 2
        assert crystal.atoms[0]["element"] == "C"
        assert crystal.atoms[0]["position"] == (0.0, 0.0, 0.0)
        assert crystal.atoms[0]["occupancy"] == 1.0
        assert crystal.atoms[1]["element"] == "O"
        assert crystal.atoms[1]["occupancy"] == 0.5

    def test_calculate_structure_factor(self):
        """Test structure factor calculation."""
        lattice_params = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)
        crystal = CrystalStructure(lattice_params)
        crystal.add_atom("C", (0.0, 0.0, 0.0))

        sf = crystal.calculate_structure_factor((1, 0, 0))
        assert isinstance(sf, complex)


class TestLoadDataFile:
    """Tests for load_data_file function."""

    def test_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_data_file("nonexistent_file.txt")


if __name__ == "__main__":
    pytest.main([__file__])
