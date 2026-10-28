"""Batch input loading must handle empty files and requested energy columns."""

import numpy as np
import pytest

from xraylabtool.data_handling.batch_processing import load_batch_input


def test_empty_batch_with_energy_header_returns_empty_inputs(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("formula,density,energy\n")

    assert load_batch_input(path, energy_column="energy") == ([], [], [])


def test_explicit_energy_column_cannot_be_silently_ignored(tmp_path):
    path = tmp_path / "batch.csv"
    path.write_text("formula,density\nSi,2.33\n")

    with pytest.raises(ValueError, match=r"Missing required columns.*energy"):
        load_batch_input(path, energy_column="energy")


def test_batch_energy_cells_allow_scalar_and_list_values(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text('formula,density,energy\nSi,2.33,10\nAl,2.7,"5,15"\n')

    formulas, densities, energies = load_batch_input(path, energy_column="energy")

    assert formulas == ["Si", "Al"]
    assert densities == [2.33, 2.7]
    assert energies is not None
    np.testing.assert_array_equal(energies[0], [10])
    np.testing.assert_array_equal(energies[1], [5, 15])
