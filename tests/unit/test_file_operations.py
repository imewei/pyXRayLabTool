"""Tests for xraylabtool.io.file_operations."""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pytest

from xraylabtool.exceptions import DataFileError
from xraylabtool.io.file_operations import load_data_file, save_calculation_results

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "xraylabtool"
    / "data"
    / "AtomicScatteringFactor"
)

SAMPLE_NFF = DATA_DIR / "ge.nff"


# ---------------------------------------------------------------------------
# load_data_file — .nff format
# ---------------------------------------------------------------------------


def test_load_valid_nff_file():
    data = load_data_file(str(SAMPLE_NFF))
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2
    assert data.shape[1] == 3  # E, f1, f2 columns
    assert data.shape[0] > 0


def test_load_nff_values_are_finite():
    data = load_data_file(str(SAMPLE_NFF))
    assert np.all(np.isfinite(data))


def test_load_nff_energy_column_monotone():
    """Energy column (col 0) should be strictly increasing."""
    data = load_data_file(str(SAMPLE_NFF))
    energies = data[:, 0]
    assert np.all(np.diff(energies) > 0)


# ---------------------------------------------------------------------------
# load_data_file — path variants
# ---------------------------------------------------------------------------


def test_load_absolute_path():
    data = load_data_file(str(SAMPLE_NFF.resolve()))
    assert data.shape[0] > 0


def test_load_path_with_spaces(tmp_path):
    dest = tmp_path / "dir with spaces" / "ge copy.nff"
    dest.parent.mkdir()
    dest.write_bytes(SAMPLE_NFF.read_bytes())
    data = load_data_file(str(dest))
    assert data.shape[0] > 0


# ---------------------------------------------------------------------------
# load_data_file — error paths
# ---------------------------------------------------------------------------


def test_load_nonexistent_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_data_file("/nonexistent/path/missing.nff")


def test_load_malformed_nff_raises_data_file_error(tmp_path):
    bad = tmp_path / "bad.nff"
    bad.write_text("not\tnumeric\tdata\nfoo bar baz\n")
    with pytest.raises(DataFileError):
        load_data_file(str(bad))


def test_load_truncated_nff_raises_data_file_error(tmp_path):
    truncated = tmp_path / "truncated.nff"
    # Header plus rows with inconsistent column counts — numpy raises ValueError
    truncated.write_text("E,f1,f2\n29.3,1.0\n30.0,2.0,extra,column\n")
    with pytest.raises(DataFileError):
        load_data_file(str(truncated))


def test_load_empty_nff_raises_data_file_error(tmp_path):
    empty = tmp_path / "empty.nff"
    empty.write_text("")
    with pytest.raises(DataFileError):
        load_data_file(str(empty))


# ---------------------------------------------------------------------------
# load_data_file — CSV format
# ---------------------------------------------------------------------------


def test_load_csv_file(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x,y\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
    data = load_data_file(str(csv_file))
    assert data.shape == (3, 2)
    assert data[0, 0] == pytest.approx(1.0)


def test_load_csv_space_separated_fallback(tmp_path):
    ssp = tmp_path / "data.csv"
    ssp.write_text("# comment\n1.0 2.0\n3.0 4.0\n")
    data = load_data_file(str(ssp))
    assert data.shape == (2, 2)


# ---------------------------------------------------------------------------
# save_calculation_results — round-trip
# ---------------------------------------------------------------------------


def test_save_and_reload_numpy_array(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = tmp_path / "out.csv"
    save_calculation_results(arr, str(out), format_type="csv")
    reloaded = np.loadtxt(str(out), delimiter=",")
    np.testing.assert_allclose(reloaded, arr, rtol=1e-5)


def test_save_dict_of_arrays_csv(tmp_path):
    data = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    out = tmp_path / "dict.csv"
    save_calculation_results(data, str(out), format_type="csv")
    assert out.exists()
    content = out.read_text()
    assert "a" in content
    assert "b" in content


def test_save_json(tmp_path):
    import json

    data = {"x": [1.0, 2.0], "y": [3.0, 4.0]}
    out = tmp_path / "out.json"
    save_calculation_results(data, str(out), format_type="json")
    loaded = json.loads(out.read_text())
    assert loaded["x"] == [1.0, 2.0]


def test_save_unsupported_format_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported format type"):
        save_calculation_results({}, str(tmp_path / "x.bin"), format_type="bin")
