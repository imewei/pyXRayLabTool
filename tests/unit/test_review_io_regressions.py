"""Regression coverage for data loss at the public I/O boundaries."""

import csv
import io
import warnings

import numpy as np
import pytest

from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.io.data_export import format_calculation_summary, format_xray_result
from xraylabtool.io.file_operations import (
    export_to_csv,
    load_data_file,
    save_calculation_results,
)


@pytest.mark.parametrize("suffix", [".csv", ".nff"])
def test_headerless_csv_preserves_first_numeric_row(tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    save_calculation_results(expected, str(path))

    np.testing.assert_array_equal(load_data_file(str(path)), expected)


def test_space_separated_header_is_supported(tmp_path):
    path = tmp_path / "data.nff"
    path.write_text("E f1 f2\n30 1 2\n40 3 4\n")

    np.testing.assert_array_equal(load_data_file(str(path)), [[30, 1, 2], [40, 3, 4]])


@pytest.fixture(scope="module")
def result():
    return calculate_single_material_properties("Si", [0.03, 10.0, 30.0], 2.33)


def test_default_result_csv_avoids_deprecated_aliases(result):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        output = format_xray_result(result, "csv")
    rows = list(csv.DictReader(io.StringIO(output)))

    assert len(rows) == 3
    assert [float(row["energy_kev"]) for row in rows] == [0.03, 10.0, 30.0]


def test_summary_csv_preserves_every_energy_and_excludes_methods(result):
    rows = list(
        csv.DictReader(io.StringIO(format_calculation_summary([result], "csv")))
    )

    assert len(rows) == 3
    assert [float(row["energy_kev"]) for row in rows] == [0.03, 10.0, 30.0]
    assert all(row["formula"] == "Si" for row in rows)
    assert "get_property_units" not in rows[0]


def test_wide_csv_honors_selected_field_order_without_inserting_formula(
    result, tmp_path
):
    path = tmp_path / "selected.csv"
    export_to_csv([result], str(path), fields=["density_g_cm3", "energy_kev"])
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows[0] == [
        "density_g_cm3",
        "energy_kev_energy_0",
        "energy_kev_energy_1",
        "energy_kev_energy_2",
    ]
    assert [float(value) for value in rows[1]] == [2.33, 0.03, 10.0, 30.0]
