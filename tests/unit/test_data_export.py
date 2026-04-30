"""Tests for io/data_export.py module.

This module tests data export and formatting utilities for XRayLabTool.
Covers formatting of XRayResult objects to different formats (JSON, CSV, table).
"""

import numpy as np
import pytest

from xraylabtool.io.data_export import (
    format_calculation_summary,
    format_xray_result,
)


class MockXRayResult:
    """Mock XRayResult for testing."""

    def __init__(
        self,
        formula: str = "SiO2",
        energy: float = 5.0,
        mass_attenuation: float = 15.2,
        linear_attenuation: float = 30.4,
    ):
        self.formula = formula
        self.energy = energy
        self.mass_attenuation = mass_attenuation
        self.linear_attenuation = linear_attenuation
        self.energy_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.attenuation_array = np.array([20.0, 18.5, 17.2, 16.1, 15.2])


class TestFormatXRayResult:
    """Tests for format_xray_result function."""

    def test_json_format_single_value(self):
        """Test JSON formatting of result with scalar values."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="json")

        # Should be valid JSON
        import json

        data = json.loads(output)
        assert data["formula"] == "SiO2"
        assert data["energy"] == 5.0

    def test_json_format_with_array(self):
        """Test JSON formatting includes array fields."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="json")

        import json

        data = json.loads(output)
        assert "energy_array" in data
        assert isinstance(data["energy_array"], list)
        assert len(data["energy_array"]) == 5

    def test_json_format_precision(self):
        """Test JSON formatting respects precision parameter."""
        result = MockXRayResult(mass_attenuation=15.123456)
        output = format_xray_result(result, format_type="json", precision=2)

        import json

        data = json.loads(output)
        # Precision should limit decimal places
        assert data["mass_attenuation"] == pytest.approx(15.12, abs=0.01)

    def test_json_format_selected_fields(self):
        """Test JSON formatting with selected fields only."""
        result = MockXRayResult()
        output = format_xray_result(
            result, format_type="json", fields=["formula", "energy"]
        )

        import json

        data = json.loads(output)
        assert "formula" in data
        assert "energy" in data
        assert "mass_attenuation" not in data

    def test_csv_format(self):
        """Test CSV formatting of result."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="csv")

        # Should contain comma-separated values
        lines = output.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row
        assert "SiO2" in output or "5.0" in output

    def test_csv_format_with_fields(self):
        """Test CSV formatting with selected fields."""
        result = MockXRayResult()
        output = format_xray_result(
            result, format_type="csv", fields=["formula", "energy"]
        )

        # Header should only have selected fields
        assert "formula" in output

    def test_table_format(self):
        """Test table formatting of result."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="table")

        # Should be human-readable text
        assert "SiO2" in output
        assert "XRay Properties" in output
        assert "=" in output  # Should have separator lines

    def test_table_format_includes_ranges(self):
        """Test table format shows range for arrays."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="table")

        # Array fields should show range info
        assert "Range:" in output or "[5 values]" in output

    def test_table_format_precision(self):
        """Test table format respects precision."""
        result = MockXRayResult(mass_attenuation=15.123456)
        output = format_xray_result(result, format_type="table", precision=2)

        # Should have limited decimal places
        assert "15.12" in output

    def test_default_format_is_table(self):
        """Test that default format is table."""
        result = MockXRayResult()
        default = format_xray_result(result)
        table = format_xray_result(result, format_type="table")

        assert default == table

    def test_case_insensitive_format_type(self):
        """Test that format type is case-insensitive."""
        result = MockXRayResult()
        json_lower = format_xray_result(result, format_type="json")
        json_upper = format_xray_result(result, format_type="JSON")

        import json

        data_lower = json.loads(json_lower)
        data_upper = json.loads(json_upper)
        assert data_lower == data_upper

    def test_unknown_format_defaults_to_table(self):
        """Test that unknown format defaults to table."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="unknown")

        # Should fall back to table format
        assert "XRay Properties" in output

    def test_empty_fields_list(self):
        """Test formatting with empty fields list."""
        result = MockXRayResult()
        output = format_xray_result(result, format_type="json", fields=[])

        import json

        data = json.loads(output)
        assert data == {}


class TestFormatCalculationSummary:
    """Tests for format_calculation_summary function."""

    def test_empty_results(self):
        """Test formatting empty results list."""
        output = format_calculation_summary([])
        assert "No results" in output

    def test_single_result_json(self):
        """Test JSON formatting of single result."""
        result = MockXRayResult()
        output = format_calculation_summary([result], format_type="json")

        import json

        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["formula"] == "SiO2"

    def test_multiple_results_json(self):
        """Test JSON formatting of multiple results."""
        results = [
            MockXRayResult("SiO2", 5.0),
            MockXRayResult("H2O", 3.0),
            MockXRayResult("Fe2O3", 7.0),
        ]
        output = format_calculation_summary(results, format_type="json")

        import json

        data = json.loads(output)
        assert len(data) == 3
        assert data[0]["formula"] == "SiO2"
        assert data[1]["formula"] == "H2O"
        assert data[2]["formula"] == "Fe2O3"

    def test_csv_format_single(self):
        """Test CSV formatting of single result."""
        result = MockXRayResult()
        output = format_calculation_summary([result], format_type="csv")

        # Should contain formula and numeric values
        assert "SiO2" in output or "5.0" in output

    def test_csv_format_multiple(self):
        """Test CSV formatting of multiple results."""
        results = [
            MockXRayResult("SiO2", 5.0),
            MockXRayResult("H2O", 3.0),
        ]
        output = format_calculation_summary(results, format_type="csv")

        lines = output.strip().split("\n")
        # Should have header + 2 data rows
        assert len(lines) >= 3

    def test_table_format_summary(self):
        """Test table formatting of summary."""
        results = [
            MockXRayResult("SiO2", 5.0),
            MockXRayResult("H2O", 3.0),
        ]
        output = format_calculation_summary(results, format_type="table")

        assert "Summary of 2 calculations" in output
        assert "SiO2" in output
        assert "H2O" in output

    def test_default_format_is_table(self):
        """Test that default format is table."""
        results = [MockXRayResult()]
        default = format_calculation_summary(results)
        table = format_calculation_summary(results, format_type="table")

        assert default == table

    def test_case_insensitive_format_type(self):
        """Test that format type is case-insensitive."""
        results = [MockXRayResult()]
        json_lower = format_calculation_summary(results, format_type="json")
        json_upper = format_calculation_summary(results, format_type="JSON")

        import json

        data_lower = json.loads(json_lower)
        data_upper = json.loads(json_upper)
        assert data_lower == data_upper

    def test_unknown_format_defaults_to_table(self):
        """Test that unknown format defaults to table."""
        results = [MockXRayResult()]
        output = format_calculation_summary(results, format_type="unknown")

        # Should fall back to table format
        assert "Summary" in output
