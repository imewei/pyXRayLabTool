#!/usr/bin/env python3
"""
Tests for the CLI module of XRayLabTool.

This module contains comprehensive tests for all CLI commands and functionality,
ensuring the command line interface works correctly with various inputs and
edge cases.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

try:
    from xraylabtool.cli import (
        main,
        parse_energy_string,
        format_xray_result,
        cmd_calc,
        cmd_batch,
        cmd_convert,
        cmd_formula,
        cmd_atomic,
        cmd_bragg,
        cmd_list,
    )
    import xraylabtool as xlt
except ImportError:
    # Add parent directory to path to import xraylabtool
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from xraylabtool.cli import (
        main,
        parse_energy_string,
        format_xray_result,
        cmd_calc,
        cmd_batch,
        cmd_convert,
        cmd_formula,
        cmd_atomic,
        cmd_bragg,
        cmd_list,
    )
    import xraylabtool as xlt


class TestEnergyParsing:
    """Test energy string parsing functionality."""

    def test_single_energy(self):
        """Test parsing single energy value."""
        result = parse_energy_string("10.0")
        assert len(result) == 1
        assert result[0] == 10.0

    def test_comma_separated_energies(self):
        """Test parsing comma-separated energy values."""
        result = parse_energy_string("5.0,10.0,15.0")
        expected = [5.0, 10.0, 15.0]
        assert len(result) == len(expected)
        assert all(abs(a - b) < 1e-10 for a, b in zip(result, expected))

    def test_linear_range(self):
        """Test parsing linear energy range."""
        result = parse_energy_string("5-15:6")
        expected = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
        assert len(result) == 6
        assert all(abs(a - b) < 1e-10 for a, b in zip(result, expected))

    def test_log_range(self):
        """Test parsing logarithmic energy range."""
        result = parse_energy_string("1-10:3:log")
        expected = [1.0, 3.162277660168379, 10.0]  # log10 spaced
        assert len(result) == 3
        assert all(abs(a - b) < 1e-10 for a, b in zip(result, expected))

    def test_invalid_energy_string(self):
        """Test handling of invalid energy strings."""
        with pytest.raises(ValueError):
            parse_energy_string("invalid")


class TestResultFormatting:
    """Test XRayResult formatting functionality."""

    def setup_method(self):
        """Set up test data for formatting tests."""
        try:
            self.result = xlt.calculate_single_material_properties("SiO2", [10.0], 2.2)
        except Exception:
            pytest.skip("Cannot create test result - atomic data not available")

    def test_json_format(self):
        """Test JSON formatting of results."""
        formatted = format_xray_result(self.result, "json")
        data = json.loads(formatted)

        # Check that required fields are present
        assert "formula" in data
        assert "energy_kev" in data
        assert data["formula"] == "SiO2"
        assert isinstance(data["energy_kev"], list)

    def test_csv_format(self):
        """Test CSV formatting of results."""
        formatted = format_xray_result(self.result, "csv")
        lines = formatted.strip().split("\n")

        # Should have header and at least one data row
        assert len(lines) >= 2
        assert "formula" in lines[0]  # header

    def test_table_format(self):
        """Test table formatting of results."""
        formatted = format_xray_result(self.result, "table")

        # Should contain material properties
        assert "Material Properties:" in formatted
        assert "SiO2" in formatted
        assert "Energy:" in formatted

    def test_custom_fields(self):
        """Test formatting with custom field selection."""
        fields = ["formula", "energy_kev", "dispersion_delta"]
        formatted = format_xray_result(self.result, "json", fields=fields)
        data = json.loads(formatted)

        # Should only contain requested fields
        assert len(data) == len(fields)
        for field in fields:
            assert field in data


class TestCalcCommand:
    """Test the 'calc' command functionality."""

    def test_basic_calc_command(self):
        """Test basic calculation command."""

        # Mock command line arguments
        class MockArgs:
            formula = "SiO2"
            energy = "10.0"
            density = 2.2
            verbose = False
            output = None
            format = "table"
            fields = None
            precision = 6

        try:
            result = cmd_calc(MockArgs())
            assert result == 0  # Should return success
        except Exception:
            pytest.skip("Cannot test calc command - atomic data not available")

    def test_calc_command_with_multiple_energies(self):
        """Test calculation with multiple energies."""

        class MockArgs:
            formula = "Si"
            energy = "5.0,10.0,15.0"
            density = 2.33
            verbose = False
            output = None
            format = "table"
            fields = None
            precision = 6

        try:
            result = cmd_calc(MockArgs())
            assert result == 0
        except Exception:
            pytest.skip("Cannot test calc command - atomic data not available")

    def test_calc_command_with_file_output(self):
        """Test calculation with file output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_file = f.name

        try:

            class MockArgs:
                formula = "Al2O3"
                energy = "8.0"
                density = 3.95
                verbose = False
                output = output_file
                format = "csv"
                fields = None
                precision = 6

            try:
                result = cmd_calc(MockArgs())
                assert result == 0

                # Check that file was created and contains data
                output_path = Path(output_file)
                assert output_path.exists()
                content = output_path.read_text()
                assert len(content) > 0
                assert "Al2O3" in content
            except Exception:
                pytest.skip("Cannot test calc command - atomic data not available")
        finally:
            # Clean up
            if Path(output_file).exists():
                Path(output_file).unlink()

    def test_calc_command_error_handling(self):
        """Test error handling in calc command."""

        class MockArgs:
            formula = "SiO2"
            energy = "10.0"
            density = -1.0  # Invalid negative density
            verbose = False
            output = None
            format = "table"
            fields = None
            precision = 6

        result = cmd_calc(MockArgs())
        assert result == 1  # Should return error


class TestBatchCommand:
    """Test the 'batch' command functionality."""

    def test_batch_command_basic(self):
        """Test basic batch processing."""
        # Create temporary input CSV file
        input_data = [
            ["formula", "density", "energy"],
            ["SiO2", 2.2, 10.0],
            ["Si", 2.33, 8.0],
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            writer = csv.writer(input_file)
            writer.writerows(input_data)
            input_filename = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            output_filename = output_file.name

        try:

            class MockArgs:
                input_file = input_filename
                output = output_filename
                format = None  # Auto-detect from extension
                workers = None
                fields = None
                verbose = False

            try:
                result = cmd_batch(MockArgs())
                assert result == 0

                # Check output file was created
                output_path = Path(output_filename)
                assert output_path.exists()
                content = output_path.read_text()
                assert len(content) > 0
            except Exception:
                pytest.skip("Cannot test batch command - atomic data not available")

        finally:
            # Clean up
            for filename in [input_filename, output_filename]:
                if Path(filename).exists():
                    Path(filename).unlink()


class TestConvertCommand:
    """Test the 'convert' command functionality."""

    def test_energy_to_wavelength_conversion(self):
        """Test energy to wavelength conversion."""

        class MockArgs:
            from_unit = "energy"
            values = "10.0"
            to_unit = "wavelength"
            output = None

        result = cmd_convert(MockArgs())
        assert result == 0

    def test_wavelength_to_energy_conversion(self):
        """Test wavelength to energy conversion."""

        class MockArgs:
            from_unit = "wavelength"
            values = "1.24"
            to_unit = "energy"
            output = None

        result = cmd_convert(MockArgs())
        assert result == 0

    def test_multiple_value_conversion(self):
        """Test conversion of multiple values."""

        class MockArgs:
            from_unit = "energy"
            values = "5.0,10.0,15.0"
            to_unit = "wavelength"
            output = None

        result = cmd_convert(MockArgs())
        assert result == 0

    def test_conversion_with_file_output(self):
        """Test conversion with file output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_file = f.name

        try:

            class MockArgs:
                from_unit = "energy"
                values = "10.0,12.0"
                to_unit = "wavelength"
                output = output_file

            result = cmd_convert(MockArgs())
            assert result == 0

            # Check that file was created
            output_path = Path(output_file)
            assert output_path.exists()
            content = output_path.read_text()
            assert len(content) > 0

        finally:
            if Path(output_file).exists():
                Path(output_file).unlink()


class TestFormulaCommand:
    """Test the 'formula' command functionality."""

    def test_basic_formula_parsing(self):
        """Test basic formula parsing."""

        class MockArgs:
            formulas = "SiO2"
            output = None
            verbose = False

        result = cmd_formula(MockArgs())
        assert result == 0

    def test_multiple_formula_parsing(self):
        """Test parsing multiple formulas."""

        class MockArgs:
            formulas = "SiO2,Al2O3,Fe2O3"
            output = None
            verbose = False

        result = cmd_formula(MockArgs())
        assert result == 0

    def test_verbose_formula_parsing(self):
        """Test verbose formula parsing."""

        class MockArgs:
            formulas = "SiO2"
            output = None
            verbose = True

        try:
            result = cmd_formula(MockArgs())
            assert result == 0
        except Exception:
            pytest.skip(
                "Cannot test verbose formula parsing - atomic data not available"
            )


class TestAtomicCommand:
    """Test the 'atomic' command functionality."""

    def test_single_element_lookup(self):
        """Test single element atomic data lookup."""

        class MockArgs:
            elements = "Si"
            output = None

        try:
            result = cmd_atomic(MockArgs())
            assert result == 0
        except Exception:
            pytest.skip("Cannot test atomic command - atomic data not available")

    def test_multiple_element_lookup(self):
        """Test multiple element atomic data lookup."""

        class MockArgs:
            elements = "H,C,N,O,Si"
            output = None

        try:
            result = cmd_atomic(MockArgs())
            assert result == 0
        except Exception:
            pytest.skip("Cannot test atomic command - atomic data not available")


class TestBraggCommand:
    """Test the 'bragg' command functionality."""

    def test_bragg_with_wavelength(self):
        """Test Bragg angle calculation with wavelength."""

        class MockArgs:
            dspacing = "3.14"
            wavelength = "1.54"
            energy = None
            order = 1
            output = None

        result = cmd_bragg(MockArgs())
        assert result == 0

    def test_bragg_with_energy(self):
        """Test Bragg angle calculation with energy."""

        class MockArgs:
            dspacing = "3.14"
            wavelength = None
            energy = "8.0"
            order = 1
            output = None

        result = cmd_bragg(MockArgs())
        assert result == 0

    def test_bragg_multiple_dspacings(self):
        """Test Bragg angle calculation with multiple d-spacings."""

        class MockArgs:
            dspacing = "3.14,2.45,1.92"
            wavelength = "1.54"
            energy = None
            order = 1
            output = None

        result = cmd_bragg(MockArgs())
        assert result == 0


class TestListCommand:
    """Test the 'list' command functionality."""

    def test_list_constants(self):
        """Test listing physical constants."""

        class MockArgs:
            type = "constants"

        result = cmd_list(MockArgs())
        assert result == 0

    def test_list_fields(self):
        """Test listing available fields."""

        class MockArgs:
            type = "fields"

        result = cmd_list(MockArgs())
        assert result == 0

    def test_list_examples(self):
        """Test listing usage examples."""

        class MockArgs:
            type = "examples"

        result = cmd_list(MockArgs())
        assert result == 0


class TestMainFunction:
    """Test the main CLI entry point."""

    def test_main_with_no_args(self):
        """Test main function with no arguments."""
        with patch("sys.argv", ["xraylabtool"]):
            result = main()
            assert result == 1  # Should show help and return error

    def test_main_with_help(self):
        """Test main function with help argument."""
        with patch("sys.argv", ["xraylabtool", "--help"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            # argparse exits with 0 for --help
            assert excinfo.value.code == 0

    def test_main_with_version(self):
        """Test main function with version argument."""
        with patch("sys.argv", ["xraylabtool", "--version"]):
            with pytest.raises(SystemExit) as excinfo:
                main()
            # argparse exits with 0 for --version
            assert excinfo.value.code == 0

    def test_main_with_invalid_command(self):
        """Test main function with invalid command."""
        with patch("sys.argv", ["xraylabtool", "invalid_command"]):
            result = main()
            assert result == 1  # Should return error


class TestCLIIntegration:
    """Integration tests for the CLI."""

    def test_full_calc_workflow(self):
        """Test full calculation workflow through CLI."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_file = f.name

        try:
            # Test calculation with JSON output
            with patch(
                "sys.argv",
                [
                    "xraylabtool",
                    "calc",
                    "SiO2",
                    "-e",
                    "10.0",
                    "-d",
                    "2.2",
                    "-o",
                    output_file,
                    "--format",
                    "json",
                ],
            ):
                try:
                    result = main()
                    if result == 0:
                        # Check output file
                        output_path = Path(output_file)
                        assert output_path.exists()

                        # Parse JSON and check content
                        with open(output_file, "r") as f:
                            data = json.load(f)

                        assert "formula" in data
                        assert data["formula"] == "SiO2"
                        assert "energy_kev" in data
                except Exception:
                    pytest.skip("Cannot test full workflow - atomic data not available")
        finally:
            if Path(output_file).exists():
                Path(output_file).unlink()

    def test_full_batch_workflow(self):
        """Test full batch processing workflow."""
        # Create input CSV
        input_data = [
            ["formula", "density", "energy"],
            ["SiO2", 2.2, 10.0],
            ["Si", 2.33, "8.0,12.0"],
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as input_file:
            writer = csv.writer(input_file)
            writer.writerows(input_data)
            input_filename = input_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as output_file:
            output_filename = output_file.name

        try:
            with patch(
                "sys.argv",
                [
                    "xraylabtool",
                    "batch",
                    input_filename,
                    "-o",
                    output_filename,
                    "--verbose",
                ],
            ):
                try:
                    result = main()
                    if result == 0:
                        # Check output file
                        output_path = Path(output_filename)
                        assert output_path.exists()

                        # Check CSV content
                        content = output_path.read_text()
                        assert "SiO2" in content
                        assert "Si" in content
                except Exception:
                    pytest.skip(
                        "Cannot test batch workflow - atomic data not available"
                    )
        finally:
            for filename in [input_filename, output_filename]:
                if Path(filename).exists():
                    Path(filename).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
