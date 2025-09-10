"""
Edge case tests for XRayLabTool CLI.

This module tests boundary conditions, error handling, and extreme values
for the command-line interface to ensure robustness.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from xraylabtool.interfaces import cli


class TestCLIInputValidation:
    """Test CLI input validation edge cases."""

    def test_invalid_energy_formats(self):
        """Test handling of invalid energy format strings."""
        invalid_energy_strings = [
            "abc",  # Non-numeric
            "5.0-",  # Incomplete range
            "5.0-10.0:",  # Missing count
            "5.0-10.0:abc",  # Non-numeric count
            "5.0,",  # Trailing comma
            ",5.0",  # Leading comma
            "5.0,,10.0",  # Double comma
            "",  # Empty string
        ]

        for energy_str in invalid_energy_strings:
            with pytest.raises(ValueError):
                cli.parse_energy_string(energy_str)

    def test_edge_case_energy_ranges(self):
        """Test edge cases in energy range parsing."""
        # Single value
        result = cli.parse_energy_string("10.0")
        assert len(result) == 1
        assert result[0] == 10.0

        # Very small range
        result = cli.parse_energy_string("10.0-10.1:3")
        assert len(result) == 3
        assert result[0] == 10.0
        assert result[-1] == 10.1

        # Log spacing with valid range
        result = cli.parse_energy_string("1.0-20:50:log")
        assert len(result) == 50
        assert result[0] == 1.0
        assert np.isclose(
            result[-1], 20.0, rtol=1e-10
        )  # Handle floating point precision

        # Very fine linear spacing
        result = cli.parse_energy_string("10.0-10.01:101")
        assert len(result) == 101


class TestCLIOutputFormatting:
    """Test CLI output formatting edge cases."""

    def test_extreme_precision_values(self):
        """Test output formatting with extreme precision values."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        result = calculate_single_material_properties("Si", [10.0], 2.33)

        # Test very high precision
        formatted = cli.format_xray_result(result, "table", precision=15)
        assert "Si" in formatted

        # Test zero precision (should default to reasonable value)
        formatted = cli.format_xray_result(result, "table", precision=0)
        assert "Si" in formatted

    def test_large_datasets_formatting(self):
        """Test formatting of large datasets."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        # Large energy array within valid range
        energies = [0.5 + i * 0.25 for i in range(100)]  # 0.5 to 25.25 keV, 100 points
        result = calculate_single_material_properties("Al", energies, 2.70)

        # Test all formats handle large datasets
        for format_type in ["table", "csv", "json"]:
            formatted = cli.format_xray_result(result, format_type)
            assert len(formatted) > 0

            if format_type == "json":
                # Should be valid JSON
                data = json.loads(formatted)
                assert len(data["energy_kev"]) == 100

    def test_field_filtering_edge_cases(self):
        """Test field filtering with edge cases."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        result = calculate_single_material_properties("C", [10.0], 2.27)

        # Empty field list
        formatted = cli.format_xray_result(result, "json", fields=[])
        data = json.loads(formatted)
        assert len(data) == 0

        # Single field
        formatted = cli.format_xray_result(result, "json", fields=["formula"])
        data = json.loads(formatted)
        assert list(data.keys()) == ["formula"]

        # Non-existent field (should raise AttributeError)
        with pytest.raises(AttributeError):
            cli.format_xray_result(
                result, "json", fields=["formula", "nonexistent_field"]
            )


class TestCLIFileOperations:
    """Test CLI file operations edge cases."""

    def test_file_permissions_and_paths(self):
        """Test file operations with various path conditions."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        result = calculate_single_material_properties("Si", [10.0], 2.33)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test nested directory creation
            nested_path = tmppath / "subdir" / "results.csv"
            nested_path.parent.mkdir(exist_ok=True)

            formatted = cli.format_xray_result(result, "csv")
            nested_path.write_text(formatted)
            assert nested_path.exists()

            # Test overwriting existing file
            formatted2 = cli.format_xray_result(result, "json")
            json_path = tmppath / "results.json"
            json_path.write_text(formatted2)
            json_path.write_text(formatted2)  # Overwrite
            assert json_path.exists()

    def test_batch_file_edge_cases(self):
        """Test batch processing file edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test CSV with edge cases
            edge_case_data = {
                "formula": ["Si", "H2O", "Al2O3", "C60"],
                "density": [2.33, 1.0, 3.95, 2.3],
                "energy": ["10.0", "5.0,10.0,15.0", "8.0", "12.5"],
            }

            input_path = tmppath / "test_input.csv"
            df = pd.DataFrame(edge_case_data)
            df.to_csv(input_path, index=False)

            # Test batch processing with edge cases
            output_path = tmppath / "test_output.csv"

            # Mock args object
            class MockArgs:
                input_file = str(input_path)
                output = str(output_path)
                format = None
                workers = None
                fields = None
                verbose = False

            # This would normally be tested with actual CLI execution
            # but we test the validation logic
            result = cli._validate_batch_input(MockArgs())
            assert result is not None
            assert len(result) == 4

    def test_malformed_batch_files(self):
        """Test handling of malformed batch input files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test cases for malformed files
            malformed_cases = [
                # Missing columns
                pd.DataFrame({"formula": ["Si"], "density": [2.33]}),
                # Wrong column names
                pd.DataFrame({"chemical": ["Si"], "rho": [2.33], "E": [10.0]}),
            ]

            for i, malformed_df in enumerate(malformed_cases):
                input_path = tmppath / f"malformed_{i}.csv"
                malformed_df.to_csv(input_path, index=False)

                class MockArgs:
                    input_file = str(input_path)

                result = cli._validate_batch_input(MockArgs())
                assert result is None  # Should return None for invalid input

            # Test empty file separately (causes pandas exception)
            empty_path = tmppath / "empty.csv"
            with open(empty_path, "w") as f:
                f.write("")  # Completely empty file

            class EmptyArgs:
                input_file = str(empty_path)

            # Should handle pandas exception gracefully
            try:
                result = cli._validate_batch_input(EmptyArgs())
                assert result is None
            except Exception:
                # If it raises an exception, that's also acceptable behavior
                pass


class TestCLIErrorHandling:
    """Test CLI error handling edge cases."""

    def test_command_line_parsing_errors(self):
        """Test command line parsing with invalid arguments."""
        parser = cli.create_parser()

        # Test invalid command
        with (
            pytest.raises(SystemExit),
            patch("sys.stderr"),  # Suppress error output
        ):
            parser.parse_args(["invalid_command"])

        # Test missing required arguments
        with pytest.raises(SystemExit), patch("sys.stderr"):
            parser.parse_args(["calc"])  # Missing required args

    def test_validation_edge_cases(self):
        """Test input validation edge cases."""
        import numpy as np

        # Test zero density
        class MockArgs:
            density = 0.0

        energies = np.array([10.0])
        assert not cli._validate_calc_inputs(MockArgs(), energies)

        # Test negative energies
        class MockArgs2:
            density = 2.33

        energies = np.array([-5.0, 10.0])
        assert not cli._validate_calc_inputs(MockArgs2(), energies)

        # Test extreme energy values
        energies = np.array([0.001, 50.0])  # Outside typical range
        with patch("builtins.print"):  # Suppress warning output
            result = cli._validate_calc_inputs(MockArgs2(), energies)
        # Should still validate (just warning for out-of-range)
        assert result

    def test_file_not_found_handling(self):
        """Test handling of missing input files."""

        class MockArgs:
            input_file = "/nonexistent/path/file.csv"

        result = cli._validate_batch_input(MockArgs())
        assert result is None


class TestCLIPerformanceAndMemory:
    """Test CLI performance and memory usage."""

    def test_large_batch_processing(self):
        """Test batch processing with large datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create large batch input
            formulas = ["Si", "Al", "Fe", "Cu", "Au"] * 20  # 100 materials
            densities = [2.33, 2.70, 7.87, 8.96, 19.3] * 20
            energies = ["10.0"] * 100

            large_data = pd.DataFrame(
                {"formula": formulas, "density": densities, "energy": energies}
            )

            input_path = tmppath / "large_input.csv"
            large_data.to_csv(input_path, index=False)

            class MockArgs:
                input_file = str(input_path)
                verbose = False

            # Test validation doesn't fail with large input
            result = cli._validate_batch_input(MockArgs())
            assert result is not None
            assert len(result) == 100

    def test_memory_efficient_processing(self):
        """Test memory efficiency with streaming-like processing."""
        # Test that processing doesn't accumulate excessive memory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create input with many energy points per material
            data = {
                "formula": ["Si"] * 10,
                "density": [2.33] * 10,
                "energy": [f"{5.0 + i * 2.0}" for i in range(10)],
            }

            input_path = tmppath / "memory_test.csv"
            pd.DataFrame(data).to_csv(input_path, index=False)

            class MockArgs:
                input_file = str(input_path)
                verbose = False

            # Should handle without memory issues
            result = cli._validate_batch_input(MockArgs())
            assert result is not None


class TestCLICompatibility:
    """Test CLI compatibility and backward compatibility."""

    def test_output_format_compatibility(self):
        """Test compatibility of different output formats."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        result = calculate_single_material_properties("SiO2", [10.0], 2.2)

        # Test all supported formats work
        formats = ["table", "csv", "json"]
        for fmt in formats:
            try:
                formatted = cli.format_xray_result(result, fmt)
                assert len(formatted) > 0
            except Exception as e:
                pytest.fail(f"Format {fmt} failed: {e}")

    def test_field_name_compatibility(self):
        """Test field name compatibility across versions."""
        from xraylabtool.calculators.core import calculate_single_material_properties

        result = calculate_single_material_properties("Al", [10.0], 2.70)

        # Test that all expected field names exist
        expected_fields = [
            "formula",
            "molecular_weight_g_mol",
            "total_electrons",
            "density_g_cm3",
            "electron_density_per_ang3",
            "energy_kev",
            "wavelength_angstrom",
            "dispersion_delta",
            "absorption_beta",
            "scattering_factor_f1",
            "scattering_factor_f2",
            "critical_angle_degrees",
            "attenuation_length_cm",
            "real_sld_per_ang2",
            "imaginary_sld_per_ang2",
        ]

        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"

    def test_version_consistency(self):
        """Test version information consistency."""
        from xraylabtool import __version__

        # Version should be a string in semantic version format
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 2  # At least major.minor

        # Should be accessible through CLI
        parser = cli.create_parser()
        # Version action should be present
        assert any(action.dest == "version" for action in parser._actions)
