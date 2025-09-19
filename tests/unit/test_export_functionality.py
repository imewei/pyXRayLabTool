#!/usr/bin/env python3
"""
Basic tests for XRayLabTool export functionality.

This module contains tests for:
- CSV export functionality
- JSON export functionality
- Basic export error handling
"""

import json
from pathlib import Path
import tempfile
import unittest

from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.export import export_to_csv, export_to_json


class TestBasicExportFunctionality(unittest.TestCase):
    """Test basic export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample results for testing
        self.result = calculate_single_material_properties("SiO2", 10.0, 2.2)
        self.results = [self.result]

    def test_export_to_csv_basic(self):
        """Test basic CSV export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.csv"

            export_to_csv(self.results, output_path)

            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertIn("SiO2", content)
            self.assertIn("formula", content)

    def test_export_to_csv_custom_fields(self):
        """Test CSV export with custom fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.csv"
            custom_fields = ["formula", "energy_kev", "critical_angle_degrees"]

            export_to_csv(self.results, output_path, fields=custom_fields)

            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertIn("critical_angle_degrees", content)

    def test_export_to_csv_empty_results(self):
        """Test CSV export with empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty.csv"

            # Should not raise an error
            export_to_csv([], output_path)

    def test_export_to_json_basic(self):
        """Test basic JSON export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.json"

            export_to_json(self.results, output_path)

            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                data = json.load(f)

            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["formula"], "SiO2")
            self.assertEqual(data[0]["density_g_cm3"], 2.2)
            self.assertIn("energy_kev", data[0])
            self.assertIn("critical_angle_degrees", data[0])

    def test_export_to_json_multiple_results(self):
        """Test JSON export with multiple results."""
        # Create multiple results
        result1 = calculate_single_material_properties("SiO2", 10.0, 2.2)
        result2 = calculate_single_material_properties("Al2O3", 10.0, 3.9)
        results = [result1, result2]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "multi.json"

            export_to_json(results, output_path)

            with open(output_path) as f:
                data = json.load(f)

            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["formula"], "SiO2")
            self.assertEqual(data[1]["formula"], "Al2O3")

    def test_export_to_json_empty_results(self):
        """Test JSON export with empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty.json"

            export_to_json([], output_path)

            with open(output_path) as f:
                data = json.load(f)

            self.assertEqual(data, [])

    def test_export_data_consistency(self):
        """Test that exported data maintains scientific accuracy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test.json"

            export_to_json(self.results, json_path)

            with open(json_path) as f:
                data = json.load(f)

            exported_result = data[0]

            # Check that critical numerical values match
            self.assertAlmostEqual(
                exported_result["critical_angle_degrees"][0],
                self.result.critical_angle_degrees[0],
                places=6,
            )
            self.assertAlmostEqual(
                exported_result["attenuation_length_cm"][0],
                self.result.attenuation_length_cm[0],
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
