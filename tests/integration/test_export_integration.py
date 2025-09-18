#!/usr/bin/env python3
"""
Integration tests for XRayLabTool basic export functionality.

This module tests the integration between basic export functions and the
XRayLabTool calculation pipeline.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.export import export_to_csv, export_to_json


class TestBasicExportIntegration(unittest.TestCase):
    """Test integration between calculations and basic export functionality."""

    def setUp(self):
        """Set up test fixtures with real calculation data."""
        self.formula = "SiO2"
        self.density = 2.2
        self.energies = np.array([5.0, 10.0, 15.0, 20.0])

        # Generate test results
        self.results = []
        for energy in self.energies:
            result = calculate_single_material_properties(
                self.formula, energy, self.density
            )
            self.results.append(result)

    def test_csv_export_integration(self):
        """Test CSV export with real calculation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.csv"

            # Export to CSV
            export_to_csv(self.results, output_path)

            # Verify file was created and has content
            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertIn("SiO2", content)
            self.assertIn("critical_angle_degrees", content)

    def test_json_export_integration(self):
        """Test JSON export with real calculation results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.json"

            # Export to JSON
            export_to_json(self.results, output_path)

            # Verify file was created and has valid JSON
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                data = json.load(f)

            self.assertEqual(len(data), len(self.results))
            self.assertEqual(data[0]["formula"], "SiO2")
            self.assertEqual(data[0]["density_g_cm3"], 2.2)
            self.assertIn("critical_angle_degrees", data[0])

    def test_export_with_empty_results(self):
        """Test export functions handle empty results gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "empty.csv"
            json_path = Path(temp_dir) / "empty.json"

            # Should not raise errors
            export_to_csv([], csv_path)
            export_to_json([], json_path)

            # JSON file should contain empty array
            with open(json_path) as f:
                data = json.load(f)
            self.assertEqual(data, [])


if __name__ == "__main__":
    unittest.main()
