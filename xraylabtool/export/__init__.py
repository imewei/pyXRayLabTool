"""
Basic data export functionality for XRayLabTool.

Simplified export capabilities for essential file formats.
"""

import csv
import json
from pathlib import Path
from typing import Any

from xraylabtool.calculators.core import XRayResult


def export_to_csv(
    results: list[XRayResult], output_path: Path, fields: list[str] | None = None
) -> None:
    """
    Export X-ray results to CSV format.

    Args:
        results: List of XRayResult objects to export
        output_path: Path to output CSV file
        fields: List of field names to export (all if None)
    """
    if not results:
        return

    # Default fields if none specified
    if fields is None:
        fields = [
            "formula",
            "energy_kev",
            "critical_angle_degrees",
            "attenuation_length_cm",
            "dispersion_delta",
            "absorption_beta",
        ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header (optimized: vectorized generation)
        n_energies = len(results[0].energy_kev)
        header_parts = []
        for field in fields[1:]:
            header_parts.extend([f"{field}_energy_{i}" for i in range(n_energies)])
        header = ["formula", *header_parts]
        writer.writerow(header)

        # Write data
        for result in results:
            row = [result.formula]
            for field in fields[1:]:
                values = getattr(result, field, [])
                if hasattr(values, "__iter__"):
                    row.extend(values)
                else:
                    row.append(values)
            writer.writerow(row)


def export_to_json(results: list[XRayResult], output_path: Path) -> None:
    """
    Export X-ray results to JSON format.

    Args:
        results: List of XRayResult objects to export
        output_path: Path to output JSON file
    """
    data = []
    for result in results:
        item = {
            "formula": result.formula,
            "molecular_weight_g_mol": result.molecular_weight_g_mol,
            "density_g_cm3": result.density_g_cm3,
            "energy_kev": result.energy_kev.tolist(),
            "critical_angle_degrees": result.critical_angle_degrees.tolist(),
            "attenuation_length_cm": result.attenuation_length_cm.tolist(),
            "dispersion_delta": result.dispersion_delta.tolist(),
            "absorption_beta": result.absorption_beta.tolist(),
        }
        data.append(item)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


__all__ = ["export_to_csv", "export_to_json"]
