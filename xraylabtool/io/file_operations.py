"""
File operations for XRayLabTool.

This module contains functions for loading and saving data files,
including atomic scattering factor data and calculation results.
"""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from xraylabtool.exceptions import DataFileError


def load_data_file(filename: str) -> np.ndarray:
    """
    Load data file with error handling.

    Args:
        filename: Path to the data file

    Returns:
        Numpy array containing the loaded data

    Raises:
        ~xraylabtool.validation.exceptions.DataFileError: If file cannot be loaded or parsed
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        # Try to load as space-separated values (common for .nff files)
        if file_path.suffix == ".nff":
            # .nff files may have a CSV header row; try skipping it first
            try:
                data = np.loadtxt(filename, delimiter=",", skiprows=1)
            except ValueError:
                # Fall back to space-separated without a header
                data = np.loadtxt(filename, comments="#")
        else:
            # For CSV files, use numpy's CSV loader
            try:
                data = np.loadtxt(filename, delimiter=",", skiprows=1)  # Skip header
            except ValueError:
                # If CSV loading fails, try space-separated
                data = np.loadtxt(filename, comments="#")

        if data.size == 0:
            raise ValueError("File contains no data")

        return data

    except (ValueError, OSError) as e:
        raise DataFileError(f"Error parsing file {filename}: {e}", filename) from e
    except Exception as e:
        raise DataFileError(
            f"Unexpected error loading file {filename}: {e}", filename
        ) from e


def save_calculation_results(
    results: Any, filename: str, format_type: str = "csv"
) -> None:
    """
    Save calculation results to file.

    Args:
        results: Calculation results to save
        filename: Output file path
        format_type: Output format ('csv', 'json')
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type.lower() == "csv":
        if hasattr(results, "to_csv"):
            results.to_csv(filename, index=False)
        # Handle different data types efficiently
        elif isinstance(results, dict):
            # Convert dict to CSV using csv module
            with open(filename, "w", newline="") as f:
                if results:
                    fieldnames = results.keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    # Handle case where values are lists/arrays
                    first_key = next(iter(results))
                    if isinstance(results[first_key], (list, np.ndarray)):
                        # Multiple rows case (optimized: vectorized operations)
                        n_rows = len(results[first_key])

                        # Pre-convert arrays to lists for efficient indexing
                        array_data = {}
                        for k, v in results.items():
                            if isinstance(v, np.ndarray):
                                array_data[k] = v.tolist()
                            else:
                                array_data[k] = (
                                    list(v) if hasattr(v, "__iter__") else [v] * n_rows
                                )

                        # Vectorized row generation
                        rows = [
                            {k: array_data[k][i] for k in array_data}
                            for i in range(n_rows)
                        ]
                        writer.writerows(rows)
                    else:
                        # Single row case
                        writer.writerow(results)
        elif isinstance(results, np.ndarray):
            # Save numpy array directly
            np.savetxt(filename, results, delimiter=",", fmt="%.6g")
        else:
            raise ValueError(f"Unsupported data type for CSV export: {type(results)}")
    elif format_type.lower() == "json":
        if hasattr(results, "to_json"):
            results.to_json(filename, orient="records", indent=2)
        else:
            import json

            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj: Any) -> Any:
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return str(obj)

            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=convert_numpy)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def export_to_csv(
    data: Any, filename: str, fields: list[str] | None = None, **_kwargs: Any
) -> None:
    """Export data to CSV format."""
    from xraylabtool.calculators.core import XRayResult

    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], XRayResult)):
        if not data:
            return
        if fields is None:
            fields = [
                "formula",
                "energy_kev",
                "critical_angle_degrees",
                "attenuation_length_cm",
                "dispersion_delta",
                "absorption_beta",
            ]
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            n_energies = len(data[0].energy_kev)
            header_parts = []
            for field in fields[1:]:
                header_parts.extend([f"{field}_energy_{i}" for i in range(n_energies)])
            header = ["formula", *header_parts]
            writer.writerow(header)
            for result in data:
                row = [result.formula]
                for field in fields[1:]:
                    values = getattr(result, field, [])
                    if hasattr(values, "__iter__") and not isinstance(values, str):
                        row.extend(values)
                    else:
                        row.append(values)
                writer.writerow(row)
        return

    save_calculation_results(data, filename, format_type="csv")


def export_to_json(data: Any, filename: str, **_kwargs: Any) -> None:
    """Export data to JSON format."""
    from xraylabtool.calculators.core import XRayResult

    if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], XRayResult)):
        res_list = []
        for result in data:
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
            res_list.append(item)
        import json

        with open(filename, "w") as f:
            json.dump(res_list, f, indent=2)
        return

    save_calculation_results(data, filename, format_type="json")
