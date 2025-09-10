"""
File operations for XRayLabTool.

This module contains functions for loading and saving data files,
including atomic scattering factor data and calculation results.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from xraylabtool.exceptions import DataFileError


def load_data_file(filename: str) -> pd.DataFrame:
    """
    Load data file with error handling.

    Args:
        filename: Path to the data file

    Returns:
        DataFrame containing the loaded data

    Raises:
        ~xraylabtool.validation.exceptions.DataFileError: If file cannot be loaded or parsed
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        # Try to load as space-separated values (common for .nff files)
        if file_path.suffix == ".nff":
            data = pd.read_csv(filename, sep=r"\s+", comment="#", header=None)
        else:
            data = pd.read_csv(filename)

        if data.empty:
            raise pd.errors.EmptyDataError("File contains no data")

        return data

    except pd.errors.ParserError as e:
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
        format_type: Output format ('csv', 'json', 'xlsx')
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format_type.lower() == "csv":
        if hasattr(results, "to_csv"):
            results.to_csv(filename, index=False)
        else:
            # Convert to DataFrame if needed
            df = pd.DataFrame(results) if isinstance(results, dict) else results
            df.to_csv(filename, index=False)
    elif format_type.lower() == "json":
        if hasattr(results, "to_json"):
            results.to_json(filename, orient="records", indent=2)
        else:
            import json

            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def export_to_csv(data: Any, filename: str, **kwargs: Any) -> None:  # noqa: ARG001
    """Export data to CSV format."""
    save_calculation_results(data, filename, format_type="csv")


def export_to_json(data: Any, filename: str, **kwargs: Any) -> None:  # noqa: ARG001
    """Export data to JSON format."""
    save_calculation_results(data, filename, format_type="json")
