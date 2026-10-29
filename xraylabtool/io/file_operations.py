"""
File operations for XRayLabTool.

This module contains functions for loading and saving data files,
including atomic scattering factor data and calculation results.
"""

import csv
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

from xraylabtool.exceptions import DataFileError
from xraylabtool.io.data_export import require_known_fields


def load_data_file(filename: str) -> np.ndarray:
    """
    Load data file with error handling.

    Args:
        filename: Path to the data file

    Returns:
        Numpy array containing the loaded data

    Raises:
        ~xraylabtool.exceptions.DataFileError: If file cannot be loaded or parsed
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    try:
        # Detect the delimiter and optional header from the first content row.
        # Retrying with skiprows=1 would silently discard a numeric first row.
        with file_path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream):  # noqa: B007 (read after the loop)
                content = line.partition("#")[0].strip()
                if content:
                    break
            else:
                raise ValueError("File contains no data")
        delimiter = "," if "," in content else None
        tokens = content.split(delimiter)
        numeric_tokens = 0
        for token in tokens:
            try:
                float(token)
                numeric_tokens += 1
            except ValueError:
                pass
        skiprows = line_number + 1 if numeric_tokens == 0 else 0
        data = np.loadtxt(
            filename, delimiter=delimiter, skiprows=skiprows, encoding="utf-8-sig"
        )

        if data.size == 0:
            raise ValueError("File contains no data")

        return data

    except (ValueError, OSError) as e:
        raise DataFileError(f"Error parsing file {filename}: {e}", filename) from e
    except Exception as e:
        raise DataFileError(
            f"Unexpected error loading file {filename}: {e}", filename
        ) from e


def _save_dict_as_csv(results: dict[str, Any], filename: str) -> None:
    """Write a dict of scalars and equal-length lists/arrays to CSV.

    Strings are scalars. Scalars are repeated on every row, so the output does not
    depend on key order. Array lengths are checked before the file is opened.
    """
    if not results:
        with open(filename, "w", newline=""):
            return

    def _is_sequence(value: Any) -> bool:
        return not isinstance(value, str | bytes) and np.ndim(value) > 0

    lengths = {len(v) for v in results.values() if _is_sequence(v)}
    if len(lengths) > 1:
        raise ValueError(
            f"Cannot write CSV: array values have different lengths {sorted(lengths)}"
        )
    n_rows = lengths.pop() if lengths else 1
    columns = {
        k: (np.asarray(v).tolist() if _is_sequence(v) else [v] * n_rows)
        for k, v in results.items()
    }
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results))
        writer.writeheader()
        writer.writerows({k: columns[k][i] for k in columns} for i in range(n_rows))


def _save_results_as_csv(results: Any, filename: str) -> None:
    """Dispatch CSV export by result type: DataFrame-like, dict, or ndarray."""
    from xraylabtool.calculators.core import XRayResult

    if hasattr(results, "to_csv"):
        results.to_csv(filename, index=False)
    elif (
        isinstance(results, dict)
        and results
        and all(isinstance(v, XRayResult) for v in results.values())
    ):
        export_to_csv(list(results.values()), filename)
    elif isinstance(results, dict):
        _save_dict_as_csv(results, filename)
    elif isinstance(results, np.ndarray):
        np.savetxt(filename, results, delimiter=",", fmt="%.6g")
    else:
        raise ValueError(f"Unsupported data type for CSV export: {type(results)}")


def _json_numpy_default(obj: Any) -> Any:
    """`json.dump(default=...)` hook: convert numpy scalars/arrays to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    return str(obj)


def _save_results_as_json(results: Any, filename: str) -> None:
    """Dispatch JSON export: DataFrame-like `.to_json`, else `json.dump`."""
    if hasattr(results, "to_json"):
        results.to_json(filename, orient="records", indent=2)
    else:
        import json

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=_json_numpy_default)


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
        _save_results_as_csv(results, filename)
    elif format_type.lower() == "json":
        _save_results_as_json(results, filename)
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
        n_energies = len(data[0].energy_kev)
        mismatched = sorted({len(r.energy_kev) for r in data} - {n_energies})
        if mismatched:
            raise ValueError(
                "Cannot export results with different numbers of energies to one wide "
                f"CSV (got {n_energies} and {mismatched}); export them separately"
            )

        require_known_fields(data[0], fields)

        def _is_array(value: Any) -> bool:
            return hasattr(value, "__iter__") and not isinstance(value, str)

        array_fields = {f for f in fields if _is_array(getattr(data[0], f))}
        header: list[str] = []
        for field in fields:
            if field in array_fields:
                header.extend(f"{field}_energy_{i}" for i in range(n_energies))
            else:
                header.append(field)
        rows = []
        for result in data:
            row: list[Any] = []
            for field in fields:
                values = getattr(result, field, [])
                if field in array_fields:
                    row.extend(values)
                else:
                    row.append(values)
            rows.append(row)
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
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
