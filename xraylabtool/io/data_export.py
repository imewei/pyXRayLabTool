"""
Data export and formatting utilities for XRayLabTool.

This module contains functions for formatting and exporting calculation results
in various formats suitable for different use cases.
"""

import json
from typing import Any

import numpy as np
import pandas as pd


def format_xray_result(
    result: Any,
    format_type: str = "table",
    fields: list[str] | None = None,
    precision: int = 6,
) -> str:
    """
    Format XRayResult for display or export.

    Args:
        result: XRayResult object to format
        format_type: Output format ('table', 'csv', 'json')
        fields: Specific fields to include (None for all)
        precision: Decimal precision for numeric values

    Returns:
        Formatted string representation
    """
    if format_type.lower() == "json":
        return _format_as_json(result, fields, precision)
    elif format_type.lower() == "csv":
        return _format_as_csv(result, fields, precision)
    else:  # table format
        return _format_as_table(result, fields, precision)


def _format_as_json(result: Any, fields: list[str] | None, precision: int) -> str:
    """Format result as JSON string."""
    data = {}

    # Get all attributes if fields not specified
    # (exclude methods and private attributes)
    if fields is None:
        fields = [
            attr
            for attr in dir(result)
            if not attr.startswith("_") and not callable(getattr(result, attr))
        ]

    for field in fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists with proper precision
                data[field] = np.round(value, precision).tolist()
            elif isinstance(value, float | np.floating):
                data[field] = round(float(value), precision)
            else:
                data[field] = value

    return json.dumps(data, indent=2)


def _format_as_csv(result: Any, fields: list[str] | None, precision: int) -> str:
    """Format result as CSV string."""
    if fields is None:
        fields = [
            attr
            for attr in dir(result)
            if not attr.startswith("_") and not callable(getattr(result, attr))
        ]

    # Create DataFrame for consistent formatting
    data: dict[str, Any] = {}
    for field in fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                data[field] = np.round(value, precision)
            elif isinstance(value, float | np.floating):
                data[field] = [round(float(value), precision)]
            else:
                data[field] = (
                    [value] if not isinstance(value, list | np.ndarray) else value
                )

    # Ensure all arrays have the same length
    max_length = max(
        len(v) if isinstance(v, list | np.ndarray) else 1 for v in data.values()
    )
    for key, value in data.items():
        if not isinstance(value, list | np.ndarray):
            data[key] = [value] * max_length
        elif len(value) == 1 and max_length > 1:
            data[key] = [value[0]] * max_length

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def _format_as_table(result: Any, fields: list[str] | None, precision: int) -> str:
    """Format result as human-readable table."""
    if fields is None:
        fields = [
            attr
            for attr in dir(result)
            if not attr.startswith("_") and not callable(getattr(result, attr))
        ]

    lines = []
    lines.append(f"XRay Properties for {getattr(result, 'formula', 'Unknown')}")
    lines.append("=" * 50)

    for field in fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                if len(value) == 1:
                    lines.append(f"{field}: {value[0]:.{precision}f}")
                else:
                    lines.append(f"{field}: [{len(value)} values]")
                    range_str = (
                        f"  Range: {value.min():.{precision}f} - "
                        f"{value.max():.{precision}f}"
                    )
                    lines.append(range_str)
            elif isinstance(value, float | np.floating):
                lines.append(f"{field}: {value:.{precision}f}")
            else:
                lines.append(f"{field}: {value}")

    return "\n".join(lines)


def format_calculation_summary(results: list[Any], format_type: str = "table") -> str:
    """
    Format a summary of multiple calculation results.

    Args:
        results: List of XRayResult objects
        format_type: Output format type

    Returns:
        Formatted summary string
    """
    if not results:
        return "No results to display"

    if format_type.lower() == "json":
        return json.dumps(
            [json.loads(format_xray_result(r, "json")) for r in results], indent=2
        )
    elif format_type.lower() == "csv":
        # Combine all results into a single CSV
        all_data = []
        for result in results:
            result_dict = {}
            for attr in dir(result):
                if not attr.startswith("_"):
                    value = getattr(result, attr)
                    if isinstance(value, np.ndarray) and len(value) == 1:
                        result_dict[attr] = value[0]
                    elif not isinstance(value, np.ndarray):
                        result_dict[attr] = value
            all_data.append(result_dict)

        df = pd.DataFrame(all_data)
        return df.to_csv(index=False)
    else:
        # Table format - show summary statistics
        lines = [f"Summary of {len(results)} calculations", "=" * 40]
        for i, result in enumerate(results, 1):
            formula = getattr(result, "formula", f"Result {i}")
            lines.append(f"{i}. {formula}")
        return "\n".join(lines)
