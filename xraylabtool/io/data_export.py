"""
Data export and formatting utilities for XRayLabTool.

This module contains functions for formatting and exporting calculation results
in various formats suitable for different use cases.
"""

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _result_fields(result: Any) -> list[str]:
    """Select result data without evaluating deprecated dataclass aliases."""
    if is_dataclass(result) and not isinstance(result, type):
        derived = ("energy_ev", "critical_angle_mrad", "linear_absorption_coefficient")
        return [field.name for field in dataclass_fields(result)] + [
            name for name in derived if hasattr(result, name)
        ]
    return [
        attr
        for attr in dir(result)
        if not attr.startswith("_") and not callable(getattr(result, attr))
    ]


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


def _format_as_json(result: Any, fields: list[str] | None, precision: int) -> str:  # noqa: ARG001  (precision applies to tables only)
    """Format result as JSON string."""
    data = {}

    # Get all attributes if fields not specified
    # (exclude methods and private attributes)
    if fields is None:
        fields = _result_fields(result)

    for field in fields:
        if hasattr(result, field):
            value = getattr(result, field)
            if isinstance(value, np.ndarray):
                data[field] = value.tolist()
            elif isinstance(value, float | np.floating):
                data[field] = float(value)
            else:
                data[field] = value

    return json.dumps(data, indent=2)


def _format_as_csv(result: Any, fields: list[str] | None, precision: int) -> str:  # noqa: ARG001  (precision applies to tables only)
    """Format result as CSV string."""
    return _result_dataframe(result, fields).to_csv(index=False)


def _result_dataframe(result: Any, fields: list[str] | None = None) -> "pd.DataFrame":
    """Build one row per energy, repeating material metadata on each row."""
    if fields is None:
        fields = _result_fields(result)

    unknown = [field for field in fields if not hasattr(result, field)]
    if unknown:
        raise ValueError(f"Unknown result fields: {unknown}")

    # Create DataFrame for consistent formatting
    data: dict[str, Any] = {}
    for field in fields:
        value = getattr(result, field)
        if isinstance(value, np.ndarray):
            data[field] = value
        elif isinstance(value, float | np.floating):
            data[field] = [float(value)]
        else:
            data[field] = [value] if not isinstance(value, list | np.ndarray) else value

    # Ensure all arrays have the same length
    max_length = max(
        (len(v) if isinstance(v, list | np.ndarray) else 1 for v in data.values()),
        default=0,
    )
    for key, value in data.items():
        if not isinstance(value, list | np.ndarray):
            data[key] = [value] * max_length
        elif len(value) == 1 and max_length > 1:
            data[key] = [value[0]] * max_length

    # Lazy import pandas only when needed
    import pandas as pd

    return pd.DataFrame(data)


def _format_as_table(result: Any, fields: list[str] | None, precision: int) -> str:
    """Format result as human-readable table."""
    if fields is None:
        fields = _result_fields(result)

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
        # Lazy import pandas only when needed
        import pandas as pd

        df = pd.concat(
            [_result_dataframe(result) for result in results], ignore_index=True
        )
        return df.to_csv(index=False)
    else:
        # Table format - show summary statistics
        lines = [f"Summary of {len(results)} calculations", "=" * 40]
        for i, result in enumerate(results, 1):
            formula = getattr(result, "formula", f"Result {i}")
            lines.append(f"{i}. {formula}")
        return "\n".join(lines)
