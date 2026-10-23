"""
Basic analysis functions for X-ray optical properties.

Simplified analysis functionality focused on core scientific calculations.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from xraylabtool.calculators.core import XRayResult

from .comparator import MaterialComparator


def find_absorption_edges(
    energies: np.ndarray, f2_values: np.ndarray, threshold: float = 0.1
) -> list[tuple[float, float]]:
    """
    Simple absorption edge detection using f2 derivative.

    Args:
        energies: Energy array in keV
        f2_values: Imaginary scattering factor values
        threshold: Minimum derivative threshold for edge detection

    Returns:
        list of (energy, derivative_magnitude) tuples for detected edges
    """
    if len(energies) < 3:
        return []

    # Calculate derivative
    derivative = np.gradient(f2_values, energies)

    # Find peaks in derivative
    peak_indices = []
    for i in range(1, len(derivative) - 1):
        if (
            derivative[i] > derivative[i - 1]
            and derivative[i] > derivative[i + 1]
            and derivative[i] > threshold
        ):
            peak_indices.append(i)

    return [(energies[i], derivative[i]) for i in peak_indices]


def compare_materials(
    results: list[XRayResult], property_name: str = "critical_angle_degrees"
) -> dict:  # type: ignore[type-arg]
    """
    Simple material comparison for a given property.

    Args:
        results: list of XRayResult objects to compare
        property_name: Property to compare across materials

    Returns:
        Dictionary with basic statistics
    """
    if not results:
        return {}

    values: list[float] = []
    for result in results:
        prop_value = getattr(result, property_name, None)
        if prop_value is not None:
            if isinstance(prop_value, np.ndarray):
                values.extend(prop_value.flatten())
            else:
                values.append(prop_value)

    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


__all__ = ["MaterialComparator", "compare_materials", "find_absorption_edges"]
