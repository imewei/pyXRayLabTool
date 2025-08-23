"""
High-performance atomic data cache system.

This module provides a pre-populated cache of atomic data for common elements
to eliminate expensive database queries to the Mendeleev library during runtime.
"""

from typing import Dict, Tuple, List
from functools import lru_cache
from .utils import UnknownElementError

# Pre-populated atomic data for the 50 most common elements in materials science
# This eliminates the need for expensive Mendeleev database queries
_ATOMIC_DATA_PRELOADED = {
    "H": {"atomic_number": 1, "atomic_weight": 1.008},
    "He": {"atomic_number": 2, "atomic_weight": 4.0026},
    "Li": {"atomic_number": 3, "atomic_weight": 6.94},
    "Be": {"atomic_number": 4, "atomic_weight": 9.0122},
    "B": {"atomic_number": 5, "atomic_weight": 10.81},
    "C": {"atomic_number": 6, "atomic_weight": 12.011},
    "N": {"atomic_number": 7, "atomic_weight": 14.007},
    "O": {"atomic_number": 8, "atomic_weight": 15.999},
    "F": {"atomic_number": 9, "atomic_weight": 18.998},
    "Ne": {"atomic_number": 10, "atomic_weight": 20.180},
    "Na": {"atomic_number": 11, "atomic_weight": 22.990},
    "Mg": {"atomic_number": 12, "atomic_weight": 24.305},
    "Al": {"atomic_number": 13, "atomic_weight": 26.982},
    "Si": {"atomic_number": 14, "atomic_weight": 28.085},
    "P": {"atomic_number": 15, "atomic_weight": 30.974},
    "S": {"atomic_number": 16, "atomic_weight": 32.06},
    "Cl": {"atomic_number": 17, "atomic_weight": 35.45},
    "Ar": {"atomic_number": 18, "atomic_weight": 39.948},
    "K": {"atomic_number": 19, "atomic_weight": 39.098},
    "Ca": {"atomic_number": 20, "atomic_weight": 40.078},
    "Sc": {"atomic_number": 21, "atomic_weight": 44.956},
    "Ti": {"atomic_number": 22, "atomic_weight": 47.867},
    "V": {"atomic_number": 23, "atomic_weight": 50.942},
    "Cr": {"atomic_number": 24, "atomic_weight": 51.996},
    "Mn": {"atomic_number": 25, "atomic_weight": 54.938},
    "Fe": {"atomic_number": 26, "atomic_weight": 55.845},
    "Co": {"atomic_number": 27, "atomic_weight": 58.933},
    "Ni": {"atomic_number": 28, "atomic_weight": 58.693},
    "Cu": {"atomic_number": 29, "atomic_weight": 63.546},
    "Zn": {"atomic_number": 30, "atomic_weight": 65.38},
    "Ga": {"atomic_number": 31, "atomic_weight": 69.723},
    "Ge": {"atomic_number": 32, "atomic_weight": 72.630},
    "As": {"atomic_number": 33, "atomic_weight": 74.922},
    "Se": {"atomic_number": 34, "atomic_weight": 78.971},
    "Br": {"atomic_number": 35, "atomic_weight": 79.904},
    "Kr": {"atomic_number": 36, "atomic_weight": 83.798},
    "Rb": {"atomic_number": 37, "atomic_weight": 85.468},
    "Sr": {"atomic_number": 38, "atomic_weight": 87.62},
    "Y": {"atomic_number": 39, "atomic_weight": 88.906},
    "Zr": {"atomic_number": 40, "atomic_weight": 91.224},
    "Nb": {"atomic_number": 41, "atomic_weight": 92.906},
    "Mo": {"atomic_number": 42, "atomic_weight": 95.95},
    "Tc": {"atomic_number": 43, "atomic_weight": 98.0},
    "Ru": {"atomic_number": 44, "atomic_weight": 101.07},
    "Rh": {"atomic_number": 45, "atomic_weight": 102.91},
    "Pd": {"atomic_number": 46, "atomic_weight": 106.42},
    "Ag": {"atomic_number": 47, "atomic_weight": 107.87},
    "Cd": {"atomic_number": 48, "atomic_weight": 112.41},
    "In": {"atomic_number": 49, "atomic_weight": 114.82},
    "Sn": {"atomic_number": 50, "atomic_weight": 118.71},
    "Sb": {"atomic_number": 51, "atomic_weight": 121.76},
    "Te": {"atomic_number": 52, "atomic_weight": 127.60},
    "I": {"atomic_number": 53, "atomic_weight": 126.90},
    "Xe": {"atomic_number": 54, "atomic_weight": 131.29},
    "Cs": {"atomic_number": 55, "atomic_weight": 132.91},
    "Ba": {"atomic_number": 56, "atomic_weight": 137.33},
    "La": {"atomic_number": 57, "atomic_weight": 138.91},
    "Ce": {"atomic_number": 58, "atomic_weight": 140.12},
    "Pr": {"atomic_number": 59, "atomic_weight": 140.91},
    "Nd": {"atomic_number": 60, "atomic_weight": 144.24},
    "Pm": {"atomic_number": 61, "atomic_weight": 145.0},
    "Sm": {"atomic_number": 62, "atomic_weight": 150.36},
    "Eu": {"atomic_number": 63, "atomic_weight": 151.96},
    "Gd": {"atomic_number": 64, "atomic_weight": 157.25},
    "Tb": {"atomic_number": 65, "atomic_weight": 158.93},
    "Dy": {"atomic_number": 66, "atomic_weight": 162.50},
    "Ho": {"atomic_number": 67, "atomic_weight": 164.93},
    "Er": {"atomic_number": 68, "atomic_weight": 167.26},
    "Tm": {"atomic_number": 69, "atomic_weight": 168.93},
    "Yb": {"atomic_number": 70, "atomic_weight": 173.05},
    "Lu": {"atomic_number": 71, "atomic_weight": 174.97},
    "Hf": {"atomic_number": 72, "atomic_weight": 178.49},
    "Ta": {"atomic_number": 73, "atomic_weight": 180.95},
    "W": {"atomic_number": 74, "atomic_weight": 183.84},
    "Re": {"atomic_number": 75, "atomic_weight": 186.21},
    "Os": {"atomic_number": 76, "atomic_weight": 190.23},
    "Ir": {"atomic_number": 77, "atomic_weight": 192.22},
    "Pt": {"atomic_number": 78, "atomic_weight": 195.08},
    "Au": {"atomic_number": 79, "atomic_weight": 196.97},
    "Hg": {"atomic_number": 80, "atomic_weight": 200.59},
    "Tl": {"atomic_number": 81, "atomic_weight": 204.38},
    "Pb": {"atomic_number": 82, "atomic_weight": 207.2},
    "Bi": {"atomic_number": 83, "atomic_weight": 208.98},
    "Po": {"atomic_number": 84, "atomic_weight": 209.0},
    "At": {"atomic_number": 85, "atomic_weight": 210.0},
    "Rn": {"atomic_number": 86, "atomic_weight": 222.0},
    "Fr": {"atomic_number": 87, "atomic_weight": 223.0},
    "Ra": {"atomic_number": 88, "atomic_weight": 226.0},
    "Ac": {"atomic_number": 89, "atomic_weight": 227.0},
    "Th": {"atomic_number": 90, "atomic_weight": 232.04},
    "Pa": {"atomic_number": 91, "atomic_weight": 231.04},
    "U": {"atomic_number": 92, "atomic_weight": 238.03},
}

# Runtime cache for elements not in the preloaded data
_RUNTIME_CACHE: Dict[str, Dict[str, float]] = {}


def get_atomic_data_fast(element: str) -> Dict[str, float]:
    """
    Fast atomic data lookup with preloaded cache and fallback to Mendeleev.

    This function first checks the preloaded cache, then the runtime cache,
    and only falls back to expensive Mendeleev queries as a last resort.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'Si')

    Returns:
        Dictionary with 'atomic_number' and 'atomic_weight' keys

    Raises:
        ValueError: If element symbol is not recognized
    """
    element_key = element.capitalize()

    # Check preloaded cache first (fastest)
    if element_key in _ATOMIC_DATA_PRELOADED:
        return _ATOMIC_DATA_PRELOADED[element_key].copy()

    # Check runtime cache second
    if element_key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[element_key].copy()

    # Fall back to Mendeleev (slowest)
    try:
        from .utils import get_atomic_number, get_atomic_weight

        atomic_data = {
            "atomic_number": get_atomic_number(element),
            "atomic_weight": get_atomic_weight(element),
        }

        # Cache for future use
        _RUNTIME_CACHE[element_key] = atomic_data.copy()
        return atomic_data

    except UnknownElementError:
        # Re-raise UnknownElementError without wrapping
        raise
    except Exception as e:
        raise ValueError(f"Cannot retrieve atomic data for element '{element}': {e}")


@lru_cache(maxsize=256)
def get_bulk_atomic_data_fast(
    elements_tuple: Tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    """
    High-performance bulk atomic data loader with caching.

    This function loads atomic data for multiple elements efficiently,
    using the preloaded cache to avoid expensive database queries.

    Args:
        elements_tuple: Tuple of element symbols

    Returns:
        Dictionary mapping element symbols to their atomic data
    """
    result = {}
    for element in elements_tuple:
        result[element] = get_atomic_data_fast(element)
    return result


def warm_up_cache(elements: List[str]) -> None:
    """
    Pre-warm the cache with specific elements.

    Args:
        elements: List of element symbols to preload
    """
    for element in elements:
        try:
            get_atomic_data_fast(element)
        except Exception:
            pass  # Skip elements that can't be loaded


def get_cache_stats() -> Dict[str, int]:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "preloaded_elements": len(_ATOMIC_DATA_PRELOADED),
        "runtime_cached_elements": len(_RUNTIME_CACHE),
        "total_cached_elements": (len(_ATOMIC_DATA_PRELOADED) + len(_RUNTIME_CACHE)),
    }


def is_element_preloaded(element: str) -> bool:
    """
    Check if an element is in the preloaded cache.

    Args:
        element: Element symbol

    Returns:
        True if element is preloaded, False otherwise
    """
    return element.capitalize() in _ATOMIC_DATA_PRELOADED
