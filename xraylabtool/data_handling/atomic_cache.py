"""
High-performance atomic data cache system.

This module provides a pre-populated cache of atomic data for common elements
to eliminate expensive database queries to the Mendeleev library during runtime.
"""

from __future__ import annotations

from functools import lru_cache
import types
from typing import TYPE_CHECKING, Any

import numpy as np

from xraylabtool.exceptions import UnknownElementError

# Cache metrics imports are done lazily to avoid circular imports

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import ComplexArray, EnergyArray

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
_RUNTIME_CACHE: dict[str, dict[str, float]] = {}


def get_atomic_data_fast(element: str) -> types.MappingProxyType[str, float]:
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

    # Check preloaded cache first (fastest) - record cache hit
    if element_key in _ATOMIC_DATA_PRELOADED:
        try:
            from xraylabtool.data_handling.cache_metrics import _record_cache_access

            _record_cache_access(element_key, "preloaded_atomic", hit=True)
        except ImportError:
            pass
        return types.MappingProxyType(_ATOMIC_DATA_PRELOADED[element_key])

    # Check runtime cache second - record cache hit
    if element_key in _RUNTIME_CACHE:
        try:
            from xraylabtool.data_handling.cache_metrics import _record_cache_access

            _record_cache_access(element_key, "runtime_atomic", hit=True)
        except ImportError:
            pass
        return types.MappingProxyType(_RUNTIME_CACHE[element_key])

    # Fall back to Mendeleev (slowest) - record cache miss
    try:
        from xraylabtool.utils import get_atomic_number, get_atomic_weight

        try:
            from xraylabtool.data_handling.cache_metrics import _record_cache_access

            _record_cache_access(element_key, "atomic_data", hit=False)
        except ImportError:
            pass

        atomic_data = {
            "atomic_number": get_atomic_number(element),
            "atomic_weight": get_atomic_weight(element),
        }

        # Cache for future use - store the actual dict in cache
        _RUNTIME_CACHE[element_key] = atomic_data
        return types.MappingProxyType(atomic_data)

    except UnknownElementError:
        # Re-raise UnknownElementError without wrapping
        raise
    except Exception as e:
        raise ValueError(
            f"Cannot retrieve atomic data for element '{element}': {e}"
        ) from e


@lru_cache(maxsize=256)
def get_bulk_atomic_data_fast(
    elements_tuple: tuple[str, ...],
) -> dict[str, types.MappingProxyType[str, float]]:
    """
    High-performance bulk atomic data loader with caching.

    This function loads atomic data for multiple elements efficiently,
    using the preloaded cache to avoid expensive database queries.

    Args:
        elements_tuple: Tuple of element symbols

    Returns:
        Dictionary mapping element symbols to their atomic data (as immutable views)
    """
    result = {}
    for element in elements_tuple:
        result[element] = get_atomic_data_fast(element)
    return result


def warm_up_cache(elements: list[str]) -> None:
    """
    Pre-warm the cache with specific elements.

    Args:
        elements: List of element symbols to preload
    """
    import contextlib

    for element in elements:
        with contextlib.suppress(Exception):
            get_atomic_data_fast(element)


def warm_cache_for_compounds(
    formulas: list[str],
    include_similar: bool = True,
    include_family: bool = True,
    timing_info: bool = False,
) -> dict[str, Any]:
    """
    Intelligently warm cache for compounds and their related elements.

    This function performs intelligent cache warming by analyzing compound
    formulas, extracting their constituent elements, and pre-loading both
    atomic data and scattering factor interpolators. It can also include
    similar compounds and compound families for comprehensive warming.

    Args:
        formulas: List of chemical formulas to warm cache for
        include_similar: Whether to include similar compounds
        include_family: Whether to include compound family members
        timing_info: Whether to return timing information

    Returns:
        Dictionary with warming results and statistics

    Examples:
        >>> result = warm_cache_for_compounds(["SiO2", "Al2O3"])
        >>> result["elements_warmed"]
        ['Si', 'O', 'Al']
        >>> result["success_rate"] > 0.9
        True
    """
    import time

    from xraylabtool.data_handling.cache_metrics import track_compound_calculation
    from xraylabtool.data_handling.compound_analysis import (
        COMPOUND_FAMILIES,
        find_similar_compounds,
        get_compound_family,
        get_elements_for_compound,
    )

    start_time = time.perf_counter() if timing_info else None

    # Collect all elements to warm
    elements_to_warm = set()
    compound_info = {}

    # Process each formula
    for formula in formulas:
        try:
            # Get constituent elements
            elements = get_elements_for_compound(formula)
            elements_to_warm.update(elements)

            compound_info[formula] = {
                "elements": elements,
                "status": "parsed",
                "similar_compounds": [],
                "family_compounds": [],
            }

            # Track this compound calculation
            calc_start = time.perf_counter()
            track_compound_calculation(formula, elements, 0.0)  # Placeholder timing
            time.perf_counter() - calc_start

            # Find similar compounds if requested
            if include_similar:
                similar = find_similar_compounds(formula, similarity_threshold=0.3)
                compound_info[formula]["similar_compounds"] = similar[:3]  # Limit to 3

                # Add elements from similar compounds
                for similar_formula in similar[:3]:
                    try:
                        similar_elements = get_elements_for_compound(similar_formula)
                        elements_to_warm.update(similar_elements)
                    except (KeyError, ValueError, ImportError):
                        # Skip invalid compounds during cache warming
                        continue

            # Find compound family members if requested
            if include_family:
                family = get_compound_family(formula)
                if family and family in COMPOUND_FAMILIES:
                    family_compounds = COMPOUND_FAMILIES[family][:5]  # Limit to 5
                    compound_info[formula]["family_compounds"] = family_compounds

                    # Add elements from family compounds
                    for family_formula in family_compounds:
                        try:
                            family_elements = get_elements_for_compound(family_formula)
                            elements_to_warm.update(family_elements)
                        except (KeyError, ValueError, ImportError):
                            # Skip invalid family compounds during cache warming
                            continue

        except Exception as e:
            compound_info[formula] = {
                "elements": [],
                "status": f"error: {e}",
                "similar_compounds": [],
                "family_compounds": [],
            }

    # Warm atomic data cache
    atomic_success = 0
    atomic_total = len(elements_to_warm)

    for element in elements_to_warm:
        try:
            get_atomic_data_fast(element)
            atomic_success += 1
        except (KeyError, ValueError, ImportError):
            # Skip elements that cannot be loaded during atomic cache warming
            continue

    # Warm scattering factor interpolators
    interpolator_success = 0
    interpolator_total = len(elements_to_warm)

    for element in elements_to_warm:
        try:
            from xraylabtool.calculators.core import (
                create_scattering_factor_interpolators,
            )

            create_scattering_factor_interpolators(element)
            interpolator_success += 1
        except (KeyError, ValueError, ImportError):
            # Skip elements that cannot create interpolators during cache warming
            continue

    # Warm bulk data cache for common combinations
    bulk_success = 0
    if len(elements_to_warm) > 1:
        try:
            # Create common element combinations
            element_list = list(elements_to_warm)
            common_combos = [
                tuple(element_list[:3]),  # First 3 elements
                tuple(element_list[:5]),  # First 5 elements
                tuple(sorted(element_list)),  # All elements sorted
            ]

            for combo in common_combos:
                if len(combo) > 0:
                    try:
                        get_bulk_atomic_data_fast(combo)
                        bulk_success += 1
                    except (KeyError, ValueError, ImportError):
                        # Skip invalid element combinations during bulk cache warming
                        continue

        except Exception:
            pass

    # Calculate timing
    end_time = time.perf_counter() if timing_info else None
    total_time_ms = (end_time - start_time) * 1000.0 if timing_info else 0.0

    # Calculate success rates
    atomic_success_rate = atomic_success / atomic_total if atomic_total > 0 else 0.0
    interpolator_success_rate = (
        interpolator_success / interpolator_total if interpolator_total > 0 else 0.0
    )
    overall_success_rate = (
        (atomic_success + interpolator_success) / (atomic_total + interpolator_total)
        if (atomic_total + interpolator_total) > 0
        else 0.0
    )

    # Update cache metrics with warming performance
    from xraylabtool.data_handling.cache_metrics import _metrics_lock, _usage_patterns

    with _metrics_lock:
        _usage_patterns["performance_metrics"]["warming_time_ms"] = total_time_ms

    return {
        "elements_warmed": sorted(elements_to_warm),
        "compounds_processed": compound_info,
        "atomic_cache": {
            "success": atomic_success,
            "total": atomic_total,
            "success_rate": atomic_success_rate,
        },
        "interpolator_cache": {
            "success": interpolator_success,
            "total": interpolator_total,
            "success_rate": interpolator_success_rate,
        },
        "bulk_cache": {
            "success": bulk_success,
            "attempts": 3 if len(elements_to_warm) > 1 else 0,
        },
        "timing": (
            {
                "total_time_ms": total_time_ms,
                "time_per_element_ms": (
                    total_time_ms / len(elements_to_warm) if elements_to_warm else 0.0
                ),
            }
            if timing_info
            else {}
        ),
        "success_rate": overall_success_rate,
        "performance_metrics": {
            "elements_per_second": (
                len(elements_to_warm) / (total_time_ms / 1000.0)
                if total_time_ms > 0
                else 0.0
            ),
            "within_target": (
                total_time_ms < 100.0 if timing_info else True
            ),  # Target: <100ms
        },
    }


def get_cache_stats() -> dict[str, int]:
    """
    Get cache statistics for monitoring.

    Returns:
        Dictionary with cache statistics
    """
    return {
        "preloaded_elements": len(_ATOMIC_DATA_PRELOADED),
        "runtime_cached_elements": len(_RUNTIME_CACHE),
        "total_cached_elements": len(_ATOMIC_DATA_PRELOADED) + len(_RUNTIME_CACHE),
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


# =====================================================================================
# AtomicDataProvider Protocol Implementation
# =====================================================================================


class FastAtomicDataProvider:
    """
    High-performance atomic data provider implementing AtomicDataProvider protocol.

    This implementation uses preloaded atomic data and interpolated scattering
    factors for maximum performance in X-ray calculations.
    """

    def __init__(self) -> None:
        """Initialize the atomic data provider."""
        self._scattering_cache: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def get_scattering_factors(
        self, element: str, energies: EnergyArray
    ) -> ComplexArray:
        """
        Get atomic scattering factors for element at given energies.

        This method loads scattering factor data and interpolates it to the
        requested energies, returning complex scattering factors (f1 + if2).

        Parameters
        ----------
        element : str
            Chemical element symbol (e.g., 'Si', 'O')
        energies : EnergyArray
            X-ray energies in keV

        Returns
        -------
        ComplexArray
            Complex scattering factors (f1 + if2)
        """
        from xraylabtool.calculators.core import create_scattering_factor_interpolators

        # Convert to numpy array if needed
        energies_arr = np.asarray(energies, dtype=np.float64)

        # Convert energy from keV to eV for interpolation
        energies_ev = energies_arr * 1000.0

        # Get interpolators for this element
        f1_interp, f2_interp = create_scattering_factor_interpolators(element)

        # Interpolate f1 and f2 values
        f1_values = f1_interp(energies_ev)
        f2_values = f2_interp(energies_ev)

        # Combine into complex array
        complex_factors = f1_values + 1j * f2_values

        return np.asarray(complex_factors, dtype=np.complex128)

    def is_element_cached(self, element: str) -> bool:
        """
        Check if element data is cached for fast access.

        Parameters
        ----------
        element : str
            Element symbol to check

        Returns
        -------
        bool
            True if element is cached for fast access
        """
        from xraylabtool.calculators.core import is_element_cached as is_core_cached

        # Check both our preloaded data and core module cache
        return is_element_preloaded(element) or is_core_cached(element)

    def preload_elements(self, elements: list[str]) -> None:
        """
        Preload scattering factor data for elements.

        Parameters
        ----------
        elements : list[str]
            List of element symbols to preload
        """
        from xraylabtool.calculators.core import create_scattering_factor_interpolators

        for element in elements:
            try:
                # This will cache the interpolators
                create_scattering_factor_interpolators(element)
            except (KeyError, ValueError, ImportError):
                # Skip elements that can't be loaded during prewarming
                continue

    def get_atomic_properties(self, element: str) -> types.MappingProxyType[str, float]:
        """
        Get basic atomic properties for an element.

        Parameters
        ----------
        element : str
            Element symbol

        Returns
        -------
        types.MappingProxyType[str, float]
            Immutable mapping with atomic properties
        """
        return get_atomic_data_fast(element)


# Global instance for easy access
_GLOBAL_PROVIDER: FastAtomicDataProvider | None = None


def get_atomic_data_provider() -> FastAtomicDataProvider:
    """
    Get the global atomic data provider instance.

    Returns
    -------
    FastAtomicDataProvider
        Shared atomic data provider instance
    """
    global _GLOBAL_PROVIDER
    if _GLOBAL_PROVIDER is None:
        _GLOBAL_PROVIDER = FastAtomicDataProvider()
    return _GLOBAL_PROVIDER
