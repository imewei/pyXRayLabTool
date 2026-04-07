"""Cache infrastructure for scattering factor data and interpolators."""

# ruff: noqa: PLW0603

from __future__ import annotations

from functools import cache, lru_cache
import types
from typing import TYPE_CHECKING, Any

from xraylabtool.backend import InterpolationFactory, ops

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import InterpolatorProtocol


# Module-level cache for f1/f2 scattering tables, keyed by element symbol
# Using Any to avoid early pandas import
_scattering_factor_cache: dict[str, Any] = {}

# Module-level cache for interpolators to avoid repeated creation
if TYPE_CHECKING:
    _interpolator_cache: dict[
        str, tuple[InterpolatorProtocol, InterpolatorProtocol]
    ] = {}
else:
    _interpolator_cache: dict[str, Any] = {}

# Cache for most commonly used elements to improve cold start performance
_PRIORITY_ELEMENTS = ["H", "C", "N", "O", "Si", "Al", "Ca", "Fe", "Cu", "Zn"]
_CACHE_WARMED = False

# Atomic data cache for bulk lookups
_atomic_data_cache: dict[str, dict[str, float]] = {}


def get_cached_elements() -> list[str]:
    """
    Get list of elements currently cached in the scattering factor cache.

    Returns:
        List of element symbols currently loaded in cache
    """
    return list(_scattering_factor_cache.keys())


@cache
def get_bulk_atomic_data(
    elements_tuple: tuple[str, ...],
) -> dict[str, types.MappingProxyType[str, float]]:
    """
    Bulk load atomic data for multiple elements with high-performance caching.

    This optimization uses a preloaded cache of common elements to eliminate
    expensive database queries to the Mendeleev library during runtime.

    Args:
        elements_tuple: Tuple of element symbols to load data for

    Returns:
        Dictionary mapping element symbols to their atomic data
    """
    from xraylabtool.data_handling.atomic_cache import get_bulk_atomic_data_fast

    return get_bulk_atomic_data_fast(elements_tuple)


def _warm_priority_cache() -> None:
    """
    Warm the cache with priority elements for improved cold start performance.

    This is called automatically on first calculation to pre-load common elements.
    Uses background thread for async warming to reduce cold start penalty.
    """
    global _CACHE_WARMED
    if _CACHE_WARMED:
        return

    # Use background thread for async warming to avoid blocking main thread
    import threading

    def _background_cache_warming():  # type: ignore[no-untyped-def]
        """Background thread function for cache warming."""
        global _CACHE_WARMED
        try:
            from xraylabtool.data_handling.atomic_cache import (
                get_bulk_atomic_data_fast,
            )

            priority_tuple = tuple(_PRIORITY_ELEMENTS)
            get_bulk_atomic_data_fast(priority_tuple)
            _CACHE_WARMED = True
        except Exception:  # noqa: S110
            # If warming fails, just continue - it's not critical
            pass

    # Start background warming but don't wait for it
    warming_thread = threading.Thread(target=_background_cache_warming, daemon=True)
    warming_thread.start()

    # Mark as "warming in progress" to avoid multiple attempts
    _CACHE_WARMED = True


def _smart_cache_warming(formula: str) -> None:
    """
    Smart cache warming that only loads elements needed for the specific calculation.

    Args:
        formula: Chemical formula to analyze for required elements (e.g., "SiO2")
    """
    try:
        from xraylabtool.utils import parse_formula

        # Parse formula to get required elements
        element_symbols, _ = parse_formula(formula)
        required_elements = element_symbols

        # Load only required elements (much faster than bulk loading)
        from xraylabtool.data_handling.atomic_cache import get_bulk_atomic_data_fast

        get_bulk_atomic_data_fast(tuple(required_elements))

        # Mark cache as warmed
        global _CACHE_WARMED
        _CACHE_WARMED = True

    except Exception:
        # If smart warming fails, fall back to traditional warming
        _warm_priority_cache()


def clear_scattering_factor_cache() -> None:
    """
    Clear the module-level scattering factor cache.

    This function removes all cached scattering factor data from memory.
    Useful for testing or memory management.
    """
    global _CACHE_WARMED
    _scattering_factor_cache.clear()
    _interpolator_cache.clear()
    _atomic_data_cache.clear()
    _CACHE_WARMED = False

    # Clear LRU caches
    get_bulk_atomic_data.cache_clear()
    create_scattering_factor_interpolators.cache_clear()


def is_element_cached(element: str) -> bool:
    """
    Check if scattering factor data for an element is already cached.

    Args:
        element: Element symbol to check

    Returns:
        True if element data is cached, False otherwise
    """
    return element.capitalize() in _scattering_factor_cache


@lru_cache(maxsize=128)
def create_scattering_factor_interpolators(
    element: str,
) -> tuple[InterpolatorProtocol, InterpolatorProtocol]:
    """
    Create PCHIP interpolators for f1 and f2 scattering factors.

    This helper function loads scattering factor data for a specific element
    and returns two callable PCHIP interpolator objects for f1 and f2 that
    behave identically to Julia interpolation behavior.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

    Returns:
        Tuple of (f1_interpolator, f2_interpolator) where each is a callable
        that takes energy values and returns interpolated scattering factors

    Raises:
        FileNotFoundError: If the .nff file for the element is not found
        ValueError: If the element symbol is invalid or data is insufficient

    Examples:
        >>> from xraylabtool.calculators.cache import create_scattering_factor_interpolators
        >>> import numpy as np
        >>> f1_interp, f2_interp = create_scattering_factor_interpolators('Si')
        >>> energy = 100.0  # eV
        >>> f1_value = f1_interp(energy)
        >>> isinstance(f1_value, (int, float, np.number, np.ndarray))
        True
        >>> f2_value = f2_interp(energy)
        >>> isinstance(f2_value, (int, float, np.number, np.ndarray))
        True
        >>> # Can also handle arrays
        >>> energies = np.array([100.0, 200.0, 300.0])
        >>> f1_values = f1_interp(energies)
        >>> len(f1_values) == 3
        True
    """
    from xraylabtool.calculators.scattering_data import load_scattering_factor_data

    # Check interpolator cache first
    if element in _interpolator_cache:
        return _interpolator_cache[element]

    # Load scattering factor data
    scattering_factor_data = load_scattering_factor_data(element)

    # Verify we have sufficient data points for PCHIP interpolation
    if len(scattering_factor_data) < 2:
        raise ValueError(
            f"Insufficient data points for element '{element}'. "
            "PCHIP interpolation requires at least 2 points, "
            f"found {len(scattering_factor_data)}."
        )

    # Extract energy, f1, and f2 data
    energy_values = ops.asarray(scattering_factor_data["E"].values)
    f1_values = ops.asarray(scattering_factor_data["f1"].values)
    f2_values = ops.asarray(scattering_factor_data["f2"].values)

    # Verify energy values are sorted (PCHIP requires sorted x values)
    if ops.any(ops.asarray(energy_values[:-1]) > ops.asarray(energy_values[1:])):
        # Sort the data if it's not already sorted
        sort_indices = ops.argsort(energy_values)
        energy_values = energy_values[sort_indices]
        f1_values = f1_values[sort_indices]
        f2_values = f2_values[sort_indices]

    # Create PCHIP interpolators via backend abstraction layer
    f1_interpolator = InterpolationFactory.create_pchip(
        energy_values, f1_values, extrapolate=False
    )
    f2_interpolator = InterpolationFactory.create_pchip(
        energy_values, f2_values, extrapolate=False
    )

    # Cache the interpolators for future use
    _interpolator_cache[element] = (f1_interpolator, f2_interpolator)

    return f1_interpolator, f2_interpolator
