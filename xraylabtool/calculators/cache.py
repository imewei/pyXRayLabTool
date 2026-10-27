"""Cache infrastructure for scattering factor data and interpolators."""

# ruff: noqa: PLW0603

from __future__ import annotations

from functools import cache, lru_cache
import sys
import types
from typing import Any

# Module-level cache for f1/f2 scattering tables, keyed by element symbol
# Using Any to avoid early pandas import
_scattering_factor_cache: dict[str, Any] = {}

# Module-level cache for interpolators to avoid repeated creation
_interpolator_cache: dict[str, tuple[TableInterpolator, TableInterpolator]] = {}

_CACHE_WARMED = False

# Atomic data cache for bulk lookups
_atomic_data_cache: dict[str, dict[str, float]] = {}


class TableInterpolator:
    """f1 or f2 of one element, read from a row of the resident device tables.

    Instances come from :func:`create_scattering_factor_interpolators`. Calling
    one evaluates the same PCHIP as the device program on an energy in eV (a
    scalar, a host array or a device array) and returns a device array, NaN
    outside the element's table. A host query costs one
    :func:`xraylabtool.backend.runtime.upload`, a device query uploads only the
    row and field indices; nothing is fetched. Each distinct query length compiles once.

    Attributes
    ----------
    element : str
        Element symbol.
    z : int
        Row of the element in the tables (atomic number minus one).
    field : int
        0 for f1, 1 for f2.
    """

    __slots__ = ("element", "field", "z")

    def __init__(self, element: str, z: int, field: int) -> None:
        self.element = element
        self.z = z
        self.field = field

    def __repr__(self) -> str:
        return f"TableInterpolator({self.element!r}, f{self.field + 1})"

    def __call__(self, energy_ev: Any) -> Any:
        import jax
        import jax.numpy as jnp
        import numpy as np

        from xraylabtool.backend import runtime
        from xraylabtool.calculators import host, optics

        runtime.ensure_x64()
        tables = host.device_tables()
        with jax.enable_x64(True):
            row = (np.int32(self.z), np.int32(self.field))
            device: tuple[Any, ...]
            if isinstance(energy_ev, jax.Array):
                device = (
                    *runtime.upload(row),
                    jnp.asarray(energy_ev, dtype=jnp.float64),
                )
            else:
                device = runtime.upload((*row, np.asarray(energy_ev, dtype=np.float64)))
            with runtime.worker_guard():
                return optics._field_program(tables, *device)


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
) -> types.MappingProxyType[str, types.MappingProxyType[str, float]]:
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


def _smart_cache_warming(formula: str) -> None:
    """
    Smart cache warming for the first calculation.

    ``calculators.warm_up`` loads the resident scattering tables and precompiles
    the default-bucket device programs, so the first call of a formula with up
    to two elements and 64 energies compiles nothing. A failure propagates: the
    calculation itself would fail the same way.

    Args:
        formula: Chemical formula of the pending calculation (the programs are
            warmed for the default ``SiO2`` buckets whatever it is)
    """
    del formula
    from xraylabtool.calculators.core import warm_up

    warm_up(("SiO2",))

    global _CACHE_WARMED
    _CACHE_WARMED = True


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

    # Drop the resident tables if they were ever loaded (importing the module
    # here just to clear it would import JAX)
    tables = sys.modules.get("xraylabtool.calculators.tables")
    if tables is not None:
        tables.clear_tables()

    # Clear LRU caches, including the shared loader that get_bulk_atomic_data wraps
    from xraylabtool.data_handling.atomic_cache import get_bulk_atomic_data_fast

    get_bulk_atomic_data_fast.cache_clear()
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
) -> tuple[TableInterpolator, TableInterpolator]:
    """
    Create the f1 and f2 interpolators of an element.

    The returned callables read the element's PCHIP row of the resident device
    tables (see :class:`TableInterpolator`); the ``.nff`` file is read and
    validated once, which also fills the element cache.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

    Returns:
        Tuple of (f1_interpolator, f2_interpolator) where each is a callable
        that takes energies in eV and returns interpolated scattering factors

    Raises:
        FileNotFoundError: If the .nff file for the element is not found
        ValueError: If the element symbol is invalid or data is insufficient

    Examples:
        >>> from xraylabtool.calculators.cache import create_scattering_factor_interpolators
        >>> import numpy as np
        >>> f1_interp, f2_interp = create_scattering_factor_interpolators('Si')
        >>> energy = 100.0  # eV
        >>> f1_value = f1_interp(energy)
        >>> bool(np.isfinite(np.asarray(f1_value)).all())  # a device array
        True
        >>> f2_value = f2_interp(energy)
        >>> bool(np.isfinite(np.asarray(f2_value)).all())
        True
        >>> # Can also handle arrays
        >>> energies = np.array([100.0, 200.0, 300.0])
        >>> f1_values = f1_interp(energies)
        >>> len(f1_values) == 3
        True
    """
    from xraylabtool.calculators.scattering_data import load_scattering_factor_data
    from xraylabtool.calculators.tables import element_index

    # Check interpolator cache first
    if element in _interpolator_cache:
        return _interpolator_cache[element]

    # Validates the element and fills the element cache
    scattering_factor_data = load_scattering_factor_data(element)

    # Verify we have sufficient data points for PCHIP interpolation
    if len(scattering_factor_data) < 2:
        raise ValueError(
            f"Insufficient data points for element '{element}'. "
            "PCHIP interpolation requires at least 2 points, "
            f"found {len(scattering_factor_data)}."
        )

    z = element_index(element.capitalize())
    interpolators = (
        TableInterpolator(element, z, 0),
        TableInterpolator(element, z, 1),
    )
    # Cache the interpolators for future use
    _interpolator_cache[element] = interpolators

    return interpolators
