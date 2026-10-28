"""
Utility functions for XRayLabTool.

This module contains helper functions for data processing, unit conversions,
mathematical operations, and other common tasks in X-ray analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import cache, lru_cache, partial
import operator
import re
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from xraylabtool.constants import (
    AVOGADRO,
    ELEMENT_CHARGE,
    PLANCK,
)
from xraylabtool.constants import (
    SPEED_OF_LIGHT as _SPEED_OF_LIGHT,
)
from xraylabtool.exceptions import AtomicDataError, UnknownElementError

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import ArrayLike, FloatLike


# Physics and signal helpers run in jax.numpy (xraylabtool/_physics.py); the
# wrappers below validate on the host and import _physics inside each call so
# that `import xraylabtool` stays JAX-free. Host by design (spec section 9):
# find_peaks, progress_bar, save_processed_data, parse_formula.


@cache
def _program(name: str, **static: Any) -> Callable[..., Any]:
    """The ``_physics`` function ``name`` with its static arguments bound (cached)."""
    from xraylabtool import _physics

    fn: Callable[..., Any] = getattr(_physics, name)
    return partial(fn, **static) if static else fn


def _map(name: str, *arrays: Any, **static: Any) -> np.ndarray:
    """Run an elementwise ``_physics`` function on host arrays (broadcast shape out)."""
    from xraylabtool.calculators.host import run_map

    for value in arrays:
        if np.iscomplexobj(value):
            # Python's own message for float(complex), which is what a complex
            # scalar raised before; NumPy would cast and warn instead.
            raise TypeError(
                "float() argument must be a string or a real number, "
                f"not {type(value).__name__!r}"
            )
    return run_map(_program(name, **static), *arrays)  # type: ignore[no-any-return]


def _real(y: ArrayLike) -> np.ndarray:
    """``y`` as a float64 array; complex data is rejected, never cast."""
    y_arr = np.asarray(y)
    if np.iscomplexobj(y_arr):
        raise TypeError("complex signal data is not supported")
    return np.asarray(y_arr, dtype=np.float64)


def _series(y: ArrayLike) -> np.ndarray:
    y_arr = _real(y)
    if y_arr.ndim != 1:
        raise ValueError("Signal data must be one-dimensional")
    return y_arr


def _run_columns(name: str, columns: np.ndarray) -> np.ndarray:
    from xraylabtool.calculators.host import run_columns

    return run_columns(_program(name), columns)


def _run_series(
    name: str, *series: np.ndarray, extra: tuple[Any, ...] = (), **static: Any
) -> np.ndarray:
    from xraylabtool.calculators.host import run_series

    return run_series(_program(name, **static), *series, extra=extra)


# Module-level constant aliases — computed once at import time.
# These thin wrappers are kept for backward API compatibility.
PLANCK_CONSTANT: float = float(PLANCK)
ELECTRON_CHARGE: float = float(ELEMENT_CHARGE)
AVOGADRO_NUMBER: float = float(AVOGADRO)


def get_planck_constant() -> float:
    """Get Planck constant (J⋅s)."""
    return PLANCK_CONSTANT


def get_speed_of_light() -> float:
    """Get speed of light (m/s)."""
    return float(_SPEED_OF_LIGHT)


def get_electron_charge() -> float:
    """Get electron charge (C)."""
    return ELECTRON_CHARGE


def get_avogadro_number() -> float:
    """Get Avogadro number (mol⁻¹)."""
    return AVOGADRO_NUMBER


# Export all public functions
__all__ = [
    "angle_from_q",
    "background_subtraction",
    "bragg_angle",
    "d_spacing_cubic",
    "d_spacing_orthorhombic",
    "d_spacing_tetragonal",
    "energy_to_wavelength",
    "find_peaks",
    "get_atomic_data",
    "get_atomic_number",
    "get_atomic_weight",
    "load_atomic_data",  # Backward compatibility
    "normalize_intensity",
    "parse_formula",  # Chemical formula parser
    "progress_bar",
    "q_from_angle",
    "save_processed_data",
    "smooth_data",
    "wavelength_to_energy",
    # Note: AtomicDataError and UnknownElementError are imported for internal use
    # but not exported to avoid duplicate documentation
]


def wavelength_to_energy(wavelength: FloatLike, units: str = "angstrom") -> float:
    """
    Convert X-ray wavelength to energy.

    Args:
        wavelength: Wavelength value
        units: Units of wavelength ('angstrom', 'nm', 'm')

    Returns:
        Energy in keV
    """
    if not np.all(np.asarray(wavelength) > 0):
        raise ValueError("Wavelength must be positive")

    if units not in ("angstrom", "nm", "m"):
        raise ValueError("Units must be 'angstrom', 'nm', or 'm'")

    return float(_map("wavelength_to_energy", wavelength, units=units))


def energy_to_wavelength(energy: FloatLike, units: str = "angstrom") -> float:
    """
    Convert X-ray energy to wavelength.

    Args:
        energy: Energy in keV
        units: Desired units of wavelength ('angstrom', 'nm', 'm')

    Returns:
        Wavelength in specified units
    """
    if not np.all(np.asarray(energy) > 0):
        raise ValueError("Energy must be positive")
    if units not in ("angstrom", "nm", "m"):
        raise ValueError("Units must be 'angstrom', 'nm', or 'm'")

    return float(_map("energy_to_wavelength", energy, units=units))


def bragg_angle(d_spacing: FloatLike, wavelength: FloatLike, order: int = 1) -> float:
    """
    Calculate Bragg angle for given d-spacing and wavelength.

    Args:
        d_spacing: d-spacing in Angstroms
        wavelength: X-ray wavelength in Angstroms
        order: Diffraction order (default: 1)

    Returns:
        Bragg angle in degrees
    """
    if d_spacing <= 0:
        raise ValueError("d-spacing must be positive")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive")
    if order <= 0:
        raise ValueError("Order must be positive")

    sin_theta = (order * wavelength) / (2 * d_spacing)

    if sin_theta > 1:
        raise ValueError("No diffraction possible for given parameters")

    return float(_map("bragg_angle", d_spacing, wavelength, order))


def _check_lattice(h: Any, k: Any, miller_l: Any, *lengths: Any) -> None:
    """Reject Miller indices (0, 0, 0) and non-positive lattice parameters."""
    zero = (np.asarray(h) == 0) & (np.asarray(k) == 0) & (np.asarray(miller_l) == 0)
    if np.any(zero):
        raise ValueError("Miller indices (0, 0, 0) have no defined d-spacing")
    if not all(np.all(np.asarray(x) > 0) for x in lengths):
        raise ValueError("Lattice parameters must be positive")


def d_spacing_cubic(h: int, k: int, miller_l: int, a: FloatLike) -> float:
    """
    Calculate d-spacing for cubic crystal system.

    Args:
        h, k, l: Miller indices
        a: Lattice parameter in Angstroms

    Returns:
        d-spacing in Angstroms
    """
    _check_lattice(h, k, miller_l, a)
    return float(_map("d_spacing_cubic", h, k, miller_l, a))


def d_spacing_tetragonal(
    h: int, k: int, miller_l: int, a: FloatLike, c: FloatLike
) -> float:
    """
    Calculate d-spacing for tetragonal crystal system.

    Args:
        h, k, l: Miller indices
        a: a lattice parameter in Angstroms
        c: c lattice parameter in Angstroms

    Returns:
        d-spacing in Angstroms
    """
    _check_lattice(h, k, miller_l, a, c)
    return float(_map("d_spacing_tetragonal", h, k, miller_l, a, c))


def d_spacing_orthorhombic(
    h: int, k: int, miller_l: int, a: FloatLike, b: FloatLike, c: FloatLike
) -> float:
    """
    Calculate d-spacing for orthorhombic crystal system.

    Args:
        h, k, l: Miller indices
        a, b, c: Lattice parameters in Angstroms

    Returns:
        d-spacing in Angstroms
    """
    _check_lattice(h, k, miller_l, a, b, c)
    return float(_map("d_spacing_orthorhombic", h, k, miller_l, a, b, c))


def q_from_angle(two_theta: FloatLike, wavelength: FloatLike) -> float:
    """
    Calculate momentum transfer q from scattering angle.

    Args:
        two_theta: Scattering angle (2θ) in degrees
        wavelength: X-ray wavelength in Angstroms

    Returns:
        Momentum transfer q in Å⁻¹
    """
    return float(_map("q_from_angle", two_theta, wavelength))


def angle_from_q(q: FloatLike, wavelength: FloatLike) -> float:
    """
    Calculate scattering angle from momentum transfer q.

    Args:
        q: Momentum transfer in Å⁻¹
        wavelength: X-ray wavelength in Angstroms

    Returns:
        Scattering angle (2θ) in degrees
    """
    sin_theta = (q * wavelength) / (4 * np.pi)

    if sin_theta > 1:
        raise ValueError("Invalid q value for given wavelength")

    return float(_map("angle_from_q", q, wavelength))


def smooth_data(x: ArrayLike, y: ArrayLike, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to one-dimensional data (``jax.numpy`` convolution).

    Edges are replicated; a window of at least ``len(y)`` returns the mean.

    Args:
        x: x-axis data (not used but kept for consistency)
        y: y-axis data to smooth; must be one-dimensional (``ValueError`` otherwise)
        window_size: Size of smoothing window

    Returns:
        Smoothed y data
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    y_arr = _series(y)
    if window_size >= y_arr.size:
        # The mean, whatever the type of the window (as before). Every such window
        # runs the same program, so its static size is the bucket, not the value.
        from xraylabtool.backend.shapes import bucket

        window = bucket(y_arr.size, minimum=64)
    else:
        # Integers only, as np.ones() accepted: not 2.9, 3.0 or True.
        if isinstance(window_size, bool | np.bool_):
            window = None
        else:
            try:
                window = operator.index(window_size)
            except TypeError:
                window = None
        if window is None:
            raise TypeError(
                f"expected a sequence of integers or a single integer, got '{window_size}'"
            )
    return _run_series("smooth_data", y_arr, window_size=window)


def find_peaks(
    x: ArrayLike, y: ArrayLike, prominence: FloatLike = 0.1, distance: int = 10
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Find peaks in diffraction data.

    Args:
        x: x-axis data (angle or q values)
        y: y-axis data (intensity)
        prominence: Minimum peak prominence
        distance: Minimum distance between peaks in data points

    Returns:
        Tuple of (peak_indices, peak_properties)
    """
    from scipy.signal import find_peaks as scipy_find_peaks

    # Convert to numpy arrays for consistent operations
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    peaks, properties = scipy_find_peaks(
        y_arr, prominence=prominence, distance=distance
    )
    properties["x_values"] = x_arr[peaks]
    properties["y_values"] = y_arr[peaks]

    return peaks, properties


def background_subtraction(
    x: ArrayLike,
    y: ArrayLike,
    method: str = "linear",
) -> np.ndarray:
    """
    Perform background subtraction on diffraction data.

    Args:
        x: x-axis data (used by ``'polynomial'`` only; one-dimensional, same
            length as ``y``)
        y: y-axis data. ``'polynomial'`` needs one dimension; ``'linear'`` also
            accepts an ``(N, k)`` array of ``k`` series in columns and subtracts
            each column's own line (one device call)
        method: Background subtraction method ('linear', 'polynomial')

    Returns:
        Background-subtracted y data, float64, the shape of ``y``
    """
    y_arr = _real(y)
    if method not in ("linear", "polynomial"):
        raise ValueError("Method must be 'linear' or 'polynomial'")
    if method == "linear" and y_arr.ndim == 2 and y_arr.shape[0] > 0:
        if y_arr.shape[1] == 0:
            return y_arr.copy()
        return _run_columns("linear_background_columns", y_arr)
    y_arr = _series(y_arr)
    if y_arr.size == 0:
        raise IndexError("index 0 is out of bounds for axis 0 with size 0")
    if method == "linear":
        return _run_series("linear_background", y_arr)
    x_arr = _series(x)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")
    # shortcut: the 3-point quadratic fit stays on the host (NumPy lstsq). Its
    # rounding noise is part of the frozen oracle and no jnp solver reproduces it;
    # the evaluation and the subtraction run on the device.
    low = int(np.argmin(y_arr))
    coefficients = np.polyfit(
        [x_arr[0], x_arr[low], x_arr[-1]], [y_arr[0], y_arr[low], y_arr[-1]], 2
    )
    return _run_series("polynomial_background", x_arr, y_arr, extra=(coefficients,))


def normalize_intensity(y: ArrayLike, method: str = "max") -> np.ndarray:
    """
    Normalize intensity data.

    Args:
        y: Intensity data. ``'max'`` and ``'standard'`` accept any shape (global
            maximum, mean and standard deviation) and a 0-d input; ``'area'``
            needs one dimension (``ValueError`` otherwise)
        method: Normalization method ('max', 'area', 'standard')

    Returns:
        Normalized intensity data, float64, the shape of ``y`` (a NumPy scalar
        for a 0-d input)
    """
    y_arr = _real(y)
    if method not in ("max", "area", "standard"):
        raise ValueError("Method must be 'max', 'area', or 'standard'")
    if method == "area":
        y_arr = _series(y_arr)
    if y_arr.size == 0 and method == "max":
        raise ValueError(
            "zero-size array to reduction operation maximum which has no identity"
        )
    out = _run_series("normalize_intensity", y_arr.ravel(), method=method)
    out = out.reshape(y_arr.shape)
    return out[()] if out.ndim == 0 else out


def progress_bar(iterable: Any, desc: str = "Processing") -> Any | Iterator[Any]:
    """
    Create a progress bar for iterations.

    Args:
        iterable: Iterable to wrap
        desc: Description for progress bar

    Returns:
        tqdm progress bar
    """
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        # Fallback if tqdm is not available
        return iterable


def save_processed_data(
    x: ArrayLike,
    y: ArrayLike,
    filename: str,
    header: str = "# X-ray diffraction data",
) -> None:
    """
    Save processed data to file.

    Args:
        x: x-axis data
        y: y-axis data
        filename: Output filename
        header: Header comment for file
    """
    data = np.column_stack((x, y))
    np.savetxt(filename, data, header=header, fmt="%.6f")


def _validate_balanced_parentheses(formula: str, original: str) -> None:
    """Raise FormulaError if `formula`'s parentheses aren't balanced."""
    from xraylabtool.exceptions import FormulaError

    depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            raise FormulaError("Unmatched closing parenthesis", original)
    if depth != 0:
        raise FormulaError("Unmatched opening parenthesis", original)


def _expand_parentheses(formula: str, original: str | None = None) -> str:
    """Expand every parenthesized group (innermost first) into a flat element string."""
    from xraylabtool.exceptions import FormulaError

    while "(" in formula:
        start = formula.rfind("(")
        end = formula.find(")", start)
        inside = formula[start + 1 : end]
        # Validate the group before rebuilding it: expansion keeps only the tokens
        # that match, so anything else (signs, empty groups) would vanish silently.
        if not inside:
            raise FormulaError("Empty parentheses in formula", original or formula)
        leftover = re.sub(r"([A-Z][a-z]*)(\d*\.?\d*)", "", inside)
        if leftover:
            raise FormulaError(
                f"Invalid characters in formula: {leftover!r}", original or formula
            )

        # Multiplier after closing paren (decimal or integer)
        rest = formula[end + 1 :]
        mult_match = re.match(r"(\d+\.?\d*)", rest)
        multiplier = float(mult_match.group(1)) if mult_match else 1.0

        # Parse the content inside parentheses
        inner_elements: dict[str, float] = {}
        for sym, cnt in re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", inside):
            count = float(cnt) if cnt else 1.0
            inner_elements[sym] = inner_elements.get(sym, 0.0) + count

        # Build expanded string
        expanded_parts: list[str] = []
        for sym, cnt in inner_elements.items():
            total = cnt * multiplier
            # Positional notation: str(1e-05) would be unparseable by the element regex
            expanded_parts.append(f"{sym}{np.format_float_positional(total, trim='-')}")

        expanded = "".join(expanded_parts)
        mult_len = len(mult_match.group(1)) if mult_match else 0
        formula = formula[:start] + expanded + formula[end + 1 + mult_len :]

    return formula


def _aggregate_element_counts(
    matches: list[tuple[str, str]],
) -> tuple[list[str], list[float]]:
    """Sum counts per element symbol, preserving first-seen order."""
    seen: dict[str, int] = {}
    element_symbols: list[str] = []
    element_counts: list[float] = []

    for sym, cnt_str in matches:
        if not sym:
            continue
        count = float(cnt_str) if cnt_str else 1.0
        if sym in seen:
            element_counts[seen[sym]] += count
        else:
            seen[sym] = len(element_symbols)
            element_symbols.append(sym)
            element_counts.append(count)

    return element_symbols, element_counts


def parse_formula(formula_str: str) -> tuple[list[str], list[float]]:
    """
    Parse a chemical formula string into element symbols and their counts.

    Canonical formula parser for the entire codebase.  Supports:
    - Element symbols with integer or decimal stoichiometry (``SiO2``, ``H0.5He0.5``)
    - Parentheses, including nested (``Ca5(PO4)3OH``, ``Ca10(PO4)6(OH)2``)

    Args:
        formula_str: Chemical formula string (e.g., "SiO2", "Al2O3", "Ca5(PO4)3OH")

    Returns:
        Tuple of (element_symbols, element_counts) where:
        - element_symbols: List of element symbols as strings
        - element_counts: List of corresponding stoichiometric counts as floats

    Raises:
        FormulaError: If formula string is empty, contains unmatched parentheses,
            or no elements are found.

    Examples:
        >>> from xraylabtool.utils import parse_formula
        >>> parse_formula("SiO2")
        (['Si', 'O'], [1.0, 2.0])
        >>> parse_formula("Ca5(PO4)3OH")
        (['Ca', 'P', 'O', 'H'], [5.0, 3.0, 13.0, 1.0])
    """
    symbols, counts = _parse_formula_cached(formula_str)
    return list(symbols), list(counts)


@lru_cache(maxsize=1024)
def _parse_formula_cached(
    formula_str: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Cached, immutable result of :func:`parse_formula` (exceptions are not cached)."""
    from xraylabtool.exceptions import FormulaError

    if not formula_str or not formula_str.strip():
        raise FormulaError("Empty formula string", formula_str)

    formula = formula_str.replace(" ", "")
    _validate_balanced_parentheses(formula, formula_str)
    formula = _expand_parentheses(formula, formula_str)

    # Final parse of the fully-expanded formula
    matches = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)

    # Reject any characters the element pattern could not consume, e.g. trailing
    # junk ("SiO2junk"), stray separators ("Si-O2"), or operators ("SiO2+").
    # re.findall() silently skips these, so without this check malformed input
    # is calculated as if the junk were absent.
    leftover = re.sub(r"([A-Z][a-z]*)(\d*\.?\d*)", "", formula)
    if leftover:
        raise FormulaError(f"Invalid characters in formula: {leftover!r}", formula_str)

    if not matches or not any(sym for sym, _ in matches):
        raise FormulaError(f"No elements found in formula: {formula_str}", formula_str)

    element_symbols, element_counts = _aggregate_element_counts(matches)

    if not element_symbols:
        raise FormulaError(f"No elements found in formula: {formula_str}", formula_str)

    return tuple(element_symbols), tuple(element_counts)


def _convert_atomic_number_to_int(atomic_num: Any) -> int:
    """Convert atomic number to integer, handling various types."""
    try:
        if isinstance(atomic_num, int | float):
            return int(atomic_num)
        return int(str(atomic_num))
    except (ValueError, TypeError):
        try:
            return int(float(str(atomic_num)))
        except (ValueError, TypeError) as e:
            raise AtomicDataError(
                f"Could not convert atomic number to int: {atomic_num}, error: {e}"
            ) from e


def _handle_mendeleev_error(e: Exception, element_symbol: str) -> NoReturn:
    """Handle errors from mendeleev package."""
    error_str = str(e).lower()
    if "not found" in error_str or "unknown" in error_str:
        raise UnknownElementError(f"Unknown element symbol: '{element_symbol}'")
    else:
        raise AtomicDataError(
            f"Could not load atomic number for element '{element_symbol}': {e}"
        )


@lru_cache(maxsize=128)
def get_atomic_number(element_symbol: str) -> int:
    """
    Get atomic number for given element symbol with LRU caching.

    Args:
        element_symbol: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Al')

    Returns:
        Atomic number as integer

    Raises:
        ~xraylabtool.validation.exceptions.UnknownElementError: If element
            symbol is not recognized
        ~xraylabtool.validation.exceptions.AtomicDataError: If there's an
            issue loading atomic data

    Examples:
        >>> from xraylabtool.utils import get_atomic_number
        >>> get_atomic_number('H')
        1
        >>> get_atomic_number('C')
        6
        >>> get_atomic_number('Si')
        14
    """
    try:
        from mendeleev import element as get_element

        elem = get_element(element_symbol)
        return int(_convert_atomic_number_to_int(elem.atomic_number))
    except ImportError as e:
        raise AtomicDataError("mendeleev package is required for atomic data") from e
    except ValueError as e:
        _handle_mendeleev_error(e, element_symbol)
        # This line should never be reached as _handle_mendeleev_error always raises
        raise  # pragma: no cover
    except Exception as e:
        raise AtomicDataError(
            "Unexpected error loading atomic number for element "
            f"'{element_symbol}': {e}"
        ) from e


@lru_cache(maxsize=128)
def get_atomic_weight(element_symbol: str) -> float:
    """
    Get atomic weight for given element symbol with LRU caching.

    Args:
        element_symbol: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Al')

    Returns:
        Atomic weight in u (atomic mass units)

    Raises:
        ~xraylabtool.validation.exceptions.UnknownElementError: If element
            symbol is not recognized
        ~xraylabtool.validation.exceptions.AtomicDataError: If there's an
            issue loading atomic data

    Examples:
        >>> from xraylabtool.utils import get_atomic_weight
        >>> round(get_atomic_weight('H'), 3)
        1.008
        >>> round(get_atomic_weight('C'), 3)
        12.011
        >>> round(get_atomic_weight('O'), 3)
        15.999
    """
    try:
        from mendeleev import element as get_element

        elem = get_element(element_symbol)
        if elem.atomic_weight is None:
            raise AtomicDataError(
                f"Atomic weight not available for element '{element_symbol}'"
            )
        atomic_weight: Any = elem.atomic_weight
        # Handle Column type from mendeleev by converting to string first
        try:
            # Try direct conversion for normal types
            if isinstance(atomic_weight, int | float):
                return float(atomic_weight)
            # For Column types and other objects, convert to string then to
            # float
            return float(str(atomic_weight))
        except (ValueError, TypeError) as e:
            raise AtomicDataError(
                f"Could not convert atomic weight to float: {atomic_weight}, error: {e}"
            ) from e
    except ImportError as e:
        raise AtomicDataError("mendeleev package is required for atomic data") from e
    except ValueError as e:
        # mendeleev raises ValueError for unknown elements
        if "not found" in str(e).lower() or "unknown" in str(e).lower():
            raise UnknownElementError(
                f"Unknown element symbol: '{element_symbol}'"
            ) from e
        else:
            raise AtomicDataError(
                f"Could not load atomic weight for element '{element_symbol}': {e}"
            ) from e
    except Exception as e:
        raise AtomicDataError(
            "Unexpected error loading atomic weight for element "
            f"'{element_symbol}': {e}"
        ) from e


@lru_cache(maxsize=128)
def get_atomic_data(element_symbol: str) -> dict[str, Any]:
    """
    Get comprehensive atomic data for given element symbol with LRU caching.

    Args:
        element_symbol: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Al')

    Returns:
        Dictionary containing atomic properties:
        - symbol: Element symbol
        - atomic_number: Atomic number
        - atomic_weight: Atomic weight in u
        - name: Element name
        - density: Density in g/cm³ (if available)

    Raises:
        ~xraylabtool.validation.exceptions.UnknownElementError: If element
            symbol is not recognized
        ~xraylabtool.validation.exceptions.AtomicDataError: If there's an
            issue loading atomic data

    Examples:
        >>> from xraylabtool.utils import get_atomic_data
        >>> data = get_atomic_data('Si')
        >>> data['atomic_number']
        14
        >>> data['symbol']
        'Si'
    """
    try:
        from mendeleev import element as get_element

        elem = get_element(element_symbol)

        return {
            "symbol": elem.symbol,
            "atomic_number": elem.atomic_number,
            "atomic_weight": elem.atomic_weight,
            "name": elem.name,
            "density": elem.density,  # May be None for some elements
        }
    except ImportError as e:
        raise AtomicDataError("mendeleev package is required for atomic data") from e
    except ValueError as e:
        # mendeleev raises ValueError for unknown elements
        if "not found" in str(e).lower() or "unknown" in str(e).lower():
            raise UnknownElementError(
                f"Unknown element symbol: '{element_symbol}'"
            ) from e
        else:
            raise AtomicDataError(
                f"Could not load atomic data for element '{element_symbol}': {e}"
            ) from e
    except Exception as e:
        raise AtomicDataError(
            f"Unexpected error loading atomic data for element '{element_symbol}': {e}"
        ) from e


# Backward compatibility alias
load_atomic_data = get_atomic_data
