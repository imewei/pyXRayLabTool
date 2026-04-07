"""
Utility functions for XRayLabTool.

This module contains helper functions for data processing, unit conversions,
mathematical operations, and other common tasks in X-ray analysis.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
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
    # Convert wavelength to meters
    if units == "angstrom":
        wavelength_m = wavelength * 1e-10
    elif units == "nm":
        wavelength_m = wavelength * 1e-9
    elif units == "m":
        wavelength_m = wavelength
    else:
        raise ValueError("Units must be 'angstrom', 'nm', or 'm'")

    # Get constants explicitly
    planck_const = get_planck_constant()
    speed_light = get_speed_of_light()
    electron_charge = get_electron_charge()

    # Calculate energy using E = hc/λ
    energy_j = (planck_const * speed_light) / wavelength_m
    energy_kev = energy_j / (electron_charge * 1000)

    return float(energy_kev)


def energy_to_wavelength(energy: FloatLike, units: str = "angstrom") -> float:
    """
    Convert X-ray energy to wavelength.

    Args:
        energy: Energy in keV
        units: Desired units of wavelength ('angstrom', 'nm', 'm')

    Returns:
        Wavelength in specified units
    """
    # Get constants explicitly
    planck_const = get_planck_constant()
    speed_light = get_speed_of_light()
    electron_charge = get_electron_charge()

    # Convert energy to Joules
    energy_j = energy * electron_charge * 1000

    # Calculate wavelength using λ = hc/E
    wavelength_m = (planck_const * speed_light) / energy_j

    # Convert to desired units
    if units == "angstrom":
        return float(wavelength_m / 1e-10)
    elif units == "nm":
        return float(wavelength_m / 1e-9)
    elif units == "m":
        return float(wavelength_m)
    else:
        raise ValueError("Units must be 'angstrom', 'nm', or 'm'")


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

    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)

    return float(theta_deg)


def d_spacing_cubic(h: int, k: int, miller_l: int, a: FloatLike) -> float:
    """
    Calculate d-spacing for cubic crystal system.

    Args:
        h, k, l: Miller indices
        a: Lattice parameter in Angstroms

    Returns:
        d-spacing in Angstroms
    """
    return float(a / np.sqrt(h**2 + k**2 + miller_l**2))


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
    return float(1 / np.sqrt((h**2 + k**2) / a**2 + miller_l**2 / c**2))


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
    return float(1 / np.sqrt(h**2 / a**2 + k**2 / b**2 + miller_l**2 / c**2))


def q_from_angle(two_theta: FloatLike, wavelength: FloatLike) -> float:
    """
    Calculate momentum transfer q from scattering angle.

    Args:
        two_theta: Scattering angle (2θ) in degrees
        wavelength: X-ray wavelength in Angstroms

    Returns:
        Momentum transfer q in Ų⁻¹
    """
    theta_rad = np.radians(two_theta / 2)
    q = (4 * np.pi * np.sin(theta_rad)) / wavelength
    return float(q)


def angle_from_q(q: FloatLike, wavelength: FloatLike) -> float:
    """
    Calculate scattering angle from momentum transfer q.

    Args:
        q: Momentum transfer in Ų⁻¹
        wavelength: X-ray wavelength in Angstroms

    Returns:
        Scattering angle (2θ) in degrees
    """
    sin_theta = (q * wavelength) / (4 * np.pi)

    if sin_theta > 1:
        raise ValueError("Invalid q value for given wavelength")

    theta_rad = np.arcsin(sin_theta)
    two_theta_deg = 2 * np.degrees(theta_rad)

    return float(two_theta_deg)


def smooth_data(x: ArrayLike, y: ArrayLike, window_size: int = 5) -> np.ndarray:
    """
    Apply moving average smoothing to data using optimized NumPy convolution.

    This optimized version uses numpy convolution instead of pandas rolling,
    providing 3-5x speedup for typical use cases.

    Args:
        x: x-axis data (not used but kept for consistency)
        y: y-axis data to smooth
        window_size: Size of smoothing window

    Returns:
        Smoothed y data
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    # Convert to numpy array for consistent operations
    y_arr = np.asarray(y)

    if window_size >= len(y_arr):
        return np.full(len(y_arr), np.mean(y_arr), dtype=np.float64)

    # Use numpy convolution - much faster than pandas rolling
    kernel = np.ones(window_size, dtype=np.float64) / window_size

    # Handle edge effects with reflection padding
    half_window = window_size // 2
    padded_y = np.pad(y_arr, half_window, mode="edge")
    convolved = np.convolve(padded_y, kernel, mode="valid")

    # Ensure we return exactly the same length as input
    return convolved[: len(y_arr)]


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
        x: x-axis data
        y: y-axis data
        method: Background subtraction method ('linear', 'polynomial')

    Returns:
        Background-subtracted y data
    """
    # Convert to numpy arrays for consistent indexing
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if method == "linear":
        # Linear background between first and last points
        background = np.linspace(y_arr[0], y_arr[-1], len(y_arr))
    elif method == "polynomial":
        # Fit quadratic polynomial using x values
        # Use endpoints and minimum point for fitting
        min_idx = np.argmin(y_arr)
        x_points = np.array([x_arr[0], x_arr[min_idx], x_arr[-1]])
        y_points = np.array([y_arr[0], y_arr[min_idx], y_arr[-1]])
        coeffs = np.polyfit(x_points, y_points, 2)
        background = np.polyval(coeffs, x_arr)
    else:
        raise ValueError("Method must be 'linear' or 'polynomial'")

    return y_arr - background


def normalize_intensity(y: ArrayLike, method: str = "max") -> np.ndarray:
    """
    Normalize intensity data.

    Args:
        y: Intensity data
        method: Normalization method ('max', 'area', 'standard')

    Returns:
        Normalized intensity data
    """
    # Convert to numpy array for consistent operations
    y_arr = np.asarray(y)

    if method == "max":
        return y_arr / float(np.max(y_arr))
    elif method == "area":
        return y_arr / float(np.trapezoid(y_arr))
    elif method == "standard":
        return (y_arr - float(np.mean(y_arr))) / float(np.std(y_arr))
    else:
        raise ValueError("Method must be 'max', 'area', or 'standard'")


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
        (['Ca', 'P', 'O', 'H'], [5.0, 3.0, 12.0, 1.0])
    """
    from xraylabtool.exceptions import FormulaError

    if not formula_str or not formula_str.strip():
        raise FormulaError("Empty formula string", formula_str)

    formula = formula_str.replace(" ", "")

    # Validate matched parentheses
    depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            raise FormulaError("Unmatched closing parenthesis", formula_str)
    if depth != 0:
        raise FormulaError("Unmatched opening parenthesis", formula_str)

    # Expand parentheses iteratively (innermost first)
    while "(" in formula:
        start = formula.rfind("(")
        end = formula.find(")", start)
        inside = formula[start + 1 : end]

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
            expanded_parts.append(f"{sym}{total}")

        expanded = "".join(expanded_parts)
        mult_len = len(mult_match.group(1)) if mult_match else 0
        formula = formula[:start] + expanded + formula[end + 1 + mult_len :]

    # Final parse of the fully-expanded formula
    matches = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)

    if not matches or not any(sym for sym, _ in matches):
        raise FormulaError(f"No elements found in formula: {formula_str}", formula_str)

    # Aggregate counts per element while preserving first-seen order
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

    if not element_symbols:
        raise FormulaError(f"No elements found in formula: {formula_str}", formula_str)

    return element_symbols, element_counts


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
        atomic_weight = elem.atomic_weight
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
