"""
Utility functions for XRayLabTool.

This module contains helper functions for data processing, unit conversions,
mathematical operations, and other common tasks in X-ray analysis.
"""

import numpy as np
from scipy import constants
from typing import List, Tuple, NoReturn, Dict, Any, Iterator, Union
import re
from functools import lru_cache


# Physical constants
PLANCK_CONSTANT: float = float(constants.h)  # J⋅s
SPEED_OF_LIGHT: float = float(constants.c)  # m/s
ELECTRON_CHARGE: float = float(constants.e)  # C
AVOGADRO_NUMBER: float = float(constants.N_A)  # mol⁻¹

# Export all public functions
__all__ = [
    "wavelength_to_energy",
    "energy_to_wavelength",
    "bragg_angle",
    "d_spacing_cubic",
    "d_spacing_tetragonal",
    "d_spacing_orthorhombic",
    "q_from_angle",
    "angle_from_q",
    "smooth_data",
    "find_peaks",
    "background_subtraction",
    "normalize_intensity",
    "progress_bar",
    "save_processed_data",
    "parse_formula",  # Chemical formula parser
    "get_atomic_number",
    "get_atomic_weight",
    "get_atomic_data",
    "load_atomic_data",  # Backward compatibility
    "AtomicDataError",
    "UnknownElementError",
]


def wavelength_to_energy(wavelength: float, units: str = "angstrom") -> float:
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

    # Calculate energy using E = hc/λ
    energy_j = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / wavelength_m
    energy_kev = energy_j / (ELECTRON_CHARGE * 1000)

    return energy_kev


def energy_to_wavelength(energy: float, units: str = "angstrom") -> float:
    """
    Convert X-ray energy to wavelength.

    Args:
        energy: Energy in keV
        units: Desired units of wavelength ('angstrom', 'nm', 'm')

    Returns:
        Wavelength in specified units
    """
    # Convert energy to Joules
    energy_j = energy * ELECTRON_CHARGE * 1000

    # Calculate wavelength using λ = hc/E
    wavelength_m = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / energy_j

    # Convert to desired units
    if units == "angstrom":
        return wavelength_m / 1e-10
    elif units == "nm":
        return wavelength_m / 1e-9
    elif units == "m":
        return wavelength_m
    else:
        raise ValueError("Units must be 'angstrom', 'nm', or 'm'")


def bragg_angle(d_spacing: float, wavelength: float, order: int = 1) -> float:
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


def d_spacing_cubic(h: int, k: int, miller_l: int, a: float) -> float:
    """
    Calculate d-spacing for cubic crystal system.

    Args:
        h, k, l: Miller indices
        a: Lattice parameter in Angstroms

    Returns:
        d-spacing in Angstroms
    """
    return float(a / np.sqrt(h**2 + k**2 + miller_l**2))


def d_spacing_tetragonal(h: int, k: int, miller_l: int, a: float, c: float) -> float:
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
    h: int, k: int, miller_l: int, a: float, b: float, c: float
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


def q_from_angle(two_theta: float, wavelength: float) -> float:
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


def angle_from_q(q: float, wavelength: float) -> float:
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


def smooth_data(x: np.ndarray, y: np.ndarray, window_size: int = 5) -> np.ndarray:
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

    if window_size >= len(y):
        return np.full(len(y), np.mean(y), dtype=np.float64)

    # Use numpy convolution - much faster than pandas rolling
    kernel = np.ones(window_size, dtype=np.float64) / window_size

    # Handle edge effects with reflection padding
    half_window = window_size // 2
    padded_y = np.pad(y, half_window, mode="edge")
    convolved = np.convolve(padded_y, kernel, mode="valid")

    # Ensure we return exactly the same length as input
    return convolved[: len(y)]


def find_peaks(
    x: np.ndarray, y: np.ndarray, prominence: float = 0.1, distance: int = 10
) -> Tuple[np.ndarray, Dict[str, Any]]:
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

    peaks, properties = scipy_find_peaks(y, prominence=prominence, distance=distance)

    # Add x-values to properties
    properties["x_values"] = x[peaks]
    properties["y_values"] = y[peaks]

    return peaks, properties


def background_subtraction(
    x: np.ndarray, y: np.ndarray, method: str = "linear"
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
    if method == "linear":
        # Simple linear background between first and last points
        background = np.linspace(y[0], y[-1], len(y))
    elif method == "polynomial":
        # Fit polynomial to endpoints and minimum values
        # This is a simplified implementation
        background = np.full_like(y, np.min(y))
    else:
        raise ValueError("Method must be 'linear' or 'polynomial'")

    return y - background


def normalize_intensity(y: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Normalize intensity data.

    Args:
        y: Intensity data
        method: Normalization method ('max', 'area', 'standard')

    Returns:
        Normalized intensity data
    """
    if method == "max":
        return y / float(np.max(y))
    elif method == "area":
        return y / float(np.trapezoid(y))
    elif method == "standard":
        return (y - float(np.mean(y))) / float(np.std(y))
    else:
        raise ValueError("Method must be 'max', 'area', or 'standard'")


def progress_bar(iterable: Any, desc: str = "Processing") -> Union[Any, Iterator[Any]]:
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
    x: np.ndarray,
    y: np.ndarray,
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


def parse_formula(formula_str: str) -> Tuple[List[str], List[float]]:
    """
    Parse a chemical formula string into element symbols and their counts.

    This function uses the identical regex pattern as the Julia implementation to ensure
    exact compatibility. The regex matches:
    - Capital letter + optional lowercase letter(s) for element symbols
    - Optional numbers (integers or decimals) for stoichiometric counts

    Args:
        formula_str: Chemical formula string (e.g., "SiO2", "Al2O3", "H0.5He0.5")

    Returns:
        Tuple of (element_symbols, element_counts) where:
        - element_symbols: List of element symbols as strings
        - element_counts: List of corresponding stoichiometric counts as floats

    Examples:
        >>> symbols, counts = parse_formula("SiO2")
        >>> print(symbols, counts)
        ['Si', 'O'] [1.0, 2.0]

        >>> symbols, counts = parse_formula("Al2O3")
        >>> print(symbols, counts)
        ['Al', 'O'] [2.0, 3.0]

        >>> symbols, counts = parse_formula("H0.5He0.5")
        >>> print(symbols, counts)
        ['H', 'He'] [0.5, 0.5]

    Raises:
        ValueError: If formula string is invalid or empty
    """
    # Regular expression identical to Julia implementation:
    # Matches: Capital letter + optional lowercase + optional number
    # (int or float)
    elements_match = re.findall(r"([A-Z][a-z]*)(\d*\.\d*|\d*)", formula_str)

    if not elements_match:
        raise ValueError(f"Invalid chemical formula: {formula_str}")

    element_symbols = []
    element_counts = []

    for element_symbol, count_str in elements_match:
        element_symbols.append(element_symbol)

        # Parse count (default to 1.0 if not specified)
        if count_str == "":
            element_counts.append(1.0)
        else:
            element_counts.append(float(count_str))

    return element_symbols, element_counts


class AtomicDataError(Exception):
    """Exception raised when atomic data cannot be retrieved."""

    pass


class UnknownElementError(AtomicDataError):
    """Exception raised when an unknown element symbol is provided."""

    pass


def _convert_atomic_number_to_int(atomic_num: Any) -> int:
    """Convert atomic number to integer, handling various types."""
    try:
        if isinstance(atomic_num, (int, float)):
            return int(atomic_num)
        return int(str(atomic_num))
    except (ValueError, TypeError):
        try:
            return int(float(str(atomic_num)))
        except (ValueError, TypeError) as e:
            raise AtomicDataError(
                f"Could not convert atomic number to int: {atomic_num}, error: {e}"
            )


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
        UnknownElementError: If element symbol is not recognized
        AtomicDataError: If there's an issue loading atomic data

    Examples:
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
    except ImportError:
        raise AtomicDataError("mendeleev package is required for atomic data")
    except ValueError as e:
        _handle_mendeleev_error(e, element_symbol)
        # This line should never be reached as _handle_mendeleev_error always raises
        raise  # pragma: no cover
    except Exception as e:
        raise AtomicDataError(
            f"Unexpected error loading atomic number for element "
            f"'{element_symbol}': {e}"
        )


@lru_cache(maxsize=128)
def get_atomic_weight(element_symbol: str) -> float:
    """
    Get atomic weight for given element symbol with LRU caching.

    Args:
        element_symbol: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Al')

    Returns:
        Atomic weight in u (atomic mass units)

    Raises:
        UnknownElementError: If element symbol is not recognized
        AtomicDataError: If there's an issue loading atomic data

    Examples:
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
            if isinstance(atomic_weight, (int, float)):
                return float(atomic_weight)
            # For Column types and other objects, convert to string then to
            # float
            return float(str(atomic_weight))
        except (ValueError, TypeError) as e:
            raise AtomicDataError(
                f"Could not convert atomic weight to float: "
                f"{atomic_weight}, error: {e}"
            )
    except ImportError:
        raise AtomicDataError("mendeleev package is required for atomic data")
    except ValueError as e:
        # mendeleev raises ValueError for unknown elements
        if "not found" in str(e).lower() or "unknown" in str(e).lower():
            raise UnknownElementError(f"Unknown element symbol: '{element_symbol}'")
        else:
            raise AtomicDataError(
                f"Could not load atomic weight for element '{element_symbol}': {e}"
            )
    except Exception as e:
        raise AtomicDataError(
            f"Unexpected error loading atomic weight for element "
            f"'{element_symbol}': {e}"
        )


@lru_cache(maxsize=128)
def get_atomic_data(element_symbol: str) -> Dict[str, Any]:
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
        UnknownElementError: If element symbol is not recognized
        AtomicDataError: If there's an issue loading atomic data

    Examples:
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
    except ImportError:
        raise AtomicDataError("mendeleev package is required for atomic data")
    except ValueError as e:
        # mendeleev raises ValueError for unknown elements
        if "not found" in str(e).lower() or "unknown" in str(e).lower():
            raise UnknownElementError(f"Unknown element symbol: '{element_symbol}'")
        else:
            raise AtomicDataError(
                f"Could not load atomic data for element " f"'{element_symbol}': {e}"
            )
    except Exception as e:
        raise AtomicDataError(
            f"Unexpected error loading atomic data for element '{element_symbol}': {e}"
        )


# Backward compatibility alias
load_atomic_data = get_atomic_data
