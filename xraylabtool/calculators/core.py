"""
Core functionality for XRayLabTool.

This module contains the main classes and functions for X-ray analysis,
including atomic scattering factors and crystallographic calculations.
"""

from collections.abc import Callable
import concurrent.futures
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
import types
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# =====================================================================================
# DATA STRUCTURES
# =====================================================================================


@dataclass
class XRayResult:
    """
    Dataclass containing complete X-ray optical property calculations for a material.

    This comprehensive data structure holds all computed X-ray properties including
    fundamental scattering factors, optical constants, and derived quantities like
    critical angles and attenuation lengths. All fields use descriptive snake_case
    names with clear units for maximum clarity.

    The dataclass is optimized for scientific workflows, supporting both single-energy
    calculations and energy-dependent analysis. All array fields are automatically
    converted to numpy arrays for efficient numerical operations.

    **Legacy Compatibility:**
    Deprecated CamelCase property aliases are available for backward compatibility
    but emit DeprecationWarning when accessed. Use the new snake_case field names
    for all new code.

    Attributes:
        Material Properties:

        formula (str): Chemical formula string exactly as provided
        molecular_weight_g_mol (float): Molecular weight in g/mol
        total_electrons (float): Total electrons per molecule (sum over all atoms)
        density_g_cm3 (float): Mass density in g/cm³
        electron_density_per_ang3 (float): Electron density in electrons/Å³

        X-ray Energy and Wavelength:

        energy_kev (np.ndarray): X-ray photon energies in keV
        wavelength_angstrom (np.ndarray): Corresponding X-ray wavelengths in Å

        Fundamental X-ray Properties:

        dispersion_delta (np.ndarray): Dispersion coefficient δ (real part of
                                      refractive index decrement: n = 1 - δ - iβ)
        absorption_beta (np.ndarray): Absorption coefficient β (imaginary part of
                                     refractive index decrement)
        scattering_factor_f1 (np.ndarray): Real part of atomic scattering factor
        scattering_factor_f2 (np.ndarray): Imaginary part of atomic scattering factor

        Derived Optical Properties:

        critical_angle_degrees (np.ndarray): Critical angles for total external
                                            reflection in degrees (θc = √(2δ))
        attenuation_length_cm (np.ndarray): 1/e penetration depths in cm
        real_sld_per_ang2 (np.ndarray): Real part of scattering length density in Å⁻²
        imaginary_sld_per_ang2 (np.ndarray): Imaginary part of scattering length
                                            density in Å⁻²

    Physical Relationships:

    - Refractive Index: n = 1 - δ - iβ where δ and β are wavelength-dependent
    - Critical Angle: θc = √(2δ) for grazing incidence geometry
    - Attenuation Length: μ^-1 = (4πβ/λ)^-1 for exponential decay
    - Dispersion/Absorption: Related to f1, f2 via classical electron radius

    Examples:
        Basic Property Access:

        >>> import xraylabtool as xlt
        >>> result = xlt.calculate_single_material_properties("SiO2", 10.0, 2.2)
        >>> print(f"Material: {result.formula}")
        Material: SiO2
        >>> print(f"MW: {result.molecular_weight_g_mol:.2f} g/mol")
        MW: 60.08 g/mol
        >>> print(result.critical_angle_degrees[0] > 0.1)  # Reasonable critical angle
        True

        Array Properties for Energy Scans:

        >>> import numpy as np
        >>> energies = np.linspace(8, 12, 5)
        >>> result = xlt.calculate_single_material_properties("Si", energies, 2.33)
        >>> print(f"Energies: {result.energy_kev}")
        Energies: [ 8.  9. 10. 11. 12.]
        >>> print(len(result.wavelength_angstrom))
        5

        Optical Constants Analysis:

        >>> print(result.dispersion_delta.min() > 0)  # δ should be positive
        True
        >>> print(result.absorption_beta.min() >= 0)  # β should be non-negative
        True

        Derived Quantities:

        >>> print(len(result.critical_angle_degrees))
        5
        >>> print(len(result.attenuation_length_cm))
        5

    Note:
        All numpy arrays have the same length as the input energy array. For scalar
        energy inputs, arrays will have length 1. Use standard numpy operations
        for analysis (e.g., np.min(), np.max(), np.argmin(), indexing).

    See Also:
        calculate_single_material_properties : Primary function returning this class
        calculate_xray_properties : Function returning Dict[str, XRayResult]
    """

    # New snake_case field names
    formula: str  # Chemical formula
    molecular_weight_g_mol: float  # Molecular weight (g/mol)
    total_electrons: float  # Electrons per molecule
    density_g_cm3: float  # Mass density (g/cm³)
    electron_density_per_ang3: float  # Electron density (electrons/Å³)
    energy_kev: np.ndarray = field()  # X-ray energy (keV)
    wavelength_angstrom: np.ndarray = field()  # X-ray wavelength (Å)
    dispersion_delta: np.ndarray = field()  # Dispersion coefficient δ
    absorption_beta: np.ndarray = field()  # Absorption coefficient β
    scattering_factor_f1: np.ndarray = field()  # Real part of atomic scattering factor
    scattering_factor_f2: np.ndarray = (
        field()
    )  # Imaginary part of atomic scattering factor
    critical_angle_degrees: np.ndarray = field()  # Critical angle (degrees)
    attenuation_length_cm: np.ndarray = field()  # Attenuation length (cm)
    real_sld_per_ang2: np.ndarray = field()  # Real part of SLD (Å⁻²)
    imaginary_sld_per_ang2: np.ndarray = field()  # Imaginary part of SLD (Å⁻²)

    def __post_init__(self) -> None:
        """Post-initialization to handle any setup after object creation."""
        # Ensure all arrays are numpy arrays - only convert if necessary
        # mypy: These checks are necessary at runtime even though types are declared
        # Runtime conversion to numpy arrays if needed
        # Convert all array fields to numpy arrays
        self.energy_kev = np.asarray(self.energy_kev)
        self.wavelength_angstrom = np.asarray(self.wavelength_angstrom)
        self.dispersion_delta = np.asarray(self.dispersion_delta)
        self.absorption_beta = np.asarray(self.absorption_beta)
        self.scattering_factor_f1 = np.asarray(self.scattering_factor_f1)
        self.scattering_factor_f2 = np.asarray(self.scattering_factor_f2)
        self.critical_angle_degrees = np.asarray(self.critical_angle_degrees)
        self.attenuation_length_cm = np.asarray(self.attenuation_length_cm)
        self.real_sld_per_ang2 = np.asarray(self.real_sld_per_ang2)
        self.imaginary_sld_per_ang2 = np.asarray(self.imaginary_sld_per_ang2)

    # Legacy property aliases (deprecated) - emit warnings when accessed
    @property
    def Formula(self) -> str:
        """Deprecated: Use 'formula' instead."""
        warnings.warn(
            "Formula is deprecated, use 'formula' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.formula

    @property
    def MW(self) -> float:
        """Deprecated: Use 'molecular_weight_g_mol' instead."""
        warnings.warn(
            "MW is deprecated, use 'molecular_weight_g_mol' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molecular_weight_g_mol

    @property
    def Number_Of_Electrons(self) -> float:
        """Deprecated: Use 'total_electrons' instead."""
        warnings.warn(
            "Number_Of_Electrons is deprecated, use 'total_electrons' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.total_electrons

    @property
    def Density(self) -> float:
        """Deprecated: Use 'density_g_cm3' instead."""
        warnings.warn(
            "Density is deprecated, use 'density_g_cm3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.density_g_cm3

    @property
    def Electron_Density(self) -> float:
        """Deprecated: Use 'electron_density_per_ang3' instead."""
        warnings.warn(
            "Electron_Density is deprecated, use 'electron_density_per_ang3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.electron_density_per_ang3

    @property
    def Energy(self) -> np.ndarray:
        """Deprecated: Use 'energy_kev' instead."""
        warnings.warn(
            "Energy is deprecated, use 'energy_kev' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.energy_kev

    @property
    def Wavelength(self) -> np.ndarray:
        """Deprecated: Use 'wavelength_angstrom' instead."""
        warnings.warn(
            "Wavelength is deprecated, use 'wavelength_angstrom' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wavelength_angstrom

    @property
    def Dispersion(self) -> np.ndarray:
        """Deprecated: Use 'dispersion_delta' instead."""
        warnings.warn(
            "Dispersion is deprecated, use 'dispersion_delta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dispersion_delta

    @property
    def Absorption(self) -> np.ndarray:
        """Deprecated: Use 'absorption_beta' instead."""
        warnings.warn(
            "Absorption is deprecated, use 'absorption_beta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.absorption_beta

    @property
    def f1(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f1' instead."""
        warnings.warn(
            "f1 is deprecated, use 'scattering_factor_f1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f1

    @property
    def f2(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f2' instead."""
        warnings.warn(
            "f2 is deprecated, use 'scattering_factor_f2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f2

    @property
    def Critical_Angle(self) -> np.ndarray:
        """Deprecated: Use 'critical_angle_degrees' instead."""
        warnings.warn(
            "Critical_Angle is deprecated, use 'critical_angle_degrees' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.critical_angle_degrees

    @property
    def Attenuation_Length(self) -> np.ndarray:
        """Deprecated: Use 'attenuation_length_cm' instead."""
        warnings.warn(
            "Attenuation_Length is deprecated, use 'attenuation_length_cm' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.attenuation_length_cm

    @property
    def reSLD(self) -> np.ndarray:
        """Deprecated: Use 'real_sld_per_ang2' instead."""
        warnings.warn(
            "reSLD is deprecated, use 'real_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.real_sld_per_ang2

    @property
    def imSLD(self) -> np.ndarray:
        """Deprecated: Use 'imaginary_sld_per_ang2' instead."""
        warnings.warn(
            "imSLD is deprecated, use 'imaginary_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imaginary_sld_per_ang2

    @classmethod
    def from_legacy(
        cls,
        Formula: str | None = None,
        MW: float | None = None,
        Number_Of_Electrons: float | None = None,
        Density: float | None = None,
        Electron_Density: float | None = None,
        Energy: np.ndarray | None = None,
        Wavelength: np.ndarray | None = None,
        Dispersion: np.ndarray | None = None,
        Absorption: np.ndarray | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        Critical_Angle: np.ndarray | None = None,
        Attenuation_Length: np.ndarray | None = None,
        reSLD: np.ndarray | None = None,
        imSLD: np.ndarray | None = None,
        **kwargs: Any,
    ) -> "XRayResult":
        """Create XRayResult from legacy field names (for internal use)."""
        return cls(
            formula=Formula or kwargs.get("formula", ""),
            molecular_weight_g_mol=MW or kwargs.get("molecular_weight_g_mol", 0.0),
            total_electrons=Number_Of_Electrons or kwargs.get("total_electrons", 0.0),
            density_g_cm3=Density or kwargs.get("density_g_cm3", 0.0),
            electron_density_per_ang3=(
                Electron_Density or kwargs.get("electron_density_per_ang3", 0.0)
            ),
            energy_kev=(
                Energy if Energy is not None else kwargs.get("energy_kev", np.array([]))
            ),
            wavelength_angstrom=(
                Wavelength
                if Wavelength is not None
                else kwargs.get("wavelength_angstrom", np.array([]))
            ),
            dispersion_delta=(
                Dispersion
                if Dispersion is not None
                else kwargs.get("dispersion_delta", np.array([]))
            ),
            absorption_beta=(
                Absorption
                if Absorption is not None
                else kwargs.get("absorption_beta", np.array([]))
            ),
            scattering_factor_f1=(
                f1
                if f1 is not None
                else kwargs.get("scattering_factor_f1", np.array([]))
            ),
            scattering_factor_f2=(
                f2
                if f2 is not None
                else kwargs.get("scattering_factor_f2", np.array([]))
            ),
            critical_angle_degrees=(
                Critical_Angle
                if Critical_Angle is not None
                else kwargs.get("critical_angle_degrees", np.array([]))
            ),
            attenuation_length_cm=(
                Attenuation_Length
                if Attenuation_Length is not None
                else kwargs.get("attenuation_length_cm", np.array([]))
            ),
            real_sld_per_ang2=(
                reSLD
                if reSLD is not None
                else kwargs.get("real_sld_per_ang2", np.array([]))
            ),
            imaginary_sld_per_ang2=(
                imSLD
                if imSLD is not None
                else kwargs.get("imaginary_sld_per_ang2", np.array([]))
            ),
        )


# =====================================================================================
# CACHING SYSTEM
# =====================================================================================

# Module-level cache for f1/f2 scattering tables, keyed by element symbol
_scattering_factor_cache: dict[str, pd.DataFrame] = {}

# Module-level cache for interpolators to avoid repeated creation
_interpolator_cache: dict[str, tuple[PchipInterpolator, PchipInterpolator]] = {}

# Pre-computed element file paths for faster access
_AVAILABLE_ELEMENTS: dict[str, Path] = {}

# Atomic data cache for bulk lookups
_atomic_data_cache: dict[str, dict[str, float]] = {}


def _initialize_element_paths() -> None:
    """
    Pre-compute all available element file paths at module load time.
    This optimization eliminates repeated file system checks.
    """

    base_paths = [
        Path.cwd() / "src" / "AtomicScatteringFactor",
        Path(__file__).parent.parent.parent
        / "src"
        / "AtomicScatteringFactor",  # For old structure compatibility
        Path(__file__).parent.parent
        / "data"
        / "AtomicScatteringFactor",  # New structure
    ]

    for base_path in base_paths:
        if base_path.exists():
            for nff_file in base_path.glob("*.nff"):
                element = nff_file.stem.capitalize()
                if element not in _AVAILABLE_ELEMENTS:
                    _AVAILABLE_ELEMENTS[element] = nff_file


def load_scattering_factor_data(element: str) -> pd.DataFrame:
    """
    Load f1/f2 scattering factor data for a specific element from .nff files.

    This function reads .nff files using pandas.read_csv and caches the results
    in a module-level dictionary keyed by element symbol.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

    Returns:
        DataFrame containing columns: E (energy), f1, f2

    Raises:
        FileNotFoundError: If the .nff file for the element is not found
        ValueError: If the element symbol is invalid or empty
        pd.errors.EmptyDataError: If the .nff file is empty or corrupted
        pd.errors.ParserError: If the .nff file format is invalid

    Examples:
        >>> from xraylabtool.calculators.core import load_scattering_factor_data
        >>> data = load_scattering_factor_data('Si')
        >>> print(data.columns.tolist())
        ['E', 'f1', 'f2']
        >>> print(len(data) > 100)  # Verify we have enough data points
        True
    """

    # Validate input
    if not element or not isinstance(element, str):
        raise ValueError(
            f"Element symbol must be a non-empty string, got: {repr(element)}"
        )

    # Normalize element symbol (capitalize first letter, lowercase rest)
    element = element.capitalize()

    # Check if already cached
    if element in _scattering_factor_cache:
        return _scattering_factor_cache[element]

    # Use pre-computed element paths for faster access
    if element not in _AVAILABLE_ELEMENTS:
        raise FileNotFoundError(
            f"Scattering factor data file not found for element '{element}'. "
            f"Available elements: {sorted(_AVAILABLE_ELEMENTS.keys())}"
        )

    file_path = _AVAILABLE_ELEMENTS[element]

    try:
        # Load .nff file using pandas.read_csv
        # .nff files are CSV format with header: E,f1,f2
        scattering_data = pd.read_csv(file_path)

        # Verify expected columns exist
        expected_columns = {"E", "f1", "f2"}
        actual_columns = set(scattering_data.columns)

        if not expected_columns.issubset(actual_columns):
            missing_cols = expected_columns - actual_columns
            raise ValueError(
                f"Invalid .nff file format for element '{element}'. "
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {list(actual_columns)}"
            )

        # Verify data is not empty
        if scattering_data.empty:
            raise ValueError(
                f"Empty scattering factor data file for element "
                f"'{element}': {file_path}"
            )

        # Cache the data
        _scattering_factor_cache[element] = scattering_data

        return scattering_data

    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(
            f"Empty or corrupted scattering factor data file for element "
            f"'{element}': {file_path}"
        ) from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(
            f"Invalid file format in scattering factor data file for element "
            f"'{element}': {file_path}. "
            f"Expected CSV format with columns: E,f1,f2"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading scattering factor data for element "
            f"'{element}' from {file_path}: {e}"
        ) from e


class AtomicScatteringFactor:
    """
    Class for handling atomic scattering factors.

    This class loads and manages atomic scattering factor data
    from .nff files using the module-level cache.
    """

    def __init__(self) -> None:
        # Maintain backward compatibility with existing tests
        self.data: dict[str, pd.DataFrame] = {}
        self.data_path = (
            Path(__file__).parent.parent / "data" / "AtomicScatteringFactor"
        )

        # Create data directory if it doesn't exist (for test compatibility)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load_element_data(self, element: str) -> pd.DataFrame:
        """
        Load scattering factor data for a specific element.

        Args:
            element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

        Returns:
            DataFrame containing scattering factor data with columns: E, f1, f2

        Raises:
            FileNotFoundError: If the .nff file for the element is not found
            ValueError: If the element symbol is invalid
        """
        return load_scattering_factor_data(element)

    def get_scattering_factor(self, element: str, q_values: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Calculate scattering factors for given q values.

        Args:
            element: Element symbol
            q_values: Array of momentum transfer values

        Returns:
            Array of scattering factor values
        """
        # Placeholder implementation
        return np.ones_like(q_values)


class CrystalStructure:
    """
    Class for representing and manipulating crystal structures.
    """

    def __init__(
        self, lattice_parameters: tuple[float, float, float, float, float, float]
    ):
        """
        Initialize crystal structure.

        Args:
            lattice_parameters: (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        """
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_parameters
        self.atoms: list[dict[str, Any]] = []

    def add_atom(
        self, element: str, position: tuple[float, float, float], occupancy: float = 1.0
    ) -> None:
        """
        Add an atom to the crystal structure.

        Args:
            element: Element symbol
            position: Fractional coordinates (x, y, z)
            occupancy: Site occupancy factor
        """
        self.atoms.append(
            {"element": element, "position": position, "occupancy": occupancy}
        )

    def calculate_structure_factor(self, hkl: tuple[int, int, int]) -> complex:  # noqa: ARG002
        """
        Calculate structure factor for given Miller indices.

        Args:
            hkl: Miller indices (h, k, l)

        Returns:
            Complex structure factor
        """
        # Placeholder implementation
        return complex(1.0, 0.0)


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


def clear_scattering_factor_cache() -> None:
    """
    Clear the module-level scattering factor cache.

    This function removes all cached scattering factor data from memory.
    Useful for testing or memory management.
    """
    _scattering_factor_cache.clear()
    _interpolator_cache.clear()
    _atomic_data_cache.clear()

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


def calculate_scattering_factors(
    energy_ev: np.ndarray,
    wavelength: np.ndarray,
    mass_density: float,
    molecular_weight: float,
    element_data: list[tuple[float, Callable[..., Any], Callable[..., Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized vectorized calculation of X-ray scattering factors and properties.

    This function performs the core calculation of dispersion, absorption, and total
    scattering factors for a material based on its elemental composition.
    Optimized with improved vectorization and memory efficiency.

    Args:
        energy_ev: X-ray energies in eV (numpy array)
        wavelength: Corresponding wavelengths in meters (numpy array)
        mass_density: Material density in g/cm³
        molecular_weight: Molecular weight in g/mol
        element_data: List of tuples (count, f1_interp, f2_interp) for each element

    Returns:
        Tuple of (dispersion, absorption, f1_total, f2_total) arrays

    Mathematical Background:
    The dispersion and absorption coefficients are calculated using:
    - δ = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f1ᵢ) / M
    - β = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f2ᵢ) / M

    Where:
    - λ: X-ray wavelength
    - rₑ: Thomson scattering length
    - ρ: Mass density
    - Nₐ: Avogadro's number
    - nᵢ: Number of atoms of element i
    - f1ᵢ, f2ᵢ: Atomic scattering factors for element i
    - M: Molecular weight
    """
    from xraylabtool.constants import SCATTERING_FACTOR

    n_energies = len(energy_ev)
    n_elements = len(element_data)

    # Pre-allocate arrays for better memory performance
    # Using specific dtypes for better numerical precision and speed
    dispersion = np.zeros(n_energies, dtype=np.float64)
    absorption = np.zeros(n_energies, dtype=np.float64)
    f1_total = np.zeros(n_energies, dtype=np.float64)
    f2_total = np.zeros(n_energies, dtype=np.float64)

    # Pre-compute common constants outside the loop
    common_factor = SCATTERING_FACTOR * mass_density / molecular_weight
    # Use np.square for better performance than ** or *
    wave_sq = np.square(wavelength)

    # Handle empty element data case
    if n_elements == 0:
        # Return zero arrays for empty element data
        return dispersion, absorption, f1_total, f2_total

    # Batch process elements for better vectorization
    if n_elements > 1:
        # For multiple elements, use vectorized operations
        f1_matrix = np.empty((n_elements, n_energies), dtype=np.float64)
        f2_matrix = np.empty((n_elements, n_energies), dtype=np.float64)
        counts = np.empty(n_elements, dtype=np.float64)

        # Vectorized interpolation for all elements
        for i, (count, f1_interp, f2_interp) in enumerate(element_data):
            f1_matrix[i] = f1_interp(energy_ev)
            f2_matrix[i] = f2_interp(energy_ev)
            counts[i] = count

        # Vectorized computation using matrix operations
        # This is much faster than individual loops
        f1_weighted = f1_matrix * counts.reshape(-1, 1)
        f2_weighted = f2_matrix * counts.reshape(-1, 1)

        # Sum across elements (axis=0) for total scattering factors
        f1_total = np.sum(f1_weighted, axis=0)
        f2_total = np.sum(f2_weighted, axis=0)

        # Calculate optical properties with vectorized operations
        wave_factor = wave_sq * common_factor
        dispersion = wave_factor * f1_total
        absorption = wave_factor * f2_total

    else:
        # Single element optimization - avoid matrix operations overhead
        count, f1_interp, f2_interp = element_data[0]

        # Direct vectorized computation for single element
        f1_values = f1_interp(energy_ev)
        f2_values = f2_interp(energy_ev)

        # Ensure arrays are float64 and contiguous for best performance
        # Only convert if not already the right type
        if not isinstance(f1_values, np.ndarray) or f1_values.dtype != np.float64:
            f1_values = np.asarray(f1_values, dtype=np.float64)
        if not isinstance(f2_values, np.ndarray) or f2_values.dtype != np.float64:
            f2_values = np.asarray(f2_values, dtype=np.float64)

        # Pre-compute factors for efficiency
        count_factor = float(count)
        wave_element_factor = wave_sq * (common_factor * count_factor)

        # Direct assignment for single element case - reuse pre-allocated arrays
        f1_total[:] = count_factor * f1_values
        f2_total[:] = count_factor * f2_values
        dispersion[:] = wave_element_factor * f1_values
        absorption[:] = wave_element_factor * f2_values

    return dispersion, absorption, f1_total, f2_total


def calculate_derived_quantities(
    wavelength: np.ndarray,
    dispersion: np.ndarray,
    absorption: np.ndarray,
    mass_density: float,
    molecular_weight: float,
    number_of_electrons: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate derived X-ray optical quantities from dispersion and absorption.

    Args:
        wavelength: X-ray wavelengths in meters (numpy array)
        dispersion: Dispersion coefficients δ (numpy array)
        absorption: Absorption coefficients β (numpy array)
        mass_density: Material density in g/cm³
        molecular_weight: Molecular weight in g/mol
        number_of_electrons: Total electrons per molecule

    Returns:
        Tuple of (electron_density, critical_angle, attenuation_length, re_sld, im_sld)
        - electron_density: Electron density in electrons/Å³ (scalar)
        - critical_angle: Critical angle in degrees (numpy array)
        - attenuation_length: Attenuation length in cm (numpy array)
        - re_sld: Real part of SLD in Å⁻² (numpy array)
        - im_sld: Imaginary part of SLD in Å⁻² (numpy array)
    """
    from xraylabtool.constants import AVOGADRO, PI

    # Numerical stability checks - consistent with existing energy validation
    if np.any(np.isnan(dispersion)) or np.any(np.isnan(absorption)):
        raise ValueError("NaN values detected in dispersion or absorption coefficients")

    if np.any(np.isinf(dispersion)) or np.any(np.isinf(absorption)):
        raise ValueError(
            "Infinite values detected in dispersion or absorption coefficients"
        )

    # Check for negative dispersion values (physically unrealistic)
    if np.any(dispersion < 0):
        raise ValueError("Negative dispersion values detected (physically unrealistic)")

    # Calculate electron density (electrons per unit volume)
    # ρₑ = ρ × Nₐ × Z / M × 10⁻³⁰ (converted to electrons/Å³)
    electron_density = (
        1e6 * mass_density / molecular_weight * AVOGADRO * number_of_electrons / 1e30
    )

    # Calculate critical angle for total external reflection
    # θc = √(2δ) (in radians), converted to degrees
    # Use np.maximum to ensure non-negative values under sqrt
    critical_angle = np.sqrt(np.maximum(2.0 * dispersion, 0.0)) * (180.0 / PI)

    # Calculate X-ray attenuation length
    # 1/e attenuation length = λ/(4πβ) (in cm)
    # Add small epsilon to prevent division by zero
    absorption_safe = np.maximum(absorption, 1e-30)  # Minimum absorption to prevent inf
    attenuation_length = wavelength / absorption_safe / (4 * PI) * 1e2

    # Calculate scattering length densities (SLD)
    # SLD = 2π × (δ + iβ) / λ² (in units of Å⁻²)
    wavelength_sq = wavelength**2
    sld_factor = 2 * PI / 1e20  # Conversion factor to Å⁻²

    re_sld = dispersion * sld_factor / wavelength_sq  # Real part of SLD
    im_sld = absorption * sld_factor / wavelength_sq  # Imaginary part of SLD

    return electron_density, critical_angle, attenuation_length, re_sld, im_sld


@lru_cache(maxsize=128)
def create_scattering_factor_interpolators(
    element: str,
) -> tuple[
    Callable[[float | np.ndarray], float | np.ndarray],
    Callable[[float | np.ndarray], float | np.ndarray],
]:
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
        >>> from xraylabtool.calculators.core import create_scattering_factor_interpolators
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
    # Check interpolator cache first
    if element in _interpolator_cache:
        return _interpolator_cache[element]

    # Load scattering factor data
    scattering_factor_data = load_scattering_factor_data(element)

    # Verify we have sufficient data points for PCHIP interpolation
    if len(scattering_factor_data) < 2:
        raise ValueError(
            f"Insufficient data points for element '{element}'. "
            f"PCHIP interpolation requires at least 2 points, "
            f"found {len(scattering_factor_data)}."
        )

    # Extract energy, f1, and f2 data
    energy_values = np.asarray(scattering_factor_data["E"].values)
    f1_values = np.asarray(scattering_factor_data["f1"].values)
    f2_values = np.asarray(scattering_factor_data["f2"].values)

    # Verify energy values are sorted (PCHIP requires sorted x values)
    if not np.all(energy_values[:-1] <= energy_values[1:]):
        # Sort the data if it's not already sorted
        sort_indices = np.argsort(energy_values)
        energy_values = energy_values[sort_indices]
        f1_values = f1_values[sort_indices]
        f2_values = f2_values[sort_indices]

    # Create PCHIP interpolators
    # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
    # and provides smooth, shape-preserving interpolation similar to Julia's
    # behavior
    f1_interpolator = PchipInterpolator(energy_values, f1_values, extrapolate=False)
    f2_interpolator = PchipInterpolator(energy_values, f2_values, extrapolate=False)

    # Cache the interpolators for future use
    _interpolator_cache[element] = (f1_interpolator, f2_interpolator)

    return f1_interpolator, f2_interpolator


def _validate_single_material_inputs(
    formula_str: str,
    energy_kev: float | list[float] | np.ndarray,
    mass_density: float,
) -> np.ndarray:
    """Validate inputs for single material calculation."""
    if not formula_str or not isinstance(formula_str, str):
        raise ValueError("Formula must be a non-empty string")

    if mass_density <= 0:
        raise ValueError("Mass density must be positive")

    # Convert and validate energy
    energy_kev = _convert_energy_input(energy_kev)

    if np.any(energy_kev <= 0):
        raise ValueError("All energies must be positive")

    if np.any(energy_kev < 0.03) or np.any(energy_kev > 30):
        raise ValueError("Energy is out of range 0.03keV ~ 30keV")

    return energy_kev


def _convert_energy_input(energy_kev: Any) -> np.ndarray:
    """Convert energy input to numpy array."""
    if np.isscalar(energy_kev):
        if isinstance(energy_kev, complex):
            energy_kev = np.array([float(energy_kev.real)], dtype=np.float64)
        elif isinstance(energy_kev, int | float | np.number):
            energy_kev = np.array([float(energy_kev)], dtype=np.float64)
        else:
            try:
                energy_kev = np.array([float(energy_kev)], dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert energy to float: {energy_kev}") from e
    else:
        energy_kev = np.array(energy_kev, dtype=np.float64)

    return np.asarray(energy_kev)


def _calculate_molecular_properties(
    element_symbols: list[str],
    element_counts: list[float],
    atomic_data_bulk: dict[str, types.MappingProxyType[str, float]],
) -> tuple[float, float]:
    """Calculate molecular weight and total electrons."""
    molecular_weight = 0.0
    number_of_electrons = 0.0

    for symbol, count in zip(element_symbols, element_counts, strict=False):
        data = atomic_data_bulk[symbol]
        atomic_number = data["atomic_number"]
        atomic_mass = data["atomic_weight"]

        molecular_weight += count * atomic_mass
        number_of_electrons += atomic_number * count

    return molecular_weight, number_of_electrons


def _prepare_element_data(
    element_symbols: list[str], element_counts: list[float]
) -> list[tuple[float, Any, Any]]:
    """Prepare element data with interpolators."""
    element_data = []

    for i, symbol in enumerate(element_symbols):
        f1_interp, f2_interp = create_scattering_factor_interpolators(symbol)
        element_data.append((element_counts[i], f1_interp, f2_interp))

    return element_data


def _calculate_single_material_xray_properties(
    formula_str: str,
    energy_kev: float | list[float] | np.ndarray,
    mass_density: float,
) -> dict[str, str | float | np.ndarray]:
    """
    Calculate X-ray optical properties for a single chemical formula.

    This function performs comprehensive X-ray optical property calculations
    for a material composition, exactly matching the Julia SubRefrac behavior.

    Args:
        formula_str: Chemical formula (e.g., "SiO2", "Al2O3")
        energy_kev: X-ray energies in keV (scalar, list, or array)
        mass_density: Mass density in g/cm³

    Returns:
        Dictionary containing calculated properties:
        - 'formula': Chemical formula string
        - 'molecular_weight': Molecular weight in g/mol
        - 'number_of_electrons': Total electrons per molecule
        - 'mass_density': Mass density in g/cm³
        - 'electron_density': Electron density in electrons/Å³
        - 'energy': X-ray energies in keV (numpy array)
        - 'wavelength': X-ray wavelengths in Å (numpy array)
        - 'dispersion': Dispersion coefficients δ (numpy array)
        - 'absorption': Absorption coefficients β (numpy array)
        - 'f1_total': Total f1 values (numpy array)
        - 'f2_total': Total f2 values (numpy array)
        - 'critical_angle': Critical angles in degrees (numpy array)
        - 'attenuation_length': Attenuation lengths in cm (numpy array)
        - 're_sld': Real part of SLD in Å⁻² (numpy array)
        - 'im_sld': Imaginary part of SLD in Å⁻² (numpy array)

    Raises:
        ValueError: If formula or energy inputs are invalid
        FileNotFoundError: If atomic scattering data is not available

    Note:
        This is an internal function. Use calculate_single_material_properties()
        for the public API that returns XRayResult objects.
    """
    from xraylabtool.constants import ENERGY_TO_WAVELENGTH_FACTOR, METER_TO_ANGSTROM
    from xraylabtool.utils import parse_formula

    energy_kev = _validate_single_material_inputs(formula_str, energy_kev, mass_density)

    element_symbols, element_counts = parse_formula(formula_str)
    elements_tuple = tuple(element_symbols)
    atomic_data_bulk = get_bulk_atomic_data(elements_tuple)

    molecular_weight, number_of_electrons = _calculate_molecular_properties(
        element_symbols, element_counts, atomic_data_bulk
    )

    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energy_kev
    energy_ev = energy_kev * 1000.0

    element_data = _prepare_element_data(element_symbols, element_counts)

    dispersion, absorption, f1_total, f2_total = calculate_scattering_factors(
        energy_ev, wavelength, mass_density, molecular_weight, element_data
    )

    electron_density, critical_angle, attenuation_length, re_sld, im_sld = (
        calculate_derived_quantities(
            wavelength,
            dispersion,
            absorption,
            mass_density,
            molecular_weight,
            number_of_electrons,
        )
    )

    return {
        "formula": formula_str,
        "molecular_weight": molecular_weight,
        "number_of_electrons": number_of_electrons,
        "mass_density": mass_density,
        "electron_density": electron_density,
        "energy": energy_kev,
        "wavelength": wavelength * METER_TO_ANGSTROM,
        "dispersion": dispersion,
        "absorption": absorption,
        "f1_total": f1_total,
        "f2_total": f2_total,
        "critical_angle": critical_angle,
        "attenuation_length": attenuation_length,
        "re_sld": re_sld,
        "im_sld": im_sld,
    }


def calculate_multiple_xray_properties(
    formula_list: list[str],
    energy_kev: float | list[float] | np.ndarray,
    mass_density_list: list[float],
) -> dict[str, dict[str, str | float | np.ndarray]]:
    """
    Calculate X-ray optical properties for multiple chemical formulas.

    This function processes multiple materials in parallel (using sequential processing
    for Python implementation, but can be extended with multiprocessing if needed).

    Args:
        formula_list: List of chemical formulas
        energy_kev: X-ray energies in keV (scalar, list, or array)
        mass_density_list: Mass densities in g/cm³

    Returns:
        Dictionary mapping formula strings to result dictionaries

    Raises:
        ValueError: If input lists have different lengths or invalid values

    Examples:
        >>> from xraylabtool.calculators.core import calculate_multiple_xray_properties
        >>> formulas = ["SiO2", "Al2O3", "Fe2O3"]
        >>> energies = [8.0, 10.0, 12.0]
        >>> densities = [2.2, 3.95, 5.24]
        >>> results = calculate_multiple_xray_properties(formulas, energies, densities)
        >>> sio2_result = results["SiO2"]
        >>> print(f"SiO2 molecular weight: {sio2_result['molecular_weight']:.2f}")
        SiO2 molecular weight: 60.08
    """
    # Input validation
    if len(formula_list) != len(mass_density_list):
        raise ValueError("Formula list and mass density list must have the same length")

    if not formula_list:
        raise ValueError("Formula list must not be empty")

    # Process each formula
    results = {}

    for formula, mass_density in zip(formula_list, mass_density_list, strict=False):
        try:
            # Calculate properties for this formula
            result = calculate_single_material_properties(
                formula, energy_kev, mass_density
            )

            # Convert XRayResult to dictionary format for backward
            # compatibility
            result_dict: dict[str, str | float | np.ndarray] = {
                "formula": result.Formula,
                "molecular_weight": result.MW,
                "number_of_electrons": result.Number_Of_Electrons,
                "mass_density": result.Density,
                "electron_density": result.Electron_Density,
                "energy": result.Energy,
                "wavelength": result.Wavelength,
                "dispersion": result.Dispersion,
                "absorption": result.Absorption,
                "f1_total": result.f1,
                "f2_total": result.f2,
                "critical_angle": result.Critical_Angle,
                "attenuation_length": result.Attenuation_Length,
                "re_sld": result.reSLD,
                "im_sld": result.imSLD,
            }
            results[formula] = result_dict
        except Exception as e:
            # Log warning but continue processing other formulas
            print(f"Warning: Failed to process formula {formula}: {e}")
            continue

    return results


def load_data_file(filename: str) -> pd.DataFrame:
    """
    Load data from various file formats commonly used in X-ray analysis.

    Args:
        filename: Path to the data file

    Returns:
        DataFrame containing the loaded data
    """
    file_path = Path(filename)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {filename}")

    # Determine file format and load accordingly
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".txt", ".dat"]:
        return pd.read_csv(file_path, delim_whitespace=True)
    else:
        # Try to load as generic text file
        return pd.read_csv(file_path, delim_whitespace=True)


# =====================================================================================
# PUBLIC API FUNCTIONS
# =====================================================================================


def calculate_single_material_properties(
    formula: str, energy_keV: float | list[float] | np.ndarray, density: float
) -> XRayResult:
    """
    Calculate X-ray optical properties for a single material composition.

    This is a pure function that calculates comprehensive X-ray optical properties
    for a single chemical formula at given energies and density. It returns an
    XRayResult dataclass containing all computed properties.

    The function supports both scalar and array energy inputs, making it suitable for
    both single-point calculations and energy-dependent analysis. All calculations
    use high-performance vectorized operations with CXRO/NIST atomic data.

    Args:
        formula: Chemical formula string (e.g., "SiO2", "Al2O3", "CaCO3")
                Case-sensitive - use proper element symbols
        energy_keV: X-ray energies in keV. Accepts:
                   - float: Single energy value
                   - list[float]: Multiple discrete energies
                   - np.ndarray: Energy array for analysis
                   Valid range: 0.03-30.0 keV
        density: Material mass density in g/cm³ (must be positive)

    Returns:
        XRayResult: Dataclass containing all calculated X-ray properties with
                   descriptive field names:

        **Material Properties:**
            - formula: Chemical formula string
            - molecular_weight_g_mol: Molecular weight (g/mol)
            - total_electrons: Total electrons per molecule
            - density_g_cm3: Mass density (g/cm³)
            - electron_density_per_ang3: Electron density (electrons/Å³)

        **X-ray Properties (Arrays):**
            - energy_kev: X-ray energies (keV)
            - wavelength_angstrom: X-ray wavelengths (Å)
            - dispersion_delta: Dispersion coefficients δ
            - absorption_beta: Absorption coefficients β
            - scattering_factor_f1: Real part of atomic scattering factor
            - scattering_factor_f2: Imaginary part of atomic scattering factor

        **Derived Quantities (Arrays):**
            - critical_angle_degrees: Critical angles for total external reflection
            - attenuation_length_cm: 1/e penetration depths (cm)
            - real_sld_per_ang2: Real scattering length density (Å⁻²)
            - imaginary_sld_per_ang2: Imaginary scattering length density (Å⁻²)

    Raises:
        FormulaError: If chemical formula cannot be parsed or contains invalid elements
        EnergyError: If energy values are outside valid range (0.03-30.0 keV)
        ValidationError: If density is not positive or other validation failures
        AtomicDataError: If atomic scattering factor data is unavailable
        CalculationError: If numerical computation fails

    Examples:
        **Basic Usage:**

        >>> import xraylabtool as xlt
        >>> result = xlt.calculate_single_material_properties("SiO2", 8.0, 2.2)
        >>> print(f"Formula: {result.formula}")
        Formula: SiO2
        >>> print(f"Molecular weight: {result.molecular_weight_g_mol:.2f} g/mol")
        Molecular weight: 60.08 g/mol
        >>> print(f"Critical angle: {result.critical_angle_degrees[0]:.3f}°")
        Critical angle: 0.218°

        **Multiple Energies:**

        >>> result = xlt.calculate_single_material_properties(
        ...     "Al2O3", [8.0, 10.0, 12.0], 3.95
        ... )
        >>> print(f"Energies: {result.energy_kev}")
        Energies: [ 8. 10. 12.]
        >>> print(f"Critical angles: {result.critical_angle_degrees}")
        Critical angles: [0.2889117  0.23075529 0.19209348]

        **Energy Range Analysis:**

        >>> import numpy as np
        >>> energies = np.linspace(5.0, 15.0, 11)  # 5-15 keV range
        >>> result = xlt.calculate_single_material_properties("Fe2O3", energies, 5.24)
        >>> print(f"Energy range: {result.energy_kev[0]:.1f} - {result.energy_kev[-1]:.1f} keV")
        Energy range: 5.0 - 15.0 keV
        >>> print(f"Attenuation range: {result.attenuation_length_cm.min():.2f} - {result.attenuation_length_cm.max():.2f} cm")
        Attenuation range: 0.00 - 0.00 cm

        **Performance Note:**
        This function is highly optimized with atomic data caching and vectorized
        operations, achieving 150,000+ calculations/second throughput.

    See Also:
        calculate_xray_properties : Calculate properties for multiple materials
        XRayResult : Complete documentation of returned dataclass
        parse_formula : Parse chemical formulas into elements and counts
    """
    # Calculate properties using the existing function
    properties = _calculate_single_material_xray_properties(
        formula, energy_keV, density
    )

    # Create and return XRayResult dataclass using new field names
    return XRayResult(
        formula=str(properties["formula"]),
        molecular_weight_g_mol=float(properties["molecular_weight"]),
        total_electrons=float(properties["number_of_electrons"]),
        density_g_cm3=float(properties["mass_density"]),
        electron_density_per_ang3=float(properties["electron_density"]),
        energy_kev=np.asarray(properties["energy"]),
        wavelength_angstrom=np.asarray(properties["wavelength"]),
        dispersion_delta=np.asarray(properties["dispersion"]),
        absorption_beta=np.asarray(properties["absorption"]),
        scattering_factor_f1=np.asarray(properties["f1_total"]),
        scattering_factor_f2=np.asarray(properties["f2_total"]),
        critical_angle_degrees=np.asarray(properties["critical_angle"]),
        attenuation_length_cm=np.asarray(properties["attenuation_length"]),
        real_sld_per_ang2=np.asarray(properties["re_sld"]),
        imaginary_sld_per_ang2=np.asarray(properties["im_sld"]),
    )


def _validate_xray_inputs(formulas: list[str], densities: list[float]) -> None:
    """Validate input formulas and densities."""
    if not isinstance(formulas, list) or not formulas:
        raise ValueError("Formulas must be a non-empty list")

    if not isinstance(densities, list) or not densities:
        raise ValueError("Densities must be a non-empty list")

    if len(formulas) != len(densities):
        raise ValueError(
            f"Number of formulas ({len(formulas)}) must match number of "
            f"densities ({len(densities)})"
        )

    for i, formula in enumerate(formulas):
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError(
                f"Formula at index {i} must be a non-empty string, got: {repr(formula)}"
            )

    for i, density in enumerate(densities):
        if not isinstance(density, int | float) or density <= 0:
            raise ValueError(
                f"Density at index {i} must be a positive number, got: {density}"
            )


def _validate_and_process_energies(energies: Any) -> np.ndarray:
    """Validate and convert energies to numpy array."""
    if np.isscalar(energies):
        if isinstance(energies, complex):
            energies_array = np.array([float(energies.real)], dtype=np.float64)
        elif isinstance(energies, int | float | np.number):
            energies_array = np.array([float(energies)], dtype=np.float64)
        else:
            try:
                energies_array = np.array([float(energies)], dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert energy to float: {energies!r}") from e
    else:
        energies_array = np.array(energies, dtype=np.float64)

    if energies_array.size == 0:
        raise ValueError("Energies array cannot be empty")

    if np.any(energies_array <= 0):
        raise ValueError("All energies must be positive")

    if np.any(energies_array < 0.03) or np.any(energies_array > 30):
        raise ValueError("Energy values must be in range 0.03-30 keV")

    return energies_array


def _restore_energy_order(
    result: XRayResult, reverse_indices: np.ndarray
) -> XRayResult:
    """Restore original energy order in XRayResult."""
    return XRayResult(
        formula=result.formula,
        molecular_weight_g_mol=result.molecular_weight_g_mol,
        total_electrons=result.total_electrons,
        density_g_cm3=result.density_g_cm3,
        electron_density_per_ang3=result.electron_density_per_ang3,
        energy_kev=result.energy_kev[reverse_indices],
        wavelength_angstrom=result.wavelength_angstrom[reverse_indices],
        dispersion_delta=result.dispersion_delta[reverse_indices],
        absorption_beta=result.absorption_beta[reverse_indices],
        scattering_factor_f1=result.scattering_factor_f1[reverse_indices],
        scattering_factor_f2=result.scattering_factor_f2[reverse_indices],
        critical_angle_degrees=result.critical_angle_degrees[reverse_indices],
        attenuation_length_cm=result.attenuation_length_cm[reverse_indices],
        real_sld_per_ang2=result.real_sld_per_ang2[reverse_indices],
        imaginary_sld_per_ang2=result.imaginary_sld_per_ang2[reverse_indices],
    )


def _create_process_formula_function(
    sorted_energies: np.ndarray, sort_indices: np.ndarray
) -> Callable[[tuple[str, float]], tuple[str, XRayResult]]:
    """Create process formula function with energy sorting logic."""

    def process_formula(
        formula_density_pair: tuple[str, float],
    ) -> tuple[str, XRayResult]:
        formula, density = formula_density_pair
        try:
            result = calculate_single_material_properties(
                formula, sorted_energies, density
            )

            if not np.array_equal(sort_indices, np.arange(len(sort_indices))):
                reverse_indices = np.argsort(sort_indices)
                result = _restore_energy_order(result, reverse_indices)

            return (formula, result)
        except Exception as e:
            raise RuntimeError(f"Failed to process formula '{formula}': {e}") from e

    return process_formula


def _process_formulas_parallel(
    formulas: list[str],
    densities: list[float],
    process_func: Callable[[tuple[str, float]], tuple[str, XRayResult]],
) -> dict[str, XRayResult]:
    """Process formulas in parallel using ThreadPoolExecutor."""
    import multiprocessing

    formula_density_pairs = list(zip(formulas, densities, strict=False))
    results = {}

    optimal_workers = min(len(formulas), max(1, multiprocessing.cpu_count() // 2), 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        future_to_formula = {
            executor.submit(process_func, pair): pair[0]
            for pair in formula_density_pairs
        }

        for future in concurrent.futures.as_completed(future_to_formula):
            formula = future_to_formula[future]
            try:
                formula_result, xray_result = future.result()
                results[formula_result] = xray_result
            except Exception as e:
                print(f"Warning: Failed to process formula '{formula}': {e}")
                continue

    return results


def calculate_xray_properties(
    formulas: list[str],
    energies: float | list[float] | np.ndarray,
    densities: list[float],
) -> dict[str, XRayResult]:
    """
    Calculate X-ray optical properties for multiple material compositions in parallel.

    This high-performance function processes multiple materials concurrently using
    parallel processing, making it ideal for batch calculations and comparative
    materials analysis. Input validation ensures data integrity and results are
    returned as a dictionary for easy access.

    The function uses optimized parallel processing with ThreadPoolExecutor for
    CPU-bound calculations, providing significant speedup for multiple materials.
    All calculations use the same high-performance atomic data caching as the
    single-material function.

    Args:
        formulas: List of chemical formula strings (e.g., ["SiO2", "Al2O3", "TiO2"])
                 Each formula must use proper element symbols and be parseable
        energies: X-ray energies in keV applied to all materials. Accepts:
                 - float: Single energy for all materials
                 - list[float]: Multiple energies for all materials
                 - np.ndarray: Energy array for all materials
                 Valid range: 0.03-30.0 keV for all energies
        densities: List of material mass densities in g/cm³
                  Must have same length as formulas list
                  Each density must be positive

    Returns:
        Dict[str, XRayResult]: Dictionary mapping chemical formula strings to
        XRayResult objects. Each XRayResult contains the complete set of X-ray
        properties calculated at all specified energies.

        Keys are the original formula strings, values are XRayResult objects
        with all the same fields as calculate_single_material_properties().

    Raises:
        ValidationError: If inputs don't match (different list lengths, empty lists)
        FormulaError: If any chemical formula cannot be parsed
        EnergyError: If energy values are outside valid range
        AtomicDataError: If atomic scattering factor data is unavailable
        BatchProcessingError: If parallel processing fails for multiple materials
        RuntimeError: If no formulas were processed successfully

    Examples:
        **Basic Multi-Material Analysis:**

        >>> import xraylabtool as xlt
        >>> formulas = ["SiO2", "Al2O3", "Fe2O3"]
        >>> energies = [8.0, 10.0, 12.0]
        >>> densities = [2.2, 3.95, 5.24]
        >>> results = xlt.calculate_xray_properties(formulas, energies, densities)
        >>>
        >>> # Access results by formula
        >>> sio2 = results["SiO2"]
        >>> print(f"SiO2 MW: {sio2.molecular_weight_g_mol:.2f} g/mol")
        SiO2 MW: 60.08 g/mol
        >>> print(f"SiO2 critical angles: {sio2.critical_angle_degrees}")
        SiO2 critical angles: [0.21775384 0.17403793 0.1446739 ]

        **Single Energy for Multiple Materials:**

        >>> results = xlt.calculate_xray_properties(
        ...     ["SiO2", "Al2O3", "C"], 10.0, [2.2, 3.95, 3.52]
        ... )
        >>> for formula, result in sorted(results.items()):
        ...     θc = result.critical_angle_degrees[0]
        ...     print(f"{formula:6}: θc = {θc:.3f}°")
        Al2O3 : θc = 0.231°
        C     : θc = 0.219°
        SiO2  : θc = 0.174°

        **Energy Range Analysis for Multiple Materials:**

        >>> import numpy as np
        >>> energy_range = np.logspace(np.log10(1), np.log10(20), 50)  # 1-20 keV
        >>> materials = ["Si", "SiO2", "Al", "Al2O3"]
        >>> densities = [2.33, 2.2, 2.70, 3.95]
        >>> results = xlt.calculate_xray_properties(materials, energy_range, densities)
        >>>
        >>> # Compare attenuation lengths at 10 keV
        >>> for formula in materials:
        ...     result = results[formula]
        ...     # Find closest energy to 10 keV
        ...     idx = np.argmin(np.abs(result.energy_kev - 10.0))
        ...     atten = result.attenuation_length_cm[idx]
        ...     # Store for analysis: print(f"{formula:6}: {atten:.2f} cm at ~10 keV")

        **Performance Comparison:**

        >>> # This is much faster than individual calls:
        >>> results = xlt.calculate_xray_properties(materials, energy_range, densities)
        >>>
        >>> # Instead of (slower):
        >>> # individual_results = {}
        >>> # for formula, density in zip(materials, densities):
        >>> #     individual_results[formula] = xlt.calculate_single_material_properties(
        >>> #         formula, energy_range, density
        >>> #     )

    Performance Notes:
        - Uses parallel processing for multiple materials
        - Shared atomic data caching across all calculations
        - Optimal for 2+ materials; use calculate_single_material_properties() for one
        - Processing time scales sub-linearly with number of materials
        - Memory usage is optimized for large material lists

    See Also:
        calculate_single_material_properties : Single material calculations
        XRayResult : Documentation of returned data structure
        validate_chemical_formula : Formula validation utility
    """
    _validate_xray_inputs(formulas, densities)
    energies_array = _validate_and_process_energies(energies)

    sort_indices = np.argsort(energies_array)
    sorted_energies = energies_array[sort_indices]

    process_func = _create_process_formula_function(sorted_energies, sort_indices)
    results = _process_formulas_parallel(formulas, densities, process_func)

    if not results:
        raise RuntimeError("Failed to process any formulas successfully")

    return results


# Initialize element paths at module import time for performance
_initialize_element_paths()
