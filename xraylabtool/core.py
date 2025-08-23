"""
Core functionality for XRayLabTool.

This module contains the main classes and functions for X-ray analysis,
including atomic scattering factors and crystallographic calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Callable, Optional, Any
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
import concurrent.futures
from functools import lru_cache

# =====================================================================================
# DATA STRUCTURES
# =====================================================================================


@dataclass
class XRayResult:
    """
    Dataclass to store complete X-ray optical property calculations for a material.

    Contains all computed properties including scattering factors, optical constants,
    and derived quantities like critical angles and attenuation lengths.

    All field names follow Python naming conventions (snake_case) with clear units.
    Legacy CamelCase field names are supported via deprecated property aliases.

    Fields:
        formula: Chemical formula string
        molecular_weight_g_mol: Molecular weight (g/mol)
        total_electrons: Total electrons per molecule
        density_g_cm3: Mass density (g/cm³)
        electron_density_per_ang3: Electron density (electrons/Å³)
        energy_kev: X-ray energies in keV (numpy.ndarray)
        wavelength_angstrom: X-ray wavelengths in Å (numpy.ndarray)
        dispersion_delta: Dispersion coefficients δ (numpy.ndarray)
        absorption_beta: Absorption coefficients β (numpy.ndarray)
        scattering_factor_f1: Real part of atomic scattering factor (numpy.ndarray)
        scattering_factor_f2: Imaginary part of atomic scattering factor (numpy.ndarray)
        critical_angle_degrees: Critical angles in degrees (numpy.ndarray)
        attenuation_length_cm: Attenuation lengths in cm (numpy.ndarray)
        real_sld_per_ang2: Real part of scattering length density in Å⁻²
        imaginary_sld_per_ang2: Imaginary part of scattering length density in Å⁻²
    """

    # New snake_case field names
    formula: str  # Chemical formula
    molecular_weight_g_mol: float  # Molecular weight (g/mol)
    total_electrons: float  # Electrons per molecule
    density_g_cm3: float  # Mass density (g/cm³)
    electron_density_per_ang3: float  # Electron density (electrons/Å³)
    energy_kev: np.ndarray  # X-ray energy (keV)
    wavelength_angstrom: np.ndarray  # X-ray wavelength (Å)
    dispersion_delta: np.ndarray  # Dispersion coefficient δ
    absorption_beta: np.ndarray  # Absorption coefficient β
    scattering_factor_f1: np.ndarray  # Real part of atomic scattering factor
    scattering_factor_f2: np.ndarray  # Imaginary part of atomic scattering factor
    critical_angle_degrees: np.ndarray  # Critical angle (degrees)
    attenuation_length_cm: np.ndarray  # Attenuation length (cm)
    real_sld_per_ang2: np.ndarray  # Real part of SLD (Å⁻²)
    imaginary_sld_per_ang2: np.ndarray  # Imaginary part of SLD (Å⁻²)

    def __post_init__(self) -> None:
        """Post-initialization to handle any setup after object creation."""
        # Ensure all arrays are numpy arrays
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
        import warnings

        warnings.warn(
            "Formula is deprecated, use 'formula' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.formula

    @property
    def MW(self) -> float:
        """Deprecated: Use 'molecular_weight_g_mol' instead."""
        import warnings

        warnings.warn(
            "MW is deprecated, use 'molecular_weight_g_mol' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.molecular_weight_g_mol

    @property
    def Number_Of_Electrons(self) -> float:
        """Deprecated: Use 'total_electrons' instead."""
        import warnings

        warnings.warn(
            "Number_Of_Electrons is deprecated, use 'total_electrons' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.total_electrons

    @property
    def Density(self) -> float:
        """Deprecated: Use 'density_g_cm3' instead."""
        import warnings

        warnings.warn(
            "Density is deprecated, use 'density_g_cm3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.density_g_cm3

    @property
    def Electron_Density(self) -> float:
        """Deprecated: Use 'electron_density_per_ang3' instead."""
        import warnings

        warnings.warn(
            "Electron_Density is deprecated, use 'electron_density_per_ang3' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.electron_density_per_ang3

    @property
    def Energy(self) -> np.ndarray:
        """Deprecated: Use 'energy_kev' instead."""
        import warnings

        warnings.warn(
            "Energy is deprecated, use 'energy_kev' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.energy_kev

    @property
    def Wavelength(self) -> np.ndarray:
        """Deprecated: Use 'wavelength_angstrom' instead."""
        import warnings

        warnings.warn(
            "Wavelength is deprecated, use 'wavelength_angstrom' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.wavelength_angstrom

    @property
    def Dispersion(self) -> np.ndarray:
        """Deprecated: Use 'dispersion_delta' instead."""
        import warnings

        warnings.warn(
            "Dispersion is deprecated, use 'dispersion_delta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.dispersion_delta

    @property
    def Absorption(self) -> np.ndarray:
        """Deprecated: Use 'absorption_beta' instead."""
        import warnings

        warnings.warn(
            "Absorption is deprecated, use 'absorption_beta' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.absorption_beta

    @property
    def f1(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f1' instead."""
        import warnings

        warnings.warn(
            "f1 is deprecated, use 'scattering_factor_f1' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f1

    @property
    def f2(self) -> np.ndarray:
        """Deprecated: Use 'scattering_factor_f2' instead."""
        import warnings

        warnings.warn(
            "f2 is deprecated, use 'scattering_factor_f2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scattering_factor_f2

    @property
    def Critical_Angle(self) -> np.ndarray:
        """Deprecated: Use 'critical_angle_degrees' instead."""
        import warnings

        warnings.warn(
            "Critical_Angle is deprecated, use 'critical_angle_degrees' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.critical_angle_degrees

    @property
    def Attenuation_Length(self) -> np.ndarray:
        """Deprecated: Use 'attenuation_length_cm' instead."""
        import warnings

        warnings.warn(
            "Attenuation_Length is deprecated, use 'attenuation_length_cm' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.attenuation_length_cm

    @property
    def reSLD(self) -> np.ndarray:
        """Deprecated: Use 'real_sld_per_ang2' instead."""
        import warnings

        warnings.warn(
            "reSLD is deprecated, use 'real_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.real_sld_per_ang2

    @property
    def imSLD(self) -> np.ndarray:
        """Deprecated: Use 'imaginary_sld_per_ang2' instead."""
        import warnings

        warnings.warn(
            "imSLD is deprecated, use 'imaginary_sld_per_ang2' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imaginary_sld_per_ang2

    @classmethod
    def from_legacy(
        cls,
        Formula: Optional[str] = None,
        MW: Optional[float] = None,
        Number_Of_Electrons: Optional[float] = None,
        Density: Optional[float] = None,
        Electron_Density: Optional[float] = None,
        Energy: Optional[np.ndarray] = None,
        Wavelength: Optional[np.ndarray] = None,
        Dispersion: Optional[np.ndarray] = None,
        Absorption: Optional[np.ndarray] = None,
        f1: Optional[np.ndarray] = None,
        f2: Optional[np.ndarray] = None,
        Critical_Angle: Optional[np.ndarray] = None,
        Attenuation_Length: Optional[np.ndarray] = None,
        reSLD: Optional[np.ndarray] = None,
        imSLD: Optional[np.ndarray] = None,
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
_scattering_factor_cache: Dict[str, pd.DataFrame] = {}

# Module-level cache for interpolators to avoid repeated creation
_interpolator_cache: Dict[str, Tuple[PchipInterpolator, PchipInterpolator]] = {}

# Pre-computed element file paths for faster access
_AVAILABLE_ELEMENTS: Dict[str, Path] = {}

# Atomic data cache for bulk lookups
_atomic_data_cache: Dict[str, Dict[str, float]] = {}


def _initialize_element_paths() -> None:
    """
    Pre-compute all available element file paths at module load time.
    This optimization eliminates repeated file system checks.
    """

    base_paths = [
        Path.cwd() / "src" / "AtomicScatteringFactor",
        Path(__file__).parent.parent / "src" / "AtomicScatteringFactor",
        Path(__file__).parent / "data" / "AtomicScatteringFactor",
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
        >>> data = load_scattering_factor_data('Si')
        >>> print(data.columns.tolist())
        ['E', 'f1', 'f2']
        >>> print(data.shape)
        (200, 3)
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
        self.data: Dict[str, pd.DataFrame] = {}
        self.data_path = Path(__file__).parent / "data" / "AtomicScatteringFactor"

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

    def get_scattering_factor(self, element: str, q_values: np.ndarray) -> np.ndarray:
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
        self, lattice_parameters: Tuple[float, float, float, float, float, float]
    ):
        """
        Initialize crystal structure.

        Args:
            lattice_parameters: (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        """
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_parameters
        self.atoms: List[Dict[str, Any]] = []

    def add_atom(
        self, element: str, position: Tuple[float, float, float], occupancy: float = 1.0
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

    def calculate_structure_factor(self, hkl: Tuple[int, int, int]) -> complex:
        """
        Calculate structure factor for given Miller indices.

        Args:
            hkl: Miller indices (h, k, l)

        Returns:
            Complex structure factor
        """
        # Placeholder implementation
        return complex(1.0, 0.0)


def get_cached_elements() -> List[str]:
    """
    Get list of elements currently cached in the scattering factor cache.

    Returns:
        List of element symbols currently loaded in cache
    """
    return list(_scattering_factor_cache.keys())


@lru_cache(maxsize=None)
def get_bulk_atomic_data(
    elements_tuple: Tuple[str, ...],
) -> Dict[str, Dict[str, float]]:
    """
    Bulk load atomic data for multiple elements with high-performance caching.

    This optimization uses a preloaded cache of common elements to eliminate
    expensive database queries to the Mendeleev library during runtime.

    Args:
        elements_tuple: Tuple of element symbols to load data for

    Returns:
        Dictionary mapping element symbols to their atomic data
    """
    from .atomic_data_cache import get_bulk_atomic_data_fast

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
    element_data: List[Tuple[float, Callable[..., Any], Callable[..., Any]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    from .constants import SCATTERING_FACTOR

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
        f1_values = np.asarray(f1_values, dtype=np.float64)
        f2_values = np.asarray(f2_values, dtype=np.float64)

        # Pre-compute factors for efficiency
        count_factor = float(count)
        wave_element_factor = wave_sq * (common_factor * count_factor)

        # Direct assignment for single element case
        f1_total = count_factor * f1_values
        f2_total = count_factor * f2_values
        dispersion = wave_element_factor * f1_values
        absorption = wave_element_factor * f2_values

    return dispersion, absorption, f1_total, f2_total


def calculate_derived_quantities(
    wavelength: np.ndarray,
    dispersion: np.ndarray,
    absorption: np.ndarray,
    mass_density: float,
    molecular_weight: float,
    number_of_electrons: float,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    from .constants import AVOGADRO, PI

    # Calculate electron density (electrons per unit volume)
    # ρₑ = ρ × Nₐ × Z / M × 10⁻³⁰ (converted to electrons/Å³)
    electron_density = (
        1e6 * mass_density / molecular_weight * AVOGADRO * number_of_electrons / 1e30
    )

    # Calculate critical angle for total external reflection
    # θc = √(2δ) (in radians), converted to degrees
    critical_angle = np.sqrt(2.0 * dispersion) * (180.0 / PI)

    # Calculate X-ray attenuation length
    # 1/e attenuation length = λ/(4πβ) (in cm)
    attenuation_length = wavelength / absorption / (4 * PI) * 1e2

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
) -> Tuple[
    Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
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
        >>> f1_interp, f2_interp = create_scattering_factor_interpolators('Si')
        >>> energy = 100.0  # eV
        >>> f1_value = f1_interp(energy)
        >>> f2_value = f2_interp(energy)
        >>> # Can also handle arrays
        >>> energies = np.array([100.0, 200.0, 300.0])
        >>> f1_values = f1_interp(energies)
        >>> f2_values = f2_interp(energies)
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
    energy_kev: Union[float, List[float], np.ndarray],
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
        elif isinstance(energy_kev, (int, float, np.number)):
            energy_kev = np.array([float(energy_kev)], dtype=np.float64)
        else:
            try:
                energy_kev = np.array([float(energy_kev)], dtype=np.float64)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert energy to float: {energy_kev}")
    else:
        energy_kev = np.array(energy_kev, dtype=np.float64)

    return np.asarray(energy_kev)


def _calculate_molecular_properties(
    element_symbols: List[str],
    element_counts: List[float],
    atomic_data_bulk: Dict[str, Dict[str, Any]],
) -> Tuple[float, float]:
    """Calculate molecular weight and total electrons."""
    molecular_weight = 0.0
    number_of_electrons = 0.0

    for symbol, count in zip(element_symbols, element_counts):
        data = atomic_data_bulk[symbol]
        atomic_number = data["atomic_number"]
        atomic_mass = data["atomic_weight"]

        molecular_weight += count * atomic_mass
        number_of_electrons += atomic_number * count

    return molecular_weight, number_of_electrons


def _prepare_element_data(
    element_symbols: List[str], element_counts: List[float]
) -> List[Tuple[float, Any, Any]]:
    """Prepare element data with interpolators."""
    element_data = []

    for i, symbol in enumerate(element_symbols):
        f1_interp, f2_interp = create_scattering_factor_interpolators(symbol)
        element_data.append((element_counts[i], f1_interp, f2_interp))

    return element_data


def _calculate_single_material_xray_properties(
    formula_str: str,
    energy_kev: Union[float, List[float], np.ndarray],
    mass_density: float,
) -> Dict[str, Union[str, float, np.ndarray]]:
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
    from .utils import parse_formula
    from .constants import ENERGY_TO_WAVELENGTH_FACTOR, METER_TO_ANGSTROM

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
    formula_list: List[str],
    energy_kev: Union[float, List[float], np.ndarray],
    mass_density_list: List[float],
) -> Dict[str, Dict[str, Union[str, float, np.ndarray]]]:
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
        >>> formulas = ["SiO2", "Al2O3", "Fe2O3"]
        >>> energies = [8.0, 10.0, 12.0]
        >>> densities = [2.2, 3.95, 5.24]
        >>> results = calculate_multiple_xray_properties(formulas, energies, densities)
        >>> sio2_result = results["SiO2"]
        >>> print(f"SiO2 molecular weight: {sio2_result['molecular_weight']:.2f}")
    """
    # Input validation
    if len(formula_list) != len(mass_density_list):
        raise ValueError("Formula list and mass density list must have the same length")

    if not formula_list:
        raise ValueError("Formula list must not be empty")

    # Process each formula
    results = {}

    for formula, mass_density in zip(formula_list, mass_density_list):
        try:
            # Calculate properties for this formula
            result = calculate_single_material_properties(
                formula, energy_kev, mass_density
            )

            # Convert XRayResult to dictionary format for backward
            # compatibility
            result_dict: Dict[str, Union[str, float, np.ndarray]] = {
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
    formula: str, energy_keV: Union[float, List[float], np.ndarray], density: float
) -> XRayResult:
    """
    Calculate X-ray optical properties for a single material composition.

    This is a pure function that calculates comprehensive X-ray optical properties
    for a single chemical formula at given energies and density. It returns an
    XRayResult dataclass containing all computed properties.

    Args:
        formula: Chemical formula string (e.g., "SiO2", "Al2O3")
        energy_keV: X-ray energies in keV (scalar, list, or numpy array)
        density: Material mass density in g/cm³

    Returns:
        XRayResult: Dataclass containing all calculated X-ray properties:
            - Formula: Chemical formula string
            - MW: Molecular weight (g/mol)
            - Number_Of_Electrons: Total electrons per molecule
            - Density: Mass density (g/cm³)
            - Electron_Density: Electron density (electrons/Å³)
            - Energy: X-ray energies (keV, numpy array)
            - Wavelength: X-ray wavelengths (Å, numpy array)
            - Dispersion: Dispersion coefficients δ (numpy array)
            - Absorption: Absorption coefficients β (numpy array)
            - f1: Real part of atomic scattering factor (numpy array)
            - f2: Imaginary part of atomic scattering factor (numpy array)
            - Critical_Angle: Critical angles (degrees, numpy array)
            - Attenuation_Length: Attenuation lengths (cm, numpy array)
            - reSLD: Real part of SLD (Å⁻², numpy array)
            - imSLD: Imaginary part of SLD (Å⁻², numpy array)

    Raises:
        ValueError: If formula, energy, or density inputs are invalid
        FileNotFoundError: If atomic scattering factor data is not available

    Examples:
        >>> result = calculate_single_material_properties("SiO2", 8.0, 2.2)
        >>> print(f"Molecular weight: {result.MW:.2f} g/mol")
        Molecular weight: 60.08 g/mol

        >>> # Multiple energies
        >>> result = calculate_single_material_properties(
        ...     "Al2O3", [8.0, 10.0, 12.0], 3.95
        ... )
        >>> print(f"Critical angles: {result.Critical_Angle}")

        >>> # Array input
        >>> energies = np.linspace(5.0, 15.0, 11)
        >>> result = calculate_single_material_properties("Fe2O3", energies, 5.24)
        >>> print(f"Energy range: {result.Energy[0]:.1f} - {result.Energy[-1]:.1f} keV")
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


def _validate_xray_inputs(formulas: List[str], densities: List[float]) -> None:
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
        if not isinstance(density, (int, float)) or density <= 0:
            raise ValueError(
                f"Density at index {i} must be a positive number, got: {density}"
            )


def _validate_and_process_energies(energies: Any) -> np.ndarray:
    """Validate and convert energies to numpy array."""
    if np.isscalar(energies):
        if isinstance(energies, complex):
            energies_array = np.array([float(energies.real)], dtype=np.float64)
        elif isinstance(energies, (int, float, np.number)):
            energies_array = np.array([float(energies)], dtype=np.float64)
        else:
            try:
                energies_array = np.array([float(energies)], dtype=np.float64)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert energy to float: {energies!r}")
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
) -> Callable[[Tuple[str, float]], Tuple[str, XRayResult]]:
    """Create process formula function with energy sorting logic."""

    def process_formula(
        formula_density_pair: Tuple[str, float],
    ) -> Tuple[str, XRayResult]:
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
    formulas: List[str],
    densities: List[float],
    process_func: Callable[[Tuple[str, float]], Tuple[str, XRayResult]],
) -> Dict[str, XRayResult]:
    """Process formulas in parallel using ThreadPoolExecutor."""
    import multiprocessing

    formula_density_pairs = list(zip(formulas, densities))
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
    formulas: List[str],
    energies: Union[float, List[float], np.ndarray],
    densities: List[float],
) -> Dict[str, XRayResult]:
    """
    Calculate X-ray optical properties for multiple material compositions in parallel.

    This function validates inputs, sorts energies, processes formulas in parallel
    using concurrent.futures.ThreadPoolExecutor, and aggregates results into a
    dictionary mapping formula strings to XRayResult objects.

    Args:
        formulas: List of chemical formula strings
        energies: X-ray energies in keV (scalar, list, or numpy array)
        densities: List of material mass densities in g/cm³

    Returns:
        Dict[str, XRayResult]: Dictionary mapping formula strings to XRayResult
        objects containing all calculated X-ray properties for each material

    Raises:
        ValueError: If input validation fails (mismatched lengths, empty inputs, etc.)
        FileNotFoundError: If atomic scattering factor data is not available

    Examples:
        >>> formulas = ["SiO2", "Al2O3", "Fe2O3"]
        >>> energies = [8.0, 10.0, 12.0]
        >>> densities = [2.2, 3.95, 5.24]
        >>> results = calculate_xray_properties(formulas, energies, densities)
        >>> sio2_result = results["SiO2"]
        >>> print(f"SiO2 MW: {sio2_result.MW:.2f} g/mol")
        SiO2 MW: 60.08 g/mol

        >>> # Single energy for all materials
        >>> results = calculate_xray_properties(["SiO2", "Al2O3"], 10.0, [2.2, 3.95])
        >>> for formula, result in results.items():
        ...     print(f"{formula}: {result.Critical_Angle[0]:.3f}°")

        >>> # Array of energies
        >>> energy_array = np.linspace(5.0, 15.0, 21)
        >>> results = calculate_xray_properties(["SiO2"], energy_array, [2.2])
        >>> print(f"Energy points: {len(results['SiO2'].Energy)}")
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
