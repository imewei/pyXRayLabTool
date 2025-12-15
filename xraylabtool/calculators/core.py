"""
Core functionality for XRayLabTool.

This module contains the main classes and functions for X-ray analysis,
including atomic scattering factors and crystallographic calculations.
"""

# ruff: noqa: RUF002, RUF003, PLC0415, PLW0603, PLW0602

from __future__ import annotations

from collections.abc import Callable

# concurrent.futures import moved to function level for parallel processing
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path
import types
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np


class ScalarFriendlyArray(np.ndarray):
    """ndarray that formats as scalar when it contains a single value."""

    def __format__(self, format_spec: str) -> str:  # pragma: no cover
        if self.size == 1:
            return format(self.item(), format_spec)
        return super().__format__(format_spec)

    def __float__(self):  # pragma: no cover
        if self.size == 1:
            return float(self.item())
        raise TypeError("Only length-1 arrays can be converted to float")

    def __getitem__(self, key):  # pragma: no cover
        result = super().__getitem__(key)
        if isinstance(result, np.ndarray) and result.size == 1:
            return float(result)
        return result

    def __array__(self, dtype=None):  # pragma: no cover
        return np.asarray(self.view(np.ndarray), dtype=dtype)


# Lazy imports for heavy dependencies to reduce startup time
# pandas and scipy imports moved to function level when needed

# Lazy loading for optimization modules to reduce memory overhead
# These will be imported only when actually needed


def _get_optimization_decorator() -> Callable[..., Any]:
    """Lazy load optimization decorator to reduce memory overhead."""
    try:
        from xraylabtool.optimization.vectorized_core import (
            ensure_c_contiguous,
        )

        return ensure_c_contiguous
    except ImportError:
        # Fallback decorator if optimization module not available
        def ensure_c_contiguous(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return ensure_c_contiguous


if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        ArrayLike,
        EnergyArray,
        FloatLike,
        InterpolatorProtocol,
        OpticalConstantArray,
        WavelengthArray,
    )

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

    # Material properties with enhanced type annotations
    formula: str  # Chemical formula string
    molecular_weight_g_mol: float  # Molecular weight (g/mol)
    total_electrons: float  # Total electrons per molecule
    density_g_cm3: float  # Mass density (g/cm³)
    electron_density_per_ang3: float  # Electron density (electrons/Å³)

    # X-ray energy and wavelength arrays (performance-optimized dtypes)
    energy_kev: EnergyArray = field()  # X-ray energies in keV
    wavelength_angstrom: WavelengthArray = field()  # X-ray wavelengths in Å

    # Fundamental optical constants (performance-critical arrays)
    dispersion_delta: OpticalConstantArray = field()  # Dispersion coefficient δ
    absorption_beta: OpticalConstantArray = field()  # Absorption coefficient β

    # Atomic scattering factors (complex arrays for scientific accuracy)
    scattering_factor_f1: OpticalConstantArray = (
        field()
    )  # Real part of scattering factor
    scattering_factor_f2: OpticalConstantArray = (
        field()
    )  # Imaginary part of scattering factor

    # Derived optical properties (performance-optimized arrays)
    critical_angle_degrees: OpticalConstantArray = field()  # Critical angles (degrees)
    attenuation_length_cm: OpticalConstantArray = field()  # Attenuation lengths (cm)
    real_sld_per_ang2: OpticalConstantArray = field()  # Real SLD (Å⁻²)
    imaginary_sld_per_ang2: OpticalConstantArray = field()  # Imaginary SLD (Å⁻²)

    def __post_init__(self) -> None:
        """Post-initialization to handle any setup after object creation."""
        # Ensure all arrays are numpy arrays - only convert if necessary
        # mypy: These checks are necessary at runtime even though types are declared
        # Runtime conversion to numpy arrays if needed
        # Convert all array fields to numpy arrays
        self.energy_kev = np.asarray(self.energy_kev)
        self.wavelength_angstrom = np.asarray(self.wavelength_angstrom).view(
            ScalarFriendlyArray
        )
        self.dispersion_delta = np.asarray(self.dispersion_delta).view(
            ScalarFriendlyArray
        )
        self.absorption_beta = np.asarray(self.absorption_beta).view(
            ScalarFriendlyArray
        )
        self.scattering_factor_f1 = np.asarray(self.scattering_factor_f1).view(
            ScalarFriendlyArray
        )
        self.scattering_factor_f2 = np.asarray(self.scattering_factor_f2).view(
            ScalarFriendlyArray
        )
        self.critical_angle_degrees = np.asarray(self.critical_angle_degrees).view(
            ScalarFriendlyArray
        )
        self.attenuation_length_cm = np.asarray(self.attenuation_length_cm).view(
            ScalarFriendlyArray
        )
        self.real_sld_per_ang2 = np.asarray(self.real_sld_per_ang2).view(
            ScalarFriendlyArray
        )
        self.imaginary_sld_per_ang2 = np.asarray(self.imaginary_sld_per_ang2).view(
            ScalarFriendlyArray
        )

    # Convenience properties used in docs/notebooks
    @property
    def energy_ev(self):
        return self.energy_kev * 1000.0

    @property
    def delta(self):
        return self.dispersion_delta

    @property
    def beta(self):
        return self.absorption_beta

    @property
    def critical_angle_mrad(self):
        return self.critical_angle_degrees * np.pi / 180.0 * 1000.0

    @property
    def linear_absorption_coefficient(self):
        # μ = 1 / attenuation length
        arr = np.where(
            self.attenuation_length_cm != 0, 1.0 / self.attenuation_length_cm, 0.0
        )
        return np.asarray(arr).view(ScalarFriendlyArray)

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
        formula: str | None = None,
        mw: float | None = None,
        number_of_electrons: float | None = None,
        density: float | None = None,
        electron_density: float | None = None,
        energy: np.ndarray | None = None,
        wavelength: np.ndarray | None = None,
        dispersion: np.ndarray | None = None,
        absorption: np.ndarray | None = None,
        f1: np.ndarray | None = None,
        f2: np.ndarray | None = None,
        critical_angle: np.ndarray | None = None,
        attenuation_length: np.ndarray | None = None,
        re_sld: np.ndarray | None = None,
        im_sld: np.ndarray | None = None,
        **kwargs: Any,
    ) -> XRayResult:
        """Create XRayResult from legacy field names (for internal use)."""
        return cls(
            formula=formula or kwargs.get("formula", ""),
            molecular_weight_g_mol=mw or kwargs.get("molecular_weight_g_mol", 0.0),
            total_electrons=number_of_electrons or kwargs.get("total_electrons", 0.0),
            density_g_cm3=density or kwargs.get("density_g_cm3", 0.0),
            electron_density_per_ang3=(
                electron_density or kwargs.get("electron_density_per_ang3", 0.0)
            ),
            energy_kev=(
                energy if energy is not None else kwargs.get("energy_kev", np.array([]))
            ),
            wavelength_angstrom=(
                wavelength
                if wavelength is not None
                else kwargs.get("wavelength_angstrom", np.array([]))
            ),
            dispersion_delta=(
                dispersion
                if dispersion is not None
                else kwargs.get("dispersion_delta", np.array([]))
            ),
            absorption_beta=(
                absorption
                if absorption is not None
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
                critical_angle
                if critical_angle is not None
                else kwargs.get("critical_angle_degrees", np.array([]))
            ),
            attenuation_length_cm=(
                attenuation_length
                if attenuation_length is not None
                else kwargs.get("attenuation_length_cm", np.array([]))
            ),
            real_sld_per_ang2=(
                re_sld
                if re_sld is not None
                else kwargs.get("real_sld_per_ang2", np.array([]))
            ),
            imaginary_sld_per_ang2=(
                im_sld
                if im_sld is not None
                else kwargs.get("imaginary_sld_per_ang2", np.array([]))
            ),
        )


# =====================================================================================
# CACHING SYSTEM
# =====================================================================================

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

# Pre-computed element file paths for faster access
_AVAILABLE_ELEMENTS: dict[str, Path] = {}

# Cache for most commonly used elements to improve cold start performance
_PRIORITY_ELEMENTS = ["H", "C", "N", "O", "Si", "Al", "Ca", "Fe", "Cu", "Zn"]
_CACHE_WARMED = False

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


def load_scattering_factor_data(element: str) -> Any:
    """
    Load f1/f2 scattering factor data for a specific element from .nff files.

    This function reads .nff files using CSV parsing and caches the results
    in a module-level dictionary keyed by element symbol. Returns a pandas-compatible
    data structure for accessing columns E, f1, f2.

    Args:
        element: Element symbol (e.g., 'H', 'C', 'N', 'O', 'Si', 'Ge')

    Returns:
        ScatteringData object with pandas-like interface containing columns: E (energy), f1, f2

    Raises:
        FileNotFoundError: If the .nff file for the element is not found
        ValueError: If the element symbol is invalid, empty, or file format is invalid

    Examples:
        >>> from xraylabtool.calculators.core import load_scattering_factor_data
        >>> data = load_scattering_factor_data('Si')
        >>> print(data.columns)
        ['E', 'f1', 'f2']
        >>> print(len(data) > 100)  # Verify we have enough data points
        True
    """

    # Validate input
    if not element or not isinstance(element, str):
        raise ValueError(f"Element symbol must be a non-empty string, got: {element!r}")

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
        # Load .nff file using numpy - faster and no pandas dependency
        # .nff files are CSV format with header: E,f1,f2
        import csv

        with open(file_path) as file:
            # Read header
            reader = csv.reader(file)
            header = next(reader)

            # Verify expected columns exist
            expected_columns = {"E", "f1", "f2"}
            actual_columns = set(header)

            if not expected_columns.issubset(actual_columns):
                missing_cols = expected_columns - actual_columns
                raise ValueError(
                    f"Invalid .nff file format for element '{element}'. "
                    f"Missing required columns: {missing_cols}. "
                    f"Found columns: {list(actual_columns)}"
                )

            # Get column indices
            e_idx = header.index("E")
            f1_idx = header.index("f1")
            f2_idx = header.index("f2")

            # Read data rows
            data_rows = []
            for row in reader:
                if len(row) >= max(e_idx, f1_idx, f2_idx) + 1:
                    data_rows.append(
                        [float(row[e_idx]), float(row[f1_idx]), float(row[f2_idx])]
                    )

        if not data_rows:
            raise ValueError(
                "Empty scattering factor data file for element "
                f"'{element}': {file_path}"
            )

        # Convert to numpy array for efficiency
        data_array = np.array(data_rows, dtype=np.float64)

        # Create a pandas-like interface using a simple class
        class ScatteringData:
            def __init__(self, data_array: np.ndarray, column_names: list[str]) -> None:
                self.data = data_array
                self.columns = column_names
                self._column_indices = {name: i for i, name in enumerate(column_names)}

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, column: str) -> Any:
                idx = self._column_indices[column]

                # Return object with .values attribute for compatibility
                class ColumnProxy:
                    def __init__(self, data: np.ndarray) -> None:
                        self.values = data

                return ColumnProxy(self.data[:, idx])

        scattering_data = ScatteringData(data_array, ["E", "f1", "f2"])

        # Cache the data
        _scattering_factor_cache[element] = scattering_data

        return scattering_data

    except (OSError, ValueError, csv.Error) as e:
        raise ValueError(
            "Error parsing scattering factor data file for element "
            f"'{element}': {file_path}. "
            f"Expected CSV format with columns: E,f1,f2. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "Unexpected error loading scattering factor data for element "
            f"'{element}' from {file_path}: {e}"
        ) from e


class AtomicScatteringFactor:
    """
    Class for handling atomic scattering factors.

    This class loads and manages atomic scattering factor data
    from NFF files using the module-level cache.
    """

    def __init__(self) -> None:
        # Maintain backward compatibility with existing tests
        self.data: dict[str, Any] = {}
        self.data_path = (
            Path(__file__).parent.parent / "data" / "AtomicScatteringFactor"
        )

        # Create data directory if it doesn't exist (for test compatibility)
        self.data_path.mkdir(parents=True, exist_ok=True)

    def load_element_data(self, element: str) -> Any:
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

    def get_scattering_factor(self, _element: str, q_values: np.ndarray) -> np.ndarray:
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

    def calculate_structure_factor(self, _hkl: tuple[int, int, int]) -> complex:
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

    def _background_cache_warming():
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

    This v0.2.5 optimization provides 90% faster cold start performance by loading
    only required elements instead of all priority elements. Reduces initial
    calculation time from 912ms (v0.2.4) to 130ms (v0.2.5).

    Features:
        - Formula-specific element loading via parse_formula
        - Automatic fallback to _warm_priority_cache on errors
        - One-time cache warming with _CACHE_WARMED flag
        - Background priority warming for comprehensive coverage

    Args:
        formula: Chemical formula to analyze for required elements (e.g., "SiO2")

    Performance:
        - 90% cold start improvement over v0.2.4
        - No impact on warm cache performance
        - Graceful error handling with fallback warming
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


def calculate_scattering_factors(
    energy_ev: EnergyArray,
    wavelength: WavelengthArray,
    mass_density: FloatLike,
    molecular_weight: FloatLike,
    element_data: list[tuple[float, InterpolatorProtocol, InterpolatorProtocol]],
) -> tuple[
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
]:
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
    - δ = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f1ᵢ) / M  # noqa: RUF002
    - β = (λ²/2π) × rₑ × ρ × Nₐ × (Σᵢ nᵢ × f2ᵢ) / M  # noqa: RUF002

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

    # Pre-allocate C-contiguous arrays for better memory performance
    # Using specific dtypes for better numerical precision and speed
    dispersion = np.zeros(n_energies, dtype=np.float64, order="C")
    absorption = np.zeros(n_energies, dtype=np.float64, order="C")
    f1_total = np.zeros(n_energies, dtype=np.float64, order="C")
    f2_total = np.zeros(n_energies, dtype=np.float64, order="C")

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
        # For multiple elements, use vectorized operations with C-contiguous arrays
        f1_matrix = np.empty((n_elements, n_energies), dtype=np.float64, order="C")
        f2_matrix = np.empty((n_elements, n_energies), dtype=np.float64, order="C")
        counts = np.empty(n_elements, dtype=np.float64, order="C")

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
    wavelength: WavelengthArray,
    dispersion: OpticalConstantArray,
    absorption: OpticalConstantArray,
    mass_density: FloatLike,
    molecular_weight: FloatLike,
    number_of_electrons: FloatLike,
) -> tuple[
    float,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
    OpticalConstantArray,
]:
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
    # Ensure scalar inputs for density calculation
    density_val = (
        np.asarray(mass_density).item()
        if np.asarray(mass_density).size == 1
        else mass_density
    )
    mol_weight_val = (
        np.asarray(molecular_weight).item()
        if np.asarray(molecular_weight).size == 1
        else molecular_weight
    )
    electrons_val = (
        np.asarray(number_of_electrons).item()
        if np.asarray(number_of_electrons).size == 1
        else number_of_electrons
    )

    electron_density = float(
        1e6 * density_val / mol_weight_val * AVOGADRO * electrons_val / 1e30
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
        # Lazy import to avoid circular imports
        try:
            from xraylabtool.data_handling.cache_metrics import _record_cache_access

            _record_cache_access(element, "interpolator_cache", hit=True)
        except ImportError:
            pass  # Silently continue if cache metrics not available
        return _interpolator_cache[element]

    # Cache miss - need to create new interpolators
    try:
        from xraylabtool.data_handling.cache_metrics import _record_cache_access

        _record_cache_access(element, "interpolator_cache", hit=False)
    except ImportError:
        pass  # Silently continue if cache metrics not available

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
    # Lazy import scipy only when needed
    from scipy.interpolate import PchipInterpolator

    f1_interpolator = PchipInterpolator(energy_values, f1_values, extrapolate=False)
    f2_interpolator = PchipInterpolator(energy_values, f2_values, extrapolate=False)

    # Cache the interpolators for future use
    _interpolator_cache[element] = (f1_interpolator, f2_interpolator)

    return f1_interpolator, f2_interpolator


def _validate_single_material_inputs(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
) -> EnergyArray:
    """Validate inputs for single material calculation."""
    if not formula_str or not isinstance(formula_str, str):
        raise ValueError("Formula must be a non-empty string")

    if np.any(np.asarray(mass_density) <= 0):
        raise ValueError("Mass density must be positive")

    # Convert and validate energy
    energy_kev = _convert_energy_input(energy_kev)

    if np.any(energy_kev <= 0):
        raise ValueError("All energies must be positive")

    if np.any(energy_kev < 0.03) or np.any(energy_kev > 30):
        raise ValueError("Energy is out of range 0.03keV ~ 30keV")

    return energy_kev


def _convert_energy_input(energy_kev: Any) -> EnergyArray:
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
) -> list[tuple[float, InterpolatorProtocol, InterpolatorProtocol]]:
    """Prepare element data with interpolators."""
    element_data = []

    for i, symbol in enumerate(element_symbols):
        f1_interp, f2_interp = create_scattering_factor_interpolators(symbol)
        element_data.append((element_counts[i], f1_interp, f2_interp))

    return element_data


# @ensure_c_contiguous  # Optimization decorator removed for compatibility
def _calculate_single_material_xray_properties(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
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

    (
        electron_density,
        critical_angle,
        attenuation_length,
        re_sld,
        im_sld,
    ) = calculate_derived_quantities(
        wavelength,
        dispersion,
        absorption,
        mass_density,
        molecular_weight,
        number_of_electrons,
    )

    return {
        "formula": formula_str,
        "molecular_weight": molecular_weight,
        "number_of_electrons": number_of_electrons,
        "mass_density": float(mass_density),
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


# @ensure_c_contiguous  # Optimization decorator removed for compatibility
def calculate_multiple_xray_properties(
    formula_list: list[str],
    energy_kev: FloatLike | ArrayLike,
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


def load_data_file(filename: str) -> Any:
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

    # Lazy import pandas only when needed
    import pandas as pd

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


# @ensure_c_contiguous  # Optimization decorator removed for compatibility
def calculate_single_material_properties(
    formula: str,
    energy_keV: FloatLike | ArrayLike | None = None,
    density: FloatLike | None = None,
    *,
    energy: FloatLike | ArrayLike | None = None,
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
        energy: Optional backward-compatible alias (eV). If provided, interpreted as
                electron-volts and converted to keV.

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

        **Performance Note (v0.2.5):**
        This function includes smart cache warming that reduces cold start time by 90%
        (from 912ms in v0.2.4 to 130ms in v0.2.5). Features formula-specific element
        loading, automatic cache state management, and graceful fallback to priority
        warming for complex cases.

    See Also:
        calculate_xray_properties : Calculate properties for multiple materials
        XRayResult : Complete documentation of returned dataclass
        parse_formula : Parse chemical formulas into elements and counts
    """
    # Backward compatibility: allow 'energy' in eV
    if energy_keV is None and energy is not None:
        energy_keV = np.asarray(energy, dtype=float) / 1000.0  # noqa: N806
    if energy_keV is None:
        raise ValueError("energy_keV or energy must be provided")
    if density is None:
        raise ValueError("density must be provided")

    # Use smart cache warming for faster cold start (only loads required elements)
    # Only warm cache if it hasn't been warmed yet
    global _CACHE_WARMED
    if not _CACHE_WARMED:
        _smart_cache_warming(formula)

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
        energy_kev=np.ascontiguousarray(properties["energy"], dtype=np.float64),
        wavelength_angstrom=np.ascontiguousarray(
            properties["wavelength"], dtype=np.float64
        ),
        dispersion_delta=np.ascontiguousarray(
            properties["dispersion"], dtype=np.float64
        ),
        absorption_beta=np.ascontiguousarray(
            properties["absorption"], dtype=np.float64
        ),
        scattering_factor_f1=np.ascontiguousarray(
            properties["f1_total"], dtype=np.float64
        ),
        scattering_factor_f2=np.ascontiguousarray(
            properties["f2_total"], dtype=np.float64
        ),
        critical_angle_degrees=np.ascontiguousarray(
            properties["critical_angle"], dtype=np.float64
        ),
        attenuation_length_cm=np.ascontiguousarray(
            properties["attenuation_length"], dtype=np.float64
        ),
        real_sld_per_ang2=np.ascontiguousarray(properties["re_sld"], dtype=np.float64),
        imaginary_sld_per_ang2=np.ascontiguousarray(
            properties["im_sld"], dtype=np.float64
        ),
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
                f"Formula at index {i} must be a non-empty string, got: {formula!r}"
            )

    for i, density in enumerate(densities):
        if not isinstance(density, int | float) or density <= 0:
            raise ValueError(
                f"Density at index {i} must be a positive number, got: {density}"
            )


def _validate_and_process_energies(energies: Any) -> EnergyArray:
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
    sorted_energies: EnergyArray, sort_indices: ArrayLike
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
    """
    Process formulas with adaptive parallelization.

    Uses sequential processing for small batches (<20 items) to avoid
    ThreadPoolExecutor overhead, and parallel processing for larger batches.
    """
    formula_density_pairs = list(zip(formulas, densities, strict=False))
    results = {}

    # Use sequential processing for small batches to avoid thread overhead
    if len(formulas) < 20:
        for pair in formula_density_pairs:
            try:
                formula_result, xray_result = process_func(pair)
                results[formula_result] = xray_result
            except Exception as e:
                print(f"Warning: Failed to process formula '{pair[0]}': {e}")
                continue
        return results

    # Use parallel processing for larger batches
    import concurrent.futures
    import multiprocessing

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
    energies: FloatLike | ArrayLike,
    densities: list[float],
) -> dict[str, XRayResult]:
    """
    Calculate X-ray optical properties for multiple material compositions.

    This high-performance function uses adaptive batch processing (v0.2.5) that
    automatically switches between sequential and parallel modes based on workload
    size. Small batches (<20 items) use sequential processing to eliminate threading
    overhead, while large batches (≥20 items) use parallel ThreadPoolExecutor for
    maximum throughput.

    The adaptive processing provides 75% performance improvement over v0.2.3 baseline
    for small batches and maintains high throughput for large datasets. All calculations
    use smart cache warming and high-performance atomic data caching.

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


# =====================================================================================
# CALCULATION ENGINE PROTOCOL IMPLEMENTATION
# =====================================================================================


class FastXRayCalculationEngine:
    """
    High-performance X-ray calculation engine implementing CalculationEngine protocol.

    This implementation provides optimized calculations for X-ray optical properties
    with performance-focused interfaces and comprehensive result handling.
    """

    def __init__(self) -> None:
        """Initialize the calculation engine."""
        self._cache_warmed = False

    def calculate_optical_constants(
        self,
        formula: str,
        energies: EnergyArray,
        density: FloatLike,
    ) -> tuple[OpticalConstantArray, OpticalConstantArray]:
        """
        Calculate dispersion and absorption coefficients.

        Parameters
        ----------
        formula : str
            Chemical formula string
        energies : EnergyArray
            X-ray energies in keV
        density : FloatLike
            Material density in g/cm³

        Returns
        -------
        tuple[OpticalConstantArray, OpticalConstantArray]
            Dispersion (δ) and absorption (β) coefficients
        """
        # Use internal calculation function
        result_dict = _calculate_single_material_xray_properties(
            formula, energies, density
        )

        # Extract optical constants
        dispersion = np.asarray(result_dict["dispersion"], dtype=np.float64)
        absorption = np.asarray(result_dict["absorption"], dtype=np.float64)

        return dispersion, absorption

    def calculate_derived_quantities(
        self,
        dispersion: OpticalConstantArray,
        absorption: OpticalConstantArray,
        energies: EnergyArray,
    ) -> dict[str, np.ndarray]:
        """
        Calculate derived quantities from optical constants.

        Parameters
        ----------
        dispersion : OpticalConstantArray
            Dispersion coefficients δ
        absorption : OpticalConstantArray
            Absorption coefficients β
        energies : EnergyArray
            X-ray energies in keV

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with derived quantities (critical_angles, attenuation_lengths, etc.)
        """
        from xraylabtool.constants import (
            ENERGY_TO_WAVELENGTH_FACTOR,
            METER_TO_ANGSTROM,
        )

        # Convert energies to wavelengths
        wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energies
        wavelength * METER_TO_ANGSTROM

        # Calculate derived quantities using internal function
        (
            _,
            critical_angle,
            attenuation_length,
            re_sld,
            im_sld,
        ) = calculate_derived_quantities(
            wavelength,
            dispersion,
            absorption,
            1.0,
            1.0,
            1.0,  # dummy values for density/MW/electrons
        )

        return {
            "critical_angles": critical_angle,
            "attenuation_lengths": attenuation_length,
            "real_sld": re_sld,
            "imaginary_sld": im_sld,
        }

    def warm_up_cache(self, common_elements: list[str] | None = None) -> None:
        """
        Pre-warm caches for improved performance.

        Parameters
        ----------
        common_elements : list[str] | None
            List of elements to preload, defaults to common materials science elements
        """
        if self._cache_warmed:
            return

        if common_elements is None:
            # Common elements in materials science
            common_elements = [
                "H",
                "C",
                "N",
                "O",
                "F",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "K",
                "Ca",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ge",
            ]

        # Preload atomic data
        from xraylabtool.data_handling.atomic_cache import (
            get_atomic_data_provider,
        )

        provider = get_atomic_data_provider()
        provider.preload_elements(common_elements)

        self._cache_warmed = True

    def get_performance_info(self) -> dict[str, Any]:
        """
        Get performance information about the calculation engine.

        Returns
        -------
        dict[str, Any]
            Performance metrics and status
        """
        from xraylabtool.data_handling.atomic_cache import (
            get_cache_stats,
        )

        cache_stats = get_cache_stats()

        return {
            "cache_warmed": self._cache_warmed,
            "cached_elements": cache_stats["total_cached_elements"],
            "preloaded_elements": cache_stats["preloaded_elements"],
            "runtime_cached": cache_stats["runtime_cached_elements"],
        }


# Global instance for easy access
_GLOBAL_ENGINE: FastXRayCalculationEngine | None = None


def get_calculation_engine() -> FastXRayCalculationEngine:
    """
    Get the global calculation engine instance.

    Returns
    -------
    FastXRayCalculationEngine
        Shared calculation engine instance
    """
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = FastXRayCalculationEngine()
        # Warm up cache for common elements
        _GLOBAL_ENGINE.warm_up_cache()
    return _GLOBAL_ENGINE


# Initialize element paths at module import time for performance
_initialize_element_paths()
