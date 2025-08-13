"""
Core functionality for XRayLabTool.

This module contains the main classes and functions for X-ray analysis,
including atomic scattering factors and crystallographic calculations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import os
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass
import concurrent.futures

# =====================================================================================
# DATA STRUCTURES
# =====================================================================================

@dataclass
class XRayResult:
    """
    Dataclass to store complete X-ray optical property calculations for a material.
    
    Contains all computed properties including scattering factors, optical constants,
    and derived quantities like critical angles and attenuation lengths.
    
    Fields are ordered to match the Julia XRayResult struct exactly.
    All vector fields use numpy.ndarray as specified in the requirements.
    
    Fields:
        Formula: Chemical formula string
        MW: Molecular weight (g/mol)
        Number_Of_Electrons: Total electrons per molecule  
        Density: Mass density (g/cm³)
        Electron_Density: Electron density (electrons/Å³)
        Energy: X-ray energies in keV (numpy.ndarray)
        Wavelength: X-ray wavelengths in Å (numpy.ndarray)
        Dispersion: Dispersion coefficients δ (numpy.ndarray)
        Absorption: Absorption coefficients β (numpy.ndarray)
        f1: Real part of atomic scattering factor (numpy.ndarray)
        f2: Imaginary part of atomic scattering factor (numpy.ndarray)
        Critical_Angle: Critical angles in degrees (numpy.ndarray)
        Attenuation_Length: Attenuation lengths in cm (numpy.ndarray)
        reSLD: Real part of scattering length density in Å⁻² (numpy.ndarray)
        imSLD: Imaginary part of scattering length density in Å⁻² (numpy.ndarray)
    """
    Formula: str                              # Chemical formula
    MW: float                                 # Molecular weight (g/mol)
    Number_Of_Electrons: float                # Electrons per molecule
    Density: float                            # Mass density (g/cm³)
    Electron_Density: float                   # Electron density (1/Å³)
    Energy: np.ndarray                        # X-ray energy (KeV)
    Wavelength: np.ndarray                    # X-ray wavelength (Å)
    Dispersion: np.ndarray                    # Dispersion coefficient
    Absorption: np.ndarray                    # Absorption coefficient
    f1: np.ndarray                            # Real part of atomic scattering factor
    f2: np.ndarray                            # Imaginary part of atomic scattering factor
    Critical_Angle: np.ndarray                # Critical angle (degrees)
    Attenuation_Length: np.ndarray            # Attenuation length (cm)
    reSLD: np.ndarray                         # Real part of SLD (Å⁻²)
    imSLD: np.ndarray                         # Imaginary part of SLD (Å⁻²)

# =====================================================================================
# CACHING SYSTEM
# =====================================================================================

# Module-level cache for f1/f2 scattering tables, keyed by element symbol
_scattering_factor_cache: Dict[str, pd.DataFrame] = {}


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
    global _scattering_factor_cache
    
    # Validate input
    if not element or not isinstance(element, str):
        raise ValueError(f"Element symbol must be a non-empty string, got: {repr(element)}")
    
    # Normalize element symbol (capitalize first letter, lowercase rest)
    element = element.capitalize()
    
    # Check if already cached
    if element in _scattering_factor_cache:
        return _scattering_factor_cache[element]
    
    # Construct file path - look for .nff files in src/AtomicScatteringFactor/
    # First check relative to current working directory
    base_paths = [
        Path.cwd() / "src" / "AtomicScatteringFactor",
        Path(__file__).parent.parent / "src" / "AtomicScatteringFactor",  # Parent of xraylabtool
        Path(__file__).parent / "data" / "AtomicScatteringFactor",  # Traditional data directory
    ]
    
    file_path = None
    for base_path in base_paths:
        candidate_path = base_path / f"{element.lower()}.nff"
        if candidate_path.exists():
            file_path = candidate_path
            break
    
    if file_path is None:
        # Create detailed error message with searched paths
        searched_paths = [str(bp / f"{element.lower()}.nff") for bp in base_paths]
        raise FileNotFoundError(
            f"Scattering factor data file not found for element '{element}'. "
            f"Searched in the following locations:\n" + 
            "\n".join(f"  - {path}" for path in searched_paths) +
            f"\n\nPlease ensure the file '{element.lower()}.nff' exists in one of these directories."
        )
    
    try:
        # Load .nff file using pandas.read_csv
        # .nff files are CSV format with header: E,f1,f2
        df = pd.read_csv(file_path)
        
        # Verify expected columns exist
        expected_columns = {'E', 'f1', 'f2'}
        actual_columns = set(df.columns)
        
        if not expected_columns.issubset(actual_columns):
            missing_cols = expected_columns - actual_columns
            raise ValueError(
                f"Invalid .nff file format for element '{element}'. "
                f"Missing required columns: {missing_cols}. "
                f"Found columns: {list(actual_columns)}"
            )
        
        # Verify data is not empty
        if df.empty:
            raise ValueError(f"Empty scattering factor data file for element '{element}': {file_path}")
        
        # Cache the data
        _scattering_factor_cache[element] = df
        
        return df
        
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(
            f"Empty or corrupted scattering factor data file for element '{element}': {file_path}"
        ) from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(
            f"Invalid file format in scattering factor data file for element '{element}': {file_path}. "
            f"Expected CSV format with columns: E,f1,f2"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading scattering factor data for element '{element}' from {file_path}: {e}"
        ) from e


class AtomicScatteringFactor:
    """
    Class for handling atomic scattering factors.
    
    This class loads and manages atomic scattering factor data
    from .nff files using the module-level cache.
    """
    
    def __init__(self):
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
    
    def __init__(self, lattice_parameters: Tuple[float, float, float, float, float, float]):
        """
        Initialize crystal structure.
        
        Args:
            lattice_parameters: (a, b, c, alpha, beta, gamma) in Angstroms and degrees
        """
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = lattice_parameters
        self.atoms: List[Dict] = []
    
    def add_atom(self, element: str, position: Tuple[float, float, float], occupancy: float = 1.0):
        """
        Add an atom to the crystal structure.
        
        Args:
            element: Element symbol
            position: Fractional coordinates (x, y, z)
            occupancy: Site occupancy factor
        """
        self.atoms.append({
            'element': element,
            'position': position,
            'occupancy': occupancy
        })
    
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


def clear_scattering_factor_cache() -> None:
    """
    Clear the module-level scattering factor cache.
    
    This function removes all cached scattering factor data from memory.
    Useful for testing or memory management.
    """
    global _scattering_factor_cache
    _scattering_factor_cache.clear()


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
    element_data: List[Tuple[float, Callable, Callable]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized calculation of X-ray scattering factors and optical properties.
    
    This function performs the core calculation of dispersion, absorption, and total
    scattering factors for a material based on its elemental composition.
    Port of Julia's calculate_scattering_factors! function.
    
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
    
    # Initialize output arrays with zeros
    dispersion = np.zeros(n_energies, dtype=np.float64)
    absorption = np.zeros(n_energies, dtype=np.float64)
    f1_total = np.zeros(n_energies, dtype=np.float64)
    f2_total = np.zeros(n_energies, dtype=np.float64)
    
    # Pre-compute density-dependent factor for efficiency
    # Factor includes: (λ²/2π) × rₑ × ρ × Nₐ / M
    common_factor = SCATTERING_FACTOR * mass_density / molecular_weight
    
    # Process each element in the formula
    for count, f1_interp, f2_interp in element_data:
        # Element-specific contribution factor
        element_contribution_factor = common_factor * count
        
        # Vectorized interpolation of scattering factors
        f1_values = f1_interp(energy_ev)
        f2_values = f2_interp(energy_ev)
        
        # Convert scalar results to arrays if necessary
        if np.isscalar(f1_values):
            f1_values = np.full(n_energies, f1_values)
        if np.isscalar(f2_values):
            f2_values = np.full(n_energies, f2_values)
        
        # Calculate wavelength-dependent factors (vectorized)
        wave_sq = wavelength ** 2
        
        # Accumulate contributions to optical properties (vectorized)
        dispersion += wave_sq * element_contribution_factor * f1_values
        absorption += wave_sq * element_contribution_factor * f2_values
        
        # Accumulate total scattering factors
        f1_total += count * f1_values
        f2_total += count * f2_values
    
    return dispersion, absorption, f1_total, f2_total


def calculate_derived_quantities(
    wavelength: np.ndarray,
    dispersion: np.ndarray,
    absorption: np.ndarray,
    mass_density: float,
    molecular_weight: float,
    number_of_electrons: float
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
    electron_density = 1e6 * mass_density / molecular_weight * AVOGADRO * number_of_electrons / 1e30
    
    # Calculate critical angle for total external reflection
    # θc = √(2δ) (in radians), converted to degrees
    critical_angle = np.sqrt(2.0 * dispersion) * (180.0 / PI)
    
    # Calculate X-ray attenuation length
    # 1/e attenuation length = λ/(4πβ) (in cm)
    attenuation_length = wavelength / absorption / (4 * PI) * 1e2
    
    # Calculate scattering length densities (SLD)
    # SLD = 2π × (δ + iβ) / λ² (in units of Å⁻²)
    wavelength_sq = wavelength ** 2
    sld_factor = 2 * PI / 1e20  # Conversion factor to Å⁻²
    
    re_sld = dispersion * sld_factor / wavelength_sq  # Real part of SLD
    im_sld = absorption * sld_factor / wavelength_sq  # Imaginary part of SLD
    
    return electron_density, critical_angle, attenuation_length, re_sld, im_sld


def create_scattering_factor_interpolators(element: str) -> Tuple[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]], Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]]:
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
    # Load scattering factor data
    data = load_scattering_factor_data(element)
    
    # Verify we have sufficient data points for PCHIP interpolation
    if len(data) < 2:
        raise ValueError(
            f"Insufficient data points for element '{element}'. "
            f"PCHIP interpolation requires at least 2 points, found {len(data)}."
        )
    
    # Extract energy, f1, and f2 data
    energies = np.asarray(data['E'].values)
    f1_values = np.asarray(data['f1'].values)
    f2_values = np.asarray(data['f2'].values)
    
    # Verify energy values are sorted (PCHIP requires sorted x values)
    if not np.all(energies[:-1] <= energies[1:]):
        # Sort the data if it's not already sorted
        sort_indices = np.argsort(energies)
        energies = energies[sort_indices]
        f1_values = f1_values[sort_indices]
        f2_values = f2_values[sort_indices]
    
    # Create PCHIP interpolators
    # PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
    # and provides smooth, shape-preserving interpolation similar to Julia's behavior
    f1_interpolator = PchipInterpolator(energies, f1_values, extrapolate=False)
    f2_interpolator = PchipInterpolator(energies, f2_values, extrapolate=False)
    
    return f1_interpolator, f2_interpolator


def calculate_xray_properties(
    formula_str: str,
    energy_kev: Union[float, List[float], np.ndarray],
    mass_density: float
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
        
    Examples:
        >>> result = calculate_xray_properties("SiO2", [8.0, 10.0, 12.0], 2.2)
        >>> print(f"Molecular weight: {result['molecular_weight']:.2f} g/mol")
        >>> print(f"Critical angles: {result['critical_angle']}")
        
        >>> # Single energy
        >>> result = calculate_xray_properties("Al2O3", 10.0, 3.95)
        >>> print(f"Attenuation length: {result['attenuation_length'][0]:.2f} cm")
    """
    from .utils import parse_formula, get_atomic_number, get_atomic_weight
    from .constants import ENERGY_TO_WAVELENGTH_FACTOR, METER_TO_ANGSTROM
    
    # Validate inputs
    if not formula_str or not isinstance(formula_str, str):
        raise ValueError("Formula must be a non-empty string")
    
    # Convert energy to numpy array
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
    
    if mass_density <= 0:
        raise ValueError("Mass density must be positive")
    
    if np.any(energy_kev <= 0):
        raise ValueError("All energies must be positive")
    
    # Validate energy range (X-ray energies typically 0.03-30 keV)
    if np.any(energy_kev < 0.03) or np.any(energy_kev > 30):
        raise ValueError("Energy is out of range 0.03keV ~ 30keV")
    
    # Parse the chemical formula into elements and their counts
    element_symbols, element_counts = parse_formula(formula_str)
    n_elements = len(element_symbols)
    n_energies = len(energy_kev)
    
    # Look up atomic data for each element in the formula
    molecular_weight = 0.0
    number_of_electrons = 0.0
    
    for i in range(n_elements):
        atomic_number = get_atomic_number(element_symbols[i])
        atomic_mass = get_atomic_weight(element_symbols[i])
        
        # Accumulate molecular weight and total electrons
        molecular_weight += element_counts[i] * atomic_mass
        number_of_electrons += atomic_number * element_counts[i]
    
    # Convert X-ray energies (keV) to wavelengths (m)
    # λ = hc/E, where h = Planck constant, c = speed of light
    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energy_kev
    
    # Convert energies to eV for scattering factor interpolation
    energy_ev = energy_kev * 1000.0
    
    # Load atomic scattering factor tables and create interpolators
    element_data = []
    
    for i in range(n_elements):
        # Create interpolators for this element
        f1_interp, f2_interp = create_scattering_factor_interpolators(element_symbols[i])
        
        # Store element count and interpolators
        element_data.append((element_counts[i], f1_interp, f2_interp))
    
    # Calculate dispersion, absorption, and total scattering factors
    dispersion, absorption, f1_total, f2_total = calculate_scattering_factors(
        energy_ev, wavelength, mass_density, molecular_weight, element_data
    )
    
    # Calculate derived quantities
    electron_density, critical_angle, attenuation_length, re_sld, im_sld = calculate_derived_quantities(
        wavelength, dispersion, absorption, mass_density, molecular_weight, number_of_electrons
    )
    
    # Return complete result structure
    return {
        'formula': formula_str,
        'molecular_weight': molecular_weight,
        'number_of_electrons': number_of_electrons,
        'mass_density': mass_density,
        'electron_density': electron_density,
        'energy': energy_kev,
        'wavelength': wavelength * METER_TO_ANGSTROM,  # Convert to Angstroms
        'dispersion': dispersion,
        'absorption': absorption,
        'f1_total': f1_total,
        'f2_total': f2_total,
        'critical_angle': critical_angle,
        'attenuation_length': attenuation_length,
        're_sld': re_sld,
        'im_sld': im_sld
    }


def calculate_multiple_xray_properties(
    formula_list: List[str],
    energy_kev: Union[float, List[float], np.ndarray],
    mass_density_list: List[float]
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
    
    for i, (formula, mass_density) in enumerate(zip(formula_list, mass_density_list)):
        try:
            # Calculate properties for this formula
            result = calculate_xray_properties(formula, energy_kev, mass_density)
            results[formula] = result
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
    if file_path.suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix.lower() in ['.txt', '.dat']:
        return pd.read_csv(file_path, delim_whitespace=True)
    else:
        # Try to load as generic text file
        return pd.read_csv(file_path, delim_whitespace=True)


# =====================================================================================
# PUBLIC API FUNCTIONS
# =====================================================================================

def SubRefrac(
    formula: str,
    energy_keV: Union[float, List[float], np.ndarray],
    density: float
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
        >>> result = SubRefrac("SiO2", 8.0, 2.2)
        >>> print(f"Molecular weight: {result.MW:.2f} g/mol")
        Molecular weight: 60.08 g/mol
        
        >>> # Multiple energies
        >>> result = SubRefrac("Al2O3", [8.0, 10.0, 12.0], 3.95)
        >>> print(f"Critical angles: {result.Critical_Angle}")
        
        >>> # Array input
        >>> energies = np.linspace(5.0, 15.0, 11)
        >>> result = SubRefrac("Fe2O3", energies, 5.24)
        >>> print(f"Energy range: {result.Energy[0]:.1f} - {result.Energy[-1]:.1f} keV")
    """
    # Calculate properties using the existing function
    properties = calculate_xray_properties(formula, energy_keV, density)
    
    # Create and return XRayResult dataclass
    return XRayResult(
        Formula=str(properties['formula']),
        MW=float(properties['molecular_weight']),
        Number_Of_Electrons=float(properties['number_of_electrons']),
        Density=float(properties['mass_density']),
        Electron_Density=float(properties['electron_density']),
        Energy=np.asarray(properties['energy']),
        Wavelength=np.asarray(properties['wavelength']),
        Dispersion=np.asarray(properties['dispersion']),
        Absorption=np.asarray(properties['absorption']),
        f1=np.asarray(properties['f1_total']),
        f2=np.asarray(properties['f2_total']),
        Critical_Angle=np.asarray(properties['critical_angle']),
        Attenuation_Length=np.asarray(properties['attenuation_length']),
        reSLD=np.asarray(properties['re_sld']),
        imSLD=np.asarray(properties['im_sld'])
    )


def Refrac(
    formulas: List[str],
    energies: Union[float, List[float], np.ndarray],
    densities: List[float]
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
        >>> results = Refrac(formulas, energies, densities)
        >>> sio2_result = results["SiO2"]
        >>> print(f"SiO2 MW: {sio2_result.MW:.2f} g/mol")
        SiO2 MW: 60.08 g/mol
        
        >>> # Single energy for all materials
        >>> results = Refrac(["SiO2", "Al2O3"], 10.0, [2.2, 3.95])
        >>> for formula, result in results.items():
        ...     print(f"{formula}: {result.Critical_Angle[0]:.3f}°")
        
        >>> # Array of energies
        >>> energy_array = np.linspace(5.0, 15.0, 21)
        >>> results = Refrac(["SiO2"], energy_array, [2.2])
        >>> print(f"Energy points: {len(results['SiO2'].Energy)}")
    """
    # Input validation
    if not isinstance(formulas, list) or not formulas:
        raise ValueError("Formulas must be a non-empty list")
    
    if not isinstance(densities, list) or not densities:
        raise ValueError("Densities must be a non-empty list")
    
    if len(formulas) != len(densities):
        raise ValueError(
            f"Number of formulas ({len(formulas)}) must match number of densities ({len(densities)})"
        )
    
    # Validate individual formulas and densities
    for i, formula in enumerate(formulas):
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError(f"Formula at index {i} must be a non-empty string, got: {repr(formula)}")
    
    for i, density in enumerate(densities):
        if not isinstance(density, (int, float)) or density <= 0:
            raise ValueError(f"Density at index {i} must be a positive number, got: {density}")
    
    # Convert and validate energies
    if np.isscalar(energies):
        if isinstance(energies, complex):
            energies_array = np.array([float(energies.real)], dtype=np.float64)
        elif isinstance(energies, (int, float, np.number)):
            energies_array = np.array([float(energies)], dtype=np.float64)
        else:
            try:
                energies_array = np.array([float(energies)], dtype=np.float64)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert energy to float: {energies}")
    else:
        energies_array = np.array(energies, dtype=np.float64)
    
    if energies_array.size == 0:
        raise ValueError("Energies array cannot be empty")
    
    if np.any(energies_array <= 0):
        raise ValueError("All energies must be positive")
    
    if np.any(energies_array < 0.03) or np.any(energies_array > 30):
        raise ValueError("Energy values must be in range 0.03-30 keV")
    
    # Sort energies for consistent processing
    # Note: We sort the energies to ensure consistent results, but we'll maintain
    # the original order in the final results by sorting back if needed
    sort_indices = np.argsort(energies_array)
    sorted_energies = energies_array[sort_indices]
    
    # Function to process a single formula
    def process_formula(formula_density_pair: Tuple[str, float]) -> Tuple[str, XRayResult]:
        formula, density = formula_density_pair
        try:
            result = SubRefrac(formula, sorted_energies, density)
            
            # If energies were sorted, we need to restore original order in results
            if not np.array_equal(sort_indices, np.arange(len(sort_indices))):
                # Create reverse mapping to restore original order
                reverse_indices = np.argsort(sort_indices)
                
                # Restore original order in all array fields
                result = XRayResult(
                    Formula=result.Formula,
                    MW=result.MW,
                    Number_Of_Electrons=result.Number_Of_Electrons,
                    Density=result.Density,
                    Electron_Density=result.Electron_Density,
                    Energy=result.Energy[reverse_indices],
                    Wavelength=result.Wavelength[reverse_indices],
                    Dispersion=result.Dispersion[reverse_indices],
                    Absorption=result.Absorption[reverse_indices],
                    f1=result.f1[reverse_indices],
                    f2=result.f2[reverse_indices],
                    Critical_Angle=result.Critical_Angle[reverse_indices],
                    Attenuation_Length=result.Attenuation_Length[reverse_indices],
                    reSLD=result.reSLD[reverse_indices],
                    imSLD=result.imSLD[reverse_indices]
                )
            
            return (formula, result)
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Failed to process formula '{formula}': {e}") from e
    
    # Process formulas in parallel using ThreadPoolExecutor
    formula_density_pairs = list(zip(formulas, densities))
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(formulas), 8)) as executor:
        # Submit all tasks
        future_to_formula = {
            executor.submit(process_formula, pair): pair[0] 
            for pair in formula_density_pairs
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_formula):
            formula = future_to_formula[future]
            try:
                formula_result, xray_result = future.result()
                results[formula_result] = xray_result
            except Exception as e:
                # Log the error but continue processing other formulas
                print(f"Warning: Failed to process formula '{formula}': {e}")
                continue
    
    if not results:
        raise RuntimeError("Failed to process any formulas successfully")
    
    return results
