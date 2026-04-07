"""
Core functionality for XRayLabTool.

This module is the orchestration layer for X-ray analysis. It re-exports
domain types and cache utilities from focused submodules and contains the
public API functions that wire everything together.
"""

# ruff: noqa: PLC0415, PLW0603

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np  # Keep for validation and array operations

from xraylabtool.backend import ops

# --- Re-exports from submodules (preserve backward-compatible import paths) ---
from xraylabtool.calculators.cache import (
    _CACHE_WARMED as _CACHE_WARMED,
)
from xraylabtool.calculators.cache import (
    _atomic_data_cache as _atomic_data_cache,
)
from xraylabtool.calculators.cache import (
    _interpolator_cache as _interpolator_cache,
)
from xraylabtool.calculators.cache import (
    _scattering_factor_cache as _scattering_factor_cache,
)
from xraylabtool.calculators.cache import (
    _smart_cache_warming,
)
from xraylabtool.calculators.cache import (
    clear_scattering_factor_cache as clear_scattering_factor_cache,
)
from xraylabtool.calculators.cache import (
    create_scattering_factor_interpolators as create_scattering_factor_interpolators,
)
from xraylabtool.calculators.cache import (
    get_bulk_atomic_data as get_bulk_atomic_data,
)
from xraylabtool.calculators.cache import (
    get_cached_elements as get_cached_elements,
)
from xraylabtool.calculators.cache import (
    is_element_cached as is_element_cached,
)
from xraylabtool.calculators.kernels import (
    _derived_quantities_kernel as _derived_quantities_kernel,
)
from xraylabtool.calculators.kernels import (
    _derived_quantities_kernel_raw as _derived_quantities_kernel_raw,
)
from xraylabtool.calculators.kernels import (
    _get_derived_kernel as _get_derived_kernel,
)
from xraylabtool.calculators.kernels import (
    _get_scattering_kernel as _get_scattering_kernel,
)
from xraylabtool.calculators.kernels import (
    _jit_cache as _jit_cache,
)
from xraylabtool.calculators.kernels import (
    _scattering_math_kernel as _scattering_math_kernel,
)
from xraylabtool.calculators.kernels import (
    _scattering_math_kernel_raw as _scattering_math_kernel_raw,
)
from xraylabtool.calculators.kernels import (
    calculate_derived_quantities as calculate_derived_quantities,
)
from xraylabtool.calculators.kernels import (
    calculate_scattering_factors as calculate_scattering_factors,
)
from xraylabtool.calculators.scattering_data import (
    _AVAILABLE_ELEMENTS as _AVAILABLE_ELEMENTS,
)
from xraylabtool.calculators.scattering_data import (
    AtomicScatteringFactor as AtomicScatteringFactor,
)
from xraylabtool.calculators.scattering_data import (
    CrystalStructure as CrystalStructure,
)
from xraylabtool.calculators.scattering_data import (
    ScatteringData as ScatteringData,
)
from xraylabtool.calculators.scattering_data import (
    _initialize_element_paths as _initialize_element_paths,
)
from xraylabtool.calculators.scattering_data import (
    load_data_file as load_data_file,
)
from xraylabtool.calculators.scattering_data import (
    load_scattering_factor_data as load_scattering_factor_data,
)
from xraylabtool.calculators.xray_result import XRayResult as XRayResult
from xraylabtool.exceptions import EnergyError, FormulaError

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        ArrayLike,
        EnergyArray,
        FloatLike,
        InterpolatorProtocol,
    )


# =====================================================================================
# VALIDATION HELPERS
# =====================================================================================


def _validate_single_material_inputs(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
) -> EnergyArray:
    """Validate inputs for single material calculation."""
    if not formula_str or not isinstance(formula_str, str):
        raise FormulaError("Formula must be a non-empty string", formula_str)

    if ops.any(ops.asarray(mass_density) <= 0):
        raise ValueError("Mass density must be positive")

    # Convert and validate energy
    energy_kev = _convert_energy_input(energy_kev)

    if np.any(energy_kev <= 0):
        raise EnergyError("All energies must be positive", valid_range="0.03-30 keV")

    if np.any(energy_kev < 0.03) or np.any(energy_kev > 30):
        raise EnergyError(
            "Energy is out of range 0.03keV ~ 30keV", valid_range="0.03-30 keV"
        )

    return energy_kev


def _convert_energy_input(energy_kev: Any) -> EnergyArray:
    """Convert energy input to array."""
    if np.isscalar(energy_kev):
        if isinstance(energy_kev, complex):
            energy_kev = ops.asarray([float(energy_kev.real)], dtype=ops.float64)
        elif isinstance(energy_kev, int | float | np.number):
            energy_kev = ops.asarray([float(energy_kev)], dtype=ops.float64)
        else:
            try:
                energy_kev = ops.asarray([float(energy_kev)], dtype=ops.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert energy to float: {energy_kev}") from e
    else:
        energy_kev = ops.asarray(energy_kev, dtype=ops.float64)

    return ops.asarray(energy_kev)  # type: ignore[no-any-return]


# =====================================================================================
# INTERNAL CALCULATION PIPELINE
# =====================================================================================


def _calculate_molecular_properties(
    element_symbols: list[str],
    element_counts: list[float],
    atomic_data_bulk: dict[str, Any],
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


def _calculate_single_material_xray_properties(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
) -> dict[str, str | float | np.ndarray]:
    """
    Calculate X-ray optical properties for a single chemical formula.

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


def calculate_multiple_xray_properties(
    formula_list: list[str],
    energy_kev: FloatLike | ArrayLike,
    mass_density_list: list[float],
) -> dict[str, dict[str, str | float | np.ndarray]]:
    """
    Calculate X-ray optical properties for multiple chemical formulas.

    Args:
        formula_list: List of chemical formulas
        energy_kev: X-ray energies in keV (scalar, list, or array)
        mass_density_list: Mass densities in g/cm³

    Returns:
        Dictionary mapping formula strings to result dictionaries

    Raises:
        ValueError: If input lists have different lengths or invalid values
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

            # Convert XRayResult to dictionary format for backward compatibility
            result_dict: dict[str, str | float | np.ndarray] = {
                "formula": result.formula,
                "molecular_weight": result.molecular_weight_g_mol,
                "number_of_electrons": result.total_electrons,
                "mass_density": result.density_g_cm3,
                "electron_density": result.electron_density_per_ang3,
                "energy": result.energy_kev,
                "wavelength": result.wavelength_angstrom,
                "dispersion": result.dispersion_delta,
                "absorption": result.absorption_beta,
                "f1_total": result.scattering_factor_f1,
                "f2_total": result.scattering_factor_f2,
                "critical_angle": result.critical_angle_degrees,
                "attenuation_length": result.attenuation_length_cm,
                "re_sld": result.real_sld_per_ang2,
                "im_sld": result.imaginary_sld_per_ang2,
            }
            results[formula] = result_dict
        except Exception as e:
            # Log warning but continue processing other formulas
            print(f"Warning: Failed to process formula {formula}: {e}")
            continue

    return results


# =====================================================================================
# PUBLIC API FUNCTIONS
# =====================================================================================


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

    Args:
        formula: Chemical formula string (e.g., "SiO2", "Al2O3", "CaCO3")
        energy_keV: X-ray energies in keV. Valid range: 0.03-30.0 keV
        density: Material mass density in g/cm³ (must be positive)
        energy: Optional backward-compatible alias (eV). If provided, interpreted as
                electron-volts and converted to keV.

    Returns:
        XRayResult: Dataclass containing all calculated X-ray properties.

    Raises:
        FormulaError: If chemical formula cannot be parsed or contains invalid elements
        EnergyError: If energy values are outside valid range (0.03-30.0 keV)
        ValidationError: If density is not positive or other validation failures

    Examples:
        >>> import xraylabtool as xlt
        >>> result = xlt.calculate_single_material_properties("SiO2", 8.0, 2.2)
        >>> print(f"Formula: {result.formula}")
        Formula: SiO2

    See Also:
        calculate_xray_properties : Calculate properties for multiple materials
        XRayResult : Complete documentation of returned dataclass
    """
    # Backward compatibility: allow 'energy' in eV
    if energy_keV is None and energy is not None:
        energy_keV = np.asarray(energy, dtype=float) / 1000.0  # noqa: N806
    if energy_keV is None:
        raise ValueError("energy_keV or energy must be provided")
    if density is None:
        raise ValueError("density must be provided")

    # Use smart cache warming for faster cold start (only loads required elements)
    from xraylabtool.calculators import cache as _cache_mod

    if not _cache_mod._CACHE_WARMED:
        _smart_cache_warming(formula)

    # Calculate properties using the existing function
    properties = _calculate_single_material_xray_properties(
        formula, energy_keV, density
    )

    # Create and return XRayResult dataclass using new field names.
    return XRayResult(
        formula=str(properties["formula"]),
        molecular_weight_g_mol=float(properties["molecular_weight"]),
        total_electrons=float(properties["number_of_electrons"]),
        density_g_cm3=float(properties["mass_density"]),
        electron_density_per_ang3=float(properties["electron_density"]),
        energy_kev=properties["energy"],
        wavelength_angstrom=properties["wavelength"],
        dispersion_delta=properties["dispersion"],
        absorption_beta=properties["absorption"],
        scattering_factor_f1=properties["f1_total"],
        scattering_factor_f2=properties["f2_total"],
        critical_angle_degrees=properties["critical_angle"],
        attenuation_length_cm=properties["attenuation_length"],
        real_sld_per_ang2=properties["re_sld"],
        imaginary_sld_per_ang2=properties["im_sld"],
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
        raise EnergyError("Energies array cannot be empty", valid_range="0.03-30 keV")

    if np.any(energies_array <= 0):
        raise EnergyError("All energies must be positive", valid_range="0.03-30 keV")

    if np.any(energies_array < 0.03) or np.any(energies_array > 30):
        raise EnergyError(
            "Energy values must be in range 0.03-30 keV", valid_range="0.03-30 keV"
        )

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
    Process formulas with adaptive strategy.

    Uses sequential processing for all batches. Threading is avoided because
    JAX's JIT-compiled kernels already parallelize at the XLA level.
    """
    formula_density_pairs = list(zip(formulas, densities, strict=False))
    results = {}

    for pair in formula_density_pairs:
        try:
            formula_result, xray_result = process_func(pair)
            results[formula_result] = xray_result
        except Exception as e:
            print(f"Warning: Failed to process formula '{pair[0]}': {e}")
            continue
    return results


def calculate_xray_properties(
    formulas: list[str],
    energies: FloatLike | ArrayLike,
    densities: list[float],
) -> dict[str, XRayResult]:
    """
    Calculate X-ray optical properties for multiple material compositions.

    Args:
        formulas: List of chemical formula strings
        energies: X-ray energies in keV applied to all materials
        densities: List of material mass densities in g/cm³

    Returns:
        Dict[str, XRayResult]: Dictionary mapping formula strings to XRayResult objects.

    Raises:
        ValidationError: If inputs don't match
        RuntimeError: If no formulas were processed successfully

    See Also:
        calculate_single_material_properties : Single material calculations
        XRayResult : Documentation of returned data structure
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
    """

    def __init__(self) -> None:
        """Initialize the calculation engine."""
        self._cache_warmed = False

    def calculate_optical_constants(
        self,
        formula: str,
        energies: EnergyArray,
        density: FloatLike,
    ) -> tuple[Any, Any]:
        """Calculate dispersion and absorption coefficients."""
        result_dict = _calculate_single_material_xray_properties(
            formula, energies, density
        )

        dispersion = np.asarray(result_dict["dispersion"], dtype=np.float64)
        absorption = np.asarray(result_dict["absorption"], dtype=np.float64)

        return dispersion, absorption

    def calculate_derived_quantities(
        self,
        dispersion: Any,
        absorption: Any,
        energies: EnergyArray,
    ) -> dict[str, np.ndarray]:
        """Calculate derived quantities from optical constants."""
        from xraylabtool.constants import (
            ENERGY_TO_WAVELENGTH_FACTOR,
            METER_TO_ANGSTROM,
        )

        wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energies
        wavelength * METER_TO_ANGSTROM

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
        """Pre-warm caches for improved performance."""
        if self._cache_warmed:
            return

        if common_elements is None:
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

        from xraylabtool.data_handling.atomic_cache import (
            get_atomic_data_provider,
        )

        provider = get_atomic_data_provider()
        provider.preload_elements(common_elements)

        self._cache_warmed = True

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance information about the calculation engine."""
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
    """Get the global calculation engine instance."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = FastXRayCalculationEngine()
        _GLOBAL_ENGINE.warm_up_cache()
    return _GLOBAL_ENGINE
