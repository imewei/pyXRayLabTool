"""
Core functionality for XRayLabTool.

This module is the orchestration layer for X-ray analysis. It re-exports
domain types and cache utilities from focused submodules and contains the
public API functions that wire everything together.
"""

# ruff: noqa: PLC0415, PLW0603

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np  # Host-side validation and result containers

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
from xraylabtool.calculators.derived_quantities import (
    calculate_derived_quantities as calculate_derived_quantities,
)
from xraylabtool.calculators.derived_quantities import (
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
from xraylabtool.logging_utils import get_logger

if TYPE_CHECKING:
    from xraylabtool.typing_extensions import (
        ArrayLike,
        EnergyArray,
        FloatLike,
        OpticalConstantArray,
    )

_logger = get_logger(__name__)


# =====================================================================================
# VALIDATION HELPERS
# =====================================================================================


def _validate_formula_and_density(formula_str: str, mass_density: FloatLike) -> None:
    """Raise today's ``FormulaError`` / ``ValueError`` for a bad formula or density."""
    if not formula_str or not isinstance(formula_str, str):
        raise FormulaError("Formula must be a non-empty string", formula_str)

    # Plain numbers skip the ndarray round trip (the common batch case).
    if isinstance(mass_density, float | int):
        non_positive = bool(mass_density <= 0)
    else:
        non_positive = bool(np.any(np.asarray(mass_density) <= 0))
    if non_positive:
        raise ValueError("Mass density must be positive")


def _validate_energies(energy_kev: FloatLike | ArrayLike) -> EnergyArray:
    """Convert ``energy_kev`` to a new 1-D float64 array; raise ``EnergyError`` if bad."""
    energy_kev = _convert_energy_input(energy_kev)

    if energy_kev.size == 0:
        raise EnergyError("Energies array cannot be empty", valid_range="0.03-30 keV")

    if np.any(energy_kev <= 0):
        raise EnergyError("All energies must be positive", valid_range="0.03-30 keV")

    if np.any(energy_kev < 0.03) or np.any(energy_kev > 30):
        raise EnergyError(
            "Energy is out of range 0.03keV ~ 30keV", valid_range="0.03-30 keV"
        )

    _require_finite_energies(energy_kev)

    return energy_kev


def _validate_single_material_inputs(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
) -> EnergyArray:
    """Validate inputs for single material calculation."""
    _validate_formula_and_density(formula_str, mass_density)
    return _validate_energies(energy_kev)


def unique_result_key(results: Mapping[str, Any], formula: str, density: float) -> str:
    """Return a key for ``formula`` that never overwrites an earlier result.

    The first occurrence keeps the plain formula; a repeat with another density
    becomes ``formula@density`` (with a ``#n`` suffix if even that is taken).
    """
    if formula not in results:
        return formula
    base = f"{formula}@{density:.3f}"
    key, n = base, 2
    while key in results:
        key = f"{base}#{n}"
        n += 1
    return key


def _coerce_energy_array(energies: Any) -> np.ndarray:
    """Return ``energies`` as a new, real, 1-D float64 NumPy array.

    The result never aliases the caller's data. Scalars and 0-d arrays become
    length-one arrays; complex values and arrays with more than one dimension are
    rejected, because discarding the imaginary part or broadcasting a grid along
    the wrong axis would silently produce wrong results.
    """
    if isinstance(energies, complex) or np.iscomplexobj(energies):
        raise EnergyError(
            "Complex energy values are not supported; pass real energies in keV",
            valid_range="0.03-30 keV",
        )
    if np.isscalar(energies):
        try:
            return np.array([float(energies)], dtype=np.float64)  # type: ignore[arg-type]
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert energy to float: {energies!r}") from e

    array = np.array(energies, dtype=np.float64)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        raise EnergyError(
            f"Energies must be a scalar or a 1-D array, got {array.ndim}-D "
            f"array with shape {array.shape}",
            valid_range="0.03-30 keV",
        )
    return array


def _require_finite_energies(energies: np.ndarray) -> None:
    """Reject NaN/Inf energies (NaN passes every ordering comparison)."""
    if not np.all(np.isfinite(energies)):
        raise EnergyError(
            "Energies must be finite (no NaN or Inf)", valid_range="0.03-30 keV"
        )


def _convert_energy_input(energy_kev: Any) -> EnergyArray:
    """Convert energy input to a new 1-D float64 NumPy array."""
    return _coerce_energy_array(energy_kev)


# =====================================================================================
# INTERNAL CALCULATION PIPELINE
# =====================================================================================


def _calculate_molecular_properties(
    element_symbols: list[str],
    element_counts: list[float],
    atomic_data_bulk: Mapping[str, Any],
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


class _MaterialPropertiesDict(TypedDict):
    """Per-key typed result of a single-material calculation.

    Replaces the previous ``dict[str, str | float | np.ndarray]`` return type:
    that union forced every reader (notably ``XRayResult`` construction) to
    either cast each value or accept a real type mismatch, since mypy can't
    tell which keys hold scalars vs. arrays. TypedDict gives each key its
    precise type instead.
    """

    formula: str
    molecular_weight: float
    number_of_electrons: float
    mass_density: float
    electron_density: float
    energy: np.ndarray
    wavelength: np.ndarray
    dispersion: np.ndarray
    absorption: np.ndarray
    f1_total: np.ndarray
    f2_total: np.ndarray
    critical_angle: np.ndarray
    attenuation_length: np.ndarray
    re_sld: np.ndarray
    im_sld: np.ndarray


_NON_FINITE_MESSAGE = (
    "Non-finite values (NaN or Inf) detected in dispersion or absorption coefficients"
)


def _density_value(mass_density: Any) -> float:
    """``mass_density`` as a float; a list or tuple keeps its legacy ``TypeError``."""
    if isinstance(mass_density, list | tuple):
        raise TypeError("can't multiply sequence by non-int of type 'float'")
    return float(mass_density)


@lru_cache(maxsize=4096)
def _encode_material(formula: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """``encode_formula`` after the checks and cache effects of the pre-device path.

    An unknown symbol raises the atomic-data lookup's ``UnknownElementError``; a
    symbol without a data file (for example ``Pu``) raises the loader's
    ``FileNotFoundError``; every element read fills the element cache that
    :func:`get_cached_elements` reports. Memoised per formula (failures are not
    cached, so they raise every time); the arrays are read-only and
    ``clear_scattering_factor_cache`` clears the memo.
    """
    from xraylabtool.calculators.optics import encode_formula
    from xraylabtool.utils import parse_formula

    symbols = parse_formula(formula)[0]
    get_bulk_atomic_data(tuple(symbols))
    for symbol in symbols:
        create_scattering_factor_interpolators(symbol)
    z, counts, molecular_weight, number_of_electrons = encode_formula(formula)
    z.flags.writeable = counts.flags.writeable = False
    return z, counts, molecular_weight, number_of_electrons


def _material_properties(
    formula: str,
    energy_kev: np.ndarray,
    mass_density: float,
    molecular_weight: float,
    number_of_electrons: float,
    got: Mapping[str, np.ndarray],
) -> _MaterialPropertiesDict:
    """Typed properties of row 0 of a downloaded program result (fresh arrays)."""
    if not got["ok"][0]:
        raise ValueError(_NON_FINITE_MESSAGE)
    out = got["out"][0]  # rows follow optics.OUT_FIELDS
    return {
        "formula": formula,
        "molecular_weight": molecular_weight,
        "number_of_electrons": number_of_electrons,
        "mass_density": mass_density,
        "electron_density": float(got["electron_density"][0]),
        "energy": energy_kev,
        "wavelength": out[0].copy(),
        "dispersion": out[1].copy(),
        "absorption": out[2].copy(),
        "f1_total": out[3].copy(),
        "f2_total": out[4].copy(),
        "critical_angle": out[5].copy(),
        "attenuation_length": out[6].copy(),
        "re_sld": out[7].copy(),
        "im_sld": out[8].copy(),
    }


def _calculate_single_material_xray_properties(
    formula_str: str,
    energy_kev: FloatLike | ArrayLike,
    mass_density: FloatLike,
) -> _MaterialPropertiesDict:
    """
    Calculate X-ray optical properties for a single chemical formula.

    This is an internal function. Use calculate_single_material_properties()
    for the public API that returns XRayResult objects. One device program call:
    one upload, one fetch.
    """
    from xraylabtool.calculators import host

    energy_kev = _validate_single_material_inputs(formula_str, energy_kev, mass_density)
    z, counts, molecular_weight, number_of_electrons = _encode_material(formula_str)
    host.check_molecular_weight(mass_density, molecular_weight)
    density = _density_value(mass_density)
    got = host.run_optical(
        [(z, counts, molecular_weight, number_of_electrons)], [density], energy_kev
    )
    return _material_properties(
        formula_str,
        energy_kev,
        density,
        molecular_weight,
        number_of_electrons,
        got,
    )


def _properties_from_totals(
    formula: str,
    energy_kev: EnergyArray,
    mass_density: FloatLike,
    molecular_weight: float,
    number_of_electrons: float,
    f1_total: OpticalConstantArray,
    f2_total: OpticalConstantArray,
) -> _MaterialPropertiesDict:
    """Shared tail: δ, β and derived quantities from per-formula f1/f2 totals.

    Used by :func:`calculate_properties_from_factors` (the device program that
    follows the table lookup of the single-material path); it does no input
    validation (callers validate).
    """
    from xraylabtool.calculators import host

    density = float(mass_density)
    got = host.run_tail(
        [molecular_weight],
        [density],
        [number_of_electrons],
        np.asarray(f1_total)[None],
        np.asarray(f2_total)[None],
        energy_kev,
    )
    return _material_properties(
        formula, energy_kev, density, molecular_weight, number_of_electrons, got
    )


def _xray_result_from_properties(properties: _MaterialPropertiesDict) -> XRayResult:
    """Build an :class:`XRayResult` from the typed properties dictionary."""
    return XRayResult(
        formula=properties["formula"],
        molecular_weight_g_mol=properties["molecular_weight"],
        total_electrons=properties["number_of_electrons"],
        density_g_cm3=properties["mass_density"],
        electron_density_per_ang3=properties["electron_density"],
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


def _calculate_materials(
    formulas: Sequence[str],
    densities: Sequence[Any],
    energy_kev: Any,
    *,
    max_block: int | None = None,
    on_block: Callable[[int], None] | None = None,
    prevalidated: bool = False,
) -> list[_MaterialPropertiesDict | Exception]:
    """Properties of many materials at shared energies as device ``F``-blocks.

    Each formula is screened on the host exactly as the single-material path does
    (formula, density, energies, parse); the survivors run through
    :func:`xraylabtool.calculators.host.run_optical_blocks`, one upload and one
    fetch per block. The outcome of material ``i`` is its properties or the
    exception that single-material call would have raised (screening failure or
    a non-finite status); outcomes follow the input order.

    Parameters
    ----------
    formulas, densities : sequence
        One entry per material.
    energy_kev : array_like
        Shared energies in keV.
    max_block : int, optional
        Upper bound of materials per block.
    on_block : callable, optional
        Called with the number of materials after each block.
    prevalidated : bool, default False
        ``energy_kev`` is already a validated array (skip the energy check).
    """
    from xraylabtool.calculators import cache as _cache_mod
    from xraylabtool.calculators import host

    if not _cache_mod._CACHE_WARMED and formulas:
        _smart_cache_warming(formulas[0])

    energies: np.ndarray | None = None
    energy_error: Exception | None = None
    try:
        energies = (
            np.asarray(energy_kev, dtype=np.float64)
            if prevalidated
            else _validate_energies(energy_kev)
        )
    except Exception as e:
        energy_error = e

    outcomes: list[_MaterialPropertiesDict | Exception | None] = [None] * len(formulas)
    screened: list[int] = []
    encoded = []
    values = []
    for i, (formula, density) in enumerate(zip(formulas, densities, strict=True)):
        try:
            _validate_formula_and_density(formula, density)
            if energy_error is not None:
                raise energy_error
            material = _encode_material(formula)
            host.check_molecular_weight(density, material[2])
            encoded.append(material)
            values.append(_density_value(density))
            screened.append(i)
        except Exception as e:
            outcomes[i] = e
    if screened:
        assert energies is not None
        got = host.run_optical_blocks(
            encoded, values, energies, max_block=max_block, on_block=on_block
        )
        for j, i in enumerate(screened):
            _z, _counts, molecular_weight, number_of_electrons = encoded[j]
            try:
                outcomes[i] = _material_properties(
                    formulas[i],
                    energies.copy(),
                    values[j],
                    molecular_weight,
                    number_of_electrons,
                    {
                        "out": got["out"][j : j + 1],
                        "electron_density": got["electron_density"][j : j + 1],
                        "ok": got["ok"][j : j + 1],
                    },
                )
            except ValueError as e:
                outcomes[i] = e
    return [o for o in outcomes if o is not None]


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

    # Process all formulas as device F-blocks
    results: dict[str, dict[str, str | float | np.ndarray]] = {}

    outcomes = _calculate_materials(formula_list, mass_density_list, energy_kev)
    for formula, mass_density, outcome in zip(
        formula_list, mass_density_list, outcomes, strict=True
    ):
        if isinstance(outcome, Exception):
            # Log warning but continue processing other formulas
            _logger.warning(
                "Failed to process formula",
                extra={"formula": formula, "error": str(outcome)},
            )
            continue
        result = _xray_result_from_properties(outcome)

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
        results[unique_result_key(results, formula, mass_density)] = result_dict

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

    return _xray_result_from_properties(properties)


def warm_up(formulas: Sequence[str] = ("SiO2",)) -> None:
    """Precompile the default-bucket device programs for ``formulas``.

    Loads the resident scattering tables, then runs the single-material
    program and the supplied-factor program once per distinct element bucket
    of ``formulas`` at one material and the smallest energy bucket, so a later
    ``calculate_single_material_properties`` or
    ``calculate_properties_from_factors`` call with up to 64 energies compiles
    nothing.

    Parameters
    ----------
    formulas : sequence of str, default ("SiO2",)
        Chemical formulas whose element counts select the programs to build.

    Raises
    ------
    FormulaError
        If a formula cannot be parsed.
    UnknownElementError
        If a formula names an element without tabulated data.

    References
    ----------
    Package addition; see ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md``.

    Examples
    --------
    >>> import xraylabtool as xlt
    >>> xlt.warm_up(("SiO2",))
    """
    from xraylabtool.backend.shapes import bucket_pow2
    from xraylabtool.calculators import host
    from xraylabtool.calculators.optics import encode_formula
    from xraylabtool.calculators.tables import load_tables

    load_tables()
    seen: set[int] = set()
    for formula in formulas:
        z, counts, molecular_weight, number_of_electrons = encode_formula(formula)
        k = bucket_pow2(len(z), minimum=2)
        if k in seen:
            continue
        seen.add(k)
        host.run_optical(
            [(z, counts, molecular_weight, number_of_electrons)], [1.0], [8.0]
        )
    host.run_tail([1.0], [1.0], [1.0], [[0.0]], [[0.0]], [8.0])


def calculate_properties_from_factors(
    formula: str,
    energy_kev: EnergyArray,
    density: float,
    f1_total: OpticalConstantArray,
    f2_total: OpticalConstantArray,
) -> XRayResult:
    """Build an :class:`XRayResult` from supplied scattering factors.

    Computes δ, β and the derived quantities from per-formula-unit ``f1`` and
    ``f2`` that the caller provides instead of reading the tabulated ``.nff``
    files. It runs the same code as :func:`calculate_single_material_properties`
    after its table lookup. This function is a package addition, not part of
    the cited databases or papers.

    Parameters
    ----------
    formula : str
        Chemical formula of one formula unit, e.g. ``"SiO2"``.
    energy_kev : numpy.ndarray, shape (m,)
        Photon energies in keV. Finite; the tabulated 0.03-30 keV range is not
        enforced here.
    density : float
        Mass density in g/cm^3; finite and positive.
    f1_total : numpy.ndarray, shape (m,)
        Real scattering factor, electrons per formula unit. Negative values
        (anomalous dispersion near an edge) are kept: δ becomes negative and
        the critical angle is 0 there.
    f2_total : numpy.ndarray, shape (m,)
        Imaginary scattering factor, electrons per formula unit. Negative
        values are kept; the attenuation length clamps β at ``1e-30``.

    Returns
    -------
    XRayResult
        All fields are arrays of shape ``(m,)`` except the scalar material
        properties; the order of ``energy_kev`` is preserved.

    Raises
    ------
    ValidationError
        If the three arrays are not non-empty 1-D arrays of equal length, any
        value is non-finite, or ``density`` is not finite and positive.
    FormulaError
        If ``formula`` cannot be parsed or names an unknown element.

    References
    ----------
    Package addition. The optical constants follow the CXRO/NIST convention
    used by :func:`calculate_single_material_properties`.

    Examples
    --------
    >>> import numpy as np
    >>> from xraylabtool.calculators.core import calculate_properties_from_factors
    >>> res = calculate_properties_from_factors(
    ...     "Si", np.array([8.0]), 2.33, np.array([14.2]), np.array([0.5])
    ... )
    >>> res.scattering_factor_f1.shape
    (1,)
    >>> bool(res.dispersion_delta[0] > 0)
    True
    """
    from xraylabtool.exceptions import ValidationError
    from xraylabtool.utils import parse_formula

    arrays: dict[str, np.ndarray] = {}
    for name, value in (
        ("energy_kev", energy_kev),
        ("f1_total", f1_total),
        ("f2_total", f2_total),
    ):
        try:
            if np.asarray(value).dtype.kind == "c":  # float64 cast drops imag part
                raise ValidationError("Must be a real numeric array", name)
            array = np.array(value, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValidationError("Must be a real numeric array", name) from e
        if array.ndim != 1 or array.size == 0:
            raise ValidationError(
                "Must be a non-empty 1-D array", name, f"shape {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValidationError("Must be finite (no NaN or Inf)", name)
        arrays[name] = array
    n = arrays["energy_kev"].size
    if arrays["f1_total"].size != n or arrays["f2_total"].size != n:
        raise ValidationError(
            "energy_kev, f1_total and f2_total must have equal length",
            "length",
            f"{n}, {arrays['f1_total'].size}, {arrays['f2_total'].size}",
        )
    if np.any(arrays["energy_kev"] <= 0):
        raise ValidationError("Energies must be positive", "energy_kev")
    try:
        density_value = float(density)
    except (TypeError, ValueError) as e:
        raise ValidationError("Density must be a real number", "density") from e
    if not np.isfinite(density_value) or density_value <= 0:
        raise ValidationError("Density must be finite and positive", "density", density)

    element_symbols, element_counts = parse_formula(formula)
    atomic_data_bulk = get_bulk_atomic_data(tuple(element_symbols))
    molecular_weight, number_of_electrons = _calculate_molecular_properties(
        element_symbols, element_counts, atomic_data_bulk
    )
    return _xray_result_from_properties(
        _properties_from_totals(
            formula,
            arrays["energy_kev"],
            density_value,
            molecular_weight,
            number_of_electrons,
            arrays["f1_total"],
            arrays["f2_total"],
        )
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
    energies_array = _coerce_energy_array(energies)

    if energies_array.size == 0:
        raise EnergyError("Energies array cannot be empty", valid_range="0.03-30 keV")

    if np.any(energies_array <= 0):
        raise EnergyError("All energies must be positive", valid_range="0.03-30 keV")

    if np.any(energies_array < 0.03) or np.any(energies_array > 30):
        raise EnergyError(
            "Energy values must be in range 0.03-30 keV", valid_range="0.03-30 keV"
        )

    _require_finite_energies(energies_array)

    return energies_array


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

    results: dict[str, XRayResult] = {}
    outcomes = _calculate_materials(
        formulas, densities, energies_array, prevalidated=True
    )
    for formula, density, outcome in zip(formulas, densities, outcomes, strict=True):
        if isinstance(outcome, Exception):
            _logger.warning(
                "Failed to process formula",
                extra={
                    "formula": formula,
                    "error": f"Failed to process formula '{formula}': {outcome}",
                },
            )
            continue
        results[unique_result_key(results, formula, density)] = (
            _xray_result_from_properties(outcome)
        )

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
