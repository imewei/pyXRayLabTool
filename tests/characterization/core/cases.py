"""Case definitions for the frozen core goldens (Task 0 of the JAX-first plan).

Imported by ``scripts/gen_core_golden.py`` (which freezes the NumPy path into
``golden.npz`` / ``golden_text.json``) and by ``test_core_golden.py`` (which
recomputes every family with the code under test and compares). Keeping the
cases in one module means the generator and the tests cannot drift apart.

Rules the cases follow:

* Every numeric input array comes from :func:`build_inputs` (explicit seed) and
  is stored in the npz under ``inputs|<name>``; the tests read the stored
  copy and a separate test checks that the builder still reproduces it.
* A *family* is a function ``(Recorder, inputs) -> None`` that calls the public
  code and records results. Arrays go to the npz, everything else (error type
  and message, result keys, python return types, warning text) to the JSON file.
* This module never calls ``set_backend``: the generator selects the NumPy path
  itself, the tests compare whatever the default backend is.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
import json
import math
from typing import Any
import warnings

import numpy as np

from xraylabtool.calculators.core import (
    calculate_derived_quantities,
    calculate_multiple_xray_properties,
    calculate_properties_from_factors,
    calculate_scattering_factors,
    calculate_single_material_properties,
    calculate_xray_properties,
    create_scattering_factor_interpolators,
)
from xraylabtool.calculators.derived_quantities import (
    calculate_attenuation_length,
    calculate_critical_angle,
    calculate_scattering_length_density,
    calculate_transmission,
)
from xraylabtool.calculators.scattering_data import (
    AtomicScatteringFactor,
    CrystalStructure,
)
from xraylabtool.constants import (
    ENERGY_TO_WAVELENGTH_FACTOR,
    attenuation_length_cm,
    critical_angle_degrees,
    energy_to_wavelength_angstrom,
    wavelength_angstrom_to_energy,
)
from xraylabtool.data_handling.atomic_cache import get_atomic_data_provider
from xraylabtool.data_handling.batch_processing import (
    BatchConfig,
    _process_chunks,
    calculate_batch_properties,
    process_single_calculation,
)
from xraylabtool.utils import (
    angle_from_q,
    background_subtraction,
    bragg_angle,
    d_spacing_cubic,
    d_spacing_orthorhombic,
    d_spacing_tetragonal,
    energy_to_wavelength,
    find_peaks,
    get_atomic_data,
    get_atomic_number,
    get_atomic_weight,
    normalize_intensity,
    parse_formula,
    q_from_angle,
    smooth_data,
    wavelength_to_energy,
)

SEP = "|"
RTOL = 1e-12
ZERO_ATOL = 1e-300  # only where the golden value is exactly 0
SEED = 20261026

RESULT_SCALARS = (
    "molecular_weight_g_mol",
    "total_electrons",
    "density_g_cm3",
    "electron_density_per_ang3",
)
RESULT_ARRAYS = (
    "energy_kev",
    "wavelength_angstrom",
    "dispersion_delta",
    "absorption_beta",
    "scattering_factor_f1",
    "scattering_factor_f2",
    "critical_angle_degrees",
    "attenuation_length_cm",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
)

SYMBOLS = (  # noqa: SIM905
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu "
    "Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs "
    "Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U"
).split()
assert len(SYMBOLS) == 92

# (formula, density g/cm3): 1 to 6 elements, repeated elements, parentheses,
# hydrates, fractional counts. CuSO4(H2O)5 is the hydrate spelling.
FORMULAS: tuple[tuple[str, float], ...] = (
    ("H2O", 1.0),
    ("SiO2", 2.2),
    ("Al2O3", 3.95),
    ("Fe2O3", 5.24),
    ("CaCO3", 2.71),
    ("Si", 2.33),
    ("Au", 19.32),
    ("C", 2.26),
    ("Ca(OH)2", 2.21),
    ("CuSO4(H2O)5", 2.286),
    ("H2OH2", 1.0),
    ("H", 0.07),
    ("Fe", 7.874),
    ("Cu", 8.96),
    ("Ge", 5.323),
    ("W", 19.25),
    ("Pb", 11.34),
    ("U", 19.05),
    ("NaCl", 2.165),
    ("LiF", 2.64),
    ("GaAs", 5.32),
    ("MgO", 3.58),
    ("TiO2", 4.23),
    ("ZnS", 4.09),
    ("SiC", 3.21),
    ("BN", 2.1),
    ("WC", 15.63),
    ("C8H8", 1.05),
    ("C5H8O2", 1.18),
    ("C2H4", 0.94),
    ("KAl(SO4)2", 2.47),
    ("Ca5(PO4)3OH", 3.16),
    ("NaAlSi3O8", 2.62),
    ("Ca3Al2Si3O12", 3.59),
    ("Mg2Al4Si5O18", 2.6),
    ("Y1Ba2Cu3O7", 6.3),
    ("La0.5Sr0.5MnO3", 6.5),
    ("Ca2Mg5Si8O22(OH)2", 3.0),
    ("KMg3AlSi3O10(OH)2", 2.8),
)
# Subset used where the cost grows with materials x grids.
FORMULAS_SMALL: tuple[tuple[str, float], ...] = (
    ("H2O", 1.0),
    ("SiO2", 2.2),
    ("Fe2O3", 5.24),
    ("Au", 19.32),
    ("CuSO4(H2O)5", 2.286),
    ("KMg3AlSi3O10(OH)2", 2.8),
)
GRID_NAMES = ("three", "lin200", "unsorted200", "edges", "absedges60")


def element_density(z: int) -> float:
    """Synthetic density for the Z sweep (only the scale of the result depends on it)."""
    return 2.0 + 0.15 * z


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def build_inputs() -> dict[str, np.ndarray]:
    """Return every numeric input array of the cases (explicit seed, no globals)."""
    rng = np.random.default_rng(SEED)
    out: dict[str, np.ndarray] = {}

    lin200 = np.linspace(0.03, 30.0, 200)
    out["grid_three"] = np.array([5.0, 10.0, 15.0])
    out["grid_lin200"] = lin200
    out["grid_unsorted200"] = lin200[rng.permutation(200)]
    out["grid_edges"] = np.array([0.03, 0.031, 29.99, 30.0])
    # Absorption edges: Au M (2.206, 2.291, 2.743, 3.148, 3.425), Fe K (7.112),
    # Au L (11.919, 13.734, 14.353) keV, each with a dense ramp around it.
    absedges = np.unique(
        np.concatenate(
            [
                np.linspace(2.15, 3.50, 17),
                [2.206, 2.291, 2.743, 3.148, 3.425],
                np.linspace(7.05, 7.18, 10),
                [7.112],
                np.linspace(11.85, 14.45, 24),
                [11.919, 13.734, 14.353],
            ]
        )
    )
    assert absedges.size == 60, absedges.size
    out["grid_absedges60"] = absedges

    # calculate_properties_from_factors: smooth, edge-like (negative f1 and f2) data.
    ff_e = np.linspace(0.5, 25.0, 120)
    out["ff_energy"] = ff_e
    out["ff_f1"] = (
        14.0 + 3.0 * np.sin(ff_e) - 20.0 * np.exp(-(((ff_e - 7.1) / 0.3) ** 2))
    )
    out["ff_f2"] = 0.4 + 0.3 * np.cos(2.0 * ff_e) - 0.2 * (ff_e > 20.0)
    out["ff_energy_unsorted"] = ff_e[rng.permutation(ff_e.size)]
    out["ff_f1_unsorted"] = rng.normal(10.0, 8.0, ff_e.size)
    out["ff_f2_unsorted"] = np.abs(rng.normal(0.5, 0.3, ff_e.size))
    out["ff_energy_wide"] = np.array([0.001, 0.02, 0.03, 8.0, 30.0, 100.0, 1000.0])
    out["ff_f1_wide"] = np.array([1.0, 5.0, 14.1, 14.2, 14.3, 14.0, 14.0])
    out["ff_f2_wide"] = np.array([0.0, 0.1, 0.5, 0.0, 1e-40, -0.3, 2.0])

    # calculate_derived_quantities: wavelengths (m) and signed delta / beta.
    lam_m = ENERGY_TO_WAVELENGTH_FACTOR / lin200
    out["dq_wavelength_m"] = lam_m
    out["dq_delta"] = 3e-5 * np.sin(np.linspace(0.0, 12.0, 200)) * np.exp(
        -np.linspace(0.0, 6.0, 200)
    ) + np.logspace(-9, -4, 200) * np.where(np.arange(200) % 7 == 0, -1.0, 1.0)
    beta = np.logspace(-12, -3, 200)
    beta[::25] = 0.0
    beta[5::40] = -1e-8
    beta[7] = 1e-31
    out["dq_beta"] = beta
    out["dq_delta_nan"] = np.array([1e-6, np.nan, 2e-6])
    out["dq_beta_inf"] = np.array([1e-8, np.inf, 2e-8])

    # standalone primitives
    out["prim_delta"] = np.array(
        [
            -1e-3,
            -1e-6,
            -1e-30,
            0.0,
            1e-30,
            1e-12,
            1e-8,
            1e-6,
            1e-4,
            1e-2,
            np.nan,
            np.inf,
        ]
    )
    out["prim_wavelength_a"] = np.array([0.1, 0.5, 1.0, 1.2398, 2.5, 12.398, 100.0])
    out["prim_beta"] = np.array(
        [-1e-6, 0.0, 1e-30, 1e-14, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, np.inf, np.nan]
    )
    out["prim_att_length"] = np.array(
        [0.0, 1e-12, 1e-6, 1e-3, 1.0, 1e3, 1e30, np.inf, -1.0]
    )

    # signal helpers
    x = np.linspace(10.0, 80.0, 400)
    peaks = (
        1.0 * np.exp(-(((x - 28.0) / 0.6) ** 2))
        + 0.6 * np.exp(-(((x - 47.0) / 0.8) ** 2))
        + 0.35 * np.exp(-(((x - 56.0) / 0.5) ** 2))
    )
    out["sig_x"] = x
    out["sig_peaks"] = peaks + 0.02 * rng.standard_normal(x.size) + 0.001 * x
    out["sig_smooth_short"] = rng.normal(0.0, 1.0, 17)
    out["sig_background"] = (
        0.002 * (x - 45.0) ** 2 + 5.0 + 2.0 * np.exp(-(((x - 30.0) / 1.5) ** 2))
    )
    out["sig_negative"] = rng.normal(0.5, 1.5, 50)
    out["sig_x_short"] = np.linspace(0.0, 1.0, 17)

    # crystal structure
    out["xtal_positions"] = rng.uniform(0.0, 1.0, (5, 3))
    out["xtal_occupancy"] = np.array([1.0, 0.5, 0.25, 0.9, 1.0])
    out["xtal_q"] = np.linspace(0.0, 6.0, 25)
    return out


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    """Describe the python-level return type (what Task 2 must keep)."""
    if isinstance(value, np.generic):
        return f"numpy.{type(value).__name__}"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"array:{type(value).__module__.split('.')[0]}"
    return type(value).__name__


def exc_record(exc: BaseException) -> dict[str, Any]:
    """Return the JSON record of an exception: type, module and message."""
    return {
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
    }


class Recorder:
    """Collects the arrays and JSON-able facts of one family."""

    def __init__(self, family: str) -> None:
        self.family = family
        self.arrays: dict[str, np.ndarray] = {}
        self.text: dict[str, Any] = {"types": {}, "errors": {}, "facts": {}}

    def put(self, key: str, value: Any, *, lenient: bool = False) -> None:
        """Record ``value`` as an array and remember its python type.

        The type facts distinguish ``array:numpy`` from ``array:jaxlib`` so a
        device array leaking into a result container is caught. ``lenient``
        (raw kernel and helper outputs, which are backend arrays today and
        NumPy after the migration) records the generic ``array``.
        """
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"{self.family}{SEP}{key}: object arrays are not allowed")
        self.arrays[f"{self.family}{SEP}{key}"] = array
        kind = _kind(value)
        self.text["types"][key] = (
            "array" if lenient and kind.startswith("array:") else kind
        )

    def put_many(self, key: str, values: Sequence[Any]) -> None:
        """Record a list of homogeneous scalars as one array (``None`` becomes NaN)."""
        kinds = sorted({_kind(v) for v in values if v is not None})
        array = np.asarray([math.nan if v is None else v for v in values])
        self.arrays[f"{self.family}{SEP}{key}"] = array
        self.text["types"][key] = kinds

    def fact(self, key: str, value: Any) -> None:
        """Record any JSON-able fact (keys of a result dict, warning text, ...)."""
        self.text["facts"][key] = json.loads(json.dumps(value))

    def error(self, key: str, exc: BaseException | None) -> None:
        """Record an exception, or that none was raised."""
        self.text["errors"][key] = (
            {"type": None, "module": None, "message": None}
            if exc is None
            else exc_record(exc)
        )

    def call(self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn``; record the value (as ``key``) or the exception raised."""
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                value = fn(*args, **kwargs)
            except Exception as exc:
                self.error(key, exc)
                return None
        return value

    def expect_error(
        self, key: str, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Call ``fn``; record the exception type and message (or that none was raised)."""
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                self.error(key, exc)
                return
        self.error(key, None)

    def put_result(self, key: str, res: Any) -> None:
        """Record every field of an ``XRayResult``."""
        self.text["facts"][f"{key}{SEP}formula"] = res.formula
        for name in RESULT_SCALARS + RESULT_ARRAYS:
            self.put(f"{key}{SEP}{name}", getattr(res, name))

    def grid(
        self, key: str, fn: Callable[..., Any], arg_rows: Sequence[tuple[Any, ...]]
    ) -> None:
        """Call ``fn`` on every row of arguments; failures become NaN plus an error record."""
        values: list[Any] = []
        for i, row in enumerate(arg_rows):
            value = self.call(f"{key}[{i}]", fn, *row)
            values.append(value)
        self.put_many(key, values)
        self.fact(f"{key}{SEP}args", [[_jsonable(a) for a in row] for row in arg_rows])


def _user_warnings(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    """Texts of the ``UserWarning`` s the code under test raised (ignores library noise)."""
    return [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


FamilyFn = Callable[[Recorder, dict[str, np.ndarray]], None]
FAMILIES: dict[str, FamilyFn] = {}


def family(fn: FamilyFn) -> FamilyFn:
    """Register ``fn`` as a family named after the function (``fam_`` stripped)."""
    FAMILIES[fn.__name__.removeprefix("fam_")] = fn
    return fn


def compute_family(
    name: str, inputs: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one family; return its arrays and its JSON-normalised text record."""
    rec = Recorder(name)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        FAMILIES[name](rec, inputs)
    return rec.arrays, json.loads(json.dumps(rec.text))


# ---------------------------------------------------------------------------
# Single-material calculations
# ---------------------------------------------------------------------------


def _single_grid(grid_name: str, rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    energies = inputs[f"grid_{grid_name}"]
    for formula, density in FORMULAS:
        res = calculate_single_material_properties(formula, energies, density)
        rec.put_result(formula, res)


for _name in GRID_NAMES:
    FAMILIES[f"single_{_name}"] = partial(_single_grid, _name)


@family
def fam_single_scalar(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Scalar energies (python float), including both range edges."""
    del inputs
    for label, energy in (("e8", 8.0), ("lo", 0.03), ("hi", 30.0), ("cu_ka", 8.048)):
        for formula, density in FORMULAS:
            res = calculate_single_material_properties(formula, energy, density)
            rec.put_result(f"{label}{SEP}{formula}", res)


@family
def fam_single_elements(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Every element Z = 1..92 over a wide grid and over the edge-dense grid."""
    for grid_name in ("lin200", "absedges60"):
        energies = inputs[f"grid_{grid_name}"]
        for z, symbol in enumerate(SYMBOLS, start=1):
            res = calculate_single_material_properties(
                symbol, energies, element_density(z)
            )
            rec.put_result(f"{grid_name}{SEP}{symbol}", res)


@family
def fam_single_input_types(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Accepted energy and density spellings; the eV alias; the keyword spelling."""
    del inputs
    cases: dict[str, tuple[Any, Any]] = {
        "int_energy": (10, 2.2),
        "np_float64": (np.float64(8.0), 2.2),
        "zero_d_array": (np.array(8.0), 2.2),
        "list": ([8.0, 9.0], 2.2),
        "tuple": ((8.0, 9.0), 2.2),
        "int_array": (np.array([8, 9]), 2.2),
        "float32_array": (np.array([8.0, 9.0], dtype=np.float32), 2.2),
        "int_density": (8.0, 2),
        "np_density": (8.0, np.float64(2.2)),
        "zero_d_density": (8.0, np.array(2.2)),
        "list_density": (8.0, [2.2]),
    }
    for label, (energy, density) in cases.items():
        res = rec.call(
            label, calculate_single_material_properties, "SiO2", energy, density
        )
        if res is not None:
            rec.put_result(label, res)
    res = calculate_single_material_properties(
        "SiO2", energy_keV=[8.0, 9.0], density=2.2
    )
    rec.put_result("keyword_energy_keV", res)
    res = calculate_single_material_properties(
        "SiO2", density=2.2, energy=[8000.0, 9000.0]
    )
    rec.put_result("alias_energy_eV", res)
    res = calculate_single_material_properties("SiO2", None, 2.2, energy=8000.0)
    rec.put_result("alias_scalar_eV", res)
    rec.fact("result_type", type(res).__name__)


@family
def fam_single_errors(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Exception type and message of every invalid-input case (single material)."""
    del inputs
    fn = calculate_single_material_properties
    ok = np.array([8.0, 9.0])
    cases: dict[str, tuple[Any, ...]] = {
        "formula_empty": ("", 8.0, 2.2),
        "formula_none": (None, 8.0, 2.2),
        "formula_int": (123, 8.0, 2.2),
        "formula_unknown_element": ("Xx2", 8.0, 2.2),
        "formula_unbalanced_open": ("Si(O2", 8.0, 2.2),
        "formula_unbalanced_close": ("SiO2)", 8.0, 2.2),
        "formula_lowercase": ("sio2", 8.0, 2.2),
        "formula_digits_only": ("123", 8.0, 2.2),
        "formula_symbol": ("Si-O", 8.0, 2.2),
        "density_zero": ("SiO2", ok, 0.0),
        "density_negative": ("SiO2", ok, -1.0),
        "density_zero_energy_bad": ("SiO2", -1.0, 0.0),
        "energy_empty_list": ("SiO2", [], 2.2),
        "energy_zero": ("SiO2", 0.0, 2.2),
        "energy_negative": ("SiO2", -8.0, 2.2),
        "energy_below_range": ("SiO2", 0.01, 2.2),
        "energy_above_range": ("SiO2", 31.0, 2.2),
        "energy_in_array_out_of_range": ("SiO2", [8.0, 31.0], 2.2),
        "energy_nan": ("SiO2", np.nan, 2.2),
        "energy_inf": ("SiO2", np.inf, 2.2),
        "energy_nan_in_array": ("SiO2", [8.0, np.nan], 2.2),
        "energy_complex": ("SiO2", 8.0 + 1.0j, 2.2),
        "energy_complex_array": ("SiO2", np.array([8.0 + 0j]), 2.2),
        "energy_2d": ("SiO2", np.ones((2, 2)) * 8.0, 2.2),
        "energy_string": ("SiO2", "abc", 2.2),
        "energy_none": ("SiO2", None, 2.2),
        "density_none": ("SiO2", 8.0, None),
    }
    for label, args in cases.items():
        rec.expect_error(label, fn, *args)
    rec.expect_error("energy_and_keV_missing", fn, "SiO2", None, 2.2)
    rec.expect_error("only_formula", fn, "SiO2")


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

_XP_FORMULAS = ["H2O", "SiO2", "Al2O3", "Fe2O3", "CaCO3", "Si", "Au", "CuSO4(H2O)5"]
_XP_DENSITIES = [1.0, 2.2, 3.95, 5.24, 2.71, 2.33, 19.32, 2.286]


def _record_results(rec: Recorder, label: str, results: dict[str, Any]) -> None:
    rec.fact(f"{label}{SEP}keys", list(results))
    for key, res in results.items():
        rec.put_result(f"{label}{SEP}{key}", res)


@family
def fam_xray_properties(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``calculate_xray_properties``: grids, unsorted input, keys, skip-and-log, errors."""
    for grid_name in GRID_NAMES:
        results = calculate_xray_properties(
            list(_XP_FORMULAS), inputs[f"grid_{grid_name}"], list(_XP_DENSITIES)
        )
        _record_results(rec, f"grid_{grid_name}", results)
    results = calculate_xray_properties(["SiO2", "Au"], 8.0, [2.2, 19.32])
    _record_results(rec, "scalar", results)
    results = calculate_xray_properties(["SiO2", "Au"], [8.0, 9.0], [2, 19])
    _record_results(rec, "list_int_density", results)

    # Keys: repeats with another density get "@density", repeats of that "#n".
    results = calculate_xray_properties(
        ["SiO2", "SiO2", "SiO2", "SiO2", "Si"],
        inputs["grid_three"],
        [2.2, 2.5, 2.2, 2.2, 2.33],
    )
    _record_results(rec, "duplicate_keys", results)

    # Skip-and-log: a failing formula is dropped, the others keep input order.
    results = calculate_xray_properties(
        ["SiO2", "Xx2", "Si", "Si(O", "Au"],
        inputs["grid_three"],
        [2.2, 1.0, 2.33, 1.0, 19.32],
    )
    _record_results(rec, "skip_failed", results)

    errors: dict[str, tuple[Any, ...]] = {
        "formulas_not_list": ("SiO2", 8.0, [2.2]),
        "formulas_empty": ([], 8.0, [2.2]),
        "densities_not_list": (["SiO2"], 8.0, 2.2),
        "densities_empty": (["SiO2"], 8.0, []),
        "length_mismatch": (["SiO2", "Si"], 8.0, [2.2]),
        "formula_empty_string": (["SiO2", ""], 8.0, [2.2, 1.0]),
        "formula_blank": (["  "], 8.0, [2.2]),
        "formula_not_str": ([14], 8.0, [2.2]),
        "density_zero": (["SiO2"], 8.0, [0.0]),
        "density_negative": (["SiO2", "Si"], 8.0, [2.2, -2.3]),
        "density_str": (["SiO2"], 8.0, ["2.2"]),
        "energy_empty": (["SiO2"], [], [2.2]),
        "energy_zero": (["SiO2"], [0.0, 8.0], [2.2]),
        "energy_below_range": (["SiO2"], 0.01, [2.2]),
        "energy_above_range": (["SiO2"], [8.0, 31.0], [2.2]),
        "energy_nan": (["SiO2"], [8.0, np.nan], [2.2]),
        "energy_complex": (["SiO2"], 8.0 + 2.0j, [2.2]),
        "energy_2d": (["SiO2"], np.ones((2, 2)) * 8.0, [2.2]),
        "energy_string": (["SiO2"], "abc", [2.2]),
        "all_formulas_fail": (["Xx2", "Qq"], 8.0, [1.0, 1.0]),
    }
    for label, args in errors.items():
        rec.expect_error(label, calculate_xray_properties, *args)


@family
def fam_multiple_properties(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``calculate_multiple_xray_properties``: dict-of-dict result, keys, ``{}`` when all fail."""
    for grid_name in ("three", "lin200", "absedges60"):
        results = calculate_multiple_xray_properties(
            list(_XP_FORMULAS), inputs[f"grid_{grid_name}"], list(_XP_DENSITIES)
        )
        rec.fact(f"grid_{grid_name}{SEP}keys", list(results))
        for key, props in results.items():
            rec.fact(f"grid_{grid_name}{SEP}{key}{SEP}formula", props["formula"])
            for name, value in props.items():
                if name != "formula":
                    rec.put(f"grid_{grid_name}{SEP}{key}{SEP}{name}", value)
    results = calculate_multiple_xray_properties(
        ["SiO2", "SiO2", "SiO2", "Xx2", "Si"],
        inputs["grid_three"],
        [2.2, 2.5, 2.2, 1.0, 2.33],
    )
    rec.fact(f"mixed{SEP}keys", list(results))
    for key, props in results.items():
        rec.put(f"mixed{SEP}{key}{SEP}dispersion", props["dispersion"])
        rec.put(f"mixed{SEP}{key}{SEP}mass_density", props["mass_density"])
    empty = calculate_multiple_xray_properties(["Xx2", "Qq"], 8.0, [1.0, 1.0])
    rec.fact("all_fail_result", empty)
    rec.fact("all_fail_type", type(empty).__name__)
    rec.expect_error(
        "length_mismatch",
        calculate_multiple_xray_properties,
        ["SiO2", "Si"],
        8.0,
        [2.2],
    )
    rec.expect_error("empty_list", calculate_multiple_xray_properties, [], 8.0, [])


# ---------------------------------------------------------------------------
# Supplied factors, derived quantities, scattering factors, primitives
# ---------------------------------------------------------------------------


@family
def fam_from_factors(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``calculate_properties_from_factors``: signed f1/f2, ranges, orders, errors."""
    e, f1, f2 = inputs["ff_energy"], inputs["ff_f1"], inputs["ff_f2"]
    rec.fact("negative_f1_points", int(np.sum(f1 < 0)))
    for formula, density in FORMULAS_SMALL:
        res = calculate_properties_from_factors(formula, e, density, f1, f2)
        rec.put_result(f"smooth{SEP}{formula}", res)
    res = calculate_properties_from_factors(
        "SiO2",
        inputs["ff_energy_unsorted"],
        2.2,
        inputs["ff_f1_unsorted"],
        inputs["ff_f2_unsorted"],
    )
    rec.put_result("unsorted", res)
    res = calculate_properties_from_factors(
        "Si", inputs["ff_energy_wide"], 2.33, inputs["ff_f1_wide"], inputs["ff_f2_wide"]
    )
    rec.put_result("wide_range_zero_f2", res)
    res = calculate_properties_from_factors(
        "H2OH2", np.array([8.0]), 1.0, np.array([-5.0]), np.array([-0.1])
    )
    rec.put_result("single_point_negative", res)
    res = calculate_properties_from_factors(
        "Au", [8.0, 9.0], 19, [70.0, 71.0], [5.0, 4.0]
    )
    rec.put_result("lists_int_density", res)

    ok_e, ok_f1, ok_f2 = (
        np.array([8.0, 9.0]),
        np.array([1.0, 2.0]),
        np.array([0.1, 0.2]),
    )
    fn = calculate_properties_from_factors
    cases: dict[str, tuple[Any, ...]] = {
        "energy_complex": ("Si", ok_e + 0j, 2.33, ok_f1, ok_f2),
        "f1_complex": ("Si", ok_e, 2.33, ok_f1 + 1j, ok_f2),
        "energy_string": ("Si", "abc", 2.33, ok_f1, ok_f2),
        "f1_ragged": ("Si", ok_e, 2.33, [[1.0], [2.0, 3.0]], ok_f2),
        "energy_2d": ("Si", np.ones((2, 2)), 2.33, ok_f1, ok_f2),
        "f2_2d": ("Si", ok_e, 2.33, ok_f1, np.ones((2, 2))),
        "energy_scalar": ("Si", 8.0, 2.33, ok_f1, ok_f2),
        "energy_empty": ("Si", np.array([]), 2.33, np.array([]), np.array([])),
        "energy_nan": ("Si", np.array([8.0, np.nan]), 2.33, ok_f1, ok_f2),
        "f1_inf": ("Si", ok_e, 2.33, np.array([1.0, np.inf]), ok_f2),
        "f2_nan": ("Si", ok_e, 2.33, ok_f1, np.array([np.nan, 0.1])),
        "length_f1": ("Si", ok_e, 2.33, np.array([1.0]), ok_f2),
        "length_f2": ("Si", ok_e, 2.33, ok_f1, np.array([0.1, 0.2, 0.3])),
        "energy_zero": ("Si", np.array([0.0, 8.0]), 2.33, ok_f1, ok_f2),
        "energy_negative": ("Si", np.array([-1.0, 8.0]), 2.33, ok_f1, ok_f2),
        "density_zero": ("Si", ok_e, 0.0, ok_f1, ok_f2),
        "density_negative": ("Si", ok_e, -2.0, ok_f1, ok_f2),
        "density_nan": ("Si", ok_e, np.nan, ok_f1, ok_f2),
        "density_inf": ("Si", ok_e, np.inf, ok_f1, ok_f2),
        "density_string": ("Si", ok_e, "abc", ok_f1, ok_f2),
        "density_none": ("Si", ok_e, None, ok_f1, ok_f2),
        "formula_empty": ("", ok_e, 2.33, ok_f1, ok_f2),
        "formula_unknown": ("Xx2", ok_e, 2.33, ok_f1, ok_f2),
        "formula_unbalanced": ("Si(O", ok_e, 2.33, ok_f1, ok_f2),
    }
    for label, args in cases.items():
        rec.expect_error(label, fn, *args)


@family
def fam_derived_quantities(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``calculate_derived_quantities``: signed delta, clamped beta, guard message."""
    lam, delta, beta = inputs["dq_wavelength_m"], inputs["dq_delta"], inputs["dq_beta"]
    rec.fact("negative_delta_points", int(np.sum(delta < 0)))
    rec.fact("nonpositive_beta_points", int(np.sum(beta <= 0)))
    materials = {"SiO2": (2.2, 60.0843, 30.0), "Au": (19.32, 196.96657, 79.0)}
    for label, (rho, mw, ne) in materials.items():
        out = calculate_derived_quantities(lam, delta, beta, rho, mw, ne)
        names = (
            "electron_density",
            "critical_angle",
            "attenuation_length",
            "re_sld",
            "im_sld",
        )
        for name, value in zip(names, out, strict=True):
            rec.put(f"{label}{SEP}{name}", value, lenient=True)
    out = calculate_derived_quantities(lam[:1], delta[:1], np.abs(beta[1:2]), 2, 28, 14)
    for name, value in zip(
        (
            "electron_density",
            "critical_angle",
            "attenuation_length",
            "re_sld",
            "im_sld",
        ),
        out,
        strict=True,
    ):
        rec.put(f"one_point_int_args{SEP}{name}", value, lenient=True)
    rec.expect_error(
        "delta_nan",
        calculate_derived_quantities,
        lam[:3],
        inputs["dq_delta_nan"],
        beta[1:4],
        2.2,
        60.0,
        30.0,
    )
    rec.expect_error(
        "beta_inf",
        calculate_derived_quantities,
        lam[:3],
        delta[:3],
        inputs["dq_beta_inf"],
        2.2,
        60.0,
        30.0,
    )


def _element_data(formula: str) -> list[tuple[float, Any, Any]]:
    symbols, counts = parse_formula(formula)
    data = []
    for symbol, count in zip(symbols, counts, strict=True):
        f1_interp, f2_interp = create_scattering_factor_interpolators(symbol)
        data.append((count, f1_interp, f2_interp))
    return data


@family
def fam_scattering_factors(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``calculate_scattering_factors`` from per-element interpolators."""
    from xraylabtool.calculators.core import get_bulk_atomic_data

    for formula, density in FORMULAS_SMALL:
        symbols, counts = parse_formula(formula)
        bulk = get_bulk_atomic_data(tuple(symbols))
        mw = sum(
            c * bulk[s]["atomic_weight"] for s, c in zip(symbols, counts, strict=True)
        )
        data = _element_data(formula)
        for grid_name in ("lin200", "absedges60", "unsorted200", "three"):
            e_kev = inputs[f"grid_{grid_name}"]
            wavelength = ENERGY_TO_WAVELENGTH_FACTOR / e_kev
            out = calculate_scattering_factors(
                e_kev * 1000.0, wavelength, density, mw, data
            )
            for name, value in zip(
                ("dispersion", "absorption", "f1_total", "f2_total"), out, strict=True
            ):
                rec.put(f"{formula}{SEP}{grid_name}{SEP}{name}", value, lenient=True)
    e_kev = inputs["grid_three"]
    out = calculate_scattering_factors(
        e_kev * 1000.0, ENERGY_TO_WAVELENGTH_FACTOR / e_kev, 2.2, 60.0, []
    )
    for name, value in zip(
        ("dispersion", "absorption", "f1_total", "f2_total"), out, strict=True
    ):
        rec.put(f"no_elements{SEP}{name}", value, lenient=True)
    # Element-level interpolators themselves (eV in, tabulated f1/f2 out).
    ev = inputs["grid_absedges60"] * 1000.0
    for symbol in ("H", "Fe", "Au", "U"):
        f1_interp, f2_interp = create_scattering_factor_interpolators(symbol)
        rec.put(f"interp{SEP}{symbol}{SEP}f1", f1_interp(ev), lenient=True)
        rec.put(f"interp{SEP}{symbol}{SEP}f2", f2_interp(ev), lenient=True)


@family
def fam_primitives(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Standalone critical angle, attenuation length, transmission and SLD helpers."""
    delta, wl, beta = (
        inputs["prim_delta"],
        inputs["prim_wavelength_a"],
        inputs["prim_beta"],
    )
    rec.put("critical_angle", calculate_critical_angle(delta), lenient=True)
    rec.put(
        "critical_angle_negative_one",
        calculate_critical_angle(np.array([-1e-6])),
        lenient=True,
    )
    rec.put(
        "critical_angle_zero", calculate_critical_angle(np.array([0.0])), lenient=True
    )
    for i, w in enumerate(wl):
        rec.put(
            f"attenuation_length{SEP}wl{i}",
            calculate_attenuation_length(np.full(beta.size, w), beta),
            lenient=True,
        )
    rec.put(
        "attenuation_length_beta_zero",
        calculate_attenuation_length(np.array([1.0]), np.array([0.0])),
        lenient=True,
    )
    rec.put(
        "attenuation_length_beta_negative",
        calculate_attenuation_length(np.array([1.0]), np.array([-1e-8])),
        lenient=True,
    )
    att = inputs["prim_att_length"]
    for thickness in (0.0, 1e-4, 1.0, 10.0):
        rec.put(
            f"transmission{SEP}t{thickness:g}",
            calculate_transmission(thickness, att),
            lenient=True,
        )
    sld = calculate_scattering_length_density(
        delta[:10], np.abs(beta[:10]), np.full(10, 1.5406)
    )
    rec.put("sld_real", sld[0], lenient=True)
    rec.put("sld_imag", sld[1], lenient=True)


@family
def fam_constants_helpers(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``constants`` conversion helpers, including their ``ValueError`` cases."""
    del inputs
    e = [
        0.03,
        0.1,
        1.0,
        5.0,
        8.048,
        10.0,
        12.398,
        29.99,
        30.0,
        100.0,
        0.0,
        -1.0,
        np.float64(8.0),
        7,
    ]
    rec.grid(
        "energy_to_wavelength_angstrom",
        energy_to_wavelength_angstrom,
        [(v,) for v in e],
    )
    w = [0.1, 0.4133, 1.0, 1.5406, 2.0, 12.398, 413.3, 0.0, -1.0, np.float64(1.0), 2]
    rec.grid(
        "wavelength_angstrom_to_energy",
        wavelength_angstrom_to_energy,
        [(v,) for v in w],
    )
    d = [1e-12, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.5, 0.0, -1e-6, np.float64(1e-6)]
    rec.grid("critical_angle_degrees", critical_angle_degrees, [(v,) for v in d])
    rows = [
        (wl, b)
        for wl in (0.5, 1.24, 12.398)
        for b in (1e-12, 1e-9, 1e-7, 1e-4, 0.0, -1e-7)
    ] + [(0.0, 1e-7), (-1.0, 1e-7), (np.float64(1.24), np.float64(1e-7))]
    rec.grid("attenuation_length_cm", attenuation_length_cm, rows)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


@family
def fam_utils_units(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Wavelength and energy conversions in every unit, python scalars and numpy scalars."""
    del inputs
    wl = [0.05, 0.1, 0.5, 1.0, 1.5406, 2.0, 12.398, 100.0, 1e-3, 3, np.float64(1.0)]
    energy = [
        0.03,
        0.1,
        1.0,
        5.0,
        8.048,
        12.398,
        29.99,
        30.0,
        100.0,
        2,
        np.float64(10.0),
    ]
    for units in ("angstrom", "nm", "m"):
        scale = {"angstrom": 1.0, "nm": 0.1, "m": 1e-10}[units]
        rec.grid(
            f"wavelength_to_energy{SEP}{units}",
            wavelength_to_energy,
            [(v * scale, units) for v in wl],
        )
        rec.grid(
            f"energy_to_wavelength{SEP}{units}",
            energy_to_wavelength,
            [(v, units) for v in energy],
        )
    rec.grid(
        f"wavelength_to_energy{SEP}default",
        wavelength_to_energy,
        [(1.5406,), (np.float64(0.5),)],
    )
    rec.grid(
        f"energy_to_wavelength{SEP}default",
        energy_to_wavelength,
        [(8.048,), (np.float64(10.0),)],
    )
    bad = [
        (0.0,),
        (-1.0,),
        (1.0, "pm"),
        (1.0, "Angstrom"),
        (np.array([1.0, 2.0]),),
        (np.array(1.0),),
        (np.array([-1.0, 1.0]),),
    ]
    rec.grid(f"wavelength_to_energy{SEP}invalid", wavelength_to_energy, bad)
    rec.grid(f"energy_to_wavelength{SEP}invalid", energy_to_wavelength, bad)


@family
def fam_utils_crystal(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Bragg angle, d-spacings, q conversions with valid and invalid arguments."""
    del inputs
    rows = [
        (d, wl, order)
        for d in (0.8, 1.0, 1.5, 3.1356, 5.0, 10.0)
        for wl in (0.5, 1.5406, 2.0, 12.398)
        for order in (1, 2, 3)
    ] + [
        (1.0, 2.0, 1),
        (1.0, 2.0000001, 1),
        (0.0, 1.0, 1),
        (-1.0, 1.0, 1),
        (1.0, 0.0, 1),
        (1.0, -1.0, 1),
        (1.0, 1.0, 0),
        (1.0, 1.0, -1),
        (np.float64(3.1356), np.float64(1.5406), 1),
    ]
    rec.grid("bragg_angle", bragg_angle, rows)
    rec.grid(f"bragg_angle{SEP}default_order", bragg_angle, [(3.1356, 1.5406)])

    miller = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (2, 2, 0),
        (3, 1, 1),
        (-1, 2, -3),
        (0, 0, 2),
        (0, 0, 0),
        (4, 4, 4),
    ]
    rec.grid(
        "d_spacing_cubic",
        d_spacing_cubic,
        [(*m, a) for m in miller for a in (3.5, 5.431, 0.5)]
        + [(1, 1, 1, 0.0), (1, 1, 1, -5.431), (1, 1, 1, np.float64(5.431))],
    )
    rec.grid(
        "d_spacing_tetragonal",
        d_spacing_tetragonal,
        [(*m, a, c) for m in miller for a, c in ((3.5, 5.0), (4.0, 2.5), (5.0, 5.0))]
        + [(1, 0, 1, 0.0, 1.0), (1, 0, 1, 1.0, -1.0)],
    )
    rec.grid(
        "d_spacing_orthorhombic",
        d_spacing_orthorhombic,
        [
            (*m, a, b, c)
            for m in miller
            for a, b, c in ((3.5, 4.5, 5.5), (5.0, 5.0, 5.0), (2.0, 7.0, 3.0))
        ]
        + [(1, 1, 1, 0.0, 1.0, 1.0), (1, 1, 1, 1.0, 1.0, -1.0)],
    )
    two_theta = [
        0.0,
        1.0,
        10.0,
        28.44,
        45.0,
        90.0,
        120.0,
        179.9,
        180.0,
        360.0,
        -10.0,
        np.float64(30.0),
    ]
    rec.grid(
        "q_from_angle",
        q_from_angle,
        [(t, wl) for t in two_theta for wl in (0.5, 1.5406, 2.0)],
    )
    q_values = [
        0.0,
        0.1,
        1.0,
        2.0,
        4.0,
        6.0,
        8.0,
        8.1,
        8.168,
        8.2,
        12.0,
        -1.0,
        np.float64(2.0),
    ]
    rec.grid(
        "angle_from_q",
        angle_from_q,
        [(q, wl) for q in q_values for wl in (0.5, 1.5406, 2.0)],
    )


@family
def fam_utils_signal(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Smoothing, peak finding, background subtraction, normalisation."""
    x, y = inputs["sig_x"], inputs["sig_peaks"]
    for window in (1, 2, 3, 4, 5, 9, 10, 11, 25):
        rec.put(f"smooth_data{SEP}w{window}", smooth_data(x, y, window))
    for window in (5, 17, 18, 100):
        rec.put(
            f"smooth_data{SEP}short_w{window}",
            smooth_data(inputs["sig_x_short"], inputs["sig_smooth_short"], window),
        )
    rec.put(f"smooth_data{SEP}default", smooth_data(x, y))
    rec.put(f"smooth_data{SEP}list_input", smooth_data(list(x), list(y), 5))
    for window in (0, -3):
        rec.expect_error(f"smooth_data_window_{window}", smooth_data, x, y, window)

    for prominence, distance in (
        (0.1, 10),
        (0.05, 5),
        (0.3, 20),
        (0.01, 1),
        (0.5, 50),
        (0.001, 3),
    ):
        peaks, props = find_peaks(x, y, prominence, distance)
        tag = f"find_peaks{SEP}p{prominence}{SEP}d{distance}"
        rec.put(f"{tag}{SEP}indices", peaks)
        rec.fact(f"{tag}{SEP}property_names", sorted(props))
        for name in sorted(props):
            rec.put(f"{tag}{SEP}{name}", props[name])
    peaks, props = find_peaks(x, y)
    rec.put(f"find_peaks{SEP}default{SEP}indices", peaks)
    peaks, props = find_peaks(list(x), list(y))
    rec.put(f"find_peaks{SEP}list_input{SEP}indices", peaks)
    peaks, props = find_peaks(x, np.zeros_like(y))
    rec.put(f"find_peaks{SEP}flat{SEP}indices", peaks)

    bg = inputs["sig_background"]
    for method in ("linear", "polynomial"):
        rec.put(
            f"background_subtraction{SEP}{method}",
            background_subtraction(x, bg, method),
        )
    rec.put(f"background_subtraction{SEP}default", background_subtraction(x, bg))
    rec.put(
        f"background_subtraction{SEP}list_input",
        background_subtraction(list(x), list(bg), "linear"),
    )
    rec.expect_error(
        "background_subtraction_bad_method", background_subtraction, x, bg, "cubic"
    )

    for name, signal in (
        ("peaks", y),
        ("negative", inputs["sig_negative"]),
        ("bg", bg),
    ):
        for method in ("max", "area", "standard"):
            rec.put(
                f"normalize_intensity{SEP}{name}{SEP}{method}",
                normalize_intensity(signal, method),
            )
    rec.put(f"normalize_intensity{SEP}default", normalize_intensity(y))
    rec.put(f"normalize_intensity{SEP}list_input", normalize_intensity(list(y), "max"))
    rec.put(
        f"normalize_intensity{SEP}zeros_max", normalize_intensity(np.zeros(5), "max")
    )
    rec.put(
        f"normalize_intensity{SEP}constant_standard",
        normalize_intensity(np.full(5, 2.0), "standard"),
    )
    rec.expect_error("normalize_intensity_bad_method", normalize_intensity, y, "sum")


@family
def fam_utils_formula(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Formula parser and atomic data lookups (they decide molecular weight and electrons)."""
    del inputs
    spellings = [f for f, _ in FORMULAS] + [
        "SiO2Si",
        "H2O2H",
        "(CH2)8",
        "Ca(OH)2(H2O)",
        "Mg3(PO4)2(H2O)8",
        "((SiO2)2)3",
        "C60",
        "Si1O2",
        "H0.5O",
    ]
    for formula in spellings:
        parsed = rec.call(f"parse_failed{SEP}{formula}", parse_formula, formula)
        if parsed is None:
            continue
        symbols, counts = parsed
        rec.fact(f"parse{SEP}{formula}{SEP}symbols", symbols)
        rec.put(f"parse{SEP}{formula}{SEP}counts", np.asarray(counts))
    for formula in (
        "",
        "Xx2",
        "Si(O",
        "SiO2)",
        "sio2",
        "123",
        "Si-O",
        "()",
        "H2O)(",
        " ",
    ):
        rec.expect_error(f"parse_error{SEP}{formula!r}", parse_formula, formula)
    rec.put("atomic_number", [get_atomic_number(s) for s in SYMBOLS])
    rec.put("atomic_weight", [get_atomic_weight(s) for s in SYMBOLS])
    rec.fact("atomic_number_type", type(get_atomic_number("Si")).__name__)
    rec.fact("atomic_weight_type", type(get_atomic_weight("Si")).__name__)
    for symbol in ("H", "Si", "Au", "U", "He"):
        data = get_atomic_data(symbol)
        rec.fact(
            f"atomic_data{SEP}{symbol}",
            {k: data[k] for k in ("symbol", "atomic_number", "name")},
        )
        rec.put(f"atomic_data{SEP}{symbol}{SEP}weight", data["atomic_weight"])
    for label, symbol in (("unknown", "Xx"), ("empty", ""), ("lower", "si")):
        rec.expect_error(f"atomic_number_{label}", get_atomic_number, symbol)
        rec.expect_error(f"atomic_weight_{label}", get_atomic_weight, symbol)
        rec.expect_error(f"atomic_data_{label}", get_atomic_data, symbol)


# ---------------------------------------------------------------------------
# Data handling and scattering_data
# ---------------------------------------------------------------------------


@family
def fam_atomic_provider(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``get_scattering_factors`` (complex128) for every element, plus out-of-range energies."""
    provider = get_atomic_data_provider()
    for grid_name in ("lin200", "absedges60"):
        for symbol in SYMBOLS:
            out = provider.get_scattering_factors(symbol, inputs[f"grid_{grid_name}"])
            rec.put(f"{grid_name}{SEP}{symbol}", out)
    outside = np.array([0.001, 0.01, 0.0293, 0.03, 30.0, 30.01, 31.0])
    for symbol in ("H", "Fe", "U"):
        rec.put(
            f"outside_table{SEP}{symbol}",
            provider.get_scattering_factors(symbol, outside),
        )
    rec.put(
        f"unsorted{SEP}Au",
        provider.get_scattering_factors("Au", inputs["grid_unsorted200"]),
    )
    rec.put(f"list_input{SEP}Si", provider.get_scattering_factors("Si", [8.0, 9.0]))
    rec.expect_error(
        "unknown_element", provider.get_scattering_factors, "Xx", np.array([8.0])
    )


@family
def fam_crystal_structure(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """``CrystalStructure.calculate_structure_factor`` and ``AtomicScatteringFactor``."""
    lattice = (4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
    structures: dict[str, list[tuple[str, tuple[float, float, float], float]]] = {
        "simple_cubic": [("Po", (0.0, 0.0, 0.0), 1.0)],
        "bcc": [("Fe", (0.0, 0.0, 0.0), 1.0), ("Fe", (0.5, 0.5, 0.5), 1.0)],
        "fcc": [
            ("Cu", (0.0, 0.0, 0.0), 1.0),
            ("Cu", (0.0, 0.5, 0.5), 1.0),
            ("Cu", (0.5, 0.0, 0.5), 1.0),
            ("Cu", (0.5, 0.5, 0.0), 1.0),
        ],
        "rocksalt_partial": [
            ("Na", (0.0, 0.0, 0.0), 1.0),
            ("Cl", (0.5, 0.0, 0.0), 0.5),
            ("Na", (0.0, 0.5, 0.5), 1.0),
            ("Cl", (0.5, 0.5, 0.5), 0.5),
        ],
        "random_five": [
            (
                "Si",
                tuple(float(v) for v in inputs["xtal_positions"][i]),
                float(inputs["xtal_occupancy"][i]),
            )
            for i in range(5)
        ],
        "empty": [],
    }
    hkl_list = [
        (h, k, ell) for h in range(-3, 4) for k in range(-3, 4) for ell in range(-3, 4)
    ]
    for label, atoms in structures.items():
        crystal = CrystalStructure(lattice)
        for element, position, occupancy in atoms:
            crystal.add_atom(element, position, occupancy)
        values = [crystal.calculate_structure_factor(hkl) for hkl in hkl_list]
        rec.put_many(f"structure_factor{SEP}{label}", values)
    asf = AtomicScatteringFactor()
    q = inputs["xtal_q"]
    rec.put("asf_placeholder", asf.get_scattering_factor("Si", q))
    rec.put("asf_placeholder_unit", asf.get_scattering_factor("H", np.array([0.0])))
    rec.put("asf_placeholder_empty", asf.get_scattering_factor("H", np.array([])))


@family
def fam_chunk_contracts(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Batch contracts: ``None`` plus warning, ``#N`` keys, key order, validation.

    Every chunk holds fewer than 8 items (thread pool) so no fork happens once
    JAX is loaded.
    """
    energies = inputs["grid_three"]
    # process_single_calculation: success, and failure -> warning text + None
    formula, res = process_single_calculation("SiO2", energies, 2.2)
    rec.fact("single_ok_formula", formula)
    rec.put_result("single_ok", res)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        formula, res = process_single_calculation("Xx999", energies, 1.0)
    rec.fact("single_fail", [formula, res is None, _user_warnings(caught)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        formula, res = process_single_calculation("SiO2", energies, -1.0)
    rec.fact("single_fail_density", [formula, res is None, _user_warnings(caught)])

    # _process_chunks: key = formula@density:.3f, "#N" for repeats, bare formula for failures
    data = [
        ("SiO2", energies, 2.2),
        ("SiO2", energies, 2.2001),
        ("SiO2", energies, 2.2),
        ("Xx999", energies, 1.0),
        ("Xx999", energies, 1.0),
        ("Al2O3", energies, 3.95),
        ("SiO2", energies, 2.5),
    ]
    config = BatchConfig(max_workers=2, chunk_size=3, enable_progress=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = _process_chunks(data, config, None)
    rec.fact("chunks_keys", list(results))
    rec.fact("chunks_none_keys", [k for k, v in results.items() if v is None])
    rec.fact("chunks_warnings", sorted(_user_warnings(caught)))
    for key, res in results.items():
        if res is not None:
            rec.put_result(f"chunks{SEP}{key}", res)

    # calculate_batch_properties end to end
    formulas = ["SiO2", "Al2O3", "Si", "SiO2", "Xx999", "H2O", "Au"]
    densities = [2.2, 3.95, 2.33, 2.5, 1.0, 1.0, 19.32]
    for grid_name in ("lin200", "absedges60"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = calculate_batch_properties(
                formulas,
                inputs[f"grid_{grid_name}"],
                densities,
                BatchConfig(max_workers=2, chunk_size=7, enable_progress=False),
            )
        rec.fact(f"batch_{grid_name}{SEP}keys", list(results))
        rec.fact(f"batch_{grid_name}{SEP}warnings", sorted(_user_warnings(caught)))
        for key, res in results.items():
            if res is not None:
                rec.put_result(f"batch_{grid_name}{SEP}{key}", res)

    cfg = BatchConfig(max_workers=2, chunk_size=3, enable_progress=False)
    errors: dict[str, tuple[Any, ...]] = {
        "length_mismatch": (["SiO2", "Si"], energies, [2.2], cfg),
        "empty": ([], energies, [], cfg),
        "energy_zero": (["SiO2"], [0.0, 8.0], [2.2], cfg),
        "energy_below": (["SiO2"], 0.01, [2.2], cfg),
        "energy_above": (["SiO2"], [8.0, 31.0], [2.2], cfg),
        "energy_nan": (["SiO2"], [8.0, np.nan], [2.2], cfg),
        "energy_complex": (["SiO2"], 8.0 + 1.0j, [2.2], cfg),
        "energy_2d": (["SiO2"], np.ones((2, 2)) * 8.0, [2.2], cfg),
    }
    for label, args in errors.items():
        rec.expect_error(label, calculate_batch_properties, *args)


def _full_list() -> tuple[list[str], list[float]]:
    """Every golden formula plus a Z sample of single elements (K from 1 to 6)."""
    pairs = list(FORMULAS) + [
        (SYMBOLS[z - 1], element_density(z)) for z in range(1, 93, 5)
    ]
    return [f for f, _ in pairs], [d for _, d in pairs]


@family
def fam_full_list(rec: Recorder, inputs: dict[str, np.ndarray]) -> None:
    """Whole formula list over the collections, so mixed-K padding is covered later.

    ``calculate_batch_properties`` runs in chunks of 7 (thread pool, no fork).
    """
    formulas, densities = _full_list()
    energies = inputs["grid_lin200"]
    results = calculate_xray_properties(formulas, energies, densities)
    _record_results(rec, "xray_properties", results)
    results = calculate_xray_properties(formulas, inputs["grid_unsorted200"], densities)
    _record_results(rec, "xray_properties_unsorted", results)
    props = calculate_multiple_xray_properties(formulas, energies, densities)
    rec.fact(f"multiple{SEP}keys", list(props))
    for key, entry in props.items():
        for name, value in entry.items():
            if name == "formula":
                rec.fact(f"multiple{SEP}{key}{SEP}formula", value)
            else:
                rec.put(f"multiple{SEP}{key}{SEP}{name}", value)
    config = BatchConfig(max_workers=2, chunk_size=7, enable_progress=False)
    batch = calculate_batch_properties(formulas, energies, densities, config)
    rec.fact(f"batch_properties{SEP}keys", list(batch))
    for key, res in batch.items():
        if res is not None:
            rec.put_result(f"batch_properties{SEP}{key}", res)
