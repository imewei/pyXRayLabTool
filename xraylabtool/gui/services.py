"""Helper utilities for the GUI layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xraylabtool.calculators.core import (
    calculate_single_material_properties,
)
from xraylabtool.logging_utils import get_logger

if TYPE_CHECKING:
    from xraylabtool.kk.nearedge import InputType
    from xraylabtool.kk.result import KKResult

logger = get_logger(__name__)

# Tabulated range of the CXRO f1 comparison (keV).
_CXRO_MIN_KEV = 0.03
_CXRO_MAX_KEV = 30.0
_EV_PER_KEV = 1000.0


@dataclass
class EnergyConfig:
    start_kev: float = 8.0
    end_kev: float = 12.0
    points: int = 50
    logspace: bool = False

    def to_array(self) -> np.ndarray:
        start = float(self.start_kev)
        end = float(self.end_kev)
        pts = max(1, int(self.points))
        if pts == 1 or start == end:
            return np.array([start], dtype=float)
        if self.logspace:
            return np.logspace(np.log10(start), np.log10(end), pts)
        return np.linspace(start, end, pts)


def compute_single(formula: str, density: float, energy_cfg: EnergyConfig):  # type: ignore[no-untyped-def]
    energies = energy_cfg.to_array()
    logger.info(
        "Compute single material",
        extra={
            "formula": formula,
            "density": density,
            "points": len(energies),
            "logspace": energy_cfg.logspace,
        },
    )
    return calculate_single_material_properties(formula, energies, density)


def compute_multiple(  # type: ignore[no-untyped-def]
    formulas: Iterable[str],
    densities: Iterable[float],
    energy_cfg: EnergyConfig,
    progress_cb: Callable[[int], None] | None = None,
):
    energies = energy_cfg.to_array()
    formulas_list = list(formulas)
    densities_list = list(densities)
    total = max(len(formulas_list), 1)
    logger.info(
        "Compute multiple materials",
        extra={
            "count": len(formulas_list),
            "points": len(energies),
            "logspace": energy_cfg.logspace,
        },
    )
    results = {}
    for idx, (formula, density) in enumerate(
        zip(formulas_list, densities_list, strict=False)
    ):
        # Key on the formula string, but disambiguate when the same formula
        # appears more than once (e.g. amorphous vs crystalline SiO2 at
        # different densities) so neither result is silently overwritten.
        key = formula
        if key in results:
            base = f"{formula} ({density:g} g/cm³)"
            key, n = base, 2
            while key in results:
                key = f"{base} #{n}"
                n += 1
        results[key] = calculate_single_material_properties(formula, energies, density)
        if progress_cb:
            pct = int(((idx + 1) / total) * 100)
            progress_cb(pct)
    return results


def linear_absorption_cm(inv_length_cm: float | None) -> float | None:
    if inv_length_cm is None:
        return None
    if inv_length_cm == 0:
        return 0.0
    return 1.0 / inv_length_cm


@dataclass(frozen=True)
class KKComputation:
    """Kramers-Kronig result plus the CXRO f1 totals to compare it with.

    Attributes
    ----------
    result : KKResult
        Output of :func:`xraylabtool.kk.kk_calculate_real`.
    cxro_energy_kev : numpy.ndarray, shape (k,)
        Energies (keV) of ``result`` inside 0.03-30 keV; empty if none.
    cxro_f1 : numpy.ndarray, shape (k,)
        CXRO f1 per formula unit (electrons) at ``cxro_energy_kev``.
    """

    result: KKResult
    cxro_energy_kev: NDArray[np.float64]
    cxro_f1: NDArray[np.float64]


def compute_kk(
    formula: str,
    nearedge_path: str | None,
    input_type: str | None,
    energy_unit: str,
    merge_points: tuple[float, float] | None,
    refine_tolerance: float | None,
    energies_kev: ArrayLike | None = None,
) -> KKComputation:
    """Run the KK transform and the CXRO f1 lookup (no Qt, safe in a worker).

    Parameters
    ----------
    formula : str
        Chemical formula, e.g. ``"SiO2"``.
    nearedge_path : str or None
        Near-edge data file; ``None`` uses the tabulated spectrum alone.
    input_type : str or None
        Kind of near-edge data (``"nexafs"``, ...); required with a file.
    energy_unit : {"eV", "keV"}
        Unit of the near-edge file energies and of ``merge_points``.
    merge_points : tuple of float or None
        ``(low, high)`` merge energies in ``energy_unit``.
    refine_tolerance : float or None
        Adaptive-refinement tolerance; ``None`` disables refinement.
    energies_kev : array_like, shape (m,), optional
        Output energies in keV; ``None`` uses the default grid.

    Returns
    -------
    KKComputation
        The transform result and CXRO f1 on its energies within 0.03-30 keV.

    Raises
    ------
    XRayLabToolError
        For invalid formula, energies or near-edge data.
    TypeError
        If near-edge arguments are inconsistent.
    """
    from xraylabtool.kk import kk_calculate_real, load_nearedge

    if energy_unit not in ("eV", "keV"):
        raise ValueError(f"energy_unit must be 'eV' or 'keV', got {energy_unit!r}")
    scale = 1.0 / _EV_PER_KEV if energy_unit == "eV" else 1.0
    merge_kev = (
        None
        if merge_points is None
        else (merge_points[0] * scale, merge_points[1] * scale)
    )
    nearedge = (
        load_nearedge(
            nearedge_path, energy_unit=cast("Literal['eV', 'keV']", energy_unit)
        )
        if nearedge_path
        else None
    )
    logger.info(
        "Compute KK",
        extra={
            "formula": formula,
            "nearedge": nearedge_path,
            "refine": refine_tolerance,
        },
    )
    result = kk_calculate_real(
        formula,
        energies_kev=energies_kev,
        nearedge=nearedge,
        input_type=cast("InputType | None", input_type),
        merge_points_kev=merge_kev,
        refine_tolerance=refine_tolerance,
    )
    in_range = (result.energy_kev >= _CXRO_MIN_KEV) & (
        result.energy_kev <= _CXRO_MAX_KEV
    )
    cxro_energy = result.energy_kev[in_range]
    if cxro_energy.size:
        # f1 is a per-formula-unit sum, so the density (1.0 here) does not enter.
        cxro_f1 = np.asarray(
            calculate_single_material_properties(
                formula, cxro_energy, 1.0
            ).scattering_factor_f1,
            dtype=np.float64,
        )
    else:
        cxro_f1 = np.empty(0, dtype=np.float64)
    return KKComputation(result, cxro_energy, cxro_f1)
