"""Helper utilities for the GUI layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from xraylabtool.calculators.core import (
    calculate_single_material_properties,
)
from xraylabtool.logging_utils import get_logger

logger = get_logger(__name__)


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


def compute_single(formula: str, density: float, energy_cfg: EnergyConfig):
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


def compute_multiple(
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
        results[formula] = calculate_single_material_properties(
            formula, energies, density
        )
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
