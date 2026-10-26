"""Seeded random valid ``kk_calculate_real`` calls: finite, ordered, consistent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xraylabtool.kk import kk_calculate_real, load_nearedge

NEAREDGE = (
    Path(__file__).resolve().parents[2]
    / "characterization"
    / "kk"
    / "nearedge_example.txt"
)
FORMULAS = [
    "H2O",
    "SiO2",
    "Fe2O3",
    "CaCO3",
    "C8H8",
    "Au",
    "Si",
    "Ar",
    "NaCl",
    "Ca5(PO4)3OH",
]
CHUNK_SIZES = [1, 7, 4096]
INPUT_TYPES = ["nexafs", "photoabsorption", "xanes"]  # same absorption scale
N_CALLS = 40


def _random_call(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    kwargs: dict[str, object] = {
        "chunk_size": int(rng.choice(CHUNK_SIZES)),
        "refine_tolerance": [None, 0.05][int(rng.integers(2))],
    }
    formula = FORMULAS[int(rng.integers(len(FORMULAS)))]
    if rng.random() < 0.5:
        m = int(rng.integers(1, 60))
        energies = np.exp(rng.uniform(np.log(0.011), np.log(499.0), m))
        if rng.random() < 0.5:  # repeats are allowed, in any order
            energies = rng.permutation(
                np.concatenate(
                    [energies, rng.choice(energies, int(rng.integers(1, 5)))]
                )
            )
        kwargs["energies_kev"] = energies
    if rng.random() < 0.5:
        kwargs["nearedge"] = load_nearedge(NEAREDGE)
        kwargs["input_type"] = INPUT_TYPES[int(rng.integers(len(INPUT_TYPES)))]
        if rng.random() < 0.5:
            kwargs["merge_points_kev"] = (0.28, 0.32)
    return {"formula": formula, **kwargs}


@pytest.mark.parametrize("seed", range(N_CALLS))
def test_fuzz_valid_calls(seed: int) -> None:
    call = _random_call(seed)
    result = kk_calculate_real(**call)  # type: ignore[arg-type]
    requested = call.get("energies_kev")

    assert np.isfinite(result.energy_kev).all()
    assert np.isfinite(result.f1).all()
    assert np.isfinite(result.f2).all()
    assert result.energy_kev.shape == result.f1.shape == result.f2.shape
    if call["refine_tolerance"] is None:
        assert result.n_refined_points == 0
        if requested is not None:
            np.testing.assert_array_equal(result.energy_kev, requested)
    else:
        assert (np.diff(result.energy_kev) > 0).all()
        if requested is not None:
            n_unique = np.unique(np.asarray(requested) * 1e3).size
            assert result.energy_kev.size == n_unique + result.n_refined_points
