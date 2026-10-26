"""f1 and f2 of a compound equal the count-weighted sum over its elements.

``compound_spectrum`` assembles the compound on the union of the element grids
while each element is evaluated on its own grid, so the two computations share
no segment partition. Agreement therefore guards the union-grid assembly. The
sums differ by rounding of the closed-form kernel (the segment partitions
differ), not by a fixed relative amount: the current NumPy code misses
``rtol=1e-11`` by up to 1.4e-7 absolute on Fe2O3 at low energies.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.kk_rounding import ROUNDING_K, kk_rounding_scale
from xraylabtool.kk import compound_spectrum, kk_calculate_real
from xraylabtool.utils import parse_formula

FORMULAS = ["SiO2", "Fe2O3", "CaCO3"]
N_RANDOM = 200
N_KNOTS = 24
KEV_LOW, KEV_HIGH = 0.0101, 499.0

# Per-energy absolute f1 tolerance = max(3 x the maximum |compound - sum|
# measured on the pre-migration NumPy code, ROUNDING_K x the summed rounding
# scales of the compound and its elements). Measured maxima (electrons), all
# at the lowest energies (10-14 eV); medians 1e-13 to 2e-12:
F1_MEASURED_MAX: dict[str, float] = {
    "SiO2": 1.2e-7,
    "Fe2O3": 4.2e-7,
    "CaCO3": 4.4e-8,
}
# Measured max relative f2 difference 2.0e-13 (CaCO3); tolerance 3.5x.
F2_RTOL = 7e-13


def additivity_energies_kev(seg_ev: np.ndarray) -> np.ndarray:
    """200 log-uniform energies plus a spread of the compound's own knots."""
    rng = np.random.default_rng(2026)
    random = np.exp(rng.uniform(np.log(KEV_LOW), np.log(KEV_HIGH), N_RANDOM))
    inside = seg_ev[(seg_ev > KEV_LOW * 1e3) & (seg_ev < KEV_HIGH * 1e3)]
    knots = inside[:: max(1, inside.size // N_KNOTS)] / 1e3
    return np.concatenate([random, knots])


@pytest.mark.parametrize("formula", FORMULAS)
def test_additivity(formula: str) -> None:
    symbols, counts = parse_formula(formula)
    seg, coeffs = compound_spectrum(symbols, counts)
    energy_kev = additivity_energies_kev(seg)
    energy_ev = energy_kev * 1e3

    whole = kk_calculate_real(formula, energies_kev=energy_kev)
    f1_sum = np.zeros_like(energy_kev)
    f2_sum = np.zeros_like(energy_kev)
    allowance = kk_rounding_scale(energy_ev, seg, coeffs)
    for symbol, count in zip(symbols, counts, strict=True):
        part = kk_calculate_real(symbol, energies_kev=energy_kev)
        f1_sum += count * part.f1
        f2_sum += count * part.f2
        seg_i, coeffs_i = compound_spectrum([symbol], [1.0])
        allowance += count * kk_rounding_scale(energy_ev, seg_i, coeffs_i)

    assert np.isfinite(whole.f1).all()
    assert np.isfinite(whole.f2).all()
    np.testing.assert_allclose(whole.f2, f2_sum, rtol=F2_RTOL, atol=0)
    bound = np.maximum(3.0 * F1_MEASURED_MAX[formula], ROUNDING_K * allowance)
    err = np.abs(whole.f1 - f1_sum)
    worst = int(np.argmax(err - bound))
    assert err[worst] <= bound[worst], (
        f"{formula} E={energy_kev[worst]!r} keV: |{whole.f1[worst]!r} - "
        f"{f1_sum[worst]!r}| = {err[worst]:.3e} exceeds {bound[worst]:.3e}"
    )
