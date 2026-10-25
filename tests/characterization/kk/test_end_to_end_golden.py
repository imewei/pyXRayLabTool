"""End-to-end ``kk_calculate_real`` against upstream kkcalc and CXRO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
import xraylabtool
from xraylabtool.backend.array_ops import JaxBackend, get_backend, set_backend
from xraylabtool.kk import kk_calculate_real
from xraylabtool.kk.database import load_asf
from xraylabtool.kk.spectrum import compound_spectrum
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction
from xraylabtool.utils import parse_formula

GOLDEN = Path(__file__).resolve().parent / "golden"
NFF_DIR = Path(xraylabtool.__file__).parent / "data" / "AtomicScatteringFactor"
CASES = [("si", "Si"), ("au", "Au"), ("sio2", "SiO2")]

# Tolerance rtol·|f1| + K·rounding scale (tests/fixtures/kk_rounding.py) for
# energies that reach the kernel bit-identically to upstream's.
# Measured on NumPy: worst relative error 3.9e-12 (Si), i.e. inside the rtol
# term. On JAX the excess over the rtol term is at most 2.0x the rounding
# scale (SiO2), 1.9x (Au), 0.65x (Si); the allowance is 32x.
RTOL = {"numpy": 1e-10, "jax": 1e-8}
# keV -> eV (x1000) of eV/1000 differs from the golden energy by 1 ulp for 12
# of 760 (Si), 6 of 510 (Au), 17 of 1202 (SiO2) knots. The kernel then
# evaluates one ulp beside the knot instead of on it; f2 is continuous there
# so f1 agrees, but only to the knot-continuity level (Task 3: the
# ln|x - E| terms one ulp from a knot are amplified ~30x). Measured worst
# absolute difference: NumPy Si 9.8e-9, SiO2 1.5e-10, Au 2.0e-5; JAX Si
# 1.2e-9, SiO2 1.5e-10, Au 2.0e-5. Bounds: Si 3x, SiO2 7x, Au 5x (the Task 3
# JAX fusion deviation reaches 6e-5).
KNOT_ULP_ATOL = {"Si": 3e-8, "SiO2": 1e-9, "Au": 1e-4}


@pytest.fixture(params=["numpy", "jax"])
def backend_name(request: pytest.FixtureRequest):
    """Run the test under each backend, restoring the original afterwards."""
    if request.param == "jax":
        pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend(request.param)
    yield request.param
    set_backend(original)


@pytest.mark.parametrize(("name", "formula"), CASES)
def test_end_to_end_matches_upstream_golden(
    name: str, formula: str, backend_name: str
) -> None:
    with np.load(GOLDEN / f"{name}.npz") as g:
        energy_ev, f1, f2 = g["energy_ev"], g["f1"], g["f2"]
        rel_corr = float(g["rel_corr"])
    res = kk_calculate_real(formula, energies_kev=energy_ev / 1000.0)
    seg, coeffs = compound_spectrum(*parse_formula(formula))
    assert res.backend == backend_name
    assert np.isfinite(res.f1).all()
    assert res.relativistic_correction == rel_corr

    # f2 is plain arithmetic on the same coefficients (measured 4.3e-16 rel).
    np.testing.assert_allclose(res.f2, f2, rtol=1e-12)

    exact = energy_ev / 1000.0 * 1000.0 == energy_ev
    assert exact.mean() > 0.97
    assert_kk_close(
        res.f1[exact],
        f1[exact],
        energy_ev[exact],
        seg,
        coeffs,
        rtol=RTOL[backend_name],
    )
    assert np.abs(res.f1[~exact] - f1[~exact]).max() < KNOT_ULP_ATOL[formula]


@pytest.mark.parametrize(("name", "formula"), CASES)
def test_default_grid_matches_upstream_golden_bit_for_bit_inputs(
    name: str, formula: str, backend_name: str
) -> None:
    """The default grid reaches the kernel in eV without a round trip."""
    with np.load(GOLDEN / f"{name}.npz") as g:
        keep = (g["energy_ev"] >= 30.0) & (g["energy_ev"] <= 30000.0)
        energy_ev, f1 = g["energy_ev"][keep], g["f1"][keep]
    res = kk_calculate_real(formula)
    seg, coeffs = compound_spectrum(*parse_formula(formula))
    assert res.energy_kev.size == energy_ev.size
    assert_kk_close(res.f1, f1, energy_ev, seg, coeffs, rtol=RTOL[backend_name])


# CXRO comparison -----------------------------------------------------------
# Elements whose tabulated Henke f2 differs from the repo's CXRO .nff f2
# (tests/unit/kk/test_database.py::test_nff_divergent_elements).
DIVERGENT = {"Sc", "Mo", "Ru", "Be", "Ti", "Si", "La", "Gd", "W"}


def _cxro_deviation(symbol: str) -> np.ndarray:
    """``|f1_kk - f1_cxro| / max(|f1_cxro|, 1)`` at the .nff energies, 30 eV-30 keV."""
    table = np.genfromtxt(
        NFF_DIR / f"{symbol.lower()}.nff", delimiter=",", skip_header=1
    )
    energy_ev, f1_cxro = table[:, 0], table[:, 1]
    window = (energy_ev >= 30.0) & (energy_ev <= 30000.0)
    res = kk_calculate_real(symbol, energies_kev=energy_ev[window] / 1000.0)
    assert np.isfinite(res.f1).all()
    return np.abs(res.f1 - f1_cxro[window]) / np.maximum(np.abs(f1_cxro[window]), 1.0)


def test_tabulated_kk_f1_close_to_cxro_f1() -> None:
    """KK of the tabulated f2 reproduces the tabulated CXRO f1 where f2 agrees.

    The published CXRO f1 values come from a different, discrete sampling of
    the piecewise f2 (the paper notes small deviations are expected), and the
    Henke edge segments are very narrow, so energies right at an absorption
    edge deviate by 10 % or much more (up to 5.4 in the metric below, 30 in
    absolute f1 for U) while the bulk of the spectrum agrees to ~1e-4. The
    bounds therefore constrain the distribution, not the maximum. Measured over
    the 83 elements outside the nine divergent ones (~36 000 points):
    pooled median 1.9e-4, pooled 95th percentile 8.9e-3, worst per-element
    median 2.8e-3 (Lu); bounds are ~3x those.
    """
    symbols = [s for s in load_asf().symbols if s not in DIVERGENT]
    assert len(symbols) == 83
    deviations = {s: _cxro_deviation(s) for s in symbols}
    pooled = np.concatenate(list(deviations.values()))
    assert np.median(pooled) < 6e-4
    assert np.quantile(pooled, 0.95) < 3e-2
    worst = max(symbols, key=lambda s: np.median(deviations[s]))
    assert np.median(deviations[worst]) < 1e-2, worst


# Truncation at 30 keV (paper Figs. 3-5) ------------------------------------
@pytest.mark.parametrize(
    ("symbol", "check"),
    [
        ("C", lambda d: np.abs(d).max() < 1e-3),
        ("Fe", lambda d: (np.abs(d) > 0.05).all() and (np.abs(d) < 0.5).all()),
        ("Pb", lambda d: (-d > 1.0).all()),
    ],
)
def test_30kev_truncation_matches_paper_trend(symbol: str, check) -> None:
    """Dropping f2 above 30 keV lowers f1, more for heavier elements.

    Measured ``f1_full - f1_trunc`` at E = 0.1, 1, 10, 29 keV (eV in the
    code): C -1.2e-4, -1.2e-4, -1.3e-4, -3.8e-4; Fe -0.097, -0.097, -0.103,
    -0.283; Pb -2.50, -2.50, -2.59, -5.23. The bounds are the review-time
    ranges (C below 1e-3, Fe between 0.05 and 0.5, Pb beyond 1 in magnitude);
    the trend (always negative, larger next to the cut) is asserted for each.
    """
    z = load_asf().symbols.index(symbol) + 1
    seg, coeffs = compound_spectrum([symbol], [1.0])
    rel_corr = relativistic_correction([z], [1.0])
    energy_ev = np.array([100.0, 1000.0, 10000.0, 29000.0])
    full = kk_real_spectrum(energy_ev, seg, coeffs, rel_corr)

    seg_t = np.append(seg[seg < 30000.0], 30000.0)
    trunc = kk_real_spectrum(energy_ev, seg_t, coeffs[: seg_t.size - 1], rel_corr)

    diff = full - trunc
    assert np.isfinite(diff).all()
    assert (diff < 0).all()
    assert check(diff)
    # The deficit is largest next to the 30 keV cut.
    assert diff[-1] < diff[0]
