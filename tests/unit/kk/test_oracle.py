"""The KK kernel against an independent longdouble quadrature of Henke's f1.

``tests.fixtures.kk_oracle.f1_reference`` integrates the tabulated f2 directly
(removable-singularity subtraction, Gauss-Legendre in ln x split at every
polynomial boundary and at the singular point); it shares no code with the
closed-form sums of ``kk_real_spectrum``. Only continuous interior knots are
compared: jump knots and the range end points are upstream conventions checked
against the locked goldens (``tests/characterization/kk``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.kk_oracle import f1_reference
from tests.fixtures.kk_rounding import ROUNDING_K, kk_rounding_scale
from xraylabtool.kk import (
    compound_spectrum,
    kk_real_spectrum,
    load_asf,
    relativistic_correction,
)
from xraylabtool.utils import parse_formula

FORMULAS = ["H", "O", "Si", "SiO2", "Fe2O3", "Au", "Pb", "U", "Mo"]
N_RANDOM = 120
LOW_EV = (10.5, 11.0, 12.0, 14.0, 18.0, 25.0)
EDGE_OFFSET = 1e-6
CONTINUITY_RTOL = 1e-9

# Per-energy absolute tolerance = max(3 x the maximum error measured on the
# pre-migration NumPy kernel, ROUNDING_K x the cancellation scale at that
# energy). Measured maximum |f1 - oracle| per compound (electrons), all at
# 10-13 eV except H (3.5e5 eV); the medians are 1e-13 to 4e-10. The errors are
# the double-precision rounding of the closed-form sum: a longdouble replica of
# the same formula agrees with the oracle to 1e-14 (H) .. 2e-8 (Au at 11 eV):
MEASURED_MAX: dict[str, float] = {
    "H": 7.3e-11,
    "O": 4.8e-9,
    "Si": 4.2e-8,
    "SiO2": 7.0e-8,
    "Fe2O3": 2.2e-7,
    "Au": 5.9e-5,
    "Pb": 1.0e-4,
    "U": 2.8e-4,
    "Mo": 3.0e-6,
}
MEDIAN_LIMIT = 1e-8


def _spectrum(formula: str) -> tuple[np.ndarray, np.ndarray, float]:
    symbols, counts = parse_formula(formula)
    seg, coeffs = compound_spectrum(symbols, counts)
    db = load_asf()
    rel = relativistic_correction([db.symbols.index(s) + 1 for s in symbols], counts)
    return seg, coeffs, rel


def _f2_mismatch(seg: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Relative f2 jump at each interior knot ``seg[1:-1]``."""
    x = seg[1:-1]

    def poly(c: np.ndarray) -> np.ndarray:
        return c[:, 0] * x + c[:, 1] + c[:, 2] / x + c[:, 3] / x**2 + c[:, 4] / x**3

    left, right = poly(coeffs[:-1]), poly(coeffs[1:])
    return np.abs(left - right) / np.maximum(np.abs(left), np.abs(right))


def oracle_energies(seg: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """120 log-uniform energies, low-energy, near-edge and continuous-knot cases."""
    n = seg.size - 1
    rng = np.random.default_rng(2026)
    random = np.exp(
        rng.uniform(np.log(seg[0] * 1.001), np.log(seg[-1] / 1.001), N_RANDOM)
    )
    mismatch = _f2_mismatch(seg, coeffs)
    edges = seg[1:-1][np.argsort(mismatch)[-2:]]  # the two largest f2 steps
    near_edge = np.concatenate([edges * (1 - EDGE_OFFSET), edges * (1 + EDGE_OFFSET)])
    knot_idx = np.array([n // 4, n // 2, 3 * n // 4])
    assert (mismatch[knot_idx - 1] <= CONTINUITY_RTOL).all()
    return np.concatenate([random, LOW_EV, near_edge, seg[knot_idx]])


def _reference(
    energy: np.ndarray, seg: np.ndarray, coeffs: np.ndarray, rel: float
) -> np.ndarray:
    return np.array([f1_reference(e, seg, coeffs, rel) for e in energy])


@pytest.mark.parametrize("formula", FORMULAS)
def test_oracle_agreement(formula: str) -> None:
    seg, coeffs, rel = _spectrum(formula)
    energy = oracle_energies(seg, coeffs)
    ref = _reference(energy, seg, coeffs, rel)
    f1 = kk_real_spectrum(energy, seg, coeffs, rel)
    assert np.isfinite(f1).all()
    err = np.abs(f1 - ref)
    bound = np.maximum(
        3.0 * MEASURED_MAX[formula],
        ROUNDING_K * kk_rounding_scale(energy, seg, coeffs),
    )
    worst = int(np.argmax(err - bound))
    assert err[worst] <= bound[worst], (
        f"{formula} E={energy[worst]!r} eV: |{f1[worst]!r} - {ref[worst]!r}| "
        f"= {err[worst]:.3e} exceeds {bound[worst]:.3e}"
    )
    assert np.median(err) < MEDIAN_LIMIT


@pytest.mark.parametrize("formula", ["H", "SiO2", "Mo"])
def test_oracle_converges(formula: str) -> None:
    """Quadrature result is independent of the starting node count."""
    # 1e-14 was measured with 80-bit longdouble (eps 1e-19); where longdouble is
    # plain float64 (Apple Silicon, Windows) the quadrature noise is ~2000x larger.
    rtol = max(1e-14, 1e3 * float(np.finfo(np.longdouble).eps))
    seg, coeffs, rel = _spectrum(formula)
    energy = oracle_energies(seg, coeffs)[::17]
    base = _reference(energy, seg, coeffs, rel)
    for n0 in (32, 128):
        other = np.array([f1_reference(e, seg, coeffs, rel, n0=n0) for e in energy])
        assert np.all(np.abs(other - base) <= rtol * np.maximum(np.abs(base), 1.0))


def test_oracle_rejects_unconverged_and_conventions() -> None:
    seg, coeffs, rel = _spectrum("SiO2")
    with pytest.raises(RuntimeError, match="not converged"):
        f1_reference(1234.5, seg, coeffs, rel, n0=2, max_nodes=2)
    with pytest.raises(ValueError, match="end point"):
        f1_reference(float(seg[0]), seg, coeffs, rel)
    seg_ar, coeffs_ar, rel_ar = _spectrum("Ar")
    jump_ar = int(_f2_mismatch(seg_ar, coeffs_ar).argmax()) + 1
    with pytest.raises(ValueError, match="jumps"):
        f1_reference(float(seg_ar[jump_ar]), seg_ar, coeffs_ar, rel_ar)
