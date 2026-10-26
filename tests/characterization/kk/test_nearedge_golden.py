"""Near-edge merge against upstream kkcalc (polystyrene example, NEXAFS input)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.kk import kk_calculate_real
from xraylabtool.kk.nearedge import load_nearedge, merge_spectra, to_asf_scale
from xraylabtool.kk.spectrum import compound_spectrum
from xraylabtool.kk.transform import kk_real_spectrum
from xraylabtool.utils import parse_formula

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
EXAMPLE = HERE / "nearedge_example.txt"

# Tolerances are set from the measured errors (C8H8, upstream example file,
# merge points 280-320 eV). The NEXAFS scale constant (hc, r_e: repo values
# differ from upstream's by ~1.3e-7 and 8e-9) cancels in the affine endpoint
# matching, so only rounding remains. Measured:
#   merged knots: identical (max |diff| 0.0 eV);
#   merged coefficients: max |diff| / column scale 6.8e-16 (max relative 3.9e-12);
#   f2 at the default grid: max relative error 1.8e-14;
#   f1 at the default grid: max relative error 6.2e-12, at most 5.7x the
#     Task 3 rounding scale;
#   f1 at the golden eval points: 3.4e-9 (XLA fusion).
# Bounds: coefficients 1e-12, f2 1e-12, f1 as in the other end-to-end goldens
# (rtol 1e-8 plus the rounding allowance).
COEFF_ATOL = 1e-12
F1_RTOL = 1e-8
F2_RTOL = 1e-12


def _load_golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN / "c8h8_nearedge.npz") as g:
        return {k: g[k] for k in g.files}


def _merged(g: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pre_ev, pre_coeffs = compound_spectrum(*parse_formula("C8H8"))
    assert np.array_equal(pre_ev, g["pre_seg_ev"])
    near = to_asf_scale(load_nearedge(EXAMPLE), "nexafs")
    return merge_spectra(
        near, pre_ev, pre_coeffs, merge_points_ev=tuple(g["merge_points"])
    )


@pytest.mark.unit
def test_merged_spectrum_matches_upstream_golden() -> None:
    g = _load_golden()
    seg, coeffs = _merged(g)
    assert np.array_equal(seg, g["seg_ev"])
    assert coeffs.shape == g["coeffs"].shape
    scale = np.abs(g["coeffs"]).max(axis=0)  # per order, avoids 0/0 on zero entries
    np.testing.assert_allclose(
        coeffs / scale, g["coeffs"] / scale, rtol=0, atol=COEFF_ATOL
    )


@pytest.mark.unit
def test_nearedge_result_matches_upstream_golden() -> None:
    g = _load_golden()
    mp_kev = tuple(float(x) / 1000.0 for x in g["merge_points"])
    res = kk_calculate_real(
        "C8H8",
        nearedge=load_nearedge(EXAMPLE),
        input_type="nexafs",
        merge_points_kev=mp_kev,
    )
    keep = (g["energy_ev"] >= 30.0) & (g["energy_ev"] <= 30000.0)
    energy_ev = g["energy_ev"][keep]
    assert res.backend == "jax"
    assert res.merge_points_kev == mp_kev
    np.testing.assert_allclose(res.energy_kev * 1000.0, energy_ev, rtol=1e-15)
    assert np.isfinite(res.f1).all()
    np.testing.assert_allclose(res.f2, g["f2"][keep], rtol=F2_RTOL, atol=1e-12)
    assert_kk_close(
        res.f1,
        g["f1"][keep],
        energy_ev,
        g["seg_ev"],
        g["coeffs"],
        rtol=F1_RTOL,
    )
    assert res.nearedge_scaled_kev_f2 is not None
    assert res.nearedge_scaled_kev_f2[0].shape == (
        load_nearedge(EXAMPLE).energy_ev.size,
    )


@pytest.mark.unit
def test_kernel_on_merged_spectrum_matches_golden_f1_points() -> None:
    g = _load_golden()
    seg, coeffs = _merged(g)
    f1 = kk_real_spectrum(g["eval_ev"], seg, coeffs, float(g["rel_corr"]))
    assert_kk_close(
        f1,
        g["f1_kk_pp"],
        g["eval_ev"],
        g["seg_ev"],
        g["coeffs"],
        rtol=F1_RTOL,
    )
