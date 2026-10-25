"""Tests for the compound f2 spectrum on the union of elemental grids."""

from __future__ import annotations

import doctest
from pathlib import Path

import numpy as np
import pytest

from xraylabtool.exceptions import FormulaError
from xraylabtool.kk import spectrum
from xraylabtool.kk.database import load_asf
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2

GOLDEN = Path(__file__).resolve().parents[2] / "characterization" / "kk" / "golden"


@pytest.mark.unit
def test_compound_grid_is_union_of_boundaries() -> None:
    db = load_asf()
    seg, coeffs = compound_spectrum(["Si", "O"], [1.0, 2.0])
    union = np.unique(np.concatenate([db.energy_ev(14), db.energy_ev(8)]))
    assert np.array_equal(seg, union)
    assert coeffs.shape == (union.size - 1, 5)
    assert np.all(np.diff(seg) > 0)
    # Both elemental grids are subsets, so the union is strictly finer than each.
    assert union.size > max(db.energy_ev(14).size, db.energy_ev(8).size)

    # The summed polynomials reproduce n_i * f2_i at every lower edge and midpoint.
    # The Laurent terms partly cancel, so summing coefficients first differs
    # from summing values at the rounding level: measured worst 1.4e-12.
    probe = np.concatenate([seg[:-1], 0.5 * (seg[:-1] + seg[1:])])
    expected = evaluate_f2(
        probe, db.energy_ev(14), db.im_coeffs(14)
    ) + 2.0 * evaluate_f2(probe, db.energy_ev(8), db.im_coeffs(8))
    got = evaluate_f2(probe, seg, coeffs)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=1e-10)


@pytest.mark.unit
def test_single_element_is_the_elemental_table_scaled_by_count() -> None:
    db = load_asf()
    seg, coeffs = compound_spectrum(["Au"], [3.0])
    assert np.array_equal(seg, db.energy_ev(79))
    assert np.array_equal(coeffs, 3.0 * db.im_coeffs(79))


@pytest.mark.unit
def test_element_order_does_not_change_the_result() -> None:
    a = compound_spectrum(["Si", "O"], [1.0, 2.0])
    b = compound_spectrum(["O", "Si"], [2.0, 1.0])
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbols", "counts"),
    [
        ([], []),
        (["Si", "O"], [1.0]),
        (["Xx"], [1.0]),
        (["si"], [1.0]),
        (["Np"], [1.0]),  # Z = 93 is outside the tables
        (["Si"], [0.0]),
        (["Si"], [-1.0]),
        (["Si"], [float("nan")]),
        (["Si"], [float("inf")]),
    ],
)
def test_compound_spectrum_rejects_bad_input(symbols, counts) -> None:
    with pytest.raises(FormulaError):
        compound_spectrum(symbols, counts)


@pytest.mark.unit
def test_evaluate_f2_matches_upstream_golden() -> None:
    with np.load(GOLDEN / "sio2.npz") as g:
        seg, coeffs = compound_spectrum(["Si", "O"], [1.0, 2.0])
        f2 = evaluate_f2(g["energy_ev"], seg, coeffs)
        np.testing.assert_allclose(f2, g["f2"], rtol=1e-12)


@pytest.mark.unit
def test_evaluate_f2_edge_conventions() -> None:
    seg = [1.0, 2.0, 4.0]
    coeffs = [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0, 0.0]]
    # Interior edge belongs to the segment that starts there (right-continuous).
    assert evaluate_f2([2.0], seg, coeffs).tolist() == [1.0]
    # The final edge uses the last polynomial.
    assert evaluate_f2([4.0], seg, coeffs).tolist() == [0.5]
    assert evaluate_f2([1.0], seg, coeffs).tolist() == [1.0]
    assert evaluate_f2([], seg, coeffs).shape == (0,)
    # All five orders of the Laurent form.
    c5 = [[2.0, 3.0, 5.0, 7.0, 11.0]]
    e = 3.0
    expected = 2.0 * e + 3.0 + 5.0 / e + 7.0 / e**2 + 11.0 / e**3
    np.testing.assert_allclose(evaluate_f2([e], [1.0, 9.0], c5), [expected], rtol=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize(
    "energy",
    [[0.5], [4.5], [float("nan")], [float("inf")], [[2.0]]],
)
def test_evaluate_f2_rejects_bad_energies(energy) -> None:
    seg = [1.0, 2.0, 4.0]
    coeffs = np.ones((2, 5))
    with pytest.raises(ValueError, match="energy_ev"):
        evaluate_f2(energy, seg, coeffs)


@pytest.mark.unit
def test_evaluate_f2_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="seg_ev"):
        evaluate_f2([1.5], [1.0, 1.0, 2.0], np.ones((2, 5)))
    with pytest.raises(ValueError, match="coeffs"):
        evaluate_f2([1.5], [1.0, 2.0], np.ones((2, 5)))
    with pytest.raises(ValueError, match="seg_ev"):
        evaluate_f2([1.5], [1.0], np.ones((0, 5)))


@pytest.mark.unit
def test_docstring_examples_run() -> None:
    results = doctest.testmod(spectrum, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0
    assert results.failed == 0
