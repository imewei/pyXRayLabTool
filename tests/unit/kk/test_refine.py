"""Tests for adaptive refinement (``xraylabtool.kk.refine``) and its API hook."""

from __future__ import annotations

import dataclasses
import doctest
import logging
import typing

import numpy as np
import pytest

from xraylabtool import kk
from xraylabtool.exceptions import ValidationError
from xraylabtool.kk import api, kk_calculate_real, refine
from xraylabtool.kk.nearedge import NearEdgeSpectrum
from xraylabtool.kk.refine import RefineOutcome, refine_spectrum
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction
from xraylabtool.utils import parse_formula


@pytest.fixture(scope="module")
def si():
    symbols, counts = parse_formula("Si")
    seg, coeffs = compound_spectrum(symbols, counts)
    rel = relativistic_correction([14], counts)
    return seg, coeffs, rel


def _refine(si, energy_ev, **kwargs):
    seg, coeffs, rel = si
    energy_ev = np.asarray(energy_ev, dtype=np.float64)
    f1 = kk_real_spectrum(energy_ev, seg, coeffs, rel)
    f2 = evaluate_f2(energy_ev, seg, coeffs)
    kwargs.setdefault("tolerance", 0.05)
    return f1, f2, refine_spectrum(energy_ev, f1, f2, seg, coeffs, rel, **kwargs)


GRID = np.geomspace(100.0, 20000.0, 12)


@pytest.mark.unit
def test_original_points_preserved(si) -> None:
    f1, f2, out = _refine(si, GRID)
    assert out.n_added > 0
    assert out.energy_ev.size == GRID.size + out.n_added
    assert (np.diff(out.energy_ev) > 0).all()
    idx = np.searchsorted(out.energy_ev, GRID)
    np.testing.assert_array_equal(out.energy_ev[idx], GRID)
    np.testing.assert_array_equal(out.f1[idx], f1)
    np.testing.assert_array_equal(out.f2[idx], f2)
    assert np.isfinite(out.f1).all()
    assert np.isfinite(out.f2).all()


@pytest.mark.unit
def test_refined_grid_meets_tolerance_at_midpoints(si) -> None:
    seg, coeffs, rel = si
    _, _, out = _refine(si, GRID, tolerance=0.02)
    assert not out.exhausted
    mid = 0.5 * (out.energy_ev[:-1] + out.energy_ev[1:])
    err = np.abs(
        0.5 * (out.f1[:-1] + out.f1[1:]) - kk_real_spectrum(mid, seg, coeffs, rel)
    )
    err2 = np.abs(0.5 * (out.f2[:-1] + out.f2[1:]) - evaluate_f2(mid, seg, coeffs))
    # Every interval that is not an original one was accepted at its parent's check.
    assert err.max() < 5 * 0.02
    assert err2.max() < 5 * 0.02


@pytest.mark.unit
def test_loose_tolerance_adds_nothing(si) -> None:
    f1, f2, out = _refine(si, GRID, tolerance=1e6)
    assert out.n_added == 0
    assert out.exhausted is False
    np.testing.assert_array_equal(out.energy_ev, GRID)
    np.testing.assert_array_equal(out.f1, f1)
    np.testing.assert_array_equal(out.f2, f2)


@pytest.mark.unit
def test_exhaustion_flagged_and_logged(si, kk_log: pytest.LogCaptureFixture) -> None:
    _, _, out = _refine(si, GRID, tolerance=1e-9, max_passes=2)
    assert out.exhausted is True
    assert out.n_added > 0
    warnings = [r for r in kk_log.records if r.levelno == logging.WARNING]
    assert warnings
    assert "max_passes" in warnings[0].getMessage()


@pytest.mark.unit
def test_point_cap_stops_and_flags(si, kk_log: pytest.LogCaptureFixture) -> None:
    _, _, out = _refine(si, GRID, tolerance=1e-9, max_points=10)
    assert out.n_added == 10
    assert out.exhausted is True
    assert (np.diff(out.energy_ev) > 0).all()
    assert any(
        r.levelno == logging.WARNING and "max_points" in r.getMessage()
        for r in kk_log.records
    )


@pytest.mark.unit
def test_midpoint_stagnation_terminates(si) -> None:
    e0 = 1000.0
    pair = np.array([e0, np.nextafter(np.nextafter(e0, np.inf), np.inf)])
    _, _, out = _refine(si, pair, tolerance=1e-300, max_passes=1000)
    assert out.exhausted is True
    assert (np.diff(out.energy_ev) > 0).all()
    assert out.energy_ev.size <= 3


@pytest.mark.unit
def test_debug_log_per_pass_and_info_total(
    si, kk_log: pytest.LogCaptureFixture
) -> None:
    kk_log.set_level(logging.DEBUG, logger="xraylabtool")
    _, _, out = _refine(si, GRID)
    msgs = [
        (r.levelno, r.getMessage())
        for r in kk_log.records
        if r.name == "xraylabtool.kk"
    ]
    assert any(lv == logging.DEBUG and "pass 1" in m for lv, m in msgs)
    assert any(lv == logging.INFO and str(out.n_added) in m for lv, m in msgs)


@pytest.mark.unit
@pytest.mark.parametrize(
    "energy",
    [
        [2000.0, 1000.0],
        [1000.0, 1000.0, 2000.0],
        [],
        [[1000.0, 2000.0]],
        [1000.0, np.nan],
    ],
)
def test_refine_rejects_non_increasing_input(si, energy) -> None:
    seg, coeffs, rel = si
    e = np.asarray(energy, dtype=np.float64)
    with pytest.raises(ValidationError):
        refine_spectrum(
            e, np.ones(e.shape), np.ones(e.shape), seg, coeffs, rel, tolerance=0.05
        )


@pytest.mark.unit
def test_refine_rejects_bad_arguments(si) -> None:
    seg, coeffs, rel = si
    e = np.array([1000.0, 2000.0])
    ones = np.ones(2)
    for kwargs in (
        {"tolerance": 0.0},
        {"tolerance": -1.0},
        {"tolerance": float("nan")},
        {"tolerance": float("inf")},
        {"tolerance": 0.05, "max_passes": 0},
        {"tolerance": 0.05, "max_points": 0},
        {"tolerance": 0.05, "max_passes": True},
    ):
        with pytest.raises(ValidationError):
            refine_spectrum(e, ones, ones, seg, coeffs, rel, **kwargs)
    with pytest.raises(ValidationError):  # length mismatch
        refine_spectrum(e, ones[:1], ones, seg, coeffs, rel, tolerance=0.05)
    with pytest.raises(ValidationError):  # outside the tabulated range
        refine_spectrum(
            np.array([1.0, 2.0]), ones, ones, seg, coeffs, rel, tolerance=0.05
        )


@pytest.mark.unit
def test_refine_with_shuffled_duplicate_energies_returns_sorted_unique_union() -> None:
    requested = np.array([20.0, 1.5, 3.0, 1.5, 0.1, 20.0, 0.3])
    res = kk_calculate_real("SiO2", energies_kev=requested, refine_tolerance=0.05)
    unique = np.unique(requested)
    assert (np.diff(res.energy_kev) > 0).all()
    assert res.n_refined_points > 0
    assert res.energy_kev.size == unique.size + res.n_refined_points
    np.testing.assert_array_equal(
        res.energy_kev[np.searchsorted(res.energy_kev, unique)], unique
    )
    plain = kk_calculate_real("SiO2", energies_kev=unique)
    np.testing.assert_allclose(
        res.f1[np.searchsorted(res.energy_kev, unique)], plain.f1, rtol=1e-10, atol=1e-8
    )
    assert np.isfinite(res.f1).all()
    assert np.isfinite(res.f2).all()


@pytest.mark.unit
def test_refine_length_one_grid() -> None:
    res = kk_calculate_real("Si", energies_kev=[2.0], refine_tolerance=0.05)
    assert res.energy_kev.tolist() == [2.0]
    assert res.n_refined_points == 0


@pytest.mark.unit
def test_n_refined_points_reported_in_result() -> None:
    plain = kk_calculate_real("Si", energies_kev=[0.1, 1.0, 10.0, 25.0])
    res = kk_calculate_real(
        "Si", energies_kev=[0.1, 1.0, 10.0, 25.0], refine_tolerance=0.01
    )
    assert plain.n_refined_points == 0
    assert res.n_refined_points == res.energy_kev.size - 4 > 0
    assert res.f1.shape == res.f2.shape == res.energy_kev.shape


@pytest.mark.unit
def test_refine_default_grid_adds_points_and_keeps_tabulated_ones() -> None:
    plain = kk_calculate_real("Si")
    res = kk_calculate_real("Si", refine_tolerance=0.05)
    assert res.n_refined_points == res.energy_kev.size - plain.energy_kev.size > 0
    np.testing.assert_array_equal(
        res.energy_kev[np.searchsorted(res.energy_kev, plain.energy_kev)],
        plain.energy_kev,
    )


@pytest.mark.unit
def test_refine_after_nearedge_merge() -> None:
    edge = NearEdgeSpectrum(
        [280.0, 290.0, 300.0, 310.0, 320.0], [0.05, 0.1, 0.3, 0.45, 0.5]
    )
    kwargs = {
        "nearedge": edge,
        "input_type": "nexafs",
        "merge_points_kev": (0.28, 0.32),
    }
    plain = kk_calculate_real("C", **kwargs)
    res = kk_calculate_real("C", refine_tolerance=0.002, **kwargs)
    assert res.merge_points_kev == (0.28, 0.32)
    assert res.n_refined_points > 0
    assert (np.diff(res.energy_kev) > 0).all()
    assert 0.29 in res.energy_kev
    assert res.nearedge_scaled_kev_f2 is not None
    assert np.isfinite(res.f1).all()
    np.testing.assert_array_equal(
        res.energy_kev[np.searchsorted(res.energy_kev, plain.energy_kev)],
        plain.energy_kev,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"refine_tolerance": 0.0},
        {"refine_tolerance": -0.1},
        {"refine_tolerance": float("nan")},
        {"refine_tolerance": float("inf")},
        {"refine_tolerance": "0.1"},
        {"refine_tolerance": 0.1, "refine_max_passes": 0},
        {"refine_tolerance": 0.1, "refine_max_passes": 1.5},
        {"refine_tolerance": 0.1, "refine_max_points": 0},
        {"refine_max_passes": 0},
        {"refine_max_points": True},
    ],
)
def test_invalid_refine_arguments_raise_validation_error(kwargs) -> None:
    with pytest.raises(ValidationError):
        kk_calculate_real("Si", energies_kev=[1.0, 2.0], **kwargs)


@pytest.mark.unit
def test_unrefined_path_keeps_caller_order() -> None:
    res = kk_calculate_real("Si", energies_kev=[3.0, 1.0, 1.0])
    assert res.energy_kev.tolist() == [3.0, 1.0, 1.0]


@pytest.mark.unit
def test_refine_names_are_exported() -> None:
    assert kk.refine_spectrum is refine_spectrum
    assert kk.RefineOutcome is RefineOutcome
    assert set(kk.__all__) == set(kk._EXPORTS)


@pytest.mark.unit
def test_docstring_examples_run() -> None:
    results = doctest.testmod(refine, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0
    assert results.failed == 0
    results = doctest.testmod(api, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0


@pytest.mark.unit
def test_type_hints_resolve_at_runtime() -> None:
    hints = typing.get_type_hints(RefineOutcome)
    assert set(hints) == {f.name for f in dataclasses.fields(RefineOutcome)}
    typing.get_type_hints(refine_spectrum)


@pytest.mark.unit
def test_interval_at_range_end_with_large_f2_is_not_bisected(si) -> None:
    """f1 is logarithmically singular at the lower end, so bisection never converges."""
    grid = np.array([si[0][0], 20.0, 100.0])
    _, f2, out = _refine(si, grid)
    assert f2[0] > 0.05
    assert not ((out.energy_ev > grid[0]) & (out.energy_ev < grid[1])).any()
    assert out.n_added > 0  # the smooth interval beyond it is still refined
