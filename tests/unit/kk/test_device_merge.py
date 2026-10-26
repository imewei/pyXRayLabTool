"""Device near-edge merge (``nearedge.merge_device``), Task 3 of the JAX-first migration.

The reference in this module is an independent NumPy statement of the splice of
B. Watts, Opt. Express 22(19), 23628 (2014) (affine end-point matching, then
the piecewise-linear near-edge curve replaces the tabulated segments between
the merge points). It is the pre-migration host algorithm, so the device merge
must reproduce it; the one deliberate difference is the source segment of an
output segment, selected by its lower boundary instead of by a midpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from xraylabtool.exceptions import NearEdgeDataError
from xraylabtool.kk import _device, kk_calculate_real, nearedge
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk._shapes import bucket, bucket_pow2
from xraylabtool.kk.nearedge import (
    MergeResult,
    MergeStatus,
    load_nearedge,
    merge_device,
    merge_spectra,
    raise_or_log,
    to_asf_scale,
    to_asf_scale_device,
)
from xraylabtool.kk.spectrum import compound_spectrum
from xraylabtool.utils import parse_formula

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "tests" / "characterization" / "kk" / "nearedge_example.txt"
BASELINE = ROOT / "tests" / "unit" / "kk" / "data" / "prejax_baseline.npz"

SEG = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
COEFFS = np.random.default_rng(7).uniform(0.5, 2.0, (4, 5))
NE_E = np.linspace(8.0, 55.0, 48)
NE_V = 0.2 + 0.8 / (1.0 + np.exp(-(NE_E - 31.0) / 3.0))


def _poly(seg: np.ndarray, coef: np.ndarray, e: np.ndarray) -> np.ndarray:
    i = np.clip(np.searchsorted(seg, e, side="right") - 1, 0, coef.shape[0] - 1)
    c = coef[i]
    return c[:, 0] * e + c[:, 1] + c[:, 2] / e + c[:, 3] / e**2 + c[:, 4] / e**3


def _reference(
    seg: np.ndarray,
    coef: np.ndarray,
    e: np.ndarray,
    v: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_low, a_high = _poly(seg, coef, np.array([low, high]))
    v_low, v_high = np.interp([low, high], e, v)
    scaled = (v - v_low) * ((a_high - a_low) / (v_high - v_low)) + a_low
    inside = (e > low) & (e < high)
    knots = np.concatenate(([low], e[inside], [high]))
    heights = np.concatenate(([a_low], scaled[inside], [a_high]))
    slope = np.diff(heights) / np.diff(knots)
    linear = np.zeros((knots.size - 1, 5))
    linear[:, 0] = slope
    linear[:, 1] = heights[:-1] - slope * knots[:-1]
    n_left = int(np.searchsorted(seg, low, side="left"))
    n_right = int(np.searchsorted(seg, high, side="right"))
    full_ev = np.concatenate((seg[:n_left], knots, seg[n_right:]))
    if n_right < seg.size:
        full_c = np.vstack(
            (coef[:n_left], linear, coef[n_right - 1 : n_right], coef[n_right:])
        )
    else:
        full_c = np.vstack((coef[:n_left], linear))
    return full_ev, full_c, scaled


def _run(
    seg: np.ndarray,
    coef: np.ndarray,
    e: np.ndarray,
    v: np.ndarray,
    low: float,
    high: float,
    *,
    capacity: int | None = None,
    spec_capacity: int | None = None,
    extra_pad: int = 0,
) -> tuple[MergeResult, MergeStatus, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``merge_device`` on padded inputs and return the unpadded host arrays."""
    n_ne = e.size
    if capacity is None:
        capacity = bucket(coef.shape[0] + n_ne + 2)
    m = bucket_pow2(n_ne) + extra_pad
    # Garbage in the padding: it must be masked, never read.
    e_p = np.concatenate([e, np.full(m - n_ne, -7.0)])
    v_p = np.concatenate([v, np.full(m - n_ne, np.nan)])
    spec = Spectrum.from_numpy(seg, coef, spec_capacity)
    e_d, v_d = _device.upload((e_p, v_p))
    res = merge_device(spec, e_d, v_d, np.int32(n_ne), low, high, capacity=capacity)
    status = _device.fetch_status(res.status)
    out = _device.fetch(res)
    n = int(out.spectrum.n_segments)
    assert n <= capacity
    assert int(status.n_segments) == n
    return (
        res,
        status,
        np.asarray(out.spectrum.seg_ev[: n + 1]),
        np.asarray(out.spectrum.coeffs[:n]),
        np.asarray(out.scaled_f2[:n_ne]),
    )


def _assert_matches_reference(
    seg: np.ndarray,
    coef: np.ndarray,
    e: np.ndarray,
    v: np.ndarray,
    low: float,
    high: float,
) -> None:
    ref_ev, ref_c, ref_scaled = _reference(seg, coef, e, v, low, high)
    _, _, got_ev, got_c, got_scaled = _run(seg, coef, e, v, low, high)
    np.testing.assert_array_equal(got_ev, ref_ev)
    np.testing.assert_allclose(got_c, ref_c, rtol=1e-13, atol=0)
    np.testing.assert_allclose(got_scaled, ref_scaled, rtol=1e-13, atol=0)
    assert np.isfinite(got_c).all()


@pytest.mark.unit
def test_merge_reproduces_goldens() -> None:
    seg, coef = compound_spectrum(*parse_formula("C8H8"))
    near = to_asf_scale(load_nearedge(EXAMPLE), "nexafs")
    _, status, got_ev, got_c, _ = _run(
        seg, coef, near.energy_ev, near.values, 280.0, 320.0
    )
    with np.load(BASELINE) as base:
        np.testing.assert_array_equal(got_ev, base["merge_seg"])
        np.testing.assert_allclose(got_c, base["merge_coeffs"], rtol=1e-13, atol=0)
    assert int(status.n_segments) == got_c.shape[0]
    # The host path (characterization goldens) goes through the same device merge.
    host_ev, host_c = merge_spectra(near, seg, coef, merge_points_ev=(280.0, 320.0))
    np.testing.assert_array_equal(host_ev, got_ev)
    np.testing.assert_array_equal(host_c, got_c)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("low", "high"),
    [
        pytest.param(22.0, 28.0, id="no-knots"),
        pytest.param(22.0, 38.0, id="one-knot"),
        pytest.param(10.5, 49.5, id="all-interior-knots"),
        pytest.param(10.0, 50.0, id="both-range-ends"),
        pytest.param(10.0, 38.0, id="low-end-only"),
        pytest.param(22.0, 50.0, id="high-end-only"),
    ],
)
def test_window_matches_reference(low: float, high: float) -> None:
    _assert_matches_reference(SEG, COEFFS, NE_E, NE_V, low, high)


@pytest.mark.unit
def test_window_covers_no_knots() -> None:
    _, _, ev, c, _ = _run(SEG, COEFFS, NE_E, NE_V, 22.0, 28.0)
    assert {10.0, 20.0, 30.0, 40.0, 50.0} <= set(ev.tolist())
    assert 22.0 in ev and 28.0 in ev
    # The tabulated segment [20, 30) is split into three; outside it is untouched.
    np.testing.assert_array_equal(c[0], COEFFS[0])
    np.testing.assert_array_equal(c[1], COEFFS[1])  # [20, 22)
    np.testing.assert_array_equal(c[-3:], COEFFS[1:])  # [28, 30) onwards


@pytest.mark.unit
def test_window_covers_all_knots() -> None:
    _, _, ev, c, _ = _run(SEG, COEFFS, NE_E, NE_V, 10.5, 49.5)
    assert not c[1:-1, 2:].any()  # only near-edge linear pieces between the ends
    assert ev[0] == 10.0 and ev[-1] == 50.0
    np.testing.assert_array_equal(c[0], COEFFS[0])
    np.testing.assert_array_equal(c[-1], COEFFS[-1])


@pytest.mark.unit
def test_window_touches_range_ends_snaps() -> None:
    near = nearedge.NearEdgeSpectrum(np.linspace(10.0, 50.0, 41), NE_V[:41] + NE_E[:41])
    exact = merge_spectra(near, SEG, COEFFS, merge_points_ev=(10.0, 50.0))
    # Within rounding of both ends: host resolution snaps to the ends exactly.
    snapped = merge_spectra(
        near, SEG, COEFFS, merge_points_ev=(10.0 * (1 + 1e-13), 50.0 * (1 - 1e-13))
    )
    np.testing.assert_array_equal(snapped[0], exact[0])
    np.testing.assert_array_equal(snapped[1], exact[1])
    assert exact[0][0] == 10.0 and exact[0][-1] == 50.0
    ref_ev, ref_c, _ = _reference(SEG, COEFFS, near.energy_ev, near.values, 10.0, 50.0)
    np.testing.assert_array_equal(exact[0], ref_ev)
    np.testing.assert_allclose(exact[1], ref_c, rtol=1e-13, atol=0)


@pytest.mark.unit
def test_adjacent_float_knots_use_lower_boundary() -> None:
    a = float(np.nextafter(1.0, np.inf))
    b = float(np.nextafter(a, np.inf))
    assert (a + b) / 2 in (a, b)  # a midpoint cannot separate them
    seg = np.array([a, b, 2.0, 3.0])
    coef = np.zeros((3, 5))
    coef[:, 1] = [10.0, 20.0, 30.0]
    coef[2, 0] = 1.0  # tabulated f2 at 2 eV and 3 eV: 32 and 33
    e = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    v = np.array([0.0, 1.0, 3.0, 6.0, 9.0])
    _, _, ev, c, _ = _run(seg, coef, e, v, 2.0, 3.0)
    np.testing.assert_array_equal(ev[:4], [a, b, 2.0, 2.5])
    assert c[0, 1] == 10.0  # [a, b): the first polynomial, not the second
    assert c[1, 1] == 20.0  # [b, 2)
    assert c[2, 0] != 0.0  # [2, 2.5): near-edge linear piece


@pytest.mark.unit
def test_merge_point_on_source_knot() -> None:
    # Both merge points coincide with tabulated knots; the segment starting at
    # the upper one is the tabulated polynomial that starts there.
    ref_ev, ref_c, _ = _reference(SEG, COEFFS, NE_E, NE_V, 20.0, 40.0)
    _, _, ev, c, _ = _run(SEG, COEFFS, NE_E, NE_V, 20.0, 40.0)
    np.testing.assert_array_equal(ev, ref_ev)
    assert np.all(np.diff(ev) > 0)
    np.testing.assert_array_equal(c[0], COEFFS[0])  # [10, 20)
    np.testing.assert_array_equal(c[-1], COEFFS[3])  # [40, 50)
    np.testing.assert_allclose(c, ref_c, rtol=1e-13, atol=0)


@pytest.mark.unit
def test_merge_bucket_invariant() -> None:
    cap = bucket(SEG.size + NE_E.size + 2)
    base = _run(SEG, COEFFS, NE_E, NE_V, 22.0, 38.0, capacity=cap)
    up1 = bucket(cap + 1)
    up2 = bucket(up1 + 1)
    assert cap < up1 < up2
    for kwargs in (
        {"capacity": up1},
        {"capacity": up2, "spec_capacity": up1},
        {"capacity": cap, "extra_pad": 64},
    ):
        other = _run(SEG, COEFFS, NE_E, NE_V, 22.0, 38.0, **kwargs)  # type: ignore[arg-type]
        for x, y in zip(base[2:], other[2:], strict=True):
            np.testing.assert_array_equal(x, y)
        assert int(other[1].n_segments) == int(base[1].n_segments)


@pytest.mark.unit
def test_padding_entries_are_masked() -> None:
    # NaN near-edge padding must not leak into the status or the output.
    _, status, _, c, scaled = _run(SEG, COEFFS, NE_E, NE_V, 22.0, 38.0, extra_pad=17)
    assert not bool(status.nonfinite)
    assert np.isfinite(c).all() and np.isfinite(scaled).all()
    assert int(status.n_inside) == int(((NE_E > 22.0) & (NE_E < 38.0)).sum())


@pytest.mark.unit
def test_scaled_f2_covers_full_nearedge_curve() -> None:
    seg, coef = compound_spectrum(*parse_formula("C8H8"))
    near = to_asf_scale(load_nearedge(EXAMPLE), "nexafs")
    _, _, _, _, scaled = _run(seg, coef, near.energy_ev, near.values, 280.0, 320.0)
    assert scaled.shape == near.energy_ev.shape
    assert (near.energy_ev < 280.0).any() and (near.energy_ev > 320.0).any()
    _, _, ref_scaled = _reference(seg, coef, near.energy_ev, near.values, 280.0, 320.0)
    np.testing.assert_allclose(scaled, ref_scaled, rtol=1e-13, atol=0)
    res = kk_calculate_real(
        "C8H8",
        nearedge=load_nearedge(EXAMPLE),
        input_type="nexafs",
        merge_points_kev=(0.28, 0.32),
    )
    assert res.nearedge_scaled_kev_f2 is not None
    np.testing.assert_array_equal(res.nearedge_scaled_kev_f2[1], scaled)


def _status_of(e: np.ndarray, v: np.ndarray, low: float, high: float) -> MergeStatus:
    return _run(SEG, COEFFS, e, v, low, high)[1]


@pytest.mark.unit
def test_status_errors_unchanged() -> None:
    low, high = 22.0, 38.0
    a_low, a_high = _poly(SEG, COEFFS, np.array([low, high]))
    # Flat measured end points.
    flat = np.full_like(NE_V, 0.5)
    with pytest.raises(NearEdgeDataError) as excinfo:
        raise_or_log(_status_of(NE_E, flat, low, high), low, high)
    assert str(excinfo.value) == (
        "Near-edge data have flat endpoints: the values at the merge points 22 "
        "and 38 eV (0.5, 0.5) differ by less than 1e-06 of their size, so the "
        "data cannot be normalised to the tabulated f2. Choose merge points on "
        "the pre-edge and post-edge."
    )
    # Flat tabulated end points: the same constant polynomial on both sides.
    coef = np.zeros((4, 5))
    coef[:, 1] = 1.5
    status = _run(SEG, coef, NE_E, NE_V, low, high)[1]
    with pytest.raises(NearEdgeDataError) as excinfo:
        raise_or_log(status, low, high)
    assert str(excinfo.value) == (
        "The tabulated f2 is equal at the merge points 22 and 38 eV (1.5, 1.5); "
        "the near-edge data cannot be normalised. Choose merge points on either "
        "side of the edge."
    )
    # Non-finite after normalisation.
    huge = NE_V.copy()
    huge[24] = 1e308
    status = _status_of(NE_E, huge, low, high)
    assert bool(status.nonfinite)
    with pytest.raises(NearEdgeDataError) as excinfo:
        raise_or_log(status, low, high)
    assert str(excinfo.value) == (
        "Near-edge data are not finite after normalisation to the tabulated f2 "
        "(values too large for the merge-point step)"
    )
    # A healthy merge raises nothing and its end points are the evaluated ones.
    ok = _status_of(NE_E, NE_V, low, high)
    raise_or_log(ok, low, high)
    assert float(ok.a_low) == a_low and float(ok.a_high) == a_high
    assert float(ok.tabulated_step) == a_high - a_low


@pytest.mark.unit
def test_negative_f2_warning_unchanged(kk_log: pytest.LogCaptureFixture) -> None:
    low, high = 22.0, 38.0
    a_low, a_high = _poly(SEG, COEFFS, np.array([low, high]))
    v_low, v_high = np.interp([low, high], NE_E, NE_V)
    ne_e = NE_E
    inside = (ne_e > low) & (ne_e < high)
    v = NE_V.copy()
    k = int(np.flatnonzero(inside)[3])
    t = -1.5 * a_low / (a_high - a_low)  # scaled value = -0.5 * a_low < 0
    v[k] = v_low + t * (v_high - v_low)
    scaled = (v - v_low) * ((a_high - a_low) / (v_high - v_low)) + a_low
    n_neg = int((scaled[inside] < 0).sum())
    assert n_neg == 1
    status = _status_of(NE_E, v, low, high)
    assert int(status.n_negative) == n_neg
    assert int(status.n_inside) == int(inside.sum())
    assert float(status.f2_min) == pytest.approx(float(scaled[inside].min()), rel=1e-12)
    raise_or_log(status, low, high)
    warnings = [r for r in kk_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage() == (
        f"{n_neg} of {int(inside.sum())} near-edge points inside the merge window "
        f"are negative after normalisation to the tabulated f2 (minimum "
        f"{float(status.f2_min):.6g}); kept unchanged, not clipped"
    )
    kk_log.clear()
    raise_or_log(_status_of(NE_E, NE_V, low, high), low, high)
    assert not [r for r in kk_log.records if r.levelno >= logging.WARNING]


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["nexafs", "photoabsorption", "xanes", "beta", "asf"])
def test_to_asf_scale_device_matches_host(kind: str) -> None:
    near = nearedge.NearEdgeSpectrum(NE_E, NE_V)
    host = to_asf_scale(near, kind, density=2.0, formula_mass=60.0)  # type: ignore[arg-type]
    m = bucket_pow2(NE_E.size)
    e_p = np.concatenate([NE_E, np.ones(m - NE_E.size)])
    v_p = np.concatenate([NE_V, np.ones(m - NE_E.size)])
    e_d, v_d = _device.upload((e_p, v_p))
    got = _device.fetch(
        to_asf_scale_device(e_d, v_d, kind, density=2.0, formula_mass=60.0)
    )
    np.testing.assert_array_equal(np.asarray(got)[: NE_E.size], host.values)
