"""Device-resident adaptive refinement (design spec §7): exact-semantics parity."""

from __future__ import annotations

import logging
from pathlib import Path
import re

import jax
import numpy as np
import pytest

from tests.fixtures.kk_compiles import KK_PROGRAM_NAMES, kk_programs, record_compiles
from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.kk import _device, kk_calculate_real, refine
from xraylabtool.kk.refine import _POLE_REL_WIDTH, refine_spectrum
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction

DATA = Path(__file__).resolve().parent / "data"
GRID = np.geomspace(100.0, 20000.0, 12)
DEBUG_LINE = re.compile(
    r"refine_spectrum: pass (\d+) added (\d+) points \(of (\d+) intervals checked\)"
)
INFO_LINE = re.compile(r"refine_spectrum: added (\d+) points in (\d+) passes")
REASON_LINE = re.compile(r"refine_spectrum exhausted \((.*)\) with (\d+) points")


@pytest.fixture(scope="module")
def si():
    seg, coeffs = compound_spectrum(["Si"], [1.0])
    return seg, coeffs, relativistic_correction([14], [1.0])


@pytest.fixture(scope="module")
def order():
    with np.load(DATA / "prejax_refine_order.npz") as data:
        return {k: data[k] for k in data.files}


def _refine(table, energy_ev, **kwargs):
    seg, coeffs, rel = table
    e = np.asarray(energy_ev, dtype=np.float64)
    kwargs.setdefault("tolerance", 0.05)
    return refine_spectrum(
        e,
        kk_real_spectrum(e, seg, coeffs, rel),
        evaluate_f2(e, seg, coeffs),
        seg,
        coeffs,
        rel,
        **kwargs,
    )


def _logged(caplog: pytest.LogCaptureFixture):
    """Parsed ``(debug passes, info (added, passes), reason)`` of one refine call."""
    debug = [
        tuple(int(g) for g in m.groups())
        for r in caplog.records
        if (m := DEBUG_LINE.search(r.getMessage()))
    ]
    info = [
        tuple(int(g) for g in m.groups())
        for r in caplog.records
        if (m := INFO_LINE.search(r.getMessage()))
    ]
    reasons = [
        m.group(1)
        for r in caplog.records
        if r.levelno == logging.WARNING and (m := REASON_LINE.search(r.getMessage()))
    ]
    return debug, info[-1], (reasons[-1] if reasons else "")


@pytest.fixture
def dlog(kk_log: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    kk_log.set_level(logging.DEBUG, logger="xraylabtool")
    return kk_log


# --- the pre-migration algorithm, as the test oracle --------------------------


def _host_reference(table, e, *, tolerance, max_passes=50, max_points=100_000):
    """The pre-migration host loop (NumPy, same kernels), for differential tests.

    Returns ``(energy, f1, f2, n_added, reason, checks, n_skipped, n_wide,
    widest)`` where ``checks`` lists ``(pass, accepted, checked)`` per pass
    that evaluated midpoints.
    """
    seg, coeffs, rel = table
    f1 = kk_real_spectrum(e, seg, coeffs, rel)
    f2 = evaluate_f2(e, seg, coeffs)
    iv = np.vstack(
        [
            e[:-1],
            e[1:],
            f1[:-1],
            f1[1:],
            f2[:-1],
            f2[1:],
            (e[:-1] == seg[0]) & (np.abs(f2[:-1]) > tolerance),
            (e[1:] == seg[-1]) & (np.abs(f2[1:]) > tolerance),
        ]
    )
    new_pts, checks, skipped, n_wide = [], [], [], 0
    added = passes = 0
    reason = ""
    while iv.shape[1] and passes < max_passes:
        passes += 1
        pole_lo, pole_hi = iv[6] > 0, iv[7] > 0
        width = iv[1] - iv[0]
        sliver = (pole_lo | pole_hi) & (
            width <= _POLE_REL_WIDTH * np.where(pole_lo, iv[0], iv[1])
        )
        if sliver.any():
            end_seg = np.where(pole_lo, seg[1] - seg[0], seg[-1] - seg[-2])
            skipped += list(zip(iv[0][sliver], width[sliver], strict=True))
            n_wide += int((sliver & (width > end_seg)).sum())
            iv = iv[:, ~sliver]
        if not iv.shape[1]:
            break
        mid = 0.5 * (iv[0] + iv[1])
        splittable = (mid > iv[0]) & (mid < iv[1])
        if not splittable.all():
            reason = reason or "float spacing"
            iv, mid = iv[:, splittable], mid[splittable]
        if not mid.size:
            break
        f1_mid = kk_real_spectrum(mid, seg, coeffs, rel)
        f2_mid = evaluate_f2(mid, seg, coeffs)
        bad = (np.abs(0.5 * (iv[2] + iv[3]) - f1_mid) > tolerance) | (
            np.abs(0.5 * (iv[4] + iv[5]) - f2_mid) > tolerance
        )
        if added + int(bad.sum()) > max_points:
            keep = np.flatnonzero(bad)[: max_points - added]
            bad = np.zeros_like(bad)
            bad[keep] = True
            reason = f"max_points={max_points}"
        checks.append((passes, int(bad.sum()), mid.size))
        added += int(bad.sum())
        new_pts.append(np.vstack([mid, f1_mid, f2_mid])[:, bad])
        if reason.startswith("max_points"):
            break
        zero = np.zeros_like(mid)
        left = np.vstack([iv[0], mid, iv[2], f1_mid, iv[4], f2_mid, iv[6], zero])
        right = np.vstack([mid, iv[1], f1_mid, iv[3], f2_mid, iv[5], zero, iv[7]])
        iv = np.hstack([left[:, bad], right[:, bad]])
    if not reason and iv.shape[1] and passes >= max_passes:
        reason = f"max_passes={max_passes}"
    pts = np.hstack([np.vstack([e, f1, f2]), *new_pts])
    pts = pts[:, np.argsort(pts[0], kind="stable")]
    widest = max(skipped, key=lambda s: s[1]) if skipped else None
    return pts[0], pts[1], pts[2], added, reason, checks, len(skipped), n_wide, widest


# --- golden / baseline -------------------------------------------------------


@pytest.mark.unit
def test_refine_matches_golden_on_device(si) -> None:
    seg, coeffs, rel = si
    with np.load(DATA / "prejax_baseline.npz") as base:
        e_in, f1_in = base["refine_input_ev"], base["refine_input_f1"]
        out = refine_spectrum(
            e_in,
            f1_in,
            evaluate_f2(e_in, seg, coeffs),
            seg,
            coeffs,
            rel,
            tolerance=0.05,
        )
        assert out.n_added == int(base["refine_n_added"]) == 32
        assert not out.exhausted
        np.testing.assert_array_equal(out.energy_ev, base["refine_energy_ev"])
        assert_kk_close(
            out.f1, base["refine_f1"], out.energy_ev, seg, coeffs, rtol=1e-8
        )
        np.testing.assert_allclose(out.f2, base["refine_f2"], rtol=1e-12)
        api = kk_calculate_real("Si", refine_tolerance=0.05)
        assert api.n_refined_points == int(base["refine_api_n_refined"]) == 31
        np.testing.assert_array_equal(api.energy_kev, base["refine_api_energy_kev"])
        np.testing.assert_allclose(api.f2, base["refine_api_f2"], rtol=1e-12)


# --- ordering ----------------------------------------------------------------


@pytest.fixture
def pass_trace(monkeypatch: pytest.MonkeyPatch):
    """Active ``(lo, hi)`` handed on by each pass, read back from the device."""
    outs: list[tuple[np.ndarray, np.ndarray]] = []

    def wrap(real):
        def spy(*args, **kwargs):
            state, status = real(*args, **kwargs)
            n = int(status.n_next_active)
            outs.append((np.asarray(state.lo)[:n], np.asarray(state.hi)[:n]))
            return state, status

        return spy

    monkeypatch.setattr(refine, "_first_pass", wrap(refine._first_pass))
    monkeypatch.setattr(refine, "_refine_pass", wrap(refine._refine_pass))
    return outs


@pytest.mark.unit
@pytest.mark.parametrize("case", ["free", "capped"])
def test_next_active_order_left_halves_then_right_halves(
    si, order, pass_trace, case: str
) -> None:
    kwargs = {"max_points": 10} if case == "capped" else {}
    out = _refine(si, GRID, **kwargs)
    n_passes = int(order[f"{case}_n_passes"])
    assert len(pass_trace) == n_passes
    # The active set of pass k+1 is the output of pass k, in the pre-migration order.
    for k in range(1, n_passes):
        lo, hi = pass_trace[k - 1]
        np.testing.assert_array_equal(lo, order[f"{case}_lo_{k + 1}"])
        np.testing.assert_array_equal(hi, order[f"{case}_hi_{k + 1}"])
        half = lo.size // 2
        # Left halves share their lower end with the parent, right halves their upper.
        assert (lo[half:] == hi[:half]).all()
    np.testing.assert_array_equal(out.energy_ev, order[f"{case}_energy_ev"])
    assert out.n_added == int(order[f"{case}_n_added"])
    assert out.exhausted is bool(order[f"{case}_exhausted"])


# --- limits ------------------------------------------------------------------


@pytest.mark.unit
def test_max_points_strict_inequality_and_exact_equality(si, order, dlog) -> None:
    total = int(order["free_n_added"])
    exact = _refine(si, GRID, max_points=total)
    assert exact.n_added == total
    assert exact.exhausted is False  # added + bad == max_points never exceeds it
    np.testing.assert_array_equal(exact.energy_ev, order["free_energy_ev"])
    one_short = _refine(si, GRID, max_points=total - 1)
    assert one_short.n_added == total - 1
    assert one_short.exhausted is True
    assert set(one_short.energy_ev) < set(order["free_energy_ev"])
    # Pass 1 adds exactly 9: equality does not fire there, pass 2 fires with 0 kept.
    dlog.clear()
    nine = _refine(si, GRID, max_points=9)
    debug, (added, passes), reason = _logged(dlog)
    assert (nine.n_added, added, passes) == (9, 9, 2)
    assert reason == "max_points=9"
    assert debug == [(1, 9, 11), (2, 0, 18)]


@pytest.mark.unit
def test_max_points_fires_on_pass_two_or_later(si, order, dlog) -> None:
    out = _refine(si, GRID, max_points=10)
    debug, (added, passes), reason = _logged(dlog)
    assert (out.n_added, added, passes) == (10, 10, 2)
    assert debug == [(1, 9, 11), (2, 1, 18)]  # one kept of the failing ones, in order
    assert reason == "max_points=10"
    np.testing.assert_array_equal(out.energy_ev, order["capped_energy_ev"])


def _spacing_grid(table):
    """An unsplittable pair (adjacent floats) next to wide intervals needing refinement."""
    e0 = 1000.0
    return np.array([e0, np.nextafter(e0, np.inf), 5000.0, 20000.0])


@pytest.mark.unit
def test_float_spacing_sticky_reason_and_continues(si, dlog) -> None:
    grid = _spacing_grid(si)
    out = _refine(si, grid, tolerance=0.01)
    _, (added, passes), reason = _logged(dlog)
    assert reason == "float spacing"
    assert out.exhausted is True
    assert passes >= 3  # the reason did not stop the other intervals
    assert out.n_added == added > 10
    assert (np.diff(out.energy_ev) > 0).all()
    # Nothing was inserted into the unsplittable pair, the rest refines normally.
    assert not ((out.energy_ev > grid[0]) & (out.energy_ev < grid[1])).any()
    rest = _refine(si, grid[1:], tolerance=0.01)
    np.testing.assert_array_equal(out.energy_ev[1:], rest.energy_ev)


@pytest.mark.unit
@pytest.mark.parametrize("cap", [1, 3, 12])
def test_max_points_overrides_spacing_reason(si, dlog, cap: int) -> None:
    out = _refine(si, _spacing_grid(si), tolerance=0.01, max_points=cap)
    _, _, reason = _logged(dlog)
    assert reason == f"max_points={cap}"
    assert out.n_added == cap


@pytest.mark.unit
def test_max_passes_exhaustion(si, order, dlog) -> None:
    n = int(order["free_n_passes"])
    full = _refine(si, GRID, max_passes=n)
    assert full.exhausted is False  # the last pass found nothing left to split
    short = _refine(si, GRID, max_passes=n - 1)
    assert short.exhausted is True
    _, (_, passes), reason = _logged(dlog)
    assert (passes, reason) == (n - 1, f"max_passes={n - 1}")
    assert short.n_added == full.n_added  # the dropped last pass would add nothing
    assert _refine(si, GRID, max_passes=2).n_added < full.n_added


@pytest.mark.unit
def test_nothing_to_refine(si, dlog) -> None:
    out = _refine(si, GRID, tolerance=1e6)
    debug, (added, passes), reason = _logged(dlog)
    assert (out.n_added, out.exhausted, reason) == (0, False, "")
    assert (added, passes) == (0, 1)
    assert debug == [(1, 0, GRID.size - 1)]
    np.testing.assert_array_equal(out.energy_ev, GRID)
    single = _refine(si, [1000.0])
    assert single.n_added == 0
    assert single.energy_ev.tolist() == [1000.0]


# --- slivers ------------------------------------------------------------------


@pytest.mark.unit
def test_sliver_only_final_pass_no_midpoint_log(si, dlog) -> None:
    seg = si[0]
    grid = np.array([seg[0], seg[0] * 1.01])
    out = _refine(si, grid)
    debug, (added, passes), _ = _logged(dlog)
    assert (out.n_added, out.n_pole_skipped, out.exhausted) == (0, 1, False)
    assert (added, passes) == (0, 1)  # the pass counts ...
    assert debug == []  # ... but has no midpoint line
    # A pass can also drop only slivers after real ones: one narrow segment whose
    # two ends are both singular, so both halves of the first midpoint are slivers.
    dlog.clear()
    table = (np.array([10.0, 10.4]), np.array([[0.0, 1e5, 0.0, 0.0, 0.0]]), 0.0)
    out = _refine(table, [10.0, 10.4])
    debug, (added, passes), _ = _logged(dlog)
    assert (out.n_added, out.n_pole_skipped, out.exhausted) == (1, 2, False)
    assert (added, passes) == (1, 2)
    assert debug == [(1, 1, 1)]  # pass 2 has no midpoint line


@pytest.mark.unit
def test_both_endpoint_slivers(si) -> None:
    seg = si[0]
    tol = 1e-5
    grid = np.array([seg[0], seg[0] * 1.01, 1000.0, seg[-1] * 0.99, seg[-1]])
    out = _refine(si, grid, tolerance=tol, max_passes=200)
    assert out.n_pole_skipped == 2
    assert not ((out.energy_ev > grid[0]) & (out.energy_ev < grid[1])).any()
    assert not ((out.energy_ev > grid[3]) & (out.energy_ev < grid[4])).any()
    assert out.n_added > 10  # the interior still refines
    ref = _host_reference(si, grid, tolerance=tol, max_passes=200)
    np.testing.assert_array_equal(out.energy_ev, ref[0])
    assert ref[6] == 2


def _narrow_last_segment_table():
    """f2 = 1 on three segments; the last real one is 5 eV wide."""
    seg = np.array([10.0, 100.0, 1000.0, 1005.0])
    coeffs = np.tile([0.0, 1.0, 0.0, 0.0, 0.0], (3, 1))
    return seg, coeffs, 0.0


@pytest.mark.unit
def test_widest_sliver_uses_real_width(dlog) -> None:
    table = _narrow_last_segment_table()
    # Sliver 15 eV wide at the upper end: wider than the real last segment (5 eV),
    # narrower than the pad width (the padded table would say 1005 eV).
    grid = np.array([100.0, 990.0, 1005.0])
    out = _refine(table, grid, tolerance=0.5)
    assert out.n_pole_skipped == 1
    warnings = [r.getMessage() for r in dlog.records if r.levelno == logging.WARNING]
    assert any("wider than the tabulated end segment" in w for w in warnings)
    assert any("widest 15 eV at 990 eV" in w for w in warnings)
    ref = _host_reference(table, grid, tolerance=0.5)
    assert (ref[6], ref[7], ref[8]) == (1, 1, (990.0, 15.0))


@pytest.mark.unit
def test_widest_sliver_is_the_widest_dropped_interval(si, dlog) -> None:
    seg = si[0]
    # Lower sliver 0.19 eV (wider than the first segment, 0.1617 eV), upper one wider.
    grid = np.array([seg[0], seg[0] + 0.19, 1000.0, seg[-1] - 8000.0, seg[-1]])
    out = _refine(si, grid, tolerance=1e-5, max_passes=200)
    assert out.n_pole_skipped == 2
    texts = [r.getMessage() for r in dlog.records if "widest" in r.getMessage()]
    assert len(texts) == 1
    assert "widest 8e+03 eV at 492000 eV" in texts[0]
    assert "1 wider than the tabulated end segment" in texts[0]


# --- logging and buffers -----------------------------------------------------


@pytest.mark.unit
def test_per_pass_debug_log_fields_unchanged(si, order, dlog) -> None:
    out = _refine(si, GRID)
    debug, info, reason = _logged(dlog)
    n = int(order["free_n_passes"])
    checked = [order[f"free_lo_{k}"].size for k in range(1, n + 1)]
    accepted = [order[f"free_lo_{k}"].size // 2 for k in range(2, n + 1)] + [0]
    assert debug == [(k + 1, accepted[k], checked[k]) for k in range(n)]
    assert info == (out.n_added, n)
    assert reason == ""
    messages = [r.getMessage() for r in dlog.records if r.name == "xraylabtool.kk"]
    assert (
        messages[0]
        == "refine_spectrum: pass 1 added 9 points (of 11 intervals checked)"
    )
    assert (
        messages[-1]
        == "refine_spectrum: added 139 points in 16 passes (tolerance 0.05)"
    )


def _device_run(table, e, *, min_active, **kwargs):
    """``refine_spectrum`` through ``_refine_device`` with a chosen buffer floor."""
    seg, coeffs, rel = table
    e = np.asarray(e, dtype=np.float64)
    f1, f2 = kk_real_spectrum(e, seg, coeffs, rel), evaluate_f2(e, seg, coeffs)
    pad = 2 ** max(3, (e.size - 1).bit_length()) - e.size
    tolerance = kwargs.pop("tolerance", 0.05)
    max_passes = kwargs.pop("max_passes", 50)
    max_points = kwargs.pop("max_points", 100_000)
    dev = _device.upload(
        (
            np.pad(e, (0, pad)),
            np.pad(f1, (0, pad)),
            np.pad(f2, (0, pad)),
            _device.host_spectrum(seg, coeffs),
            np.float64(rel),
            refine._Knobs(
                np.float64(tolerance), np.int64(max_points), np.int32(e.size)
            ),
        )
    )
    arrays, info = refine._refine_device(
        *dev,
        m=e.size,
        max_passes=max_passes,
        max_points=max_points,
        min_active=min_active,
    )
    return _device.fetch(arrays), info


@pytest.mark.unit
@pytest.mark.parametrize("kwargs", [{}, {"max_points": 10}, {"max_passes": 4}])
def test_refine_buffer_invariance(si, kwargs) -> None:
    seg, coeffs, _ = si
    base, base_info = _device_run(si, GRID, min_active=256, **kwargs)
    for floor in (512, 4096):
        big, info = _device_run(si, GRID, min_active=floor, **kwargs)
        assert info == base_info
        n = info.n_total
        np.testing.assert_array_equal(big.energy[:n], base.energy[:n])
        assert_kk_close(big.f1[:n], base.f1[:n], base.energy[:n], seg, coeffs, 1e-10)
        np.testing.assert_allclose(big.f2[:n], base.f2[:n], rtol=1e-12)
        assert (big.n_sliver, big.n_wide) == (base.n_sliver, base.n_wide)


@pytest.mark.unit
def test_no_padded_point_in_result(si) -> None:
    seg = si[0]
    arrays, info = _device_run(si, GRID, min_active=256)
    n = info.n_total
    assert n == GRID.size + info.added
    assert arrays.energy.size >= n
    assert np.isfinite(arrays.energy[:n]).all()
    assert (np.diff(arrays.energy[:n]) > 0).all()
    assert np.isinf(arrays.energy[n:]).all()  # the padding sorts last as +inf
    out = _refine(si, GRID)
    assert out.energy_ev.size == n
    assert np.isfinite(out.energy_ev).all()
    assert np.isfinite(out.f1).all()
    assert np.isfinite(out.f2).all()
    assert out.energy_ev[0] == GRID[0]
    assert out.energy_ev[-1] == GRID[-1]
    assert (out.energy_ev >= seg[0]).all()
    assert (out.energy_ev <= seg[-1]).all()


@pytest.mark.unit
def test_one_status_fetch_per_pass(si, dlog) -> None:
    seg, coeffs, rel = si
    f1 = kk_real_spectrum(GRID, seg, coeffs, rel)
    f2 = evaluate_f2(GRID, seg, coeffs)
    _device.reset_transfer_counts()
    refine_spectrum(GRID, f1, f2, seg, coeffs, rel, tolerance=0.05)
    counts = _device.transfer_counts()
    _, (_, passes), _ = _logged(dlog)
    # No assembly status without a pipeline: one upload, one fetch per pass, one final.
    assert passes > 2
    assert counts == {"upload": 1, "fetch_status": passes, "fetch": 1}


@pytest.mark.unit
def test_device_loop_runs_under_transfer_guard(si) -> None:
    seg, coeffs, rel = si
    e = GRID
    f1, f2 = kk_real_spectrum(e, seg, coeffs, rel), evaluate_f2(e, seg, coeffs)
    dev = _device.upload(
        (
            np.pad(e, (0, 4)),
            np.pad(f1, (0, 4)),
            np.pad(f2, (0, 4)),
            _device.host_spectrum(seg, coeffs),
            np.float64(rel),
            refine._Knobs(np.float64(0.05), np.int64(100_000), np.int32(e.size)),
        )
    )
    with jax.transfer_guard("disallow_explicit"):
        arrays, info = refine._refine_device(
            *dev, m=e.size, max_passes=50, max_points=100_000
        )
    out = _device.fetch(arrays)
    assert info.added == 139
    assert out.energy[: info.n_total].size == GRID.size + 139


# --- the whole algorithm against the pre-migration loop ----------------------


def _grid_cases():
    return [
        pytest.param({}, GRID, id="plain"),
        pytest.param({"tolerance": 0.02}, GRID, id="tight"),
        pytest.param({"max_points": 1}, GRID, id="cap1"),
        pytest.param({"max_points": 25}, GRID, id="cap25"),
        pytest.param({"max_passes": 1}, GRID, id="passes1"),
        pytest.param({"max_passes": 5}, GRID, id="passes5"),
        pytest.param({"tolerance": 0.01}, "spacing", id="spacing"),
        pytest.param({"tolerance": 0.01, "max_points": 7}, "spacing", id="spacing-cap"),
        pytest.param({"tolerance": 1e-5, "max_passes": 200}, "ends", id="ends"),
        pytest.param({"tolerance": 0.05}, "segments", id="segments"),
        pytest.param({"tolerance": 1e-5, "max_points": 40}, "ends", id="ends-cap"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(("kwargs", "grid"), _grid_cases())
def test_device_matches_pre_migration_loop(si, dlog, kwargs, grid) -> None:
    seg, coeffs, _ = si
    if isinstance(grid, str):
        grid = {
            "spacing": _spacing_grid(si),
            "ends": np.array([seg[0], seg[0] * 1.01, 1000.0, seg[-1] * 0.99, seg[-1]]),
            "segments": seg[:300],
        }[grid]
    kwargs = {"tolerance": 0.05, **kwargs}
    out = _refine(si, grid, **kwargs)
    ref = _host_reference(si, grid, **kwargs)
    debug, (added, _), reason = _logged(dlog)
    np.testing.assert_array_equal(out.energy_ev, ref[0])
    assert_kk_close(out.f1, ref[1], ref[0], seg, coeffs, rtol=1e-10)
    np.testing.assert_allclose(out.f2, ref[2], rtol=1e-12)
    assert (out.n_added, added) == (ref[3], ref[3])
    assert reason == ref[4]
    assert out.exhausted is bool(ref[4])
    assert [(p, a, c) for p, a, c in debug] == ref[5]
    assert out.n_pole_skipped == ref[6]


# --- compile budget -----------------------------------------------------------


@pytest.mark.unit
def test_compile_recorder_sees_a_fresh_function_at_two_shapes() -> None:
    """A silent zero cannot pass: a fresh jit compiled at two shapes is two events."""

    def kk_recorder_probe(x):
        return x * 3.0 + 1.0

    probe = jax.jit(kk_recorder_probe)
    with record_compiles() as names:
        probe(np.ones(3))
        probe(np.ones(5))
        probe(np.ones(3))  # cached: no event
    assert names.count("kk_recorder_probe") == 2


@pytest.mark.unit
def test_refined_sio2_compiles_few_top_level_programs() -> None:
    jax.clear_caches()
    with record_compiles() as names:
        first = kk_calculate_real("SiO2", refine_tolerance=0.01)
    programs = kk_programs(names)
    assert first.n_refined_points > 0
    assert 0 < len(programs) <= 6, programs
    assert {"_first_pass", "_refine_pass", "_finalize"} <= set(programs)
    # A new underscore-named kk program missing from the allow-list fails loudly.
    assert {n for n in names if n.startswith("_")} <= KK_PROGRAM_NAMES
    with record_compiles() as again:
        second = kk_calculate_real("SiO2", refine_tolerance=0.0099)
    assert second.n_refined_points > 0
    assert kk_programs(again) == []  # same buckets, other values: no compile


@pytest.mark.unit
@pytest.mark.parametrize("block", [1, 3, 7, 64, 640, 2048])
def test_midpoint_loop_never_evaluates_more_than_block_per_call(
    si, monkeypatch: pytest.MonkeyPatch, block: int
) -> None:
    seg, coeffs, rel = si
    spec = _device.Spectrum.from_numpy(seg, coeffs)
    probe = np.geomspace(100.0, 20000.0, 200)
    seen: list[int] = []
    real = refine.kk_piecewise_poly

    def spy(chunk, *args):
        seen.append(chunk.shape[0])
        return real(chunk, *args)

    monkeypatch.setattr(refine, "kk_piecewise_poly", spy)
    out = np.asarray(
        refine._midpoint_f1(jax.numpy.asarray(probe), spec, rel, 150, block=block)
    )
    assert seen
    assert max(seen) <= block
    # Only blocks holding active entries run; the rest stays zero.
    assert out[-1] == 0
    # The active prefix agrees with the public kernel.
    assert_kk_close(
        out[:150],
        kk_real_spectrum(probe[:150], seg, coeffs, rel),
        probe[:150],
        seg,
        coeffs,
        1e-10,
    )
