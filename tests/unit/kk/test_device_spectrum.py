"""Device-resident compound spectrum, f2 evaluation and Z* correction.

Spec §5 and Verification 3-4 (``2026-10-25-kk-jax-first-design.md``). The
pre-migration spectra in ``data/prejax_baseline.npz`` are the reference.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xraylabtool.kk._device import Spectrum, fetch, upload
from xraylabtool.kk._shapes import _next_bucket, bucket, bucket_pow2
from xraylabtool.kk.database import DeviceTables, load_asf
from xraylabtool.kk.spectrum import compound_spectrum_device, evaluate_f2_device
from xraylabtool.kk.transform import relativistic_correction

BASELINE = Path(__file__).parent / "data" / "prejax_baseline.npz"
FORMULAS: dict[str, tuple[list[str], list[float]]] = {
    "H": (["H"], [1.0]),
    "Mo": (["Mo"], [1.0]),
    "U": (["U"], [1.0]),
    "SiO2": (["Si", "O"], [1.0, 2.0]),
    "Fe2O3": (["Fe", "O"], [2.0, 3.0]),
}


def _z(symbols: list[str]) -> list[int]:
    return [load_asf().symbols.index(s) + 1 for s in symbols]


def _assemble(
    symbols: list[str], counts: list[float], capacity: int | None = None
) -> Spectrum:
    db = load_asf()
    z = _z(symbols)
    cap = db.compound_capacity(z) if capacity is None else capacity
    k = bucket_pow2(len(z))
    z_pad = np.ones(k, np.int32)
    z_pad[: len(z)] = z
    n_pad = np.zeros(k)
    n_pad[: len(z)] = counts
    z_d, n_d, ne_d = upload((z_pad, n_pad, np.int32(len(z))))
    return compound_spectrum_device(z_d, n_d, ne_d, db.device_tables(), capacity=cap)


def _poly(c: np.ndarray, e: float) -> float:
    return float(c[0] * e + c[1] + c[2] / e + c[3] / e**2 + c[4] / e**3)


@pytest.mark.unit
@pytest.mark.parametrize("case", list(FORMULAS))
def test_device_matches_baseline(case: str) -> None:
    symbols, counts = FORMULAS[case]
    seg, coeffs = _assemble(symbols, counts).to_numpy()
    with np.load(BASELINE) as base:
        ref_seg, ref_coeffs = base[f"{case}_seg"], base[f"{case}_coeffs"]
    assert seg.shape == ref_seg.shape
    assert coeffs.shape == ref_coeffs.shape
    np.testing.assert_allclose(seg, ref_seg, rtol=1e-13)
    np.testing.assert_allclose(coeffs, ref_coeffs, rtol=1e-13, atol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("case", ["H", "U", "SiO2"])
def test_capacity_independent(case: str) -> None:
    symbols, counts = FORMULAS[case]
    cap = load_asf().compound_capacity(_z(symbols))
    assert cap == bucket(cap)
    nxt = _next_bucket(cap)
    ref = _assemble(symbols, counts, cap).to_numpy()
    for bigger in (nxt, _next_bucket(nxt)):
        got = _assemble(symbols, counts, bigger)
        assert got.seg_ev.shape == (bigger + 1,)
        assert np.isfinite(np.asarray(got.seg_ev)).all()
        for a, b in zip(got.to_numpy(), ref, strict=True):
            assert np.array_equal(a, b)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbols", "counts"),
    [
        (["H"], [1.0]),
        (["Mo"], [1.0]),
        (["U"], [1.0]),
        (["H", "O", "H"], [2.0, 1.0, 1.0]),
    ],
)
def test_union_no_dummy_or_fill_leak(symbols: list[str], counts: list[float]) -> None:
    db = load_asf()
    spec = _assemble(symbols, counts)
    real = np.unique(np.concatenate([db.energy_ev(z) for z in _z(symbols)]))
    seg, coeffs = spec.to_numpy()
    assert np.array_equal(seg, real)
    full = np.asarray(spec.seg_ev)
    assert np.isfinite(full).all()
    assert (np.diff(full) > 0).all()
    n = int(spec.n_segments)
    # Pad rule x_n * (1 + k) and exactly-zero pad coefficients.
    np.testing.assert_array_equal(
        full[n + 1 :], full[n] * (1.0 + np.arange(1, full.size - n))
    )
    assert not np.asarray(spec.coeffs)[n:].any()
    assert np.isfinite(coeffs).all()


@pytest.mark.unit
def test_repeated_element_equals_summed_count() -> None:
    twice = _assemble(["H", "O", "H"], [2.0, 1.0, 1.0]).to_numpy()
    once = _assemble(["H", "O"], [3.0, 1.0]).to_numpy()
    assert np.array_equal(twice[0], once[0])
    np.testing.assert_allclose(twice[1], once[1], rtol=1e-14, atol=0.0)


@pytest.mark.unit
def test_count_within_capacity() -> None:
    db = load_asf()
    sample = ["H", "C", "O", "Si", "Fe", "Mo", "Au", "U"]
    for i, a in enumerate(sample):
        for b in sample[i:]:
            cap = db.compound_capacity(_z([a, b]))
            spec = _assemble([a, b], [1.0, 1.0])
            union = np.unique(np.concatenate([db.energy_ev(z) for z in _z([a, b])]))
            assert int(spec.n_segments) == union.size - 1
            assert int(spec.n_segments) <= cap
            assert spec.coeffs.shape == (cap, 5)


@pytest.mark.unit
def test_compound_capacity_is_bucketed_segment_sum() -> None:
    db = load_asf()
    mo, u = db.energy_ev(42).size, db.energy_ev(92).size
    assert db.compound_capacity([42, 92]) == bucket(mo - 1 + u - 1)
    assert db.compound_capacity([1]) == bucket(db.energy_ev(1).size - 1)


@pytest.mark.unit
def test_device_tables_layout_and_single_flight() -> None:
    db = load_asf()
    tables = db.device_tables()
    assert isinstance(tables, DeviceTables)
    assert db.device_tables() is tables
    kmax = max(db.energy_ev(z).size for z in range(1, 93))
    assert tables.knots.shape == (92, kmax)
    assert tables.coeffs.shape == (92, kmax - 1, 5)
    assert tables.n_knots.shape == (92,)
    assert tables.n_knots.dtype == jnp.int32
    knots = np.asarray(tables.knots)
    assert np.isfinite(knots).all()
    assert (np.diff(knots, axis=1) > 0).all()
    for z in (1, 42, 92):
        n = db.energy_ev(z).size
        assert int(tables.n_knots[z - 1]) == n
        assert np.array_equal(knots[z - 1, :n], db.energy_ev(z))
        assert not np.asarray(tables.coeffs[z - 1, n - 1 :]).any()


def _reference_f2(seg: np.ndarray, coeffs: np.ndarray, e: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(seg, e, side="right") - 1, 0, coeffs.shape[0] - 1)
    return np.array([_poly(coeffs[i], x) for i, x in zip(idx, e, strict=True)])


@pytest.mark.unit
def test_evaluate_f2_device_edges() -> None:
    with np.load(BASELINE) as base:
        seg, coeffs = base["Ar_seg"], base["Ar_coeffs"]
    n = coeffs.shape[0]
    spec = Spectrum.from_numpy(seg, coeffs)
    assert spec.coeffs.shape[0] > n  # padding present

    # The tables are continuous (no value jump anywhere); the largest mismatch
    # between the two polynomials at a shared edge is the Ar K-edge knot, about
    # 4e-12 relative, which still tells a right-continuous lookup (the segment
    # starting at the knot) from a left-continuous one.
    jump = max(
        range(1, n),
        key=lambda j: abs(_poly(coeffs[j - 1], seg[j]) - _poly(coeffs[j], seg[j])),
    )
    right = _poly(coeffs[jump], seg[jump])
    left = _poly(coeffs[jump - 1], seg[jump])
    assert abs(left - right) > 1e-12 * abs(right)

    probes = np.array(
        [
            seg[0],  # first knot
            seg[jump],  # jump knot: right-continuous
            seg[n // 2],  # interior knot
            seg[-1],  # last real knot uses the last real polynomial
            *(seg[-1] * (1.0 + np.arange(1, 4))),  # pad knots: never selected
            0.5 * (seg[0] + seg[1]),
        ]
    )
    got = fetch(evaluate_f2_device(upload(probes), spec))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[1], right, rtol=1e-13)
    assert abs(got[1] - right) < abs(got[1] - left)
    np.testing.assert_allclose(got[3], _poly(coeffs[n - 1], seg[-1]), rtol=1e-13)
    for pad_value, x in zip(got[4:7], probes[4:7], strict=True):
        np.testing.assert_allclose(pad_value, _poly(coeffs[n - 1], x), rtol=1e-13)
    np.testing.assert_allclose(got, _reference_f2(seg, coeffs, probes), rtol=1e-13)

    one = fetch(evaluate_f2_device(upload(probes[:1]), spec))
    assert one.shape == (1,)
    empty = fetch(evaluate_f2_device(upload(np.empty(0)), spec))
    assert empty.shape == (0,)


@pytest.mark.unit
def test_evaluate_f2_device_is_jittable_and_bucket_independent() -> None:
    with np.load(BASELINE) as base:
        seg, coeffs = base["SiO2_seg"], base["SiO2_coeffs"]
        e = base["SiO2_energy_ev"]
    n = coeffs.shape[0]
    f = jax.jit(evaluate_f2_device)
    ref = fetch(f(upload(e), Spectrum.from_numpy(seg, coeffs)))
    bigger = fetch(
        f(upload(e), Spectrum.from_numpy(seg, coeffs, _next_bucket(bucket(n))))
    )
    assert np.array_equal(ref, bigger)
    np.testing.assert_allclose(ref, _reference_f2(seg, coeffs, e), rtol=1e-13)


@pytest.mark.unit
def test_relativistic_correction_device() -> None:
    z, n = [14, 8], [1.0, 2.0]
    host = relativistic_correction(z, n)
    assert isinstance(host, float)
    dev = relativistic_correction(jnp.asarray(z), jnp.asarray(n))
    assert isinstance(dev, jax.Array)
    np.testing.assert_allclose(float(dev), host, rtol=1e-15)
    traced = jax.jit(relativistic_correction)(jnp.asarray(z), jnp.asarray(n))
    np.testing.assert_allclose(float(traced), host, rtol=1e-15)
    # Zero-count padding contributes nothing.
    padded = relativistic_correction(
        jnp.asarray([14, 8, 1, 1]), jnp.asarray([1.0, 2.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(float(padded), host, rtol=1e-15)
    with pytest.raises(ValueError, match="same length"):
        relativistic_correction(jnp.asarray([14, 8]), jnp.asarray([1.0]))
    with pytest.raises(ValueError, match="same length"):
        relativistic_correction([14, 8], [1.0])


@pytest.mark.unit
def test_evaluate_f2_device_rounds_like_upstream_at_cancellation() -> None:
    # Si at 1838.8 eV: a1*E + a0 cancels to 0.37 from terms of 1.7e4, so a
    # fused multiply-add would differ from upstream's NumPy rounding by 3e-12.
    db = load_asf()
    seg, coeffs = db.energy_ev(14), db.im_coeffs(14)
    e = np.array([1838.8])
    got = fetch(evaluate_f2_device(upload(e), Spectrum.from_numpy(seg, coeffs)))
    np.testing.assert_allclose(got, _reference_f2(seg, coeffs, e), rtol=1e-15)


@pytest.mark.unit
def test_device_tables_concurrent_first_call_uploads_once() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from xraylabtool.kk import database
    from xraylabtool.kk._device import transfer_counts

    db = load_asf()
    cached = dict(database._tables_cache)  # restored below
    database._tables_cache.clear()
    try:
        before = transfer_counts()["upload"]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda _: db.device_tables(), range(8)))
        assert all(t is got[0] for t in got)
        assert transfer_counts()["upload"] == before + 1
    finally:
        database._tables_cache.clear()
        database._tables_cache.update(cached)
