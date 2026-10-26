"""Padded ``Spectrum`` and the jnp-only kernel ``kk_f1`` (design spec §2-§4)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.kk import _device
from xraylabtool.kk._shapes import bucket
from xraylabtool.kk.transform import kk_f1, kk_real_spectrum

BASELINE = Path(__file__).resolve().parent / "data" / "prejax_baseline.npz"


def _case(name: str) -> dict[str, np.ndarray]:
    with np.load(BASELINE) as data:
        return {
            "seg": data[f"{name}_seg"],
            "coeffs": data[f"{name}_coeffs"],
            "energy": data[f"{name}_energy_ev"],
            "f1": data[f"{name}_f1"],
            "rel": float(data[f"{name}_rel_corr"]),
        }


def _eval(case: dict[str, np.ndarray], capacity: int, energy: np.ndarray) -> np.ndarray:
    spec = _device.Spectrum.from_numpy(case["seg"], case["coeffs"], capacity)
    assert spec.coeffs.shape == (capacity, 5)
    out = np.asarray(kk_f1(jnp.asarray(energy), spec, case["rel"]))
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    return out


@pytest.fixture
def x64_restored():
    """Restore ``jax_enable_x64`` after a test that switches it off."""
    yield
    jax.config.update("jax_enable_x64", True)


# --- (a) padding is algebraically neutral ------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", ["Au", "SiO2"])
def test_padding_neutral_at_knots(name: str) -> None:
    case = _case(name)
    seg, coeffs = case["seg"], case["coeffs"]
    n = coeffs.shape[0]
    # Baseline energies hold the first/last/interior knots, midpoints and the
    # three x_n(1+k) pad knots; add random ones.
    rng = np.random.default_rng(11)
    extra = np.exp(rng.uniform(np.log(10.0), np.log(5e5), 10))
    energy = np.concatenate([case["energy"], extra])
    k = case["energy"].size
    base = _eval(case, n, energy)
    assert_kk_close(base[:k], case["f1"], case["energy"], seg, coeffs, rtol=1e-8)
    for capacity in (bucket(n), bucket(n + 1), bucket(bucket(n + 1) + 1)):
        out = _eval(case, capacity, energy)
        assert_kk_close(out, base, energy, seg, coeffs, rtol=1e-8)
        assert_kk_close(out[:k], case["f1"], case["energy"], seg, coeffs, rtol=1e-8)


@pytest.mark.unit
def test_padding_at_a_pad_knot_is_finite_and_matches() -> None:
    case = _case("SiO2")
    n = case["coeffs"].shape[0]
    capacity = bucket(n + 3)
    spec = _device.Spectrum.from_numpy(case["seg"], case["coeffs"], capacity)
    pad_knots = np.asarray(spec.seg_ev)[n + 1 : n + 3]
    assert pad_knots[0] == case["seg"][-1] * 2.0
    energy = np.concatenate([pad_knots, case["energy"][:-3]])
    out = _eval(case, capacity, energy)
    plain = _eval(case, n, energy)
    assert_kk_close(out, plain, energy, case["seg"], case["coeffs"], rtol=1e-8)


@pytest.mark.unit
def test_padding_neutral_synthetic() -> None:
    rng = np.random.default_rng(5)
    seg = np.array([2.0, 3.5, 5.0, 8.0, 13.0, 21.0])
    coeffs = rng.uniform(0.5, 2.0, size=(5, 5)) * rng.choice([-1.0, 1.0], size=(5, 5))
    assert np.all(coeffs != 0.0)
    energy = np.concatenate([[1.0, 2.7, 4.2, 6.1, 10.0, 20.0, 50.0, 300.0], seg])
    case = {"seg": seg, "coeffs": coeffs, "rel": 1.5}
    base = _eval(case, 5, energy)
    for capacity in (5, 8, 64, 640):
        np.testing.assert_allclose(_eval(case, capacity, energy), base, rtol=1e-12)
    pad = _device.Spectrum.from_numpy(seg, coeffs, 64).seg_ev
    assert pad[6] == 21.0 * 2.0  # first pad knot x_n(1+k), k = 1, at index n+1
    assert pad[7] == 21.0 * 3.0  # second pad knot, k = 2
    # Pad knots as evaluation energies leave the real-segment sum untouched.
    knots = np.asarray(pad)[6:12]
    ref = kk_real_spectrum(knots, seg, coeffs, 1.5)
    np.testing.assert_allclose(_eval(case, 64, knots), ref, rtol=1e-12)


# --- (b) jit / vmap -------------------------------------------------------------


@pytest.mark.unit
def test_kk_f1_jit_vmap() -> None:
    case = _case("SiO2")
    energy = jnp.asarray(case["energy"])
    spec = _device.Spectrum.from_numpy(case["seg"], case["coeffs"])
    eager = kk_f1(energy, spec, case["rel"], block=64)
    jitted = jax.jit(kk_f1, static_argnames="block")(
        energy, spec, case["rel"], block=64
    )
    np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))

    # Two spectra, equal capacity, different true segment counts.
    cap = 16
    rng = np.random.default_rng(3)
    specs, rels = [], [0.5, 1.0]
    for n in (5, 9):
        seg = np.cumsum(rng.uniform(0.5, 2.0, n + 1)) + 1.0
        c = rng.uniform(-1.0, 1.0, (n, 5))
        specs.append(_device.Spectrum.from_numpy(seg, c, cap))
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *specs)
    assert stacked.n_segments.tolist() == [5, 9]
    e = jnp.asarray([0.7, 3.3, 5.5, 9.9, 20.0])
    batched = jax.vmap(lambda s, r: kk_f1(e, s, r))(stacked, jnp.asarray(rels))
    for i, (s, r) in enumerate(zip(specs, rels, strict=True)):
        np.testing.assert_allclose(
            np.asarray(batched[i]), np.asarray(kk_f1(e, s, r)), rtol=1e-13
        )


# --- (c) gradients --------------------------------------------------------------


@pytest.mark.unit
def test_kk_f1_grad() -> None:
    n, cap = 4, 8
    seg = np.array([2.0, 3.5, 5.0, 8.0, 13.0])
    rng = np.random.default_rng(9)
    c0 = rng.uniform(0.5, 1.5, (n, 5)) * rng.choice([-1.0, 1.0], (n, 5))
    host = _device.host_spectrum(seg, c0, cap)
    seg_p = jnp.asarray(host.seg_ev)
    c_pad = jnp.asarray(host.coeffs) + jnp.where(
        jnp.arange(cap)[:, None] >= n, 0.7, 0.0
    )
    energy = jnp.asarray([1.2, 2.8, 4.1, 6.5, 10.0, 25.0])

    def loss(c: jax.Array, e: jax.Array) -> jax.Array:
        return kk_f1(e, _device.Spectrum(seg_p, c, jnp.int32(n)), 0.3).sum()

    g_c, g_e = jax.grad(loss, argnums=(0, 1))(c_pad, energy)
    assert np.isfinite(np.asarray(g_c)).all()
    assert np.isfinite(np.asarray(g_e)).all()
    np.testing.assert_array_equal(np.asarray(g_c)[n:], 0.0)  # masked padded rows

    steps = np.array([1e-6, 1e-6, 1e-6, 1e-5, 1e-5])  # per basis term
    fd = np.zeros((n, 5))
    base = np.asarray(c_pad)
    for i in range(n):
        for j in range(5):
            hi, lo = base.copy(), base.copy()
            hi[i, j] += steps[j]
            lo[i, j] -= steps[j]
            fd[i, j] = (
                float(loss(jnp.asarray(hi), energy))
                - float(loss(jnp.asarray(lo), energy))
            ) / (2 * steps[j])
    np.testing.assert_allclose(np.asarray(g_c)[:n], fd, rtol=1e-6, atol=1e-7)

    h = 1e-6 * np.asarray(energy)
    fd_e = np.array(
        [
            (
                float(loss(c_pad, energy.at[k].add(h[k])))
                - float(loss(c_pad, energy.at[k].add(-h[k])))
            )
            / (2 * h[k])
            for k in range(energy.size)
        ]
    )
    np.testing.assert_allclose(np.asarray(g_e), fd_e, rtol=1e-6, atol=1e-7)


# --- (d) dtype and shapes -------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("m", [0, 1, 7, 8, 63, 64, 65, 213])
@pytest.mark.parametrize("block", [None, 8])
def test_output_float64_and_shapes(m: int, block: int | None) -> None:
    case = _case("SiO2")
    spec = _device.Spectrum.from_numpy(case["seg"], case["coeffs"])
    energy = case["energy"][:m]
    out = kk_f1(jnp.asarray(energy), spec, case["rel"], block=block)
    assert out.dtype == jnp.float64
    assert out.shape == (m,)
    if m:
        assert_kk_close(
            np.asarray(out), case["f1"][:m], energy, case["seg"], case["coeffs"],
            rtol=1e-8,
        )  # fmt: skip


@pytest.mark.unit
def test_kk_f1_keeps_input_shape_and_rejects_bad_block() -> None:
    case = _case("Au")
    spec = _device.Spectrum.from_numpy(case["seg"], case["coeffs"])
    grid = jnp.asarray(case["energy"][:12].reshape(3, 4))
    assert kk_f1(grid, spec, case["rel"]).shape == (3, 4)
    with pytest.raises(ValueError, match="block"):
        kk_f1(grid, spec, case["rel"], block=0)


# --- (e) x64 policy ---------------------------------------------------------------


@pytest.mark.unit
def test_x64_restored_by_entry(x64_restored) -> None:
    case = _case("SiO2")
    expected = kk_real_spectrum(
        case["energy"], case["seg"], case["coeffs"], case["rel"]
    )
    jax.config.update("jax_enable_x64", False)
    out = kk_real_spectrum(
        case["energy"].astype(np.float32),
        case["seg"].astype(np.float32),
        case["coeffs"].astype(np.float32),
        np.float32(case["rel"]),
    )
    assert jax.config.jax_enable_x64
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    assert out.shape == expected.shape


# --- Spectrum container ------------------------------------------------------------


@pytest.mark.unit
def test_spectrum_roundtrip_and_pad_rule() -> None:
    seg = np.array([2.0, 5.0, 9.0])
    coeffs = np.arange(10.0).reshape(2, 5)
    spec = _device.Spectrum.from_numpy(seg, coeffs, 6)
    assert spec.seg_ev.shape == (7,)
    assert spec.n_segments.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(spec.seg_ev)[3:], 9.0 * np.arange(2, 6))
    np.testing.assert_array_equal(np.asarray(spec.coeffs)[2:], 0.0)
    out_seg, out_c = spec.to_numpy()
    np.testing.assert_array_equal(out_seg, seg)
    np.testing.assert_array_equal(out_c, coeffs)
    leaves, tree = jax.tree_util.tree_flatten(spec)
    assert len(leaves) == 3
    assert isinstance(jax.tree_util.tree_unflatten(tree, leaves), _device.Spectrum)
    with pytest.raises(ValueError, match="capacity"):
        _device.Spectrum.from_numpy(seg, coeffs, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        _device.Spectrum.from_numpy(seg[::-1], coeffs)


@pytest.mark.unit
def test_transfer_helpers_are_counted() -> None:
    _device.reset_transfer_counts()
    tree = _device.upload({"a": np.arange(3.0)})
    _device.fetch_status(tree)
    _device.fetch(tree)
    assert _device.transfer_counts() == {"upload": 1, "fetch_status": 1, "fetch": 1}
    _device.reset_transfer_counts()
    assert sum(_device.transfer_counts().values()) == 0
    case = _case("SiO2")
    kk_real_spectrum(case["energy"], case["seg"], case["coeffs"], case["rel"])
    assert _device.transfer_counts() == {"upload": 1, "fetch_status": 0, "fetch": 1}


@pytest.mark.unit
def test_x64_restored_by_entry_on_empty_input(x64_restored) -> None:
    case = _case("SiO2")
    jax.config.update("jax_enable_x64", False)
    out = kk_real_spectrum(np.array([]), case["seg"], case["coeffs"], case["rel"])
    assert jax.config.jax_enable_x64
    assert out.dtype == np.float64
    assert out.shape == (0,)
