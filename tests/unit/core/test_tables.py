"""Resident scattering tables: shipped npz integrity and the ``load_tables`` loader.

Spec section 2 / plan steps 1.2 and 1.3. The npz is derived from the ``.nff``
files by ``scripts/gen_scattering_tables.py``; ``--check`` fails on drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gen_scattering_tables.py"
NPZ = REPO / "xraylabtool" / "data" / "scattering_tables.npz"
NFF_DIR = REPO / "xraylabtool" / "data" / "AtomicScatteringFactor"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_scattering_tables", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shipped():
    with np.load(NPZ) as f:
        return {k: f[k] for k in f.files}


def test_regenerated_equals_shipped(gen, shipped):
    built = gen.build_tables()
    assert sorted(built) == sorted(shipped)
    for key, value in built.items():
        assert value.dtype == shipped[key].dtype, key
        if value.dtype.kind == "f":
            # XLA rounds the slope arithmetic differently across platforms in the last ulps
            np.testing.assert_allclose(
                value, shipped[key], rtol=1e-13, atol=0.0, err_msg=key
            )
        else:
            np.testing.assert_array_equal(value, shipped[key], err_msg=key)


def test_check_flag_passes_on_the_shipped_file():
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr


def test_check_flag_fails_on_drift(gen, shipped, tmp_path):
    bad = dict(shipped)
    bad["f1"] = shipped["f1"].copy()
    bad["f1"][3, 0] += 1e-9
    out = tmp_path / "drifted.npz"
    np.savez(out, **bad)
    assert gen.main(["--check", "--out", str(out)]) == 1


def test_shapes_and_dtypes(shipped):
    n = shipped["n"]
    width = int(n.max())
    assert width == 449
    for key in ("x", "f1", "f2", "d1", "d2"):
        assert shipped[key].shape == (92, width), key
        assert shipped[key].dtype == np.float64, key
    assert n.shape == (92,)
    assert n.dtype == np.int32
    assert shipped["symbols"].shape == (92,)


def test_symbols_are_atomic_number_order_and_match_files(shipped):
    from xraylabtool.data_handling.atomic_cache import _ATOMIC_DATA_PRELOADED

    symbols = [str(s) for s in shipped["symbols"]]
    numbers = [_ATOMIC_DATA_PRELOADED[s]["atomic_number"] for s in symbols]
    assert numbers == list(range(1, 93))
    assert set(symbols) == {p.stem.capitalize() for p in NFF_DIR.glob("*.nff")}


def test_padding_knots_strictly_increase_and_pad_values(shipped):
    for z, m in enumerate(shipped["n"]):
        x = shipped["x"][z]
        assert (np.diff(x) > 0).all(), z
        for key in ("f1", "f2"):
            assert (shipped[key][z, m:] == shipped[key][z, m - 1]).all(), (z, key)
        for key in ("d1", "d2"):
            assert (shipped[key][z, m:] == 0).all(), (z, key)


def test_slopes_match_scipy_pchip(shipped):
    from scipy.interpolate import PchipInterpolator

    for z, m in enumerate(shipped["n"]):
        x = shipped["x"][z, :m]
        for f, d in (("f1", "d1"), ("f2", "d2")):
            ref = PchipInterpolator(x, shipped[f][z, :m]).derivative()(x)
            np.testing.assert_allclose(
                shipped[d][z, :m], ref, rtol=1e-12, atol=1e-14, err_msg=f"{z} {d}"
            )


def _hermite(x, y, d, n, q):
    """Cubic Hermite from the stored (padded) row, as the device program evaluates it."""
    i = np.clip(np.searchsorted(x, q, side="right") - 1, 0, n - 2)
    h = x[i + 1] - x[i]
    t = (q - x[i]) / h
    t2 = t * t
    t3 = t2 * t
    return (
        (2 * t3 - 3 * t2 + 1) * y[i]
        + (t3 - 2 * t2 + t) * h * d[i]
        + (3 * t2 - 2 * t3) * y[i + 1]
        + (t3 - t2) * h * d[i + 1]
    )


def test_padded_rows_evaluate_like_the_unpadded_pchip(shipped):
    """All 92 rows: a cubic Hermite built from the STORED x, f, d (padding included)
    equals SciPy's PCHIP on the unpadded row over [x[0], x[n-1]].

    Queries are every knot, every midpoint and 777 uniform points. The bound is
    rtol 1e-12 plus an absolute 1e-14 of the row's largest |value| (measured
    worst case: 6e-14 relative at a near-zero value, <1e-14 of the scale).
    """
    from scipy.interpolate import PchipInterpolator

    for z, n in enumerate(shipped["n"]):
        n = int(n)
        x = shipped["x"][z]
        q = np.concatenate(
            [x[:n], 0.5 * (x[: n - 1] + x[1:n]), np.linspace(x[0], x[n - 1], 777)]
        )
        for f, d in (("f1", "d1"), ("f2", "d2")):
            y = shipped[f][z]
            ref = PchipInterpolator(x[:n], y[:n])(q)
            got = _hermite(x, y, shipped[d][z], n, q)
            np.testing.assert_allclose(
                got,
                ref,
                rtol=1e-12,
                atol=1e-14 * np.abs(y[:n]).max(),
                err_msg=f"row {z} {f}",
            )
            # the knots themselves are reproduced exactly
            np.testing.assert_array_equal(got[:n], y[:n], err_msg=f"row {z} {f}")


# ---------------------------------------------------------------- loader


@pytest.fixture
def tables_mod():
    from xraylabtool.calculators import tables

    tables.clear_tables()
    yield tables
    tables.clear_tables()


def test_load_tables_uploads_once(tables_mod):
    import jax

    from xraylabtool.backend import runtime

    runtime.reset_transfer_counts()
    first = tables_mod.load_tables()
    for _ in range(5):
        assert tables_mod.load_tables() is first
    assert runtime.transfer_counts()["upload"] == 1
    assert first.x.dtype == np.float64
    assert first.n.dtype == np.int32
    assert first.x.shape == (92, 449)
    assert isinstance(first.x, jax.Array)


def test_load_tables_single_upload_from_threads(tables_mod):
    from xraylabtool.backend import runtime

    runtime.reset_transfer_counts()
    barrier = threading.Barrier(8)
    seen: list[object] = []

    def work() -> None:
        barrier.wait()
        seen.append(tables_mod.load_tables())

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 8
    assert all(t is seen[0] for t in seen)
    assert runtime.transfer_counts()["upload"] == 1


def test_clear_tables_forces_reload(tables_mod):
    from xraylabtool.backend import runtime

    first = tables_mod.load_tables()
    runtime.reset_transfer_counts()
    tables_mod.clear_tables()
    second = tables_mod.load_tables()
    assert second is not first
    assert runtime.transfer_counts()["upload"] == 1
    np.testing.assert_array_equal(np.asarray(second.x), np.asarray(first.x))


def test_tables_are_a_pytree_without_symbols_leaf(tables_mod):
    import jax

    t = tables_mod.load_tables()
    leaves = jax.tree_util.tree_leaves(t)
    assert len(leaves) == 7
    assert t.symbols[0] == "H"
    assert t.symbols[91] == "U"
    assert len(t.symbols) == 92


def test_element_index(tables_mod):
    from xraylabtool.exceptions import UnknownElementError

    assert tables_mod.element_index("H") == 0
    assert tables_mod.element_index("Si") == 13
    assert tables_mod.element_index("U") == 91
    with pytest.raises(UnknownElementError, match="Unknown element symbol: 'Xx'"):
        tables_mod.element_index("Xx")


@pytest.mark.parametrize("symbol", ["H", "C", "Si", "Fe", "Au", "U"])
def test_packed_coefficients_reproduce_scipy_pchip(tables_mod, symbol):
    """The per-interval cubics evaluate like SciPy's PCHIP on the real knots (rtol 1e-12)."""
    from scipy.interpolate import PchipInterpolator

    from xraylabtool.calculators.cache import create_scattering_factor_interpolators

    with np.load(tables_mod._NPZ) as f:
        z = list(f["symbols"]).index(symbol)
        n = int(f["n"][z])
        x, y1, y2 = f["x"][z, :n], f["f1"][z, :n], f["f2"][z, :n]
    q = np.random.default_rng(0).uniform(x[0], x[-1], 400)
    got1, got2 = (
        np.asarray(g(q)) for g in create_scattering_factor_interpolators(symbol)
    )
    for got, y in ((got1, y1), (got2, y2)):
        want = PchipInterpolator(x, y)(q)
        np.testing.assert_allclose(
            got, want, rtol=1e-12, atol=1e-12 * np.max(np.abs(y))
        )
