"""Parity of ``refine_spectrum`` with upstream kkcalc ``improve_accuracy`` (commit f331ca31)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.backend.array_ops import JaxBackend, get_backend, set_backend
from xraylabtool.kk.refine import refine_spectrum
from xraylabtool.kk.spectrum import evaluate_f2

GOLDEN = Path(__file__).resolve().parent / "golden"


@pytest.fixture(params=["numpy", "jax"])
def backend_name(request: pytest.FixtureRequest):
    """Run the test under each backend, restoring the original afterwards."""
    if request.param == "jax":
        pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend(request.param)
    yield request.param
    set_backend(original)


# Measured: the refined grids are bit-identical to upstream on both backends;
# f1 differs by at most 6.8e-14 (NumPy) / 9.3e-10 (JAX), 1.5x the rounding scale.
@pytest.mark.parametrize("name", ["si", "sio2"])
def test_matches_upstream_improve_accuracy_golden(name: str, backend_name: str) -> None:
    with np.load(GOLDEN / f"refine_{name}.npz") as g:
        e_in, f1_in = g["input_ev"], g["f1_input"]
        seg, coeffs, rel = g["seg_ev"], g["coeffs"], float(g["rel_corr"])
        tol = float(g["tolerance"])
        out = refine_spectrum(
            e_in,
            f1_in,
            evaluate_f2(e_in, seg, coeffs),
            seg,
            coeffs,
            rel,
            tolerance=tol,
            max_passes=int(g["recursion"]),
        )
        assert out.n_added == g["energy_ev"].size - e_in.size
        assert not out.exhausted
        # Same midpoints (pure arithmetic on the input grid), up to rounding.
        np.testing.assert_allclose(out.energy_ev, g["energy_ev"], rtol=1e-14, atol=0)
        assert_kk_close(
            out.f1,
            g["f1"],
            g["energy_ev"],
            seg,
            coeffs,
            rtol=1e-8 if backend_name == "jax" else 1e-10,
        )
        np.testing.assert_allclose(out.f2, g["f2"], rtol=1e-12)
