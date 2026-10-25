"""Kernel parity with upstream kkcalc ``KK_PP`` (golden outputs, commit f331ca31)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.backend.array_ops import JaxBackend, get_backend, set_backend
from xraylabtool.kk.transform import kk_real_spectrum

GOLDEN = Path(__file__).resolve().parent / "golden"
TRANSFORM_CASES = ["synthetic", "si", "au", "sio2", "ar", "ne", "c8h8_nearedge"]
# Tolerance rtol·|f1| + K·rounding scale (tests/fixtures/kk_rounding.py).
# Measured: NumPy worst relative error 1.2e-12 (Ne, |f1| = 4.4e-3); JAX
# worst absolute 6.0e-5 (Au at 13.1 eV, |f1| = 0.020, where the summed terms
# reach 7.9e12), i.e. 2.1x the rounding scale.
RTOL = {"numpy": 1e-10, "jax": 1e-8}


@pytest.fixture(params=["numpy", "jax"])
def backend_name(request: pytest.FixtureRequest):
    """Run the test under each backend, restoring the original afterwards."""
    if request.param == "jax":
        pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend(request.param)
    yield request.param
    set_backend(original)


@pytest.mark.parametrize("name", TRANSFORM_CASES)
def test_matches_upstream_kk_pp_golden(name: str, backend_name: str) -> None:
    with np.load(GOLDEN / f"{name}.npz") as g:
        args = (g["eval_ev"], g["seg_ev"], g["coeffs"])
        out = kk_real_spectrum(*args, float(g["rel_corr"]))
        expected = g["f1_kk_pp"]
    assert_kk_close(out, expected, *args, rtol=RTOL[backend_name])
