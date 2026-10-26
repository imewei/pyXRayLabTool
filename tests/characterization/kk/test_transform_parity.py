"""Kernel parity with upstream kkcalc ``KK_PP`` (golden outputs, commit f331ca31)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.kk.transform import kk_real_spectrum

GOLDEN = Path(__file__).resolve().parent / "golden"
TRANSFORM_CASES = ["synthetic", "si", "au", "sio2", "ar", "ne", "c8h8_nearedge"]
# Tolerance rtol·|f1| + K·rounding scale (tests/fixtures/kk_rounding.py).
# Measured worst absolute error 6.0e-5 (Au at 13.1 eV, |f1| = 0.020, where the
# summed terms reach 7.9e12), i.e. 2.1x the rounding scale.
RTOL = 1e-8


@pytest.mark.parametrize("name", TRANSFORM_CASES)
def test_matches_upstream_kk_pp_golden(name: str) -> None:
    with np.load(GOLDEN / f"{name}.npz") as g:
        args = (g["eval_ev"], g["seg_ev"], g["coeffs"])
        out = kk_real_spectrum(*args, float(g["rel_corr"]))
        expected = g["f1_kk_pp"]
    assert_kk_close(out, expected, *args, rtol=RTOL)
