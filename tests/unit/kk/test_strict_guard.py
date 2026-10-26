"""Warmed device entries run under ``transfer_guard("disallow_explicit")``; single-flight loads.

Spec §1 (``2026-10-25-kk-jax-first-design.md``): the device-only region makes
no host/device transfer of its own. Inputs are uploaded through the boundary
helper first, the entry is called once to warm it, and a second call must
pass the strict guard (a Python scalar passed to a jitted function, or an eager
op on a device array with a Python constant, would be an implicit transfer).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import jax
import numpy as np
import pytest

from xraylabtool.kk import _device
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk.database import load_asf
from xraylabtool.kk.nearedge import merge_device, to_asf_scale_device
from xraylabtool.kk.spectrum import compound_spectrum_device, evaluate_f2_device
from xraylabtool.kk.transform import kk_f1, relativistic_correction

ROOT = Path(__file__).resolve().parents[3]


def _warm_then_guarded(call: Callable[[], Any]) -> None:
    jax.block_until_ready(call())
    with jax.transfer_guard("disallow_explicit"):
        jax.block_until_ready(call())


@pytest.mark.unit
def test_compound_spectrum_device_under_strict_guard() -> None:
    db = load_asf()
    tables = db.device_tables()
    z, n, ne = _device.upload(
        (np.array([14, 8], np.int32), np.array([1.0, 2.0]), np.int32(2))
    )
    cap = db.compound_capacity([14, 8])
    _warm_then_guarded(lambda: compound_spectrum_device(z, n, ne, tables, capacity=cap))


@pytest.mark.unit
def test_evaluate_f2_device_under_strict_guard() -> None:
    spec = Spectrum.from_numpy([2.0, 5.0, 9.0], np.ones((2, 5)))
    e = _device.upload(np.array([2.5, 6.0]))
    _warm_then_guarded(lambda: evaluate_f2_device(e, spec))


@pytest.mark.unit
def test_relativistic_correction_device_arrays_under_strict_guard() -> None:
    z, n = _device.upload((np.array([14.0, 8.0]), np.array([1.0, 2.0])))
    _warm_then_guarded(lambda: relativistic_correction(z, n))
    expected = relativistic_correction([14, 8], [1.0, 2.0])
    got = float(_device.fetch(relativistic_correction(z, n)))
    assert got == pytest.approx(expected, rel=1e-15)


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["nexafs", "beta", "asf"])
def test_to_asf_scale_device_under_strict_guard(kind: str) -> None:
    e, v, rho, mass = _device.upload(
        (
            np.array([280.0, 300.0, 320.0, 340.0]),
            np.array([0.1, 0.4, 0.8, 0.9]),
            2.0,
            60.0,
        )
    )
    _warm_then_guarded(
        lambda: to_asf_scale_device(e, v, kind, density=rho, formula_mass=mass)
    )
    _warm_then_guarded(lambda: to_asf_scale_device(e, v, kind))  # defaults


@pytest.mark.unit
def test_merge_device_under_strict_guard() -> None:
    seg = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    coeffs = np.random.default_rng(7).uniform(0.5, 2.0, (4, 5))
    ne_e = np.linspace(8.0, 55.0, 48)
    ne_v = 0.2 + 0.8 / (1.0 + np.exp(-(ne_e - 31.0) / 3.0))
    spec = Spectrum.from_numpy(seg, coeffs)
    e, v, n, low, high = _device.upload(
        (ne_e, ne_v, np.int32(ne_e.size), np.float64(22.0), np.float64(38.0))
    )
    _warm_then_guarded(lambda: merge_device(spec, e, v, n, low, high, capacity=512))


@pytest.mark.unit
def test_cold_concurrent_first_call_uploads_the_tables_once() -> None:
    """Fresh process, barrier-released threads each doing ``load_asf().device_tables()``."""
    script = textwrap.dedent(
        """
        import threading
        from xraylabtool.kk import _device
        from xraylabtool.kk.database import load_asf

        n = 8
        barrier = threading.Barrier(n)
        got = []

        def run():
            barrier.wait()
            got.append(load_asf().device_tables())

        before = _device.transfer_counts()["upload"]
        threads = [threading.Thread(target=run) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(_device.transfer_counts()["upload"] - before, all(t is got[0] for t in got), len(got))
        """
    )
    env = {"JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**env, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "1 True 8"


@pytest.mark.unit
def test_host_to_asf_scale_beta_passes_the_strict_guard() -> None:
    from xraylabtool.kk.nearedge import NearEdgeSpectrum, to_asf_scale

    near = NearEdgeSpectrum([280.0, 300.0, 320.0], [0.1, 0.4, 0.8])

    def call() -> Any:
        return to_asf_scale(near, "beta", density=2.0, formula_mass=60.0)

    call()
    with jax.transfer_guard("disallow_explicit"):
        out = call()
    assert np.isfinite(out.values).all()


@pytest.mark.unit
@pytest.mark.parametrize("shape", [(0,), (3,), (2, 5), (700,)])
def test_public_kk_f1_under_strict_guard(shape: tuple[int, ...]) -> None:
    """The bare public ``kk_f1`` (pad, constants and slice included) makes no transfer of its own."""
    spec = _device.upload(Spectrum.from_numpy([2.0, 5.0, 9.0], np.ones((2, 5))))
    e, rel = _device.upload(
        (np.linspace(2.5, 8.5, int(np.prod(shape))).reshape(shape), np.float64(0.0))
    )  # a Python float rel_corr would be a caller-side implicit transfer
    _warm_then_guarded(lambda: kk_f1(e, spec, rel))
    got = _device.fetch(kk_f1(e, spec, rel))
    assert got.shape == shape
