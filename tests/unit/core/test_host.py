"""Host wrappers of the device programs: x64 policy, transfer budget, table check."""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import jax
import numpy as np
import pytest

from xraylabtool.backend import runtime
from xraylabtool.calculators import host, optics
from xraylabtool.calculators.tables import ScatteringTables, load_tables

_ENERGIES = np.linspace(1.0, 20.0, 50)

_NO_ENV_PROBE = """
import numpy as np
import xraylabtool
import jax

assert not jax.config.jax_enable_x64, "the package must not set x64 at import"
from xraylabtool.calculators import host, optics

got = host.run_optical([optics.encode_formula("SiO2")], [2.2], [8.0, 9.0])
assert got["out"].dtype == np.float64, got["out"].dtype
assert jax.config.jax_enable_x64
print("OK")
"""


def test_wrappers_enable_x64_without_the_environment_variable():
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    done = subprocess.run(
        [sys.executable, "-c", _NO_ENV_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("OK")


def test_wrapper_runs_under_an_outer_explicit_transfer_guard():
    host.run_optical([optics.encode_formula("SiO2")], [2.2], _ENERGIES)  # warm
    with jax.transfer_guard("disallow_explicit"):
        got = host.run_optical([optics.encode_formula("SiO2")], [2.2], _ENERGIES)
    assert got["ok"].all()


def test_wrapper_budget_is_one_upload_and_one_fetch_per_call():
    encoded = [optics.encode_formula("SiO2")]
    host.run_optical(encoded, [2.2], _ENERGIES)  # warm: tables upload, compile
    before = runtime.thread_transfer_counts()
    host.run_optical(encoded, [2.2], _ENERGIES)
    after = runtime.thread_transfer_counts()
    assert after["upload"] - before["upload"] == 1
    assert after["fetch"] - before["fetch"] == 1
    assert after["fetch_status"] == before["fetch_status"]


def test_an_extra_upload_inside_the_program_call_is_rejected():
    def sneaky(dev):
        runtime.upload(np.ones(1))
        return dev

    with pytest.raises(RuntimeError, match="1 upload and 1 fetch"):
        host._run(sneaky, np.ones(2))


def test_thread_counts_ignore_other_threads():
    before = runtime.thread_transfer_counts()
    worker = threading.Thread(target=lambda: runtime.upload(np.ones(2)))
    worker.start()
    worker.join()
    assert runtime.thread_transfer_counts() == before
    assert runtime.transfer_counts()["upload"] >= 1


def test_host_tables_are_rejected(monkeypatch):
    on_host = ScatteringTables(
        *(np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(load_tables())),
        symbols=load_tables().symbols,
    )
    monkeypatch.setattr(host, "load_tables", lambda: on_host)
    with pytest.raises(TypeError, match="device arrays"):
        host.run_optical([optics.encode_formula("SiO2")], [2.2], _ENERGIES)
