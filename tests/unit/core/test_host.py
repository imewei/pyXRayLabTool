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
import os
import xraylabtool

os.environ.pop("JAX_ENABLE_X64", None)  # after the package import, before jax
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
        host.execute(sneaky, np.ones(2))


def test_thread_counts_ignore_other_threads():
    before = runtime.thread_transfer_counts()
    worker = threading.Thread(target=lambda: runtime.upload(np.ones(2)))
    worker.start()
    worker.join()
    assert runtime.thread_transfer_counts() == before
    assert runtime.transfer_counts()["upload"] >= 1


def test_host_tables_are_rejected(monkeypatch):
    on_host = jax.tree_util.tree_map(np.asarray, load_tables())
    monkeypatch.setattr(host, "load_tables", lambda: on_host)
    with pytest.raises(TypeError, match="device arrays"):
        host.run_optical([optics.encode_formula("SiO2")], [2.2], _ENERGIES)


def test_pack_materials_places_every_column_where_the_program_reads_it():
    encoded = [optics.encode_formula("SiO2"), optics.encode_formula("H")]
    rows = optics.pack_materials(encoded, [2.2, 0.07], f_pad=4, k_pad=4)
    k = 4
    assert rows.shape == (4, optics.row_width(k))
    z, counts, mw, n_el = encoded[0]
    np.testing.assert_array_equal(rows[0, : len(z)], z)
    np.testing.assert_array_equal(rows[0, len(z) : k], 0.0)  # padded elements
    np.testing.assert_array_equal(rows[0, k : k + len(counts)], counts)
    assert rows[0, 2 * k : 2 * k + 3].tolist() == [mw, 2.2, n_el]
    assert rows[1, 2 * k + 1] == 0.07
    # padded materials: mw = rho = 1, no electrons, no elements
    assert rows[3].tolist() == [0.0] * (2 * k) + [1.0, 1.0, 0.0]


def test_a_batch_of_mixed_materials_equals_each_material_alone_bit_for_bit():
    formulas = ["H", "SiO2", "Al2O3", "UO2", "C6H12O6Ca2Fe"]  # K = 1..5, Z up to 92
    densities = [0.07, 2.2, 3.95, 10.9, 1.5]
    encoded = [optics.encode_formula(f) for f in formulas]
    together = host.run_optical(encoded, densities, _ENERGIES)
    for i, (enc, rho) in enumerate(zip(encoded, densities, strict=True)):
        alone = host.run_optical([enc], [rho], _ENERGIES)
        np.testing.assert_array_equal(together["out"][i], alone["out"][0])
        assert together["electron_density"][i] == alone["electron_density"][0]
        assert bool(together["ok"][i]) is bool(alone["ok"][0]) is True
