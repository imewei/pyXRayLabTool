"""Transfer, compile and bookkeeping budgets of the ``kk_calculate_real`` device pipeline.

Spec §1 (one upload, one status fetch after assembly/merge, one per refinement
pass, one final download), §8 (energy bookkeeping), §10 (threads, x64) and the
compile rows of the acceptance table. Counters, not the transfer guard, enforce
the budget on CPU (no fetch is ever guarded there); the guard test proves that
nothing crosses the boundary outside the ``_device`` helpers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading

import jax
import numpy as np
import pytest

from tests.fixtures.kk_compiles import kk_programs, numerical_programs, record_compiles
from xraylabtool.backend.array_ops import JaxBackend, get_backend, set_backend
from xraylabtool.exceptions import CalculationError, NearEdgeDataError
from xraylabtool.kk import (
    _device,
    _shapes,
    api,
    database,
    nearedge,
    refine,
    transform,
)
from xraylabtool.kk._device import Spectrum
from xraylabtool.kk.nearedge import NearEdgeSpectrum

ROOT = Path(__file__).resolve().parents[3]
EDGE = NearEdgeSpectrum(
    [280.0, 290.0, 300.0, 310.0, 320.0], [0.05, 0.1, 0.3, 0.45, 0.5]
)
INFO_PASSES = "refine_spectrum: added"

# Call shapes of the budget: (formula, kwargs).
CALLS = {
    "explicit": ("SiO2", {"energies_kev": np.geomspace(0.05, 20.0, 64)}),
    "default": ("SiO2", {}),
    "refined": ("SiO2", {"refine_tolerance": 0.05}),
    "refined_explicit": (
        "Si",
        {"energies_kev": np.geomspace(0.05, 20.0, 40), "refine_tolerance": 0.05},
    ),
    "nearedge": (
        "C",
        {"nearedge": EDGE, "input_type": "nexafs", "merge_points_kev": (0.28, 0.32)},
    ),
    "nearedge_refined": (
        "C",
        {
            "nearedge": EDGE,
            "input_type": "nexafs",
            "merge_points_kev": (0.28, 0.32),
            "refine_tolerance": 0.05,
        },
    ),
}


def _run(name: str, **extra):
    formula, kwargs = CALLS[name]
    return api.kk_calculate_real(formula, **kwargs, **extra)


def _passes(caplog: pytest.LogCaptureFixture) -> int:
    """Refinement passes of the call, from its INFO summary (0 without refinement)."""
    for record in caplog.records:
        if record.getMessage().startswith(INFO_PASSES):
            return int(record.getMessage().split(" in ")[1].split(" passes")[0])
    return 0


@pytest.fixture
def x64_restored():
    """Restore ``jax_enable_x64`` after a test that switches it off."""
    yield
    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def backend_restored():
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    yield
    set_backend(original)


def _subprocess(code: str, timeout: float = 240, **env_extra: str) -> dict:
    """Run ``code`` in a fresh CPU-only interpreter; it prints one JSON line last."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "PYTHONPATH": str(ROOT),
        **env_extra,
    }
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# --- transfer budget ---------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", list(CALLS))
def test_pipeline_under_disallow_explicit_guard(name: str) -> None:
    """Every crossing goes through a ``_device`` helper (each opens its own scope)."""
    expected = _run(name)  # warm: compiles and uploads the tables outside the guard
    with jax.transfer_guard("disallow_explicit"):
        guarded = _run(name)
    np.testing.assert_array_equal(guarded.f1, expected.f1)
    np.testing.assert_array_equal(guarded.energy_kev, expected.energy_kev)


@pytest.mark.unit
@pytest.mark.parametrize("name", list(CALLS))
def test_transfer_budget_warm_call(name: str, kk_log: pytest.LogCaptureFixture) -> None:
    _run(name)  # warm
    kk_log.clear()
    _device.reset_transfer_counts()
    _run(name)
    counts = _device.transfer_counts()
    assert counts == {
        "upload": 1,
        "fetch_status": 1 + _passes(kk_log),
        "fetch": 1,
    }


_CONVERSIONS = (
    "__array__",
    "__float__",
    "__int__",
    "__bool__",
    "__index__",
    "item",
    "tolist",
    "block_until_ready",
)
_NUMPY_CONVERSIONS = ("asarray", "array", "asanyarray", "ascontiguousarray")


def _outside_device_helpers() -> bool:
    """True unless the nearest ``xraylabtool`` frame on the stack is ``kk._device``."""
    frame = sys._getframe(2)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith("xraylabtool"):
            return module != "xraylabtool.kk._device"
        frame = frame.f_back
    return True


@pytest.fixture
def stray_conversions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Names of host conversions of a device array made outside ``kk._device``.

    ``jax.transfer_guard`` does not stop ``np.asarray`` or ``float`` on a device
    array on the CPU platform, and ``np.asarray`` reads the buffer directly
    without calling ``ArrayImpl.__array__``. The probe therefore wraps both the
    array methods and the NumPy constructors that package code calls.
    """
    array_type = type(jax.numpy.zeros(1))
    found: list[str] = []

    def patch_method(name: str) -> None:
        original = getattr(array_type, name)

        def wrapper(self, *args, **kwargs):
            if _outside_device_helpers():
                found.append(name)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(array_type, name, wrapper)

    def patch_numpy(name: str) -> None:
        original = getattr(np, name)

        def wrapper(obj, *args, **kwargs):
            if isinstance(obj, jax.Array) and _outside_device_helpers():
                found.append(f"np.{name}")
            return original(obj, *args, **kwargs)

        monkeypatch.setattr(np, name, wrapper)

    for name in _CONVERSIONS:
        patch_method(name)
    for name in _NUMPY_CONVERSIONS:
        patch_numpy(name)
    return found


@pytest.mark.unit
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("name", list(CALLS))
def test_no_host_conversion_outside_the_device_helpers(
    name: str,
    cold: bool,
    stray_conversions: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(name)  # compile outside the probe
    stray_conversions.clear()
    if cold:
        monkeypatch.setattr(database, "_tables_cache", {})
    result = _run(name)
    assert stray_conversions == []
    assert result.f1.size > 0


@pytest.mark.unit
def test_conversion_probe_sees_a_stray_conversion(
    stray_conversions: list[str],
) -> None:
    """The probe has teeth: a bare ``np.asarray`` on a device array is recorded."""
    np.asarray(jax.numpy.arange(3.0))
    float(jax.numpy.float64(2.0))
    assert stray_conversions == ["np.asarray", "__float__"]


@pytest.mark.unit
def test_cold_table_upload_is_counted_once_and_separately(monkeypatch) -> None:
    monkeypatch.setattr(database, "_tables_cache", {})
    _device.reset_transfer_counts()
    _run("explicit")
    assert _device.transfer_counts()["upload"] == 2  # tables + inputs
    _device.reset_transfer_counts()
    _run("explicit")
    assert _device.transfer_counts()["upload"] == 1


# --- compile budget ----------------------------------------------------------


@pytest.mark.unit
def test_second_call_no_recompile() -> None:
    """Same buckets, different values: no kk program compiles again."""
    energies = np.geomspace(0.05, 20.0, 64)
    api.kk_calculate_real("SiO2", energies_kev=energies)
    api.kk_calculate_real("SiO2", energies_kev=energies, refine_tolerance=0.05)
    api.kk_calculate_real("C", nearedge=EDGE, input_type="nexafs")
    with record_compiles() as events:
        res = api.kk_calculate_real("SiO2", energies_kev=energies[::-1] * 0.99)
        refined = api.kk_calculate_real(
            "SiO2", energies_kev=energies, refine_tolerance=0.05
        )
        edge = NearEdgeSpectrum(EDGE.energy_ev, EDGE.values * 2.0 + 0.01)
        api.kk_calculate_real("C", nearedge=edge, input_type="nexafs")
    assert res.f1.size == 64
    assert refined.n_refined_points > 0
    assert kk_programs(events) == []
    assert numerical_programs(events) == []


@pytest.mark.unit
def test_energy_length_bucket_recompiles_only_the_evaluation() -> None:
    """The output grid stays out of ``_assemble``: only ``_evaluate`` is shape-specific."""
    small = np.geomspace(0.05, 20.0, 100)
    large = np.geomspace(0.05, 20.0, 3000)
    api.kk_calculate_real("SiO2", energies_kev=small)
    api.kk_calculate_real("SiO2", energies_kev=large)  # compile the large bucket too
    with record_compiles() as events:
        api.kk_calculate_real("SiO2", energies_kev=small * 1.01)
        api.kk_calculate_real("SiO2", energies_kev=large * 0.99)
    assert kk_programs(events) == []
    api.kk_calculate_real("C", energies_kev=small)
    with record_compiles() as events:
        api.kk_calculate_real("C", energies_kev=large)
    assert kk_programs(events) == ["_evaluate"]


@pytest.mark.unit
def test_kernel_evaluates_only_blocks_holding_real_energies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Padded blocks cost nothing: 50 real energies in 4 blocks of 40 run 2 blocks."""
    spectrum = Spectrum.from_numpy(
        np.array([10.0, 100.0, 1000.0, 5000.0]),
        np.array([[0.0, 1.0, 0.0, 0.0, 0.0]] * 3),
        512,
    )
    calls: list[int] = []
    original = transform.kk_piecewise_poly

    def counted(e, seg, coeffs, rel):
        jax.debug.callback(lambda: calls.append(1))
        return original(e, seg, coeffs, rel)

    monkeypatch.setattr(transform, "kk_piecewise_poly", counted)
    padded = jax.numpy.asarray(np.linspace(20.0, 900.0, 160))
    out = transform._kk_f1_blocks(padded, spectrum, 0.0, 50, block=40)
    jax.block_until_ready(out)
    assert len(calls) == 2
    assert np.all(np.asarray(out)[50:] == 0.0)
    assert np.all(np.isfinite(np.asarray(out)[:50]))


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "energies_kev": np.geomspace(0.05, 29.0, 3100)
        },  # two blocks on an accelerator
        {"energies_kev": np.geomspace(0.05, 29.0, 3100), "chunk_size": 500},
        {},
        {"refine_tolerance": 0.05},
    ],
    ids=["explicit-3100", "chunk-500", "default-grid", "refined"],
)
def test_accelerator_block_rule_gives_the_same_result(
    kwargs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GPU shape path (fine-ladder blocks) runs in CI by forcing the predicate."""
    cpu = api.kk_calculate_real("SiO2", **kwargs)
    chunk, m = kwargs.get("chunk_size"), cpu.energy_kev.size
    cpu_shape = _shapes.energy_blocks(chunk, 1248, m)
    monkeypatch.setattr(_shapes, "_on_cpu", lambda: False)
    assert _shapes.energy_blocks(chunk, 1248, m) != cpu_shape
    accelerator = api.kk_calculate_real("SiO2", **kwargs)
    np.testing.assert_array_equal(accelerator.energy_kev, cpu.energy_kev)
    np.testing.assert_allclose(accelerator.f1, cpu.f1, rtol=1e-12, atol=1e-11)
    np.testing.assert_array_equal(accelerator.f2, cpu.f2)


_BUDGET_SCRIPT = """
    import json
    from tests.fixtures.kk_compiles import kk_programs, numerical_programs, record_compiles
    with record_compiles() as events:
        from xraylabtool.kk import kk_calculate_real
        res = kk_calculate_real("SiO2", refine_tolerance=0.01)
    print(json.dumps({"kk": kk_programs(events), "all": numerical_programs(events),
                      "n": int(res.energy_kev.size)}))
"""

# The refined default run of the acceptance table: <= 6 top-level programs.
BUDGET = 6
EXPECTED_PROGRAMS = [
    "_assemble",
    "_evaluate",
    "_finalize",
    "_first_pass",
    "_refine_pass",
]


@pytest.mark.unit
def test_refined_run_program_budget() -> None:
    """Fresh interpreter: a cold refined SiO2 run compiles only the budgeted programs."""
    out = _subprocess(_BUDGET_SCRIPT)
    assert out["n"] > 1000
    assert len(out["kk"]) <= BUDGET
    assert sorted(out["kk"]) == EXPECTED_PROGRAMS
    # No unlisted numerical program hides behind the trivial-primitive filter.
    assert sorted(out["all"]) == sorted(out["kk"])


# --- energy bookkeeping ------------------------------------------------------


@pytest.mark.unit
def test_unsorted_duplicate_energies_order_preserved() -> None:
    requested = np.array([5.0, 0.1, 5.0, 2.0, 0.1])
    res = api.kk_calculate_real("Si", energies_kev=requested)
    np.testing.assert_array_equal(res.energy_kev, requested)
    assert res.f1[0] == res.f1[2]
    assert res.f1[1] == res.f1[4]
    single = api.kk_calculate_real("Si", energies_kev=[2.0])
    np.testing.assert_allclose(res.f1[3], single.f1[0], rtol=1e-12)


def _colliding_pair() -> tuple[float, float]:
    """Two distinct keV floats that are the same energy in eV."""
    for k in np.linspace(1.5, 1.99, 5000):
        other = float(np.nextafter(k, np.inf))
        if other != k and other * 1000.0 == k * 1000.0:
            return float(k), other
    raise AssertionError("no colliding keV pair found")


@pytest.mark.unit
@pytest.mark.parametrize("reverse", [False, True])
def test_ev_collision_first_representative_kept(reverse: bool) -> None:
    first, second = _colliding_pair()
    pair = [second, first] if reverse else [first, second]
    res = api.kk_calculate_real(
        "Si", energies_kev=[0.5, *pair, 9.0], refine_tolerance=0.5
    )
    kept = pair[0]
    assert kept in res.energy_kev
    assert pair[1] not in res.energy_kev
    assert (np.diff(res.energy_kev) > 0).all()


@pytest.mark.unit
def test_requested_kev_bit_exact() -> None:
    rng = np.random.default_rng(7)
    requested = np.sort(rng.uniform(0.04, 25.0, 33))
    plain = api.kk_calculate_real("SiO2", energies_kev=requested)
    np.testing.assert_array_equal(plain.energy_kev, requested)
    refined = api.kk_calculate_real(
        "SiO2", energies_kev=requested, refine_tolerance=0.05
    )
    assert np.isin(requested, refined.energy_kev).all()
    default = api.kk_calculate_real("SiO2", refine_tolerance=0.2)
    assert (np.diff(default.energy_kev) > 0).all()


@pytest.mark.unit
def test_default_grid_is_exact_ev_over_1000() -> None:
    res = api.kk_calculate_real("SiO2")
    db = database.load_asf()
    knots = np.unique(np.concatenate([db.energy_ev(14), db.energy_ev(8)]))
    window = knots[(knots >= 30.0) & (knots <= 30000.0)]
    np.testing.assert_array_equal(res.energy_kev, window / 1000.0)


# --- backend / x64 -----------------------------------------------------------


@pytest.mark.unit
def test_backend_field_is_jax() -> None:
    assert api.kk_calculate_real("Si", energies_kev=[1.0]).backend == "jax"


@pytest.mark.unit
def test_set_backend_numpy_does_not_affect_kk(backend_restored) -> None:
    energies = np.geomspace(0.05, 20.0, 30)
    set_backend("jax")
    reference = api.kk_calculate_real("SiO2", energies_kev=energies)
    set_backend("numpy")
    res = api.kk_calculate_real("SiO2", energies_kev=energies)
    assert res.backend == "jax"
    np.testing.assert_array_equal(res.f1, reference.f1)
    np.testing.assert_array_equal(res.f2, reference.f2)


@pytest.mark.unit
def test_x64_disabled_after_import_still_float64(x64_restored) -> None:
    expected = api.kk_calculate_real("SiO2", refine_tolerance=0.05)
    jax.config.update("jax_enable_x64", False)
    res = api.kk_calculate_real("SiO2", refine_tolerance=0.05)
    assert jax.config.jax_enable_x64
    assert res.f1.dtype == np.float64
    assert res.f2.dtype == np.float64
    assert res.energy_kev.dtype == np.float64
    np.testing.assert_array_equal(res.f1, expected.f1)


@pytest.mark.unit
def test_runtime_is_logged_once_per_process(
    monkeypatch, kk_log: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(api, "_runtime_logged", threading.Event())
    api.kk_calculate_real("Si", energies_kev=[1.0])
    api.kk_calculate_real("Si", energies_kev=[2.0])
    lines = [r for r in kk_log.records if "jax " in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert jax.__version__ in lines[0].getMessage()


# --- threads -----------------------------------------------------------------


def _threaded(n: int, fn) -> list:
    """Run ``fn(i)`` on ``n`` threads released together; each opens its own guard."""
    barrier = threading.Barrier(n)
    out: list = [None] * n

    def work(i: int) -> None:
        barrier.wait()
        # The guard is thread-local: each worker establishes its own scope.
        with jax.transfer_guard("disallow_explicit"):
            try:
                out[i] = fn(i)
            except BaseException as exc:  # surfaced in the main thread
                out[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for item in out:
        if isinstance(item, BaseException):
            raise item
    return out


@pytest.mark.unit
def test_threaded_calls_agree() -> None:
    """Warm: four concurrent calls are bit-equal to the serial call."""
    energies = np.geomspace(0.05, 20.0, 50)
    serial = api.kk_calculate_real("SiO2", energies_kev=energies)
    serial_refined = api.kk_calculate_real("SiO2", refine_tolerance=0.05)
    results = _threaded(
        4,
        lambda i: (
            api.kk_calculate_real("SiO2", energies_kev=energies),
            api.kk_calculate_real("SiO2", refine_tolerance=0.05),
        ),
    )
    for plain, refined in results:
        np.testing.assert_array_equal(plain.f1, serial.f1)
        np.testing.assert_array_equal(refined.energy_kev, serial_refined.energy_kev)
        np.testing.assert_array_equal(refined.f1, serial_refined.f1)


_COLD_THREADS_SCRIPT = """
    import json, threading
    import jax
    import numpy as np
    from tests.fixtures.kk_compiles import kk_programs, record_compiles
    from xraylabtool.kk import _device, database, kk_calculate_real

    packs = []
    original = database.AsfDatabase._pack_host
    def counted(self):
        packs.append(1)
        return original(self)
    database.AsfDatabase._pack_host = counted

    jax.config.update("jax_enable_x64", False)  # off after import: entries re-enable it
    energies = np.geomspace(0.05, 20.0, 50)
    barrier = threading.Barrier(4)
    out = [None] * 4
    def work(i):
        barrier.wait()
        with jax.transfer_guard("disallow_explicit"):
            try:
                out[i] = kk_calculate_real("SiO2", energies_kev=energies)
            except BaseException as exc:
                out[i] = exc
    with record_compiles() as events:
        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    errors = [repr(o) for o in out if isinstance(o, BaseException)]
    print(json.dumps({
        "errors": errors,
        "packs": len(packs),
        "uploads": _device.transfer_counts()["upload"],
        "equal": not errors and all(np.array_equal(o.f1, out[0].f1) for o in out),
        "dtype": "" if errors else str(out[0].f1.dtype),
        "kk": sorted(kk_programs(events)),
    }))
"""


@pytest.mark.unit
def test_cold_concurrent_first_calls_upload_the_tables_once() -> None:
    out = _subprocess(_COLD_THREADS_SCRIPT)
    assert out["errors"] == []
    assert out["packs"] == 1  # one table build, however many first callers
    assert out["uploads"] == 1 + 4  # the tables once, then one input upload each
    assert out["equal"]
    assert out["dtype"] == "float64"
    assert out["kk"] == ["_assemble", "_evaluate"]  # each compiled once, not per thread


_QT_SCRIPT = """
    import json, os, time
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import jax
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QThreadPool
    from xraylabtool.gui.services import compute_kk
    from xraylabtool.gui.workers import CalculationWorker

    app = QCoreApplication([])
    jax.config.update("jax_enable_x64", False)
    pool = QThreadPool()
    pool.setMaxThreadCount(4)
    results, errors, workers = [], [], []
    for _ in range(4):
        w = CalculationWorker(compute_kk, "SiO2", None, None, "eV", None, 0.05)
        w.setAutoDelete(False)
        workers.append(w)  # the signals die with their worker
        w.signals.finished.connect(results.append)
        w.signals.error.connect(errors.append)
        pool.start(w)
    pool.waitForDone(120000)
    deadline = time.time() + 30
    while len(results) + len(errors) < 4 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    print(json.dumps({
        "errors": [e[:300] for e in errors],
        "n": len(results),
        "equal": all(
            np.array_equal(r.result.f1, results[0].result.f1) for r in results
        ),
        "dtype": str(results[0].result.f1.dtype) if results else "",
    }))
"""


@pytest.mark.unit
def test_qthreadpool_workers_agree() -> None:
    pytest.importorskip("PySide6")
    # The worker entry opens its own (thread-local) disallow_explicit scope.
    out = _subprocess(_QT_SCRIPT, XRAYLABTOOL_TRANSFER_GUARD="1")
    assert out["errors"] == []
    assert out["n"] == 4
    assert out["equal"]
    assert out["dtype"] == "float64"


# --- block size ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_chunk_size_is_upper_bound(monkeypatch, chunk: int) -> None:
    """The block the drivers see never exceeds the request (kernel and refinement)."""
    seen: list[int] = []
    for owner, attr in (
        (api, "_evaluate"),
        (refine, "_first_pass"),
        (refine, "_refine_pass"),
    ):
        original = getattr(owner, attr)

        def spy(*args, __original=original, **kwargs):
            seen.append(kwargs["block"])
            return __original(*args, **kwargs)

        monkeypatch.setattr(owner, attr, spy)
    energies = np.geomspace(0.05, 20.0, 12)
    api.kk_calculate_real(
        "Si", energies_kev=energies, chunk_size=chunk, refine_tolerance=0.5
    )
    assert seen
    assert max(seen) <= chunk
    plain = api.kk_calculate_real("Si", energies_kev=energies)
    chunked = api.kk_calculate_real("Si", energies_kev=energies, chunk_size=chunk)
    np.testing.assert_allclose(chunked.f1, plain.f1, rtol=1e-9, atol=1e-9)


# --- capacity overflow and error precedence -----------------------------------


@pytest.mark.unit
def test_undersized_capacity_raises_instead_of_truncating(monkeypatch) -> None:
    host = _device.host_spectrum([2.0, 5.0, 9.0], np.zeros((2, 5)), 2)
    over = Spectrum(host.seg_ev, host.coeffs, np.int32(7))
    with pytest.raises(CalculationError, match=r"capacity 2.*7|7.*capacity 2"):
        over.to_numpy()

    monkeypatch.setattr(api, "_compound_capacity", lambda n: 64)
    with pytest.raises(CalculationError, match="capacity 64"):
        api.kk_calculate_real("SiO2", energies_kev=[1.0, 2.0])
    # With near-edge data the merge capacity is sized separately and can hide a
    # truncated compound spectrum, so the compound count is checked on its own.
    with pytest.raises(CalculationError, match="capacity 64"):
        _run("nearedge")


@pytest.mark.unit
def test_merge_wrapper_raises_on_undersized_capacity(monkeypatch) -> None:
    from xraylabtool.kk.spectrum import compound_spectrum

    original = nearedge.merge_device

    def small(*args, capacity, **kwargs):
        return original(*args, capacity=16, **kwargs)

    monkeypatch.setattr(nearedge, "merge_device", small)
    seg, coeffs = compound_spectrum(["C"], [1.0])
    with pytest.raises(CalculationError, match="capacity 16"):
        nearedge.merge_spectra(EDGE, seg, coeffs, merge_points_ev=(280.0, 320.0))


@pytest.mark.unit
def test_nearedge_error_precedes_energy_error() -> None:
    """A data error in the merge wins over bad energies, as before the rewrite."""
    flat = NearEdgeSpectrum([280.0, 290.0, 300.0], [0.5, 0.7, 0.5])
    with pytest.raises(NearEdgeDataError, match="flat endpoints"):
        api.kk_calculate_real("C", energies_kev=[1e6], nearedge=flat, input_type="asf")


@pytest.mark.unit
def test_merge_points_reported_are_the_used_values() -> None:
    res = api.kk_calculate_real(
        "C", nearedge=EDGE, input_type="nexafs", merge_points_kev=(0.28, 0.32)
    )
    assert res.merge_points_kev == (0.28, 0.32)
    whole = api.kk_calculate_real("C", nearedge=EDGE, input_type="nexafs")
    assert whole.merge_points_kev == (0.28, 0.32)


@pytest.mark.unit
@pytest.mark.parametrize("energies", [[1.0], [1.0, 1.0], [0.03], [500.0, 500.0]])
def test_single_point_grid_with_refinement(energies: list[float]) -> None:
    """A grid of one unique energy reaches the refinement as a length-1 array."""
    plain = api.kk_calculate_real("Si", energies_kev=energies)
    refined = api.kk_calculate_real("Si", energies_kev=energies, refine_tolerance=0.05)
    assert refined.n_refined_points == 0
    unique = np.unique(np.asarray(energies) * 1000.0)
    assert refined.energy_kev.size == unique.size
    np.testing.assert_array_equal(refined.f1, plain.f1[: unique.size])
