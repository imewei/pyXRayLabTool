"""GUI core compute path: idle warm-up, worker guard, QThreadPool agreement (serial).

``test_gui_*`` files run outside xdist (Makefile ``GUI_TESTS``, ``ci.yml``
``IGNORE_GUI``): they share one Qt application and one global thread pool.
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
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication
import pytest

from xraylabtool.backend import runtime
from xraylabtool.gui import services
from xraylabtool.gui.services import EnergyConfig, compute_multiple, compute_single
from xraylabtool.gui.workers import CalculationWorker

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "dispersion_delta",
    "absorption_beta",
    "critical_angle_degrees",
    "attenuation_length_cm",
)
CASES = [("SiO2", 2.2), ("Au", 19.32), ("H2O", 1.0), ("Si", 2.33), ("Fe2O3", 5.24)]


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


def test_worker_threads_agree_with_the_calling_thread(app: QApplication) -> None:
    """Twenty jobs on four pool threads equal the same calls made here, bit for bit."""
    cfg = EnergyConfig(5.0, 12.0, 37, False)
    expected = {c: compute_single(*c, cfg) for c in CASES}
    got: dict[int, Any] = {}
    failures: list[BaseException] = []

    def job(index: int, case: tuple[str, float]) -> None:
        try:
            got[index] = compute_single(*case, cfg)
        except BaseException as exc:  # reported through the assert below
            failures.append(exc)

    pool = QThreadPool()
    pool.setMaxThreadCount(4)
    jobs = [(i, CASES[i % len(CASES)]) for i in range(20)]
    for index, case in jobs:
        pool.start(CalculationWorker(job, index, case))
    assert pool.waitForDone(120_000)

    assert not failures, failures
    assert sorted(got) == list(range(20))
    for index, case in jobs:
        for field in FIELDS:
            np.testing.assert_array_equal(
                getattr(got[index], field), getattr(expected[case], field)
            )


def test_multiple_materials_on_a_worker_match_direct_calls(app: QApplication) -> None:
    cfg = EnergyConfig(8.0, 12.0, 5, False)
    out: list[Any] = []
    pool = QThreadPool()
    pool.setMaxThreadCount(2)
    pool.start(
        CalculationWorker(
            lambda: out.append(compute_multiple(["SiO2", "Au"], [2.2, 19.32], cfg))
        )
    )
    assert pool.waitForDone(60_000)
    direct = compute_multiple(["SiO2", "Au"], [2.2, 19.32], cfg)
    assert list(out[0]) == list(direct)
    for name, result in direct.items():
        np.testing.assert_array_equal(
            out[0][name].dispersion_delta, result.dispersion_delta
        )


def test_compute_runs_inside_the_worker_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service opens the guard; the host wrapper's own guard nests inside it."""
    depths: list[int] = []
    open_scopes = [0]

    class Probe:
        def __enter__(self) -> None:
            open_scopes[0] += 1
            depths.append(open_scopes[0])

        def __exit__(self, *exc: object) -> None:
            open_scopes[0] -= 1

    monkeypatch.setattr(runtime, "worker_guard", Probe)
    cfg = EnergyConfig(8.0, 9.0, 3, False)
    for call in (
        lambda: compute_single("SiO2", 2.2, cfg),
        lambda: compute_multiple(["SiO2"], [2.2], cfg),
    ):
        depths.clear()
        call()
        assert max(depths) == 2, depths
    assert open_scopes == [0]


def test_core_warm_up_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(
        "xraylabtool.calculators.core.warm_up", MagicMock(side_effect=OSError("boom"))
    )
    logger = logging.getLogger(services.logger.name)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            services.warm_up_core()  # must not raise
    finally:
        logger.removeHandler(caplog.handler)
    assert any(
        r.levelno == logging.WARNING and "core warm-up failed" in r.getMessage()
        for r in caplog.records
    )


def test_start_core_warm_up_runs_one_untracked_worker(
    app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xraylabtool.gui.main_window import MainWindow

    seen: list[str] = []
    monkeypatch.setattr(
        "xraylabtool.gui.main_window.single_tab.warm_up_core", seen.append
    )
    win = MainWindow()
    try:
        win.single_form.formula.setText("Al2O3")
        before = len(win._workers)
        win.start_core_warm_up()
        assert QThreadPool.globalInstance().waitForDone(20_000)
        assert seen == ["Al2O3"]
        assert len(win._workers) == before
    finally:
        win.close()


_SCRIPT = """
    import json
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import jax
    jax.config.update("jax_log_compiles", True)  # worker threads: not thread-local
    from PySide6.QtCore import QThreadPool
    from tests.fixtures.kk_compiles import record_compiles
    from xraylabtool.gui import services
    from xraylabtool.gui.services import EnergyConfig
    from xraylabtool.gui.workers import CalculationWorker
    from xraylabtool.utils import energy_to_wavelength, wavelength_to_energy

    if {warm}:
        pool = QThreadPool.globalInstance()
        pool.start(CalculationWorker(services.warm_up_core, "SiO2"))
        assert pool.waitForDone(120000)
    with record_compiles() as events:
        services.compute_single("SiO2", 2.2, EnergyConfig(8.0, 12.0, 50, False))
        energy_to_wavelength(10.0)
        wavelength_to_energy(1.24)
    mine = {{"_optical_program", "energy_to_wavelength", "wavelength_to_energy"}}
    print(json.dumps(sorted(set(events) & mine)))
"""


def _fresh(code: str) -> list[str]:
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=240,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_warm_up_covers_the_first_compute_click() -> None:
    """Fresh interpreters: after the idle warm-up the default click compiles nothing."""
    cold = _fresh(_SCRIPT.format(warm=False))
    assert "_optical_program" in cold  # control: the cold click compiles
    assert _fresh(_SCRIPT.format(warm=True)) == []


def test_the_pool_threads_are_not_the_main_thread(app: QApplication) -> None:
    """Self-check of the agreement test: jobs really run off the calling thread."""
    names: list[str] = []
    pool = QThreadPool()
    pool.start(CalculationWorker(lambda: names.append(threading.current_thread().name)))
    assert pool.waitForDone(10_000)
    assert names and names[0] != threading.main_thread().name
