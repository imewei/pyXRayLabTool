"""Kramers-Kronig GUI on the worker thread, unmocked compute (offscreen, serial)."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PySide6.QtCore import QThreadPool
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
import pytest

from tests import test_gui_kk_tab
from xraylabtool.kk import kk_calculate_real, load_nearedge

# Reuse the module's offscreen QApplication and stubbed-dialog window fixtures.
app = test_gui_kk_tab.app
win = test_gui_kk_tab.win
EDGE = Path(__file__).parent / "characterization" / "kk" / "nearedge_example.txt"


def _wait_idle(app: QApplication, win: Any, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        QTest.qWait(20)
        if not win._workers:
            app.processEvents()
            return
    raise AssertionError("KK worker did not finish")


def _setup(win: Any, formula: str = "SiO2") -> None:
    win.kk_formula.setText(formula)
    win.kk_energy_start.setValue(1.0)
    win.kk_energy_end.setValue(10.0)
    win.kk_points.setValue(12)
    win.kk_refine.setValue(0.0)


def _check_table(win: Any) -> None:
    res = win.kk_result.result
    assert win.kk_table.rowCount() == res.energy_kev.size
    last = win.kk_table.rowCount() - 1
    for col, ref in enumerate((res.energy_kev, res.f1, res.f2)):
        item = win.kk_table.item(last, col)
        np.testing.assert_allclose(float(item.text()), ref[-1], rtol=1e-5)
    x, y = win.kk_plot.curves["kk_f1"].getData()
    assert x.size == y.size == res.energy_kev.size and np.isfinite(y).all()


def test_plain_run_matches_api(app: QApplication, win: Any) -> None:
    _setup(win)
    win.kk_run_btn.click()
    _wait_idle(app, win)

    ref = kk_calculate_real("SiO2", energies_kev=np.linspace(1.0, 10.0, 12))
    res = win.kk_result.result
    np.testing.assert_array_equal(res.energy_kev, ref.energy_kev)
    np.testing.assert_allclose(res.f1, ref.f1, rtol=1e-12)
    np.testing.assert_allclose(res.f2, ref.f2, rtol=1e-12)
    _check_table(win)


def test_refined_run_matches_api(app: QApplication, win: Any) -> None:
    _setup(win)
    win.kk_refine.setValue(0.05)
    win.kk_run_btn.click()
    _wait_idle(app, win)

    ref = kk_calculate_real(
        "SiO2", energies_kev=np.linspace(1.0, 10.0, 12), refine_tolerance=0.05
    )
    res = win.kk_result.result
    assert res.n_refined_points > 0
    np.testing.assert_array_equal(res.energy_kev, ref.energy_kev)
    np.testing.assert_allclose(res.f1, ref.f1, rtol=1e-12)
    _check_table(win)


def test_nearedge_run_matches_api(app: QApplication, win: Any) -> None:
    _setup(win, "C8H8")
    win.kk_nearedge.setText(str(EDGE))
    win.kk_input_type.setCurrentText("nexafs")
    win.kk_energy_unit.setCurrentText("eV")
    win.kk_merge.setText("280, 320")
    win.kk_run_btn.click()
    _wait_idle(app, win)

    res = win.kk_result.result
    assert res.merge_points_kev == (0.28, 0.32)
    ref = kk_calculate_real(
        "C8H8",
        energies_kev=res.energy_kev,
        nearedge=load_nearedge(str(EDGE), energy_unit="eV"),
        input_type="nexafs",
        merge_points_kev=(0.28, 0.32),
    )
    np.testing.assert_allclose(res.f1, ref.f1, rtol=1e-12)
    _check_table(win)


def test_error_path_shows_dialog_and_stays_responsive(
    app: QApplication,
    win: Any,
) -> None:
    _setup(win, "Xx")  # unknown element -> FormulaError on the worker
    win.kk_run_btn.click()
    _wait_idle(app, win)

    QMessageBox.critical.assert_called_once()  # stubbed by the `win` fixture
    assert win.kk_result is None
    assert win.kk_run_btn.isEnabled()

    _setup(win)  # the next run still works
    win.kk_run_btn.click()
    _wait_idle(app, win)
    assert win.kk_result is not None


def test_close_during_warm_up_does_not_hang(
    app: QApplication,
    win: Any,
) -> None:
    win.start_kk_warm_up()
    t0 = time.perf_counter()
    win.close()
    assert time.perf_counter() - t0 < 5.0  # close never waits for the running job
    assert QThreadPool.globalInstance().waitForDone(20_000)  # and it does finish


def test_warm_up_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    from xraylabtool.gui import services

    monkeypatch.setattr(
        "xraylabtool.kk.kk_calculate_real", MagicMock(side_effect=OSError("boom"))
    )
    logger = logging.getLogger(services.logger.name)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            services.warm_up_kk()  # must not raise
    finally:
        logger.removeHandler(caplog.handler)
    assert any(
        r.levelno == logging.WARNING and "kk warm-up failed" in r.getMessage()
        for r in caplog.records
    )


_WARM_GUI_SCRIPT = """
    import json, os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import jax
    jax.config.update("jax_log_compiles", True)  # worker threads: not thread-local
    from PySide6.QtCore import QThreadPool
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMessageBox
    from tests.fixtures.kk_compiles import kk_programs, record_compiles
    from xraylabtool.gui.main_window import MainWindow

    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    app = QApplication([])
    win = MainWindow()
    if {warm}:
        win.start_kk_warm_up()
        assert QThreadPool.globalInstance().waitForDone(120000)
    with record_compiles() as events:
        win.kk_run_btn.click()  # default form: SiO2, 500 points, 1-30 keV, refine off
        for _ in range(3000):
            app.processEvents()
            QTest.qWait(10)
            if not win._workers:
                break
        assert win.kk_result is not None
    print(json.dumps(kk_programs(events)))
"""


def test_warm_up_covers_the_first_compute_click() -> None:
    """Fresh interpreters: after the idle warm-up the default click compiles nothing."""
    from tests.unit.kk.test_pipeline_device import _subprocess

    assert _subprocess(
        _WARM_GUI_SCRIPT.format(warm=False)
    )  # control: cold click compiles
    assert _subprocess(_WARM_GUI_SCRIPT.format(warm=True)) == []


def test_launcher_schedules_the_warm_up(monkeypatch: pytest.MonkeyPatch) -> None:
    from xraylabtool.gui import __main__ as launcher

    timer = MagicMock()
    window = MagicMock()
    monkeypatch.setattr(launcher, "QTimer", timer)
    monkeypatch.setattr(launcher, "QApplication", MagicMock())
    monkeypatch.setattr(launcher, "ThemeManager", MagicMock())
    monkeypatch.setattr(launcher, "MainWindow", MagicMock(return_value=window))
    with pytest.raises(SystemExit):
        launcher.main([])
    assert [c.args for c in timer.singleShot.call_args_list] == [
        (0, window.start_core_warm_up),
        (0, window.start_kk_warm_up),
    ]

    timer.reset_mock()
    with pytest.raises(SystemExit):
        launcher.main(["--test-launch"])  # smoke mode quits after 500 ms, no warm-up
    assert [c.args[0] for c in timer.singleShot.call_args_list] == [500]
