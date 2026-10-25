"""Kramers-Kronig tab: wiring, worker errors, CSV export, theme (offscreen, serial)."""

from __future__ import annotations

import csv
import os
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
import pytest

from xraylabtool.gui.logging_filters import suppress_qt_noise
from xraylabtool.gui.main_window import MainWindow, kk_tab
from xraylabtool.gui.style import DARK_THEME, LIGHT_THEME, apply_styles


@pytest.fixture(scope="module")
def app() -> QApplication:
    inst = QApplication.instance()
    if inst is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        inst = QApplication([])
        apply_styles(inst)
    assert isinstance(inst, QApplication)
    return inst


@pytest.fixture
def win(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> Any:
    # A modal QMessageBox would block forever without a user.
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(MagicMock()))
    with suppress_qt_noise():
        window = MainWindow()
    window.toast.show_toast = MagicMock()  # type: ignore[method-assign]
    yield window
    window.close()


def _wait_for_worker(app: QApplication, win: Any, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        QTest.qWait(20)
        if not win._workers:
            app.processEvents()
            return
    raise AssertionError("KK worker did not finish")


def _small_run(win: Any) -> None:
    win.kk_formula.setText("SiO2")
    win.kk_energy_start.setValue(1.0)
    win.kk_energy_end.setValue(10.0)
    win.kk_points.setValue(12)


def test_tab_present_and_named(win: Any) -> None:
    assert win.main_tabs.count() == 3
    assert win.main_tabs.tabText(2) == "Kramers-Kronig"


def test_run_without_file_populates_plot_and_table(app: QApplication, win: Any) -> None:
    _small_run(win)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    assert win.kk_result is not None
    assert win.kk_table.rowCount() == 12
    assert win.kk_table.columnCount() == 3
    assert win.kk_run_btn.isEnabled()
    f1_item = win.kk_table.item(3, 1)
    assert f1_item is not None
    assert np.isfinite(float(f1_item.text()))
    kk_f1 = win.kk_plot.curves["kk_f1"]
    x, y = kk_f1.getData()
    assert x.size == 12 and np.isfinite(y).all()
    assert win.kk_export_csv.isEnabled()


def test_energy_range_controls_reach_the_service(
    app: QApplication, win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real = kk_tab.compute_kk

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(kk_tab, "compute_kk", spy)
    win.kk_formula.setText("SiO2")
    win.kk_energy_start.setValue(2.0)
    win.kk_energy_end.setValue(4.0)
    win.kk_points.setValue(5)
    win.kk_refine.setValue(0.0)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("SiO2", None, None, "eV", None, None)
    np.testing.assert_allclose(kwargs["energies_kev"], np.linspace(2.0, 4.0, 5))
    assert win.kk_table.rowCount() == 5


@pytest.mark.parametrize(("start", "end"), [(0.012, 0.02), (40.0, 60.0)])
def test_energies_outside_cxro_range_reach_the_service(
    app: QApplication,
    win: Any,
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
) -> None:
    calls: list[dict[str, Any]] = []
    real = kk_tab.compute_kk

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(kk_tab, "compute_kk", spy)
    win.kk_formula.setText("Si")
    win.kk_energy_start.setValue(start)
    win.kk_energy_end.setValue(end)
    win.kk_points.setValue(3)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["energies_kev"], np.linspace(start, end, 3))
    assert win.kk_table.rowCount() == 3
    assert win.kk_plot.curves["kk_f1"].isVisible()
    assert not win.kk_plot.curves["cxro_f1"].isVisible()
    assert win.kk_result.cxro_f1.size == 0


def test_nearedge_options_reach_the_service(
    app: QApplication, win: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        raise ValueError("stop here")

    monkeypatch.setattr(kk_tab, "compute_kk", fake)
    win.kk_formula.setText("C")
    win.kk_nearedge.setText(str(tmp_path / "edge.txt"))
    win.kk_input_type.setCurrentText("beta")
    win.kk_energy_unit.setCurrentText("keV")
    win.kk_merge.setText("0.28, 0.32")
    win.kk_refine.setValue(0.05)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    assert captured["args"] == (
        "C",
        str(tmp_path / "edge.txt"),
        "beta",
        "keV",
        (0.28, 0.32),
        0.05,
    )


@pytest.mark.parametrize(
    ("unit", "merge_text", "window_kev"),
    [("keV", "0.28, 0.32", (0.28, 0.32)), ("eV", "280, 320", (0.28, 0.32))],
)
def test_merge_window_is_always_sampled(
    app: QApplication,
    win: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    unit: str,
    merge_text: str,
    window_kev: tuple[float, float],
) -> None:
    captured: dict[str, Any] = {}

    def fake(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(kk_tab, "compute_kk", fake)
    win.kk_formula.setText("C")
    win.kk_nearedge.setText(str(tmp_path / "edge.txt"))
    win.kk_energy_unit.setCurrentText(unit)
    win.kk_merge.setText(merge_text)
    win.kk_energy_start.setValue(1.0)  # range excludes the window
    win.kk_energy_end.setValue(10.0)
    win.kk_points.setValue(7)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    grid = captured["energies_kev"]
    assert np.all(np.diff(grid) > 0)
    np.testing.assert_allclose(
        grid, np.unique(np.r_[np.linspace(1, 10, 7), np.linspace(*window_kev, 7)])
    )


@pytest.mark.parametrize(
    ("setup", "needle"),
    [
        ({"merge": "1,2"}, "near-edge"),
        ({"merge": "oops"}, "merge"),
        ({"start": 5.0, "end": 2.0}, "End energy"),
        ({"formula": ""}, "formula"),
    ],
)
def test_invalid_inputs_are_reported_without_a_worker(
    win: Any, setup: dict[str, Any], needle: str
) -> None:
    win.kk_formula.setText(setup.get("formula", "SiO2"))
    win.kk_merge.setText(setup.get("merge", ""))
    win.kk_energy_start.setValue(setup.get("start", 1.0))
    win.kk_energy_end.setValue(setup.get("end", 10.0))
    win.kk_run_btn.click()

    assert not win._workers
    assert win.kk_result is None
    message = win.toast.show_toast.call_args.args[0]
    assert needle.lower() in message.lower()
    assert win.toast.show_toast.call_args.args[1] == "error"


def test_worker_error_shows_toast_not_crash(app: QApplication, win: Any) -> None:
    win.kk_formula.setText("Xx")  # unknown element -> FormulaError in the worker
    win.kk_energy_start.setValue(1.0)
    win.kk_energy_end.setValue(10.0)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    kinds = [c.args[1] for c in win.toast.show_toast.call_args_list]
    assert "error" in kinds
    assert win.kk_result is None
    assert win.kk_run_btn.isEnabled()
    assert win.kk_run_btn.text() == "Compute"


def test_gui_input_types_are_valid_api_types() -> None:
    from typing import get_args

    from xraylabtool.kk.nearedge import InputType

    assert set(kk_tab.KK_INPUT_TYPES) <= set(get_args(InputType))


def test_tab_order_includes_kk_widgets(win: Any) -> None:
    chain = [
        win.kk_formula,
        win.kk_nearedge,
        win.kk_browse_btn,
        win.kk_input_type,
        win.kk_energy_unit,
        win.kk_merge,
        win.kk_refine,
        win.kk_energy_start,
        win.kk_energy_end,
        win.kk_points,
        win.kk_run_btn,
        win.kk_export_csv,
    ]
    # Walk the window's focus ring once from the first widget; the KK widgets
    # must appear in the declared order (other widgets may sit in between).
    ring = [chain[0]]
    while (nxt := ring[-1].nextInFocusChain()) is not chain[0]:
        ring.append(nxt)
    positions = [ring.index(w) for w in chain]
    assert positions == sorted(positions)


def test_set_tab_order_chains_kk_widgets(win: Any) -> None:
    pairs: list[tuple[Any, Any]] = []
    win.setTabOrder = lambda first, second: pairs.append((first, second))
    win._set_tab_order()
    assert (win.kk_formula, win.kk_nearedge) in pairs
    assert (win.kk_run_btn, win.kk_export_csv) in pairs


def test_export_csv_columns(
    app: QApplication, win: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from xraylabtool.gui import main_window as mw

    assert win._export_kk_csv() is None  # nothing computed yet
    _small_run(win)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)

    monkeypatch.setattr(
        mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    path = win._export_kk_csv()
    assert path is not None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["energy_keV", "f1", "f2"]
    assert len(rows) == 1 + 12
    res = win.kk_result.result
    np.testing.assert_allclose(
        [float(v) for v in rows[1]], [res.energy_kev[0], res.f1[0], res.f2[0]]
    )


def test_export_file_name_uses_computed_formula(
    app: QApplication, win: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from xraylabtool.gui import main_window as mw

    _small_run(win)
    win.kk_formula.setText("Si")
    win.kk_run_btn.click()
    _wait_for_worker(app, win)
    win.kk_formula.setText("Au")  # edited after the compute, not recomputed

    monkeypatch.setattr(
        mw.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
    )
    path = win._export_kk_csv()
    assert path is not None
    name = os.path.basename(path)
    assert "Si" in name
    assert "Au" not in name


def test_export_csv_io_error_is_reported(
    app: QApplication, win: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xraylabtool.gui import main_window as mw

    _small_run(win)
    win.kk_run_btn.click()
    _wait_for_worker(app, win)
    monkeypatch.setattr(
        mw.QFileDialog, "getExistingDirectory", lambda *a, **k: "/nonexistent-xrlt/x"
    )
    assert win._export_kk_csv() is None
    assert win.toast.show_toast.call_args.args[1] == "error"


def test_theme_toggle_updates_kk_plot(app: QApplication, win: Any) -> None:
    original = app.palette()
    try:
        for window_color, theme in (("#101010", DARK_THEME), ("#f5f5f5", LIGHT_THEME)):
            pal = QPalette(original)
            pal.setColor(QPalette.ColorRole.Window, QColor(window_color))
            app.setPalette(pal)
            win._on_theme_changed("dark" if theme is DARK_THEME else "light")
            bg = win.kk_plot.plot_widget.backgroundBrush().color().name()
            assert bg == QColor(theme.plot_bg).name()
            pen = win.kk_plot.curves["kk_f1"].opts["pen"].color().name()
            assert pen == QColor(theme.plot_cycle[0]).name()
    finally:
        app.setPalette(original)
