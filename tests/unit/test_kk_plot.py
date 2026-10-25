"""GUI tests for KKPlot.

Requires PySide6. Run with QT_QPA_PLATFORM=offscreen to avoid needing a display.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

PySide6 = pytest.importorskip("PySide6")

if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

from xraylabtool.gui.services import KKComputation
from xraylabtool.kk import KKResult

_app: QApplication | None = None


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)  # type: ignore[assignment]
    return _app  # type: ignore[return-value]


@pytest.fixture
def plot(qt_app: QApplication):
    from xraylabtool.gui.widgets.kk_plot import KKPlot

    widget = KKPlot()
    yield widget
    widget.close()


def _computation(*, nearedge: bool, cxro: bool) -> KKComputation:
    energy = np.linspace(0.1, 2.0, 7)
    result = KKResult(
        formula="C",
        energy_kev=energy,
        f1=np.linspace(5.0, 6.0, 7),
        f2=np.linspace(0.1, 0.2, 7),
        relativistic_correction=0.0,
        merge_points_kev=None,
        n_refined_points=0,
        backend="numpy",
        nearedge_scaled_kev_f2=(
            (np.linspace(0.28, 0.32, 11), np.linspace(0.0, 1.0, 11))
            if nearedge
            else None
        ),
    )
    cxro_e = np.linspace(0.2, 1.8, 4) if cxro else np.empty(0)
    return KKComputation(result, cxro_e, np.linspace(5.0, 6.0, cxro_e.size))


def _xy(curve) -> tuple[np.ndarray, np.ndarray]:
    x, y = curve.getData()
    return np.asarray(x), np.asarray(y)


@pytest.mark.unit
def test_plot_draws_curves_on_unequal_grids(plot) -> None:
    plot.set_data(_computation(nearedge=True, cxro=True))
    curves = plot.curves
    assert set(curves) == {"measured_f2", "spliced_f2", "kk_f1", "cxro_f1"}
    assert [
        _xy(curves[k])[0].size
        for k in ("measured_f2", "spliced_f2", "kk_f1", "cxro_f1")
    ] == [11, 7, 7, 4]
    assert all(curves[k].isVisible() for k in curves)


@pytest.mark.unit
def test_plot_hides_missing_curves(plot) -> None:
    plot.set_data(_computation(nearedge=False, cxro=True))
    assert not plot.curves["measured_f2"].isVisible()
    assert plot.curves["spliced_f2"].isVisible()
    plot.set_data(_computation(nearedge=True, cxro=False))
    assert plot.curves["measured_f2"].isVisible()
    assert not plot.curves["cxro_f1"].isVisible()


@pytest.mark.unit
def test_plot_follows_theme_toggle(plot, qt_app: QApplication) -> None:
    from PySide6.QtGui import QColor, QPalette

    from xraylabtool.gui.style import DARK_THEME, LIGHT_THEME
    from xraylabtool.gui.widgets import current_palette

    original = qt_app.palette()
    try:
        plot.set_data(_computation(nearedge=True, cxro=True))
        dark = QPalette(original)
        dark.setColor(QPalette.ColorRole.Window, QColor("#101010"))
        qt_app.setPalette(dark)
        plot.update_theme()
        assert current_palette() is DARK_THEME
        assert (
            plot.plot_widget.backgroundBrush().color().name()
            == QColor(DARK_THEME.plot_bg).name()
        )
        light = QPalette(original)
        light.setColor(QPalette.ColorRole.Window, QColor("#f5f5f5"))
        qt_app.setPalette(light)
        plot.update_theme()
        assert current_palette() is LIGHT_THEME
        assert (
            plot.plot_widget.backgroundBrush().color().name()
            == QColor(LIGHT_THEME.plot_bg).name()
        )
        # curves survive the toggle with their data
        assert _xy(plot.curves["kk_f1"])[0].size == 7
    finally:
        qt_app.setPalette(original)
