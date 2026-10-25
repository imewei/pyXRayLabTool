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
    comp = _computation(nearedge=True, cxro=True)
    plot.set_data(comp)
    curves = plot.curves
    assert set(curves) == {"measured_f2", "spliced_f2", "kk_f1", "cxro_f1"}
    measured = comp.result.nearedge_scaled_kev_f2
    assert measured is not None
    expected = {
        "measured_f2": measured,
        "spliced_f2": (comp.result.energy_kev, comp.result.f2),
        "kk_f1": (comp.result.energy_kev, comp.result.f1),
        "cxro_f1": (comp.cxro_energy_kev, comp.cxro_f1),
    }
    assert [expected[k][0].size for k in expected] == [11, 7, 7, 4]
    for key, (ex, ey) in expected.items():
        x, y = _xy(curves[key])
        assert curves[key].isVisible()
        # Log-x mode: PyQtGraph hands back log10(energy).
        np.testing.assert_allclose(10**x, ex)
        np.testing.assert_allclose(y, ey)


@pytest.mark.unit
def test_plot_hides_missing_curves(plot) -> None:
    plot.set_data(_computation(nearedge=False, cxro=True))
    assert not plot.curves["measured_f2"].isVisible()
    assert plot.curves["spliced_f2"].isVisible()
    plot.set_data(_computation(nearedge=True, cxro=False))
    assert plot.curves["measured_f2"].isVisible()
    assert not plot.curves["cxro_f1"].isVisible()


_CYCLE_INDEX = {"measured_f2": 3, "spliced_f2": 1, "kk_f1": 0, "cxro_f1": 2}


def _pen_colors(plot) -> dict[str, str]:
    return {k: c.opts["pen"].color().name() for k, c in plot.curves.items()}


@pytest.mark.unit
def test_plot_follows_theme_toggle(plot, qt_app: QApplication) -> None:
    from PySide6.QtGui import QColor, QPalette

    from xraylabtool.gui.style import DARK_THEME, LIGHT_THEME
    from xraylabtool.gui.widgets import current_palette

    original = qt_app.palette()
    seen: dict[str, tuple[dict[str, str], str]] = {}
    try:
        plot.set_data(_computation(nearedge=True, cxro=True))
        for name, window, theme in (
            ("dark", "#101010", DARK_THEME),
            ("light", "#f5f5f5", LIGHT_THEME),
        ):
            pal = QPalette(original)
            pal.setColor(QPalette.ColorRole.Window, QColor(window))
            qt_app.setPalette(pal)
            plot.update_theme()
            assert current_palette() is theme
            assert (
                plot.plot_widget.backgroundBrush().color().name()
                == QColor(theme.plot_bg).name()
            )
            pens = _pen_colors(plot)
            for key, idx in _CYCLE_INDEX.items():
                assert pens[key] == QColor(theme.plot_cycle[idx]).name()
            symbol = plot.curves["measured_f2"].opts["symbolBrush"].color().name()
            assert symbol == QColor(theme.plot_cycle[3]).name()
            seen[name] = (pens, symbol)
        assert seen["dark"][0] != seen["light"][0]
        assert seen["dark"][1] != seen["light"][1]
        # curves survive the toggle with their data
        assert _xy(plot.curves["kk_f1"])[0].size == 7
    finally:
        qt_app.setPalette(original)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("energy", "expect_log"),
    [
        (np.array([0.1, 1.0, 2.0]), True),
        (np.array([0.0, 1.0, 2.0]), False),
        (np.array([-1.0, 1.0, 2.0]), False),
        (np.array([1.0]), False),
    ],
)
def test_plot_log_x_rule(plot, energy: np.ndarray, expect_log: bool) -> None:
    result = KKResult(
        formula="C",
        energy_kev=energy,
        f1=np.ones_like(energy),
        f2=np.ones_like(energy),
        relativistic_correction=0.0,
        merge_points_kev=None,
        n_refined_points=0,
        backend="numpy",
    )
    plot.set_data(KKComputation(result, np.empty(0), np.empty(0)))
    assert plot.curves["kk_f1"].opts["logMode"][0] is expect_log
    np.testing.assert_allclose(_xy(plot.curves["kk_f1"])[0].size, energy.size)
