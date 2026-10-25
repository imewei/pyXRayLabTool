"""Kramers-Kronig plot widget backed by PyQtGraph."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import QVBoxLayout, QWidget

from xraylabtool.gui.services import KKComputation
from xraylabtool.gui.widgets import apply_palette_to_widget, current_palette

# curve key -> (legend label, palette-cycle index, dashed, symbols)
_CURVES: dict[str, tuple[str, int, bool, bool]] = {
    "measured_f2": ("measured f2 (near-edge)", 3, False, True),
    "spliced_f2": ("spliced f2", 1, False, False),
    "kk_f1": ("KK f1", 0, False, False),
    "cxro_f1": ("CXRO f1", 2, True, False),
}


class KKPlot(QWidget):
    """Plot of measured/spliced f2, KK f1 and CXRO f1, each on its own grid."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.plot_widget = pg.PlotWidget()
        apply_palette_to_widget(self.plot_widget, current_palette())
        self._legend = self.plot_widget.addLegend()
        self.curves: dict[str, Any] = {
            key: self.plot_widget.plot([], [], name=None) for key in _CURVES
        }
        self._restyle()
        for curve in self.curves.values():
            curve.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
        self.plot_widget.setLabel("bottom", "Energy (keV)")
        self.plot_widget.setLabel("left", "Scattering factor (e)")

    def set_data(self, computation: KKComputation) -> None:
        """Draw the four curves; curves without data are hidden.

        Parameters
        ----------
        computation : KKComputation
            Result and CXRO comparison; every curve keeps its own energy grid.
        """
        res = computation.result
        measured = res.nearedge_scaled_kev_f2
        empty = (np.empty(0), np.empty(0))
        data = {
            "measured_f2": measured if measured is not None else empty,
            "spliced_f2": (res.energy_kev, res.f2),
            "kk_f1": (res.energy_kev, res.f1),
            "cxro_f1": (computation.cxro_energy_kev, computation.cxro_f1),
        }
        self._legend.clear()
        for key, (x, y) in data.items():
            curve = self.curves[key]
            has_data = np.size(x) > 0
            curve.setData(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
            curve.setVisible(has_data)
            if has_data:
                self._legend.addItem(curve, _CURVES[key][0])
        # Log-x like F1F2Plot; only valid for positive energies.
        all_x = [np.asarray(x) for x, _ in data.values() if np.size(x)]
        log_x = (
            bool(all_x)
            and all(xs.min() > 0 for xs in all_x)
            and max(xs.size for xs in all_x) > 1
        )
        self.plot_widget.setLogMode(x=log_x, y=False)
        self.plot_widget.autoRange()

    def update_theme(self) -> None:
        """Re-apply colors from the currently active palette."""
        apply_palette_to_widget(self.plot_widget, current_palette())
        self._restyle()

    def _restyle(self) -> None:
        colors = current_palette().plot_cycle
        for key, (_, idx, dashed, symbols) in _CURVES.items():
            color = colors[idx % len(colors)]
            style = (
                pg.QtCore.Qt.PenStyle.DashLine
                if dashed
                else pg.QtCore.Qt.PenStyle.SolidLine
            )
            pen = pg.mkPen(color=color, width=1.5, style=style)
            curve = self.curves[key]
            curve.setPen(pen)
            if symbols:
                curve.setSymbol("o")
                curve.setSymbolSize(5)
                curve.setSymbolBrush(pg.mkBrush(color))
                curve.setSymbolPen(pg.mkPen(None))
