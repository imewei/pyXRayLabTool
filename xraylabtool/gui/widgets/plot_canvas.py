"""PyQtGraph-based plot canvas widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import QVBoxLayout, QWidget

from xraylabtool.gui.widgets import apply_palette_to_widget, current_palette


class PlotCanvas(QWidget):
    """Single-panel plot widget backed by PyQtGraph."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        self.log_x = False
        self.log_y = False
        self.setMinimumHeight(320)

        palette = current_palette()
        self._colors: list[str] = list(palette.plot_cycle)
        apply_palette_to_widget(self.plot_widget, palette)

        # Legend must be added once; clearing the plot removes items but keeps it
        self._legend = self.plot_widget.addLegend()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public API (matches main_window.py contract)
    # ------------------------------------------------------------------

    def set_scales(self, log_x: bool, log_y: bool) -> None:
        self.log_x = log_x
        self.log_y = log_y

    def clear(self) -> None:
        self.plot_widget.clear()
        # Re-attach legend after clear() removes it
        self._legend = self.plot_widget.addLegend()

    def plot_single(
        self,
        result: Any,
        property_name: str,
        ylabel: str | None = None,
    ) -> None:
        self.clear()
        x = np.array(result.energy_kev, ndmin=1, copy=False)
        y = np.array(getattr(result, property_name), ndmin=1, copy=False)
        label = (
            f"{result.formula} "
            f"({getattr(result, 'density_g_cm3', 0):.3g} g/cm\u00b3)"
        )
        color = self._colors[0]
        self.plot_widget.plot(
            x,
            y,
            pen=pg.mkPen(color=color, width=1.5),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(color),
            symbolPen=pg.mkPen(None),
            name=label,
        )
        self.plot_widget.setLabel("bottom", "Energy (keV)")
        self.plot_widget.setLabel("left", ylabel or property_name.replace("_", " "))
        self.plot_widget.setLogMode(x=self.log_x, y=self.log_y)

    def plot_multi(
        self,
        results: Mapping[str, Any],
        property_name: str,
        ylabel: str | None = None,
    ) -> None:
        self.clear()
        palette = current_palette()
        colors = list(palette.plot_cycle)
        for idx, (formula, res) in enumerate(results.items()):
            x = np.array(res.energy_kev, ndmin=1, copy=False)
            y = np.array(getattr(res, property_name), ndmin=1, copy=False)
            label = (
                f"{formula} "
                f"({getattr(res, 'density_g_cm3', 0):.3g} g/cm\u00b3)"
            )
            color = colors[idx % len(colors)]
            self.plot_widget.plot(
                x,
                y,
                pen=pg.mkPen(color=color, width=1.3),
                symbol="o",
                symbolSize=5,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(None),
                name=label,
            )
        self.plot_widget.setLabel("bottom", "Energy (keV)")
        self.plot_widget.setLabel("left", ylabel or property_name.replace("_", " "))
        self.plot_widget.setLogMode(x=self.log_x, y=self.log_y)

    def update_theme(self) -> None:
        """Re-apply colors from the currently active palette."""
        palette = current_palette()
        self._colors = list(palette.plot_cycle)
        apply_palette_to_widget(self.plot_widget, palette)
