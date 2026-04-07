"""Scattering factor plot widgets backed by PyQtGraph.

This module contains small PyQtGraph-based widgets used by the GUI.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import QVBoxLayout, QWidget

from xraylabtool.gui.widgets import (
    apply_palette_to_item,
    apply_palette_to_widget,
    current_palette,
)


class F1F2Plot(QWidget):
    """Single-panel plot showing f1 and f2 vs energy for one material."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        palette = current_palette()
        apply_palette_to_widget(self.plot_widget, palette)
        self._legend = self.plot_widget.addLegend()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.plot_widget.clear()
        self._legend = self.plot_widget.addLegend()

    def render_result(self, result: Any) -> None:
        self.clear()
        palette = current_palette()
        colors = palette.plot_cycle
        energy = np.array(result.energy_kev, ndmin=1, copy=False)

        self.plot_widget.plot(
            energy,
            result.scattering_factor_f1,
            pen=pg.mkPen(color=colors[0], width=1.5),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(colors[0]),
            symbolPen=pg.mkPen(None),
            name="f1",
        )
        self.plot_widget.plot(
            energy,
            result.scattering_factor_f2,
            pen=pg.mkPen(color=colors[1], width=1.5),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(colors[1]),
            symbolPen=pg.mkPen(None),
            name="f2",
        )

        if energy.size > 1:
            self.plot_widget.setLogMode(x=True, y=False)
        else:
            self.plot_widget.setLogMode(x=False, y=False)

        self.plot_widget.setLabel("bottom", "Energy (keV)")
        self.plot_widget.setLabel("left", "Scattering factors (e)")

    def update_theme(self) -> None:
        """Re-apply colors from the currently active palette."""
        apply_palette_to_widget(self.plot_widget, current_palette())


class MultiF1F2Plot(QWidget):
    """Two-panel plot comparing f1 (top) and f2 (bottom) for multiple materials."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)

        self.layout_widget = pg.GraphicsLayoutWidget()
        palette = current_palette()
        self.layout_widget.setBackground(palette.plot_bg)

        self._ax1: Any = self.layout_widget.addPlot(row=0, col=0)
        self._ax2: Any = self.layout_widget.addPlot(row=1, col=0)
        self._ax2.setXLink(self._ax1)

        apply_palette_to_item(self._ax1, palette)
        apply_palette_to_item(self._ax2, palette)

        self._ax1.addLegend()
        self._ax2.addLegend()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.layout_widget)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._ax1.clear()
        self._ax2.clear()
        self._ax1.addLegend()
        self._ax2.addLegend()

    def render_multi(self, results: Mapping[str, Any]) -> None:
        """Render f1 and f2 vs energy for multiple materials.

        Parameters
        ----------
        results
            Mapping of formula -> XRayResult-like objects.
        """
        self.clear()
        palette = current_palette()
        colors = palette.plot_cycle

        for idx, (formula, res) in enumerate(results.items()):
            energy = np.array(res.energy_kev, ndmin=1, copy=False)
            color = colors[idx % len(colors)]
            label = str(formula)

            self._ax1.plot(
                energy,
                res.scattering_factor_f1,
                pen=pg.mkPen(color=color, width=1.3),
                symbol="o",
                symbolSize=4,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(None),
                name=label,
            )
            self._ax2.plot(
                energy,
                res.scattering_factor_f2,
                pen=pg.mkPen(color=color, width=1.3),
                symbol="o",
                symbolSize=4,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(None),
                name=label,
            )

        # Log x-axis when energy is swept over multiple points
        any_res = next(iter(results.values()), None)
        use_log = any_res is not None and len(getattr(any_res, "energy_kev", [])) > 1
        self._ax1.setLogMode(x=use_log, y=False)
        self._ax2.setLogMode(x=use_log, y=False)

        self._ax1.setLabel("left", "f1 (e)")
        self._ax2.setLabel("left", "f2 (e)")
        self._ax2.setLabel("bottom", "Energy (keV)")

    def update_theme(self) -> None:
        """Re-apply colors from the currently active palette."""
        palette = current_palette()
        self.layout_widget.setBackground(palette.plot_bg)
        apply_palette_to_item(self._ax1, palette)
        apply_palette_to_item(self._ax2, palette)
