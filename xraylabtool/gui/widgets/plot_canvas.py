"""Matplotlib canvas helpers."""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class PlotCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.log_x = False
        self.log_y = False
        self.setMinimumHeight(320)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def set_scales(self, log_x: bool, log_y: bool) -> None:
        self.log_x = log_x
        self.log_y = log_y

    def _apply_axes(self, ax) -> None:
        if self.log_x:
            ax.set_xscale("log")
        if self.log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        self.figure.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def plot_single(
        self, result, property_name: str, ylabel: str | None = None
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        import numpy as np

        x = np.atleast_1d(result.energy_kev)
        y = np.atleast_1d(getattr(result, property_name))
        ax.plot(x, y, label=f"{result.formula} ({result.density_g_cm3:.3g} g/cm³)")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel or property_name.replace("_", " "))
        self._apply_axes(ax)
        ax.legend()
        self.canvas.draw_idle()

    def plot_multi(
        self,
        results: Mapping[str, object],
        property_name: str,
        ylabel: str | None = None,
    ) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for formula, res in results.items():
            import numpy as np

            x = np.atleast_1d(res.energy_kev)
            y = np.atleast_1d(getattr(res, property_name))
            label = f"{formula} ({getattr(res, 'density_g_cm3', 0):.3g} g/cm³)"
            ax.plot(x, y, label=label)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel or property_name.replace("_", " "))
        self._apply_axes(ax)
        ax.legend()
        self.canvas.draw_idle()
