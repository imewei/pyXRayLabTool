"""Matplotlib canvas helpers."""

from __future__ import annotations

# isort: off
from collections.abc import Mapping

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

# isort: on


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
        x = np.array(result.energy_kev, ndmin=1, copy=False)
        y = np.array(getattr(result, property_name), ndmin=1, copy=False)
        ax.plot(
            x,
            y,
            label=f"{result.formula} ({result.density_g_cm3:.3g} g/cm³)",
            marker="o",
            markersize=6,
            linewidth=1.5,
        )
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
            x = np.array(res.energy_kev, ndmin=1, copy=False)
            y = np.array(getattr(res, property_name), ndmin=1, copy=False)
            label = f"{formula} ({getattr(res, 'density_g_cm3', 0):.3g} g/cm³)"
            ax.plot(x, y, label=label, marker="o", markersize=5, linewidth=1.3)
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(ylabel or property_name.replace("_", " "))
        self._apply_axes(ax)
        ax.legend()
        self.canvas.draw_idle()

    def update_theme(self) -> None:
        """Update figure appearance based on current RcParams."""
        import matplotlib as mpl

        rc = mpl.rcParams
        self.figure.set_facecolor(rc["figure.facecolor"])

        for ax in self.figure.axes:
            ax.set_facecolor(rc["axes.facecolor"])
            ax.grid(color=rc["grid.color"], alpha=rc["grid.alpha"])
            ax.title.set_color(rc["text.color"])
            ax.xaxis.label.set_color(rc["text.color"])
            ax.yaxis.label.set_color(rc["text.color"])
            ax.tick_params(colors=rc["xtick.color"], which="both")

            for spine in ax.spines.values():
                spine.set_edgecolor(rc["axes.edgecolor"])

            legend = ax.get_legend()
            if legend:
                # Force legend update if needed, usually automatic on draw
                pass

        self.canvas.draw_idle()
