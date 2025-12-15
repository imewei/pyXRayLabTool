"""Scattering factor plot widgets.

This module contains small Matplotlib-based widgets used by the GUI.
"""

from __future__ import annotations

from collections.abc import Mapping

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class F1F2Plot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def render_result(self, result) -> None:
        import numpy as np

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        energy = np.array(result.energy_kev, ndmin=1, copy=False)
        ax.plot(
            energy,
            result.scattering_factor_f1,
            label="f1",
            marker="o",
            markersize=5,
            linewidth=1.5,
        )
        ax.plot(
            energy,
            result.scattering_factor_f2,
            label="f2",
            marker="o",
            markersize=5,
            linewidth=1.5,
        )
        if energy.size > 1:
            ax.set_xscale("log", nonpositive="clip")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Scattering factors (e)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.figure.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.14)
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
        self.canvas.draw_idle()


class MultiF1F2Plot(QWidget):
    """Compare f1 and f2 across multiple materials."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.setMinimumHeight(320)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def render_multi(self, results: Mapping[str, object]) -> None:
        """Render f1 and f2 vs energy for multiple materials.

        Parameters
        ----------
        results
            Mapping of formula -> XRayResult-like objects.
        """

        import numpy as np

        self.figure.clear()
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212, sharex=ax1)

        for formula, res in results.items():
            energy = np.array(res.energy_kev, ndmin=1, copy=False)
            ax1.plot(
                energy,
                res.scattering_factor_f1,
                label=str(formula),
                marker="o",
                markersize=4,
                linewidth=1.3,
            )
            ax2.plot(
                energy,
                res.scattering_factor_f2,
                label=str(formula),
                marker="o",
                markersize=4,
                linewidth=1.3,
            )

        # Use log x-axis when energy is swept
        any_energy = next(iter(results.values()), None)
        if any_energy is not None and len(getattr(any_energy, "energy_kev", [])) > 1:
            ax1.set_xscale("log", nonpositive="clip")
            ax2.set_xscale("log", nonpositive="clip")

        ax1.set_ylabel("f1 (e)")
        ax2.set_ylabel("f2 (e)")
        ax2.set_xlabel("Energy (keV)")
        for ax in (ax1, ax2):
            ax.grid(True, alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()

        self.figure.subplots_adjust(
            left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.28
        )
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
        self.canvas.draw_idle()
