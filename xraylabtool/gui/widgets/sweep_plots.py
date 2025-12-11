"""Energy sweep multi-axes plotting."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class SweepPlots(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.setMinimumHeight(320)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def render(self, result) -> None:
        self.figure.clear()
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        import numpy as np

        energy = np.array(result.energy_kev, ndmin=1, copy=False)
        ax1.plot(energy, result.dispersion_delta, label="δ", marker="o", markersize=5)
        if energy.size > 1:
            ax1.set_xscale("log")
            ax1.set_yscale("log")
        ax1.set_ylabel("Dispersion δ")
        ax1.set_xlabel("Energy (keV)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            energy,
            result.absorption_beta,
            label="β",
            color="orange",
            marker="o",
            markersize=5,
        )
        if energy.size > 1:
            ax2.set_xscale("log")
            ax2.set_yscale("log")
        ax2.set_ylabel("Absorption β")
        ax2.set_xlabel("Energy (keV)")
        ax2.grid(True, alpha=0.3)

        ax3.plot(
            energy, result.critical_angle_degrees, label="θc", marker="o", markersize=5
        )
        if energy.size > 1:
            ax3.set_xscale("log")
        ax3.set_ylabel("Critical angle (deg)")
        ax3.set_xlabel("Energy (keV)")
        ax3.grid(True, alpha=0.3)
        # Highlight max critical angle
        if len(result.critical_angle_degrees) > 0:
            idx = result.critical_angle_degrees.argmax()
            ax3.plot(
                energy[idx],
                result.critical_angle_degrees[idx],
                "o",
                color="red",
                markersize=6,
                label="max θc",
            )

        ax4.plot(
            energy,
            result.attenuation_length_cm,
            label="Atten",
            color="green",
            marker="o",
            markersize=5,
        )
        if energy.size > 1:
            ax4.set_xscale("log")
            ax4.set_yscale("log")
        ax4.set_ylabel("Attenuation length (cm)")
        ax4.set_xlabel("Energy (keV)")
        ax4.grid(True, alpha=0.3)

        for ax in (ax1, ax2, ax3, ax4):
            if not ax.get_legend_handles_labels()[0]:
                continue
            ax.legend()

        self.figure.subplots_adjust(
            left=0.1, right=0.98, top=0.93, bottom=0.12, hspace=0.35, wspace=0.25
        )
        self.canvas.draw_idle()


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

    def render(self, result) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        import numpy as np

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
            ax.set_xscale("log")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Scattering factors (e)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.figure.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.14)
        self.canvas.draw_idle()
