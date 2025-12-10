"""Bar chart helpers for multi-material comparisons."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ComparisonBars(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.setMinimumHeight(240)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot(self, labels, values, ylabel: str, logy: bool = False) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        def _to_scalar(val):
            try:
                if val is None:
                    return None
                if (
                    hasattr(val, "__len__")
                    and not isinstance(val, (str, bytes))
                    and len(val) == 1
                ):
                    val = val[0]
                return float(val)
            except (TypeError, ValueError):
                return None

        sanitized = []
        for v in values:
            scalar = _to_scalar(v)
            if scalar is None or scalar <= 0:
                sanitized.append(None)
            else:
                sanitized.append(scalar)

        effective_log = logy and all(v is not None and v > 0 for v in sanitized)
        plot_values = [v if v is not None else 0 for v in sanitized]

        bars = ax.bar(labels, plot_values)
        if effective_log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, plot_values, strict=False):
            label = "n/a" if val == 0 and not effective_log else f"{val:.3g}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val if val > 0 else (bar.get_height() * 0.1 + 1e-9),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
        self.figure.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.18)
        self.canvas.draw_idle()
