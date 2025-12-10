"""Shared Qt styling for the XRayLabTool GUI."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

try:
    import matplotlib as _mpl
except Exception:
    _mpl = None


_QSS = """
/* Base typography + spacing */
* {
    font-size: 14px;
}

QMainWindow, QWidget {
    background: #f7f9fb;
}

QGroupBox {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: 600;
}

QLabel {
    color: #0f172a;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    padding: 6px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background: #ffffff;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #2563eb;
}

QLineEdit[validation="invalid"], QSpinBox[validation="invalid"], QDoubleSpinBox[validation="invalid"] {
    border: 1px solid #dc2626;
    background: #fee2e2;
}

/* Buttons */
QPushButton {
    padding: 8px 12px;
    border-radius: 6px;
    background: #e2e8f0;
    border: 1px solid #cbd5e1;
    color: #0f172a;
    font-weight: 500;
}

QPushButton:hover {
    background: #dbeafe;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background: #bfdbfe;
}

QPushButton:disabled {
    color: #94a3b8;
    background: #e2e8f0;
    border-color: #cbd5e1;
}

QPushButton[class="primary"] {
    background: #2563eb;
    border-color: #1e55cb;
    color: #ffffff;
}

QPushButton[class="primary"]:hover {
    background: #1e55cb;
}

QPushButton[class="secondary"] {
    background: #ffffff;
    color: #2563eb;
    border-color: #2563eb;
}

/* Tables */
QTableWidget {
    alternate-background-color: #f8fafc;
    gridline-color: #e2e8f0;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}

QHeaderView::section {
    background: #eef2f7;
    padding: 6px;
    border: 1px solid #e2e8f0;
    font-weight: 600;
}

QStatusBar {
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
}
"""


def apply_styles(app) -> None:
    """Apply palette and stylesheet to the QApplication."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f7f9fb"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f8fafc"))
    palette.setColor(QPalette.Text, QColor("#0f172a"))
    palette.setColor(QPalette.Button, QColor("#e2e8f0"))
    palette.setColor(QPalette.ButtonText, QColor("#0f172a"))
    palette.setColor(QPalette.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(_QSS)


def apply_matplotlib_theme() -> None:
    """Apply a Matplotlib theme aligned with the GUI palette."""

    if _mpl is None:
        return
    rc = _mpl.rcParams
    rc.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Source Sans Pro",
                "Arial",
                "Helvetica",
                "sans-serif",
            ],
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": "#cbd5e1",
            "grid.alpha": 0.45,
            "xtick.color": "#0f172a",
            "ytick.color": "#0f172a",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "figure.facecolor": "#f7f9fb",
            "savefig.facecolor": "#ffffff",
            "axes.prop_cycle": _mpl.cycler(
                "color",
                ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#0ea5e9", "#dc2626"],
            ),
        }
    )
