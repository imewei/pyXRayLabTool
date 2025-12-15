"""Shared Qt styling for the XRayLabTool GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette

try:
    import matplotlib as _mpl
except ImportError:
    _mpl = None


@dataclass
class ColorPalette:
    """Semantic color tokens for the GUI."""

    name: str
    window_bg: str
    panel_bg: str
    input_bg: str
    text_primary: str
    text_secondary: str
    border: str
    border_focus: str
    accent: str
    accent_hover: str
    accent_text: str
    error: str
    error_bg: str
    success: str
    plot_bg: str
    mpl_cycle: list[str]

    def to_qpalette(self) -> QPalette:
        """Convert to QPalette."""
        p = QPalette()
        p.setColor(QPalette.Window, QColor(self.window_bg))
        p.setColor(QPalette.Base, QColor(self.input_bg))
        p.setColor(QPalette.AlternateBase, QColor(self.panel_bg))
        p.setColor(QPalette.Text, QColor(self.text_primary))
        p.setColor(QPalette.WindowText, QColor(self.text_primary))
        p.setColor(QPalette.Button, QColor(self.panel_bg))
        p.setColor(QPalette.ButtonText, QColor(self.text_primary))
        p.setColor(QPalette.Highlight, QColor(self.accent))
        p.setColor(QPalette.HighlightedText, QColor(self.accent_text))
        p.setColor(QPalette.PlaceholderText, QColor(self.text_secondary))
        return p


LIGHT_THEME = ColorPalette(
    name="light",
    window_bg="#f7f9fb",
    panel_bg="#f1f5f9",  # Slightly darker than window for contrast
    input_bg="#ffffff",
    text_primary="#0f172a",
    text_secondary="#64748b",
    border="#cbd5e1",
    border_focus="#2563eb",
    accent="#2563eb",
    accent_hover="#1d4ed8",
    accent_text="#ffffff",
    error="#dc2626",
    error_bg="#fee2e2",
    success="#16a34a",
    plot_bg="#ffffff",
    mpl_cycle=["#2563eb", "#f97316", "#16a34a", "#9333ea", "#0ea5e9", "#dc2626"],
)

DARK_THEME = ColorPalette(
    name="dark",
    window_bg="#0f172a",
    panel_bg="#1e293b",
    input_bg="#1e293b",  # Matches panel for unified look or slightly lighter
    text_primary="#f8fafc",
    text_secondary="#94a3b8",
    border="#334155",
    border_focus="#3b82f6",
    accent="#3b82f6",  # Lighter blue for dark mode
    accent_hover="#2563eb",
    accent_text="#ffffff",
    error="#ef4444",
    error_bg="#450a0a",
    success="#22c55e",
    plot_bg="#1e293b",  # Match panel
    mpl_cycle=["#3b82f6", "#fb923c", "#4ade80", "#a855f7", "#38bdf8", "#f87171"],
)


def get_qss(t: ColorPalette) -> str:
    """Generate QSS string from palette."""
    return f"""
    /* Base typography + spacing */
    * {{
        font-size: 14px;
        color: {t.text_primary};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}

    QMainWindow, QWidget {{
        background: {t.window_bg};
        color: {t.text_primary};
    }}

    QGroupBox {{
        border: 1px solid {t.border};
        border-radius: 6px;
        margin-top: 12px;
        padding: 10px;
        font-weight: 600;
        background: {t.window_bg};
    }}

    QGroupBox::title {{
        color: {t.text_primary};
    }}

    QLabel {{
        color: {t.text_primary};
    }}

    QLabel[role="hint"] {{
        color: {t.text_secondary};
    }}

    QLabel[role="success"] {{
        color: {t.success};
    }}

    QLabel[role="error"] {{
        color: {t.error};
    }}

    /* Form controls */
    QCheckBox {{
        color: {t.text_primary};
        font-weight: 600;
        background: transparent;
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        padding: 6px;
        border: 1px solid {t.border};
        border-radius: 4px;
        background: {t.input_bg};
        color: {t.text_primary};
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {t.border_focus};
    }}

    QLineEdit[validation="invalid"], QSpinBox[validation="invalid"], QDoubleSpinBox[validation="invalid"] {{
        border: 1px solid {t.error};
        background: {t.error_bg};
    }}

    QComboBox::drop-down {{
        border: 0px;
    }}

    QComboBox QAbstractItemView {{
        background: {t.input_bg};
        border: 1px solid {t.border};
        color: {t.text_primary};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}

    /* Buttons */
    QPushButton {{
        padding: 8px 12px;
        border-radius: 6px;
        background: {t.panel_bg};
        border: 1px solid {t.border};
        color: {t.text_primary};
        font-weight: 500;
    }}

    QPushButton:hover {{
        background: {t.border}; /* Slightly simplistic, but usually panel darker */
        border-color: {t.text_secondary};
    }}

    QPushButton:pressed {{
        background: {t.accent};
        color: {t.accent_text};
    }}

    QPushButton:disabled {{
        color: {t.text_secondary};
        background: {t.panel_bg};
        border-color: {t.border};
    }}

    QPushButton[class="primary"] {{
        background: {t.accent};
        border-color: {t.accent};
        color: {t.accent_text};
    }}

    QPushButton[class="primary"]:hover {{
        background: {t.accent_hover};
    }}

    /* Tabs */
    QTabWidget::pane {{
        border: 1px solid {t.border};
        border-radius: 6px;
        padding: 4px;
        margin-top: 6px;
        background: {t.window_bg};
    }}

    QTabBar::tab {{
        background: {t.panel_bg};
        color: {t.text_primary};
        padding: 8px 14px;
        border: 1px solid {t.border};
        border-bottom: 0;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 700;
        min-width: 140px;
    }}

    QTabBar::tab:hover {{
        background: {t.border}; /* Reuse border color as hover shade */
        color: {t.text_primary};
    }}

    QTabBar::tab:selected {{
        background: {t.window_bg};
        color: {t.accent};
        border-bottom: 2px solid {t.window_bg}; /* Blend with pane */
    }}

    QTabBar::tab:!selected {{
        margin-top: 2px;
    }}

    /* Tables */
    QTableWidget {{
        background: {t.input_bg};
        color: {t.text_primary};
        alternate-background-color: {t.panel_bg};
        gridline-color: {t.border};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
        border: 1px solid {t.border};
    }}

    QHeaderView::section {{
        background: {t.panel_bg};
        padding: 6px;
        border: 1px solid {t.border};
        color: {t.text_primary};
        font-weight: 600;
    }}

    QStatusBar {{
        background: {t.panel_bg};
        color: {t.text_primary};
        border-top: 1px solid {t.border};
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}
    """


def apply_theme(app, theme: ColorPalette) -> None:
    """Apply palette and stylesheet to the QApplication."""
    app.setPalette(theme.to_qpalette())
    app.setStyleSheet(get_qss(theme))


def apply_styles(app) -> None:
    """Legacy entry point: Defaults to Light Theme."""
    apply_theme(app, LIGHT_THEME)


def apply_matplotlib_theme(theme: ColorPalette = LIGHT_THEME) -> None:
    """Apply a Matplotlib theme aligned with the GUI palette."""
    if _mpl is None:
        return

    rc = _mpl.rcParams
    rc.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Source Sans Pro", "Arial", "sans-serif"],
            "axes.facecolor": theme.plot_bg,
            "axes.edgecolor": theme.border,
            "axes.labelcolor": theme.text_primary,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": theme.border,
            "grid.alpha": 0.45,
            "xtick.color": theme.text_primary,
            "ytick.color": theme.text_primary,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "figure.facecolor": theme.window_bg,
            "savefig.facecolor": theme.window_bg,
            "text.color": theme.text_primary,
            "axes.prop_cycle": _mpl.cycler("color", theme.mpl_cycle),
        }
    )
