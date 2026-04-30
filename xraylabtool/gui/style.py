"""Shared Qt styling for the XRayLabTool GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


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
    plot_cycle: list[str]

    def to_qpalette(self) -> QPalette:
        """Convert to QPalette."""
        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(self.window_bg))
        p.setColor(QPalette.ColorRole.Base, QColor(self.input_bg))
        p.setColor(QPalette.ColorRole.AlternateBase, QColor(self.panel_bg))
        p.setColor(QPalette.ColorRole.Text, QColor(self.text_primary))
        p.setColor(QPalette.ColorRole.WindowText, QColor(self.text_primary))
        p.setColor(QPalette.ColorRole.Button, QColor(self.panel_bg))
        p.setColor(QPalette.ColorRole.ButtonText, QColor(self.text_primary))
        p.setColor(QPalette.ColorRole.Highlight, QColor(self.accent))
        p.setColor(QPalette.ColorRole.HighlightedText, QColor(self.accent_text))
        p.setColor(QPalette.ColorRole.PlaceholderText, QColor(self.text_secondary))
        return p


# Enforce Dark Industrial Aesthetic (Overwriting both to ensure uniformity)
DARK_INDUSTRIAL = ColorPalette(
    name="dark_industrial",
    window_bg="#0a0a0a",  # Deep void black
    panel_bg="#141414",  # Industrial grey
    input_bg="#050505",  # Inset darkness for inputs
    text_primary="#f8f9fa",  # Stark white
    text_secondary="#737373",  # Muted technical grey
    border="#2a2a2a",  # Subtle machined edges
    border_focus="#00f0ff",  # Cyber-cyan focus
    accent="#00f0ff",  # Cyber-cyan primary
    accent_hover="#5cffff",  # Bright cyan hover
    accent_text="#000000",  # Black text on cyan buttons
    error="#ff9d00",  # Vibrant amber/orange
    error_bg="#2a1a00",  # Dark amber background
    success="#00ff9d",  # Tech green
    plot_bg="#050505",  # Deep background for plots
    plot_cycle=["#00f0ff", "#ff00aa", "#00ff9d", "#ff9d00", "#b700ff", "#ffee00"],
)

# Maintain legacy references pointing to the new unified aesthetic
LIGHT_THEME = DARK_INDUSTRIAL
DARK_THEME = DARK_INDUSTRIAL


def get_qss(t: ColorPalette) -> str:
    """Generate QSS string from palette."""
    return f"""
    /* Base typography + spacing */
    * {{
        font-family: "Roboto", "Segoe UI", -apple-system, sans-serif;
        font-size: 13px;
        color: {t.text_primary};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
    }}

    QMainWindow, QWidget {{
        background: {t.window_bg};
        color: {t.text_primary};
    }}

    /* Machined GroupBoxes */
    QGroupBox {{
        border: 1px solid {t.border};
        border-radius: 0px; /* Sharp corners */
        margin-top: 18px;
        padding: 14px;
        background: {t.panel_bg};
    }}

    QGroupBox::title {{
        color: {t.accent};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 1px;
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding-left: 4px;
        padding-right: 4px;
        top: 0px;
    }}

    QLabel {{
        color: {t.text_primary};
        font-weight: 500;
    }}

    QLabel[role="hint"] {{
        color: {t.text_secondary};
        font-size: 11px;
    }}

    QLabel[role="success"] {{
        color: {t.success};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
    }}

    QLabel[role="error"] {{
        color: {t.error};
        font-weight: bold;
    }}

    /* Form controls - Monospaced data entry */
    QCheckBox {{
        color: {t.text_primary};
        font-weight: 600;
        background: transparent;
    }}

    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border: 1px solid {t.border};
        background: {t.input_bg};
    }}

    QCheckBox::indicator:checked {{
        background: {t.accent};
        border: 1px solid {t.accent};
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        padding: 10px 12px;
        min-height: 20px;
        border: 1px solid {t.border};
        border-radius: 2px;
        background: {t.input_bg};
        color: {t.accent};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 13px;
    }}

    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 1px solid {t.border_focus};
        background: {t.window_bg};
    }}

    QLineEdit[validation="invalid"], QSpinBox[validation="invalid"], QDoubleSpinBox[validation="invalid"] {{
        border: 1px solid {t.error};
        background: {t.error_bg};
        color: {t.error};
    }}

    QComboBox::drop-down {{
        border: 0px;
        width: 20px;
    }}

    QComboBox QAbstractItemView {{
        background: {t.panel_bg};
        border: 1px solid {t.accent};
        color: {t.text_primary};
        selection-background-color: {t.accent};
        selection-color: {t.accent_text};
        font-family: "Roboto", sans-serif;
    }}

    /* Buttons - Hard edged */
    QPushButton {{
        padding: 10px 16px;
        min-height: 20px;
        border-radius: 2px;
        background: {t.panel_bg};
        border: 1px solid {t.border};
        color: {t.text_primary};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-weight: bold;
        font-size: 11px;
        text-transform: uppercase;
    }}

    QPushButton:hover {{
        background: {t.border};
        border-color: {t.accent};
        color: {t.accent};
    }}

    QPushButton:pressed {{
        background: {t.accent};
        color: {t.accent_text};
    }}

    QPushButton:disabled {{
        color: {t.text_secondary};
        background: {t.window_bg};
        border-color: {t.window_bg};
        opacity: 0.5;
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
        border-radius: 0px;
        padding: 8px;
        background: {t.window_bg};
    }}

    QTabBar::tab {{
        background: {t.panel_bg};
        color: {t.text_secondary};
        padding: 12px 20px;
        border: 1px solid {t.border};
        border-bottom: 0;
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
        min-width: 120px;
    }}

    QTabBar::tab:hover {{
        color: {t.text_primary};
    }}

    QTabBar::tab:selected {{
        background: {t.window_bg};
        color: {t.accent};
        border-top: 2px solid {t.accent};
    }}

    /* Tables - Data grid look */
    QTableWidget {{
        background: {t.input_bg};
        color: {t.text_primary};
        alternate-background-color: {t.panel_bg};
        gridline-color: {t.window_bg};
        selection-background-color: {t.border};
        selection-color: {t.accent};
        border: 1px solid {t.border};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 12px;
    }}

    QHeaderView::section {{
        background: {t.panel_bg};
        padding: 6px;
        border: none;
        border-bottom: 1px solid {t.border};
        border-right: 1px solid {t.window_bg};
        color: {t.text_secondary};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 11px;
        font-weight: bold;
        text-transform: uppercase;
    }}

    QStatusBar {{
        background: {t.panel_bg};
        color: {t.accent};
        font-family: "Fira Code", "JetBrains Mono", "Consolas", monospace;
        font-size: 11px;
        border-top: 1px solid {t.border};
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        background: {t.window_bg};
        width: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:vertical {{
        background: {t.border};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {t.text_secondary};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    QScrollBar:horizontal {{
        background: {t.window_bg};
        height: 10px;
        margin: 0px 0px 0px 0px;
    }}
    QScrollBar::handle:horizontal {{
        background: {t.border};
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {t.text_secondary};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}
    """


def apply_theme(app: QApplication, theme: ColorPalette) -> None:
    """Apply palette and stylesheet to the QApplication."""
    app.setPalette(theme.to_qpalette())
    app.setStyleSheet(get_qss(theme))


def apply_styles(app: QApplication) -> None:
    """Entry point: Defaults to Dark Industrial Theme."""
    apply_theme(app, DARK_INDUSTRIAL)


def apply_pyqtgraph_theme(theme: ColorPalette = DARK_INDUSTRIAL) -> None:
    """Apply a PyQtGraph theme aligned with the GUI palette."""
    try:
        import pyqtgraph as pg  # type: ignore[import-untyped]
    except ImportError:
        return

    pg.setConfigOptions(
        background=theme.plot_bg,
        foreground=theme.text_secondary,  # Subdued axes
        antialias=True,
    )

    # Custom PyQtGraph default styles for lines
    pg.setConfigOption("leftButtonPan", False)
