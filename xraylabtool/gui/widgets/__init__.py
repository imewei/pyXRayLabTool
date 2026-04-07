"""Custom widgets for the XRayLabTool GUI."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from xraylabtool.gui.style import DARK_THEME, LIGHT_THEME, ColorPalette


def current_palette() -> ColorPalette:
    """Detect the active palette by checking the application background color."""
    app = QGuiApplication.instance()
    if isinstance(app, QApplication):
        bg = app.palette().window().color()
        if bg.lightness() < 64:
            return DARK_THEME
    return LIGHT_THEME


def apply_palette_to_widget(pw: Any, palette: ColorPalette) -> None:
    """Apply background and foreground colors from palette to a PlotWidget."""
    pw.setBackground(palette.plot_bg)
    for axis_name in ("bottom", "left", "top", "right"):
        axis = pw.getAxis(axis_name)
        axis.setTextPen(pg.mkPen(color=palette.text_primary))
        axis.setPen(pg.mkPen(color=palette.border))
    pw.showGrid(x=True, y=True, alpha=0.3)


def apply_palette_to_item(pi: Any, palette: ColorPalette) -> None:
    """Apply background/foreground colors from palette to a PlotItem."""
    pi.getViewBox().setBackgroundColor(palette.plot_bg)
    for axis_name in ("bottom", "left", "top", "right"):
        axis = pi.getAxis(axis_name)
        axis.setTextPen(pg.mkPen(color=palette.text_primary))
        axis.setPen(pg.mkPen(color=palette.border))
    pi.showGrid(x=True, y=True, alpha=0.3)
