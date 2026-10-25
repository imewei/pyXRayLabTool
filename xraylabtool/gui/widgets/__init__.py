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


def legend_kwargs(palette: ColorPalette | None = None) -> dict[str, Any]:
    """Return ``addLegend`` styling: theme text and a translucent plot-colored box."""
    palette = palette or current_palette()
    fill = pg.mkColor(palette.plot_bg)
    fill.setAlpha(210)
    return {
        "labelTextColor": palette.text_primary,
        "brush": pg.mkBrush(fill),
        "pen": pg.mkPen(color=palette.border),
    }


def _restyle_legend(legend: Any, palette: ColorPalette) -> None:
    if legend is None:
        return
    style = legend_kwargs(palette)
    legend.setLabelTextColor(style["labelTextColor"])
    for _sample, label in legend.items:
        label.setText(label.text)  # setLabelTextColor alone does not repaint
    legend.setBrush(style["brush"])
    legend.setPen(style["pen"])


def _recolor_axis_label(axis: Any, color: str) -> None:
    """Set the axis-title color, keeping the text (setLabel(color=) alone blanks it)."""
    axis.setLabel(axis.labelText, axis.labelUnits, axis.labelUnitPrefix, color=color)


def apply_palette_to_widget(pw: Any, palette: ColorPalette) -> None:
    """Apply background and foreground colors from palette to a PlotWidget."""
    pw.setBackground(palette.plot_bg)
    for axis_name in ("bottom", "left", "top", "right"):
        axis = pw.getAxis(axis_name)
        axis.setTextPen(pg.mkPen(color=palette.text_primary))
        axis.setPen(pg.mkPen(color=palette.border))
        _recolor_axis_label(axis, palette.text_primary)
    pw.showGrid(x=True, y=True, alpha=0.3)
    _restyle_legend(pw.getPlotItem().legend, palette)


def apply_palette_to_item(pi: Any, palette: ColorPalette) -> None:
    """Apply background/foreground colors from palette to a PlotItem."""
    pi.getViewBox().setBackgroundColor(palette.plot_bg)
    for axis_name in ("bottom", "left", "top", "right"):
        axis = pi.getAxis(axis_name)
        axis.setTextPen(pg.mkPen(color=palette.text_primary))
        axis.setPen(pg.mkPen(color=palette.border))
        _recolor_axis_label(axis, palette.text_primary)
    pi.showGrid(x=True, y=True, alpha=0.3)
    _restyle_legend(pi.legend, palette)
