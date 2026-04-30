# ADR-002: PyQtGraph vs Matplotlib for GUI Plotting

**Status:** ACCEPTED
**Date:** 2026-04-06
**Deciders:** Architecture Team
**Supersedes:** None

---

## Context

The pyXRayLabTool GUI currently uses matplotlib embedded in PySide6 via `FigureCanvasQTAgg` and `NavigationToolbar2QT`. There are three matplotlib-based plot widgets:

| Widget | File | Purpose | Complexity |
|--------|------|---------|------------|
| `PlotCanvas` | `gui/widgets/plot_canvas.py` | Primary property plot (energy vs. property) | Medium -- log axes, grid, legend, markers |
| `F1F2Plot` | `gui/widgets/sweep_plots.py` | Single-material f1/f2 scattering factors | Simple -- two line plots |
| `MultiF1F2Plot` | `gui/widgets/sweep_plots.py` | Multi-material f1/f2 comparison (2 subplots) | Medium -- shared x-axis, dual subplot |

All three widgets follow the same pattern:
1. `self.figure = Figure(figsize=...)`
2. `self.canvas = FigureCanvasQTAgg(self.figure)`
3. `self.figure.clear()` then `self.figure.add_subplot(111)`
4. `ax.plot(x, y, label=..., marker=..., linewidth=...)`
5. `ax.set_xlabel/ylabel/xscale`
6. `self.canvas.draw_idle()`

**Pain points with matplotlib in the GUI:**
1. **Rendering speed:** `canvas.draw_idle()` redraws the entire figure. For energy sweeps with 500+ points across 5+ materials, plot updates feel sluggish.
2. **Theme integration:** The `update_theme()` method on each widget manually iterates over axes, spines, ticks, and legend to apply Qt palette colors via `mpl.rcParams`. This is fragile and requires synchronization with the `ColorPalette` dataclass.
3. **Interactivity:** matplotlib's `NavigationToolbar` provides zoom/pan but no smooth real-time interaction (e.g., hover tooltips, cursor tracking).
4. **Memory:** Each `FigureCanvasQTAgg` embeds a full Agg renderer. Three plot widgets means three independent renderers in memory.
5. **Import time:** matplotlib is one of the heaviest imports, adding ~500ms to GUI startup.

**What PyQtGraph offers:**
1. **Native Qt rendering:** Renders directly via QPainter, so it shares the Qt event loop and palette natively. No Agg renderer overhead.
2. **Real-time performance:** Optimized for live data display (oscilloscope-style). Can handle 10k+ points at 60fps.
3. **Built-in interactivity:** Mouse-wheel zoom, drag-to-pan, hover crosshair, region-of-interest selection -- all built in.
4. **Theme integration:** Uses Qt's QPalette directly. The existing `ColorPalette.to_qpalette()` method works without additional matplotlib synchronization.
5. **Lighter weight:** ~50MB vs matplotlib's ~100MB. Faster import.

## Decision

**Replace matplotlib with PyQtGraph for all interactive GUI plots. Retain matplotlib as an optional dependency for publication-quality static export.**

Migration strategy:
1. Define a `PlotWidget` protocol matching the current interface (`clear()`, `plot_single()`, `plot_multi()`, `update_theme()`, `set_scales()`).
2. Implement `PyQtGraphPlotCanvas`, `PyQtGraphF1F2Plot`, `PyQtGraphMultiF1F2Plot` behind this protocol.
3. Swap imports in `gui/main_window.py` from matplotlib widgets to PyQtGraph widgets.
4. Remove `apply_matplotlib_theme()` from `gui/style.py`.
5. Add an optional `export_publication_plot()` function that uses matplotlib for high-DPI vector export (PDF/SVG).

## Consequences

### Positive
- **Rendering performance:** Plot updates for 500+ point energy sweeps will be near-instantaneous instead of the ~100ms matplotlib redraw.
- **Theme consistency:** PyQtGraph inherits the Qt palette automatically. The manual `update_theme()` methods (25 lines each across 3 widgets) become trivial or unnecessary.
- **Interactivity:** Users get smooth zoom/pan, hover crosshair with value readout, and real-time cursor tracking without custom code.
- **Startup time:** Removing matplotlib from the GUI import chain saves ~500ms of startup time.
- **Code reduction:** The three matplotlib widget classes (~165 lines total) can be replaced with simpler PyQtGraph equivalents (~100 lines total), since PyQtGraph handles more out of the box.

### Negative
- **Log-scale axes:** PyQtGraph's `setLogMode(x=True)` works but tick formatting is less polished than matplotlib. May need custom `AxisItem` subclass for publication-quality log ticks.
- **Legend placement:** PyQtGraph's `LegendItem` has less automatic layout intelligence than matplotlib's `ax.legend()`. May need manual positioning.
- **Publication export:** PyQtGraph produces raster output by default. For vector (PDF/SVG) export, either use PyQtGraph's `exportFile()` or fall back to matplotlib.
- **Familiarity:** Team members experienced with matplotlib will need to learn PyQtGraph's API (PlotWidget, PlotItem, PlotDataItem).

### Migration Details

**Current matplotlib pattern:**
```python
# plot_canvas.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class PlotCanvas(QWidget):
    def __init__(self):
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

    def plot_single(self, result, property_name, ylabel):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, label=label, marker='o', linewidth=1.5)
        ax.set_xlabel("Energy (keV)")
        self.canvas.draw_idle()
```

**Target PyQtGraph pattern:**
```python
# plot_canvas.py
import pyqtgraph as pg

class PlotCanvas(QWidget):
    def __init__(self):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(None)  # Use Qt palette
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

    def plot_single(self, result, property_name, ylabel):
        self.plot_widget.clear()
        self.plot_widget.plot(x, y, name=label,
            pen=pg.mkPen(width=1.5), symbol='o', symbolSize=6)
        self.plot_widget.setLabel('bottom', "Energy (keV)")
```

**Log scale handling:**
```python
# PyQtGraph log mode
self.plot_widget.setLogMode(x=self.log_x, y=self.log_y)
# Custom tick formatting for nice log labels
if self.log_x:
    self.plot_widget.getAxis('bottom').setStyle(tickTextOffset=4)
```

---

## Appendix: Feature Parity Checklist

| Feature | matplotlib (current) | PyQtGraph (target) | Notes |
|---------|---------------------|--------------------|----|
| Line plot | `ax.plot()` | `pw.plot()` | Direct equivalent |
| Markers | `marker='o', markersize=6` | `symbol='o', symbolSize=6` | Direct equivalent |
| Log axes | `ax.set_xscale('log')` | `pw.setLogMode(x=True)` | Direct equivalent |
| Grid | `ax.grid(True, alpha=0.3)` | `pw.showGrid(x=True, y=True, alpha=0.3)` | Direct equivalent |
| Legend | `ax.legend()` | `pw.addLegend()` | PyQtGraph needs manual call |
| Axis labels | `ax.set_xlabel()` | `pw.setLabel('bottom', ...)` | Different API |
| Dual subplot | `fig.add_subplot(211)` | Two `PlotWidget` in `QSplitter` | Architectural change |
| Theme colors | Manual `rcParams` sync | Qt QPalette (automatic) | Simplification |
| Zoom/Pan | `NavigationToolbar` | Built-in mouse interaction | Feature upgrade |
| Export | `fig.savefig()` | `pg.exportFile()` or matplotlib fallback | May need both |
