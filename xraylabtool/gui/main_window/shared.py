from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QObject, QStandardPaths, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QScrollArea,
    QWidget,
)

from xraylabtool.logging_utils import get_log_file_path, get_logger

from ..widgets.scrollbar_helper import OverlayScrollbarMarginHelper

if TYPE_CHECKING:
    from ._protocol import MainWindowProtocol

logger = get_logger("xraylabtool.gui.main_window")


class SharedHelpersMixin:
    """Window-lifecycle, layout-tuning, and export helpers shared by both tabs."""

    def _handle_theme_toggle_click(self: MainWindowProtocol) -> None:
        if self.theme_manager:
            self.theme_manager.toggle_theme()

    def _on_theme_changed(self: MainWindowProtocol, mode: str) -> None:
        is_dark = mode == "dark"
        self.theme_toggle.setChecked(is_dark)
        self.theme_toggle.setText("Dark Mode" if is_dark else "Light Mode")
        self._refresh_plots()

    def _refresh_plots(self: MainWindowProtocol) -> None:
        """Force update of all plots to match new theme."""
        # Find all widgets with update_theme capability (PlotCanvas, F1F2Plot, etc.)
        # We search recursively
        for widget in self.findChildren(QWidget):
            if hasattr(widget, "update_theme"):
                widget.update_theme()

    def _tune_table_headers(self: MainWindowProtocol) -> None:
        def tune(
            table: Any,
            default_size: int = 110,
            min_size: int = 80,
            stretch_last: bool = True,
        ) -> None:
            hdr = table.horizontalHeader()
            hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            hdr.setDefaultSectionSize(default_size)
            hdr.setMinimumSectionSize(min_size)
            hdr.setStretchLastSection(stretch_last)
            hdr.setTextElideMode(Qt.TextElideMode.ElideMiddle)

        tune(self.single_table, default_size=110, min_size=80, stretch_last=False)
        self.single_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.single_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        tune(self.single_summary, default_size=120, min_size=90)
        tune(self.multi_full_table, default_size=120, min_size=90)

    def _reserve_overlay_scrollbar_space(
        self: MainWindowProtocol, scroll_area: QScrollArea
    ) -> None:
        """Avoid overlay scrollbars clipping the scroll area viewport.

        Some Qt styles render scrollbars as overlays (not consuming layout width).
        If the vertical scrollbar overlaps the viewport, reserve space via viewport
        margins so plot canvases and labels aren't clipped.
        """
        if not hasattr(self, "_scroll_overlay_helpers"):
            self._scroll_overlay_helpers: list[QObject] = []

        self._scroll_overlay_helpers.append(
            OverlayScrollbarMarginHelper(cast(QObject, self), scroll_area)
        )

    def _set_tab_order(self: MainWindowProtocol) -> None:
        # Single tab order
        self.setTabOrder(self.single_preset, self.energy_preset)
        self.setTabOrder(self.energy_preset, self.single_form.formula)
        self.setTabOrder(self.single_form.formula, self.single_form.density)
        self.setTabOrder(self.single_form.density, self.single_form.energy_start)
        self.setTabOrder(self.single_form.energy_start, self.single_form.energy_end)
        self.setTabOrder(self.single_form.energy_end, self.single_form.energy_points)
        self.setTabOrder(self.single_form.energy_points, self.single_form.logspace)
        self.setTabOrder(self.single_form.logspace, self.single_form.compute_button)
        self.setTabOrder(self.single_form.compute_button, self.single_property)
        self.setTabOrder(self.single_property, self.single_logx)
        self.setTabOrder(self.single_logx, self.single_logy)
        self.setTabOrder(self.single_logy, self.single_save_png)
        self.setTabOrder(self.single_save_png, self.single_export_csv)
        # Multi tab order (kept simple left-to-right, top-to-bottom)
        self.setTabOrder(self.multi_formula, self.multi_density)
        self.setTabOrder(self.multi_density, self.multi_preset)
        self.setTabOrder(self.multi_preset, self.multi_table)
        self.setTabOrder(self.multi_table, self.multi_energy_start)
        self.setTabOrder(self.multi_energy_start, self.multi_energy_end)
        self.setTabOrder(self.multi_energy_end, self.multi_energy_points)
        self.setTabOrder(self.multi_energy_points, self.multi_logspace)
        self.setTabOrder(self.multi_logspace, self.multi_compute_btn)
        self.setTabOrder(self.multi_compute_btn, self.multi_property)
        self.setTabOrder(self.multi_property, self.multi_logx)
        self.setTabOrder(self.multi_logx, self.multi_logy)
        self.setTabOrder(self.multi_logy, self.multi_save_png)
        self.setTabOrder(self.multi_save_png, self.multi_export_csv)

    # ------------------------------------------------------------------
    # Status helpers
    def _info(self: MainWindowProtocol, message: str) -> None:
        self.status_bar.showMessage(message, 5000)
        self.toast.show_toast(message, "info")

    def _error(self: MainWindowProtocol, message: str) -> None:
        self.status_bar.showMessage(message, 10000)
        self.toast.show_toast(message, "error", duration_ms=3500)
        QMessageBox.critical(cast(QWidget, self), "Error", message)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Drop queued-but-unstarted compute jobs so the app shuts down
        promptly instead of blocking on the global QThreadPool destructor
        while a large sweep is in flight."""
        win = cast("MainWindowProtocol", self)
        if win.threadpool is not None:
            win.threadpool.clear()
        super().closeEvent(event)  # type: ignore[misc]

    def _show_progress(self: MainWindowProtocol, active: bool, value: int = 0) -> None:
        if active:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, value)))
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _toggle_log_path(self: MainWindowProtocol) -> None:
        path = get_log_file_path()
        if path:
            if self.log_path_toggle.isChecked():
                self.log_path_label.setText(f"Log: {path}")
                self.log_path_label.setVisible(True)
                logger.info("log_path_shown", extra={"path": path})
            else:
                self.log_path_label.clear()
                self.log_path_label.setVisible(False)
        else:
            self.status_bar.showMessage("File logging is disabled", 5000)
            logger.info("log_path_missing")

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)  # type: ignore[misc]
        if hasattr(self, "toast"):
            cast("MainWindowProtocol", self).toast._reposition()

    # ------------------------------------------------------------------
    # Export helpers

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        """Replace characters unsafe for filenames with underscores."""
        return re.sub(r"[^\w\-.]", "_", text)

    def _save_plot(
        self: MainWindowProtocol, plot_widget: QWidget, suggested: str
    ) -> None:
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        )
        if not default_dir:
            default_dir = str(Path.home())
        path, _ = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Save plot",
            str(Path(default_dir) / suggested),
            "PNG Files (*.png)",
        )
        if not path:
            logger.info("save_plot_cancelled", extra={"suggested": suggested})
            return
        if not self._render_plot_to_png(plot_widget, path):
            self._error("Could not render plot to image")
            logger.error("plot_save_failed", extra={"path": path})
            return
        logger.info("plot_saved", extra={"path": path, "suggested": suggested})
        self._info(f"Saved plot to {path}")

    def _render_plot_to_png(self, plot_widget: QWidget, path: str) -> bool:
        """Export a PyQtGraph plot widget to a PNG file.

        The plot widgets wrap either a ``pg.PlotWidget`` (``.plot_widget``) or a
        ``pg.GraphicsLayoutWidget`` (``.layout_widget``); both expose a scene that
        ImageExporter renders offscreen, so this works without the window shown
        (e.g. headless smoke tests). Falls back to grabbing the rendered widget.
        """
        view = getattr(plot_widget, "plot_widget", None) or getattr(
            plot_widget, "layout_widget", None
        )
        if view is not None:
            try:
                from pyqtgraph.exporters import ImageExporter

                exporter = ImageExporter(view.scene())
                exporter.parameters()["width"] = 1200
                exporter.export(path)
                return Path(path).exists() and Path(path).stat().st_size > 0
            except Exception:
                logger.exception("plot_export_failed", extra={"path": path})
        pixmap = plot_widget.grab()
        return bool(not pixmap.isNull() and pixmap.save(path, "PNG"))

    def _label_for_property(self, prop: str) -> str:
        labels = {
            "attenuation_length_cm": "Attenuation length (cm)",
            "dispersion_delta": "Dispersion δ",
            "absorption_beta": "Absorption β",
            "critical_angle_degrees": "Critical angle (deg)",
            "real_sld_per_ang2": "Real SLD (Å⁻²)",
            "imaginary_sld_per_ang2": "Imag SLD (Å⁻²)",
        }
        return labels.get(prop, prop.replace("_", " "))

    def _track_worker(self: MainWindowProtocol, worker):  # type: ignore[no-untyped-def]
        self._workers.append(worker)
        worker.signals.finished.connect(lambda _res, w=worker: self._cleanup_worker(w))
        worker.signals.error.connect(lambda _msg, w=worker: self._cleanup_worker(w))

    def _cleanup_worker(self: MainWindowProtocol, worker):  # type: ignore[no-untyped-def]
        if worker in self._workers:
            self._workers.remove(worker)
