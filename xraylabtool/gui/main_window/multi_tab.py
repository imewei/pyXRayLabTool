from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PySide6.QtCore import QStandardPaths, Qt, QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xraylabtool.logging_utils import get_logger

from ..services import EnergyConfig, compute_multiple
from ..table_formatter import TableFormatter
from ..widgets.material_table import MaterialTable
from ..widgets.plot_canvas import PlotCanvas
from ..widgets.sweep_plots import MultiF1F2Plot
from ..workers import CalculationWorker
from .single_tab import PROPERTIES

if TYPE_CHECKING:
    from ._protocol import MainWindowProtocol

logger = get_logger("xraylabtool.gui.main_window")


class MultiTabMixin:
    """Widgets and behavior for the 'Multiple Materials' tab."""

    # See the matching declaration in SingleTabMixin for why this is needed.
    threadpool: QThreadPool | None

    def _build_multi_tab(self: MainWindowProtocol) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Material entry row
        self.multi_formula = QLineEdit()
        self.multi_formula.setPlaceholderText("e.g. SiO2")
        self.multi_formula.setToolTip("Enter chemical formula for the material")
        self.multi_density = QDoubleSpinBox()
        self.multi_density.setRange(0.001, 100.0)
        self.multi_density.setDecimals(4)
        self.multi_density.setValue(2.2)
        self.multi_density.setSuffix(" g/cm³")
        self.multi_density.setToolTip("Mass density in g/cm³")

        add_btn = QPushButton("Add material")
        add_btn.clicked.connect(self._add_material)
        add_btn.setShortcut("Alt+A")
        add_btn.setToolTip("Add the formula/density to the list (Alt+A)")
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_material)
        remove_btn.setShortcut("Alt+R")
        remove_btn.setToolTip("Remove selected rows (Alt+R)")

        self.multi_preset = QComboBox()
        self.multi_preset.addItem("Add preset material")
        for name in self.material_presets:
            self.multi_preset.addItem(name)
        self.multi_preset.currentTextChanged.connect(self._add_multi_preset)
        self.multi_preset.setToolTip("Quickly add a common material")

        add_btn.setProperty("class", "primary")
        remove_btn.setProperty("class", "secondary")

        material_box = QGroupBox("Materials")
        entry_row = QGridLayout()
        entry_row.setHorizontalSpacing(10)
        entry_row.addWidget(QLabel("Formula"), 0, 0)
        entry_row.addWidget(self.multi_formula, 0, 1)
        entry_row.addWidget(QLabel("Density"), 0, 2)
        entry_row.addWidget(self.multi_density, 0, 3)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(10)
        buttons_row.addStretch(1)
        buttons_row.addWidget(add_btn)
        buttons_row.addWidget(remove_btn)

        entry_row.addLayout(buttons_row, 1, 0, 1, 6)
        entry_row.addWidget(QLabel("Preset"), 2, 0)
        entry_row.addWidget(self.multi_preset, 2, 1, 1, 5)
        material_box.setLayout(entry_row)

        # Energy controls
        self.multi_energy_start = QDoubleSpinBox()
        self.multi_energy_start.setRange(0.03, 30.0)
        self.multi_energy_start.setDecimals(3)
        self.multi_energy_start.setValue(8.0)
        self.multi_energy_start.setSuffix(" keV")

        self.multi_energy_end = QDoubleSpinBox()
        self.multi_energy_end.setRange(0.03, 30.0)
        self.multi_energy_end.setDecimals(3)
        self.multi_energy_end.setValue(12.0)
        self.multi_energy_end.setSuffix(" keV")

        self.multi_energy_points = QSpinBox()
        self.multi_energy_points.setRange(1, 5000)
        self.multi_energy_points.setValue(50)

        self.multi_logspace = QCheckBox("Log-spaced energies")
        self.multi_logspace.setToolTip("Use logarithmic spacing for the energy grid")

        energy_box = QGroupBox("Energy range")
        energy_layout = QHBoxLayout()
        energy_layout.setSpacing(10)
        energy_layout.addWidget(QLabel("Start"))
        energy_layout.addWidget(self.multi_energy_start)
        energy_layout.addWidget(QLabel("End"))
        energy_layout.addWidget(self.multi_energy_end)
        energy_layout.addWidget(QLabel("Points"))
        energy_layout.addWidget(self.multi_energy_points)
        energy_layout.addWidget(self.multi_logspace)
        energy_layout.addStretch(1)
        energy_box.setLayout(energy_layout)

        self.multi_table = MaterialTable()
        self.multi_table.setMinimumHeight(160)

        self.multi_property = QComboBox()
        self.multi_property.addItems(PROPERTIES)
        self.multi_property.currentTextChanged.connect(self._refresh_multi_views)
        self.multi_property.setToolTip(
            "Choose which property to compare across materials"
        )

        self.multi_compute_btn = QPushButton("Compute comparison")
        self.multi_compute_btn.setProperty("class", "primary")
        self.multi_compute_btn.setShortcut("Ctrl+Shift+Return")
        self.multi_compute_btn.setToolTip(
            "Compute properties for listed materials (Ctrl+Shift+Enter)"
        )
        self.multi_compute_btn.clicked.connect(self._run_multi)
        self.multi_save_png = QPushButton("Save plot PNG")
        self.multi_save_png.setProperty("class", "secondary")
        self.multi_save_png.setShortcut("Ctrl+Alt+S")
        self.multi_save_png.setToolTip("Export comparison plot (Ctrl+Alt+S)")
        self.multi_save_png.clicked.connect(self._save_multi_png)
        self.multi_export_csv = QPushButton("Export CSV")
        self.multi_export_csv.setProperty("class", "secondary")
        self.multi_export_csv.setShortcut("Ctrl+Alt+E")
        self.multi_export_csv.setToolTip("Export comparison data (Ctrl+Alt+E)")
        self.multi_export_csv.clicked.connect(self._export_multi_csv)

        self.multi_logx = QCheckBox("Log X")
        self.multi_logy = QCheckBox("Log Y")
        self.multi_logx.stateChanged.connect(self._refresh_multi_views)
        self.multi_logy.stateChanged.connect(self._refresh_multi_views)

        # Plot tabs
        self.multi_plot = PlotCanvas()
        self.multi_f1f2_plot = MultiF1F2Plot()
        self.multi_plot_tabs = QTabWidget()
        self.multi_plot_tabs.setMinimumHeight(260)
        self.multi_plot_tabs.addTab(self.multi_plot, "Property plot")
        self.multi_plot_tabs.addTab(self.multi_f1f2_plot, "f1 / f2")

        # Full-parameter table (long-form): same parameters as Single, with Material/Density
        self.multi_full_table = QTableWidget(0, 14)
        self.multi_full_table.setAlternatingRowColors(True)
        self.multi_full_table.setHorizontalHeaderLabels(
            [
                "Material",
                "Density (g/cm³)",
                "Energy (keV)",
                "Wavelength (Å)",
                "δ",
                "β",
                "θc (deg)",
                "θc (mrad)",
                "Atten. length (cm)",
                "μ (1/cm)",
                "f1 (e)",
                "f2 (e)",
                "Re SLD (Å⁻²)",
                "Im SLD (Å⁻²)",
            ]
        )
        self.multi_full_table.verticalHeader().setVisible(False)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        header_row.addWidget(QLabel("Property"))
        header_row.addWidget(self.multi_property)
        header_row.addWidget(self.multi_logx)
        header_row.addWidget(self.multi_logy)
        header_row.addStretch(1)
        header_row.addWidget(self.multi_save_png)
        header_row.addWidget(self.multi_export_csv)

        multi_plot_container = QWidget()
        # Give the scroll area real overflow so the scrollbar can actually scroll.
        multi_plot_container.setMinimumHeight(720)
        multi_plot_layout = QVBoxLayout(multi_plot_container)
        multi_plot_layout.setContentsMargins(0, 0, 0, 0)
        multi_plot_layout.setSpacing(0)
        multi_plot_layout.addWidget(self.multi_plot_tabs)

        self.multi_plot_scroll = QScrollArea()
        self.multi_plot_scroll.setWidgetResizable(True)
        self.multi_plot_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.multi_plot_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.multi_plot_scroll.setWidget(multi_plot_container)
        self._reserve_overlay_scrollbar_space(self.multi_plot_scroll)

        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(24)
        left_layout.addWidget(material_box)
        left_layout.addWidget(energy_box)
        left_layout.addWidget(self.multi_table)
        left_layout.addWidget(self.multi_compute_btn)
        left_layout.addStretch(1)

        right_layout = QGridLayout()
        right_layout.setHorizontalSpacing(24)
        right_layout.setVerticalSpacing(16)
        right_layout.addLayout(header_row, 0, 0)
        right_layout.addWidget(self.multi_plot_scroll, 1, 0)
        right_layout.setRowStretch(1, 2)

        layout = QGridLayout()
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(20)
        layout.addWidget(left_panel, 0, 0, 1, 1)
        layout.addLayout(right_layout, 0, 1, 1, 1)
        layout.addWidget(self.multi_full_table, 1, 0, 1, 2)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(1, 1)
        outer.addLayout(layout)
        return container

    def _add_material(self: MainWindowProtocol) -> None:
        formula = self.multi_formula.text().strip()
        density = float(self.multi_density.value())
        try:
            self.multi_table.add_material(formula, density)
            self.multi_formula.clear()
            self.multi_formula.setFocus()
            logger.info(
                "multi_add_material", extra={"formula": formula, "density": density}
            )
        except ValueError as exc:
            self.toast.show_toast(str(exc), "error")

    def _add_multi_preset(self: MainWindowProtocol, name: str) -> None:
        if name in self.material_presets:
            self.multi_formula.setText(name)
            self.multi_density.setValue(self.material_presets[name])
            self._add_material()
        self.multi_preset.setCurrentIndex(0)

    def _remove_material(self: MainWindowProtocol) -> None:
        self.multi_table.remove_selected()
        logger.info(
            "multi_remove_material",
            extra={"remaining": len(self.multi_table.materials()[0])},
        )

    def _multi_energy_cfg(self: MainWindowProtocol) -> EnergyConfig:
        return EnergyConfig(
            start_kev=self.multi_energy_start.value(),
            end_kev=self.multi_energy_end.value(),
            points=self.multi_energy_points.value(),
            logspace=self.multi_logspace.isChecked(),
        )

    def _run_multi(self: MainWindowProtocol) -> None:
        formulas, densities = self.multi_table.materials()
        if not formulas:
            self._error("Add at least one material")
            return
        energy_cfg = self._multi_energy_cfg()
        logger.info(
            "multi_compute_clicked",
            extra={
                "count": len(formulas),
                "points": energy_cfg.points,
                "logspace": energy_cfg.logspace,
            },
        )
        self._info("Computing…")
        self._show_progress(True, 0)
        self.multi_compute_btn.setEnabled(False)
        self.multi_compute_btn.setText("Computing...")
        self.multi_save_png.setEnabled(False)
        self.multi_export_csv.setEnabled(False)
        if self.threadpool is None:
            self.threadpool = QThreadPool.globalInstance()
        worker = CalculationWorker(
            compute_multiple,
            formulas,
            densities,
            energy_cfg,
        )
        worker.signals.progress.connect(self._progress_multi)
        worker.signals.finished.connect(self._on_multi_finished)
        worker.signals.error.connect(self._on_multi_error)
        self._track_worker(worker)
        if self.threadpool is not None:
            self.threadpool.start(worker)

    def _on_multi_finished(self: MainWindowProtocol, results: Any) -> None:
        self.multi_compute_btn.setEnabled(True)
        self.multi_compute_btn.setText("Compute comparison")
        self.multi_save_png.setEnabled(True)
        self.multi_export_csv.setEnabled(True)
        self._show_progress(False, 0)
        self.multi_results = results
        logger.info(
            "multi_compute_complete",
            extra={"count": len(results), "first": next(iter(results.keys()), "")},
        )
        self.multi_comparison = None
        self._info("Multi-material comparison complete")
        self.toast.show_toast("Comparison done", "success")
        self._refresh_multi_views()

    def _on_multi_error(self: MainWindowProtocol, message: str) -> None:
        self.multi_compute_btn.setEnabled(True)
        self.multi_compute_btn.setText("Compute comparison")
        self.multi_save_png.setEnabled(True)
        self.multi_export_csv.setEnabled(True)
        self._show_progress(False, 0)
        logger.error("multi_compute_failed", extra={"message": message})
        self._error(message)

    def _progress_multi(self: MainWindowProtocol, value: int) -> None:
        self._show_progress(True, value)

    def _refresh_multi_views(self: MainWindowProtocol) -> None:
        if not self.multi_results:
            return
        prop = self.multi_property.currentText()
        self.multi_plot.set_scales(
            self.multi_logx.isChecked(), self.multi_logy.isChecked()
        )
        ylabel = self._label_for_property(prop)
        self.multi_plot.plot_multi(self.multi_results, prop, ylabel)

        # f1/f2 plot
        self.multi_f1f2_plot.render_multi(self.multi_results)

        # Full-parameter table (long-form)
        total_rows = sum(len(res.energy_kev) for res in self.multi_results.values())
        self.multi_full_table.setRowCount(total_rows)
        row_idx = 0
        for formula, res in self.multi_results.items():
            for i in range(len(res.energy_kev)):
                cells = TableFormatter.format_multi_row(formula, res, i)
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col >= 1:
                        item.setTextAlignment(
                            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                        )
                    self.multi_full_table.setItem(row_idx, col, item)
                row_idx += 1
        self.multi_full_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Export helpers
    def _save_multi_png(self: MainWindowProtocol) -> None:
        if not self.multi_results:
            self._error("No data to export yet")
            return
        prop = self.multi_property.currentText()
        logger.info("multi_save_png_clicked", extra={"property": prop})
        current_plot = self.multi_plot_tabs.currentWidget()
        self._save_plot(current_plot, f"multi_{prop}.png")

    def _export_multi_csv(self: MainWindowProtocol) -> str | None:
        if not self.multi_results:
            self._error("No data to export yet")
            return None
        logger.info("multi_export_csv_clicked")
        fname = "multi_full.csv"
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        if not default_dir:
            default_dir = str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            cast(QWidget, self), "Select folder to save CSV", default_dir
        )
        if not folder:
            logger.info("export_multi_cancelled", extra={"suggested": fname})
            return None
        path = str(Path(folder) / fname)

        headers = [
            "material",
            "density_g_cm3",
            "energy_kev",
            "wavelength_angstrom",
            "delta",
            "beta",
            "critical_angle_deg",
            "critical_angle_mrad",
            "attenuation_length_cm",
            "mu_1_per_cm",
            "f1",
            "f2",
            "real_sld_per_ang2",
            "imag_sld_per_ang2",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for formula, res in self.multi_results.items():
                    density = getattr(res, "density_g_cm3", 0.0)
                    for i, e in enumerate(res.energy_kev):
                        crit_deg = res.critical_angle_degrees[i]
                        crit_mrad = crit_deg * 3.141592653589793 / 180.0 * 1000.0
                        atten = res.attenuation_length_cm[i]
                        mu = 1.0 / atten if atten != 0 else 0.0
                        writer.writerow(
                            [
                                formula,
                                density,
                                e,
                                res.wavelength_angstrom[i],
                                res.dispersion_delta[i],
                                res.absorption_beta[i],
                                crit_deg,
                                crit_mrad,
                                atten,
                                mu,
                                res.scattering_factor_f1[i],
                                res.scattering_factor_f2[i],
                                res.real_sld_per_ang2[i],
                                res.imaginary_sld_per_ang2[i],
                            ]
                        )
        except OSError as exc:
            self._error(f"Could not save CSV: {exc}")
            logger.exception("export_multi_csv_failed", extra={"path": path})
            return None
        self._info(f"Saved CSV to {path}")
        logger.info(
            "export_multi_csv",
            extra={"path": path, "materials": len(self.multi_results)},
        )
        return path
