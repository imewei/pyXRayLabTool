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
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xraylabtool.logging_utils import get_logger
from xraylabtool.utils import energy_to_wavelength, wavelength_to_energy

from ..services import compute_single
from ..table_formatter import TableFormatter
from ..widgets.material_form import MaterialInputForm
from ..widgets.plot_canvas import PlotCanvas
from ..widgets.sweep_plots import F1F2Plot
from ..workers import CalculationWorker

if TYPE_CHECKING:
    from ._protocol import MainWindowProtocol

PROPERTIES = [
    "attenuation_length_cm",
    "dispersion_delta",
    "absorption_beta",
    "critical_angle_degrees",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
]

logger = get_logger("xraylabtool.gui.main_window")


class SingleTabMixin:
    """Widgets and behavior for the 'Single Material' tab."""

    # Declared here (matching MainWindowProtocol/MainWindow.__init__) so mypy's
    # implicit-attribute inference from `self.threadpool = QThreadPool.globalInstance()`
    # below doesn't narrow it to non-Optional and conflict with __init__'s declaration.
    threadpool: QThreadPool | None

    def _build_single_tab(self: MainWindowProtocol) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # Inputs
        self.single_form = MaterialInputForm()
        self.single_form.compute_button.clicked.connect(self._run_single)

        # Presets
        self.single_preset = QComboBox()
        self.single_preset.addItem("Select material preset")
        for name in self.material_presets:
            self.single_preset.addItem(name)
        self.single_preset.currentTextChanged.connect(self._apply_single_preset)
        self.single_preset.setToolTip("Apply a common material formula and density")

        self.energy_preset = QComboBox()
        self.energy_preset.addItem("Select energy preset")
        for name in self.energy_presets:
            self.energy_preset.addItem(name)
        self.energy_preset.currentTextChanged.connect(self._apply_energy_preset)
        self.energy_preset.setToolTip(
            "Pick a frequently used energy sweep or single energy"
        )

        # Property chooser + export buttons
        self.single_property = QComboBox()
        self.single_property.addItems(PROPERTIES)
        self.single_property.currentTextChanged.connect(self._refresh_single_views)
        self.single_logx = QCheckBox("Log X")
        self.single_logy = QCheckBox("Log Y")
        self.single_logx.stateChanged.connect(self._refresh_single_views)
        self.single_logy.stateChanged.connect(self._refresh_single_views)
        self.single_property.setToolTip("Select which property to plot and export")
        self.single_logx.setToolTip("Toggle logarithmic X axis for plots")
        self.single_logy.setToolTip("Toggle logarithmic Y axis for plots")

        self.single_save_png = QPushButton("Save plot PNG")
        self.single_save_png.setProperty("class", "secondary")
        self.single_save_png.setShortcut("Ctrl+Shift+S")
        self.single_save_png.setToolTip("Export the current plot to PNG (Ctrl+Shift+S)")
        self.single_save_png.clicked.connect(self._save_single_png)
        self.single_export_csv = QPushButton("Export CSV")
        self.single_export_csv.setProperty("class", "secondary")
        self.single_export_csv.setShortcut("Ctrl+Shift+E")
        self.single_export_csv.setToolTip("Export table data to CSV (Ctrl+Shift+E)")
        self.single_export_csv.clicked.connect(self._export_single_csv)

        plot_header = QHBoxLayout()
        plot_header.setSpacing(12)
        plot_header.addWidget(QLabel("Property"))
        plot_header.addWidget(self.single_property)
        plot_header.addWidget(self.single_logx)
        plot_header.addWidget(self.single_logy)
        plot_header.addStretch(1)
        plot_header.addWidget(self.single_save_png)
        plot_header.addWidget(self.single_export_csv)

        self.single_summary = QTableWidget(1, 5)
        self.single_summary.setHorizontalHeaderLabels(
            [
                "Formula",
                "MW (g/mol)",
                "Density (g/cm³)",
                "Electron density (e/Å³)",
                "Total electrons",
            ]
        )
        self.single_summary.verticalHeader().setVisible(False)
        self.single_summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.single_summary.setMaximumHeight(64)
        self.single_summary.setMinimumHeight(48)
        self.single_summary.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        # Table
        # 12 columns: energy, wavelength, delta, beta, critical angles, attenuation, mu, f1/f2, SLDs
        self.single_table = QTableWidget(0, 12)
        self.single_table.setAlternatingRowColors(True)
        self.single_table.setHorizontalHeaderLabels(
            [
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
        self.single_table.verticalHeader().setVisible(False)

        # Plot tabs
        self.single_plot = PlotCanvas()
        self.single_f1f2 = F1F2Plot()

        # Converter
        converter = QGroupBox("Energy ↔ Wavelength")
        conv_layout = QHBoxLayout()
        conv_layout.setSpacing(10)
        self.conv_energy = QDoubleSpinBox()
        self.conv_energy.setRange(0.01, 100.0)
        self.conv_energy.setDecimals(4)
        self.conv_energy.setSuffix(" keV")
        self.conv_wavelength = QDoubleSpinBox()
        self.conv_wavelength.setRange(0.01, 10_000.0)
        self.conv_wavelength.setDecimals(4)
        self.conv_wavelength.setSuffix(" Å")
        btn_e2w = QPushButton("E→λ")
        btn_w2e = QPushButton("λ→E")
        btn_e2w.clicked.connect(self._convert_e2w)
        btn_w2e.clicked.connect(self._convert_w2e)
        conv_layout.addWidget(QLabel("Energy"))
        conv_layout.addWidget(self.conv_energy)
        conv_layout.addWidget(btn_e2w)
        conv_layout.addWidget(QLabel("Wavelength"))
        conv_layout.addWidget(self.conv_wavelength)
        conv_layout.addWidget(btn_w2e)
        converter.setLayout(conv_layout)

        presets_box = QGroupBox("Presets")
        presets_row = QHBoxLayout()
        presets_row.setSpacing(10)
        presets_row.addWidget(QLabel("Material"))
        presets_row.addWidget(self.single_preset)
        presets_row.addWidget(QLabel("Energy"))
        presets_row.addWidget(self.energy_preset)
        presets_row.addStretch(1)
        presets_box.setLayout(presets_row)

        input_box = QGroupBox("Material input")
        input_layout = QVBoxLayout()
        self.single_form.compute_button.setProperty("class", "primary")
        self.single_form.compute_button.setShortcut("Ctrl+Return")
        self.single_form.compute_button.setToolTip(
            "Compute properties for this material (Ctrl+Enter)"
        )
        input_layout.addWidget(self.single_form)
        input_box.setLayout(input_layout)

        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(24)
        left_layout.addWidget(presets_box)
        left_layout.addWidget(input_box)
        left_layout.addStretch(1)

        self.single_plot_tabs = QTabWidget()
        self.single_plot_tabs.setMinimumHeight(260)
        self.single_plot_tabs.addTab(self.single_plot, "Property plot")
        self.single_plot_tabs.addTab(self.single_f1f2, "f1 / f2")

        single_plot_container = QWidget()
        # Give the scroll area real overflow so the scrollbar can actually scroll.
        single_plot_container.setMinimumHeight(720)
        single_plot_layout = QVBoxLayout(single_plot_container)
        single_plot_layout.setContentsMargins(0, 0, 0, 0)
        single_plot_layout.setSpacing(0)
        single_plot_layout.addWidget(self.single_plot_tabs)

        self.single_plot_scroll = QScrollArea()
        self.single_plot_scroll.setWidgetResizable(True)
        self.single_plot_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.single_plot_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.single_plot_scroll.setWidget(single_plot_container)
        self._reserve_overlay_scrollbar_space(self.single_plot_scroll)

        right_layout = QGridLayout()
        right_layout.setHorizontalSpacing(24)
        right_layout.setVerticalSpacing(16)
        right_layout.addLayout(plot_header, 0, 0)
        right_layout.addWidget(self.single_summary, 1, 0)
        right_layout.addWidget(self.single_plot_scroll, 2, 0)
        right_layout.setRowStretch(2, 2)

        layout = QGridLayout()
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(20)
        layout.addWidget(left_panel, 0, 0, 1, 1)
        layout.addLayout(right_layout, 0, 1, 1, 1)
        # Full-width rows below. The converter's controls are compact, so keep
        # it left-aligned within its full-width cell instead of stretching a
        # small toolset across the whole window.
        layout.addWidget(converter, 1, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.single_table, 2, 0, 1, 2)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 3)
        layout.setRowStretch(2, 1)
        outer.addLayout(layout)
        return container

    def _run_single(self: MainWindowProtocol) -> None:
        formula, density, energy_cfg = self.single_form.values()
        if not formula:
            self._error("Please enter a chemical formula")
            return
        logger.info(
            "single_compute_clicked",
            extra={
                "formula": formula,
                "density": density,
                "points": energy_cfg.points,
                "logspace": energy_cfg.logspace,
            },
        )
        self._info("Computing…")
        self.single_form.compute_button.setEnabled(False)
        self.single_form.compute_button.setText("Computing...")
        self.single_save_png.setEnabled(False)
        self.single_export_csv.setEnabled(False)
        if self.threadpool is None:
            self.threadpool = QThreadPool.globalInstance()
        worker = CalculationWorker(compute_single, formula, density, energy_cfg)
        worker.signals.finished.connect(self._on_single_finished)
        worker.signals.error.connect(self._on_single_error)
        self._track_worker(worker)
        if self.threadpool is not None:
            self.threadpool.start(worker)

    def _on_single_finished(self: MainWindowProtocol, result: Any) -> None:
        self.single_form.compute_button.setEnabled(True)
        self.single_form.compute_button.setText("Compute")
        self.single_result = result
        self.single_save_png.setEnabled(True)
        self.single_export_csv.setEnabled(True)
        logger.info(
            "single_compute_complete",
            extra={"formula": result.formula, "points": len(result.energy_kev)},
        )
        self._info("Single calculation complete")
        self.toast.show_toast("Single calculation done", "success")
        self._refresh_single_views()

    def _on_single_error(self: MainWindowProtocol, message: str) -> None:
        self.single_form.compute_button.setEnabled(True)
        self.single_form.compute_button.setText("Compute")
        self.single_save_png.setEnabled(True)
        self.single_export_csv.setEnabled(True)
        logger.error("single_compute_failed", extra={"message": message})
        self._error(message)

    def _refresh_single_views(self: MainWindowProtocol) -> None:
        if self.single_result is None:
            return
        prop = self.single_property.currentText()
        self.single_plot.set_scales(
            self.single_logx.isChecked(), self.single_logy.isChecked()
        )
        ylabel = self._label_for_property(prop)
        # Update plot
        self.single_plot.plot_single(self.single_result, prop, ylabel)
        # Update table with multiple properties
        energies = self.single_result.energy_kev
        self.single_table.setRowCount(len(energies))
        for i in range(len(energies)):
            cells = TableFormatter.format_single_row(self.single_result, i)
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col != 0 and col != 1:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.single_table.setItem(i, col, item)
        self.single_table.resizeColumnsToContents()

        # Summary row
        summary = TableFormatter.format_summary(self.single_result)
        for col, text in enumerate(summary):
            self.single_summary.setItem(0, col, QTableWidgetItem(text))
        self.single_summary.resizeColumnsToContents()

        # Plot f1/f2 only if >1 point
        if len(energies) > 1:
            self.single_f1f2.render_result(self.single_result)
        else:
            self.single_f1f2.clear()

    # ------------------------------------------------------------------
    # Presets helpers
    def _apply_single_preset(self: MainWindowProtocol, name: str) -> None:
        if name in self.material_presets:
            self.single_form.formula.setText(name)
            self.single_form.density.setValue(self.material_presets[name])
            logger.info("single_preset_applied", extra={"preset": name})
        else:
            return

    def _apply_energy_preset(self: MainWindowProtocol, name: str) -> None:
        if name not in self.energy_presets:
            return
        start, end, pts, logspace = self.energy_presets[name]
        self.single_form.energy_start.setValue(start)
        self.single_form.energy_end.setValue(end)
        self.single_form.energy_points.setValue(pts)
        self.single_form.logspace.setChecked(logspace)
        logger.info(
            "single_energy_preset_applied",
            extra={
                "preset": name,
                "start": start,
                "end": end,
                "points": pts,
                "logspace": logspace,
            },
        )

    def _convert_e2w(self: MainWindowProtocol) -> None:
        energy = self.conv_energy.value()
        try:
            wl = energy_to_wavelength(energy)
            self.conv_wavelength.setValue(wl)
            self._info("Converted energy to wavelength")
            logger.info("convert_e2w", extra={"energy": energy, "wavelength": wl})
        except Exception as exc:
            logger.error(
                "convert_e2w_failed", extra={"energy": energy, "error": str(exc)}
            )
            self._error(str(exc))

    def _convert_w2e(self: MainWindowProtocol) -> None:
        wl = self.conv_wavelength.value()
        try:
            energy = wavelength_to_energy(wl)
            self.conv_energy.setValue(energy)
            self._info("Converted wavelength to energy")
            logger.info("convert_w2e", extra={"wavelength": wl, "energy": energy})
        except Exception as exc:
            logger.error(
                "convert_w2e_failed", extra={"wavelength": wl, "error": str(exc)}
            )
            self._error(str(exc))

    # ------------------------------------------------------------------
    # Export helpers
    def _save_single_png(self: MainWindowProtocol) -> None:
        if self.single_result is None:
            self._error("No data to export yet")
            return
        prop = self.single_property.currentText()
        logger.info("single_save_png_clicked", extra={"property": prop})
        suggested = (
            f"single_{self._sanitize_filename(self.single_result.formula)}_{prop}.png"
        )
        current_plot = self.single_plot_tabs.currentWidget()
        self._save_plot(current_plot, suggested)

    def _export_single_csv(self: MainWindowProtocol) -> str | None:
        if self.single_result is None:
            self._error("No data to export yet")
            return None
        prop = self.single_property.currentText()
        logger.info("single_export_csv_clicked", extra={"property": prop})
        fname = (
            f"single_{self._sanitize_filename(self.single_result.formula)}_{prop}.csv"
        )
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        if not default_dir:
            default_dir = str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            cast(QWidget, self), "Select folder to save CSV", default_dir
        )
        if not folder:
            logger.info("export_single_cancelled", extra={"suggested": fname})
            return None
        path = str(Path(folder) / fname)
        energies = self.single_result.energy_kev
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
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
                )
                for i, e in enumerate(energies):
                    crit_deg = self.single_result.critical_angle_degrees[i]
                    crit_mrad = crit_deg * 3.141592653589793 / 180.0 * 1000.0
                    atten = self.single_result.attenuation_length_cm[i]
                    mu = 1.0 / atten if atten != 0 else 0.0
                    writer.writerow(
                        [
                            e,
                            self.single_result.wavelength_angstrom[i],
                            self.single_result.dispersion_delta[i],
                            self.single_result.absorption_beta[i],
                            crit_deg,
                            crit_mrad,
                            atten,
                            mu,
                            self.single_result.scattering_factor_f1[i],
                            self.single_result.scattering_factor_f2[i],
                            self.single_result.real_sld_per_ang2[i],
                            self.single_result.imaginary_sld_per_ang2[i],
                        ]
                    )
        except OSError as exc:
            self._error(f"Could not save CSV: {exc}")
            logger.exception("export_single_csv_failed", extra={"path": path})
            return None
        self._info(f"Saved CSV to {path}")
        logger.info(
            "export_single_csv",
            extra={
                "path": path,
                "formula": self.single_result.formula,
                "property": prop,
                "rows": len(energies),
            },
        )
        return path
