from __future__ import annotations

import csv
import os

from PySide6.QtCore import QStandardPaths, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xraylabtool.analysis.comparator import MaterialComparator
from xraylabtool.logging_utils import get_log_file_path, get_logger
from xraylabtool.utils import energy_to_wavelength, wavelength_to_energy

from .services import EnergyConfig, compute_multiple, compute_single
from .widgets.bar_plots import ComparisonBars
from .widgets.material_form import MaterialInputForm
from .widgets.material_table import MaterialTable
from .widgets.plot_canvas import PlotCanvas
from .widgets.sweep_plots import F1F2Plot, SweepPlots
from .workers import CalculationWorker


class Toast(QLabel):
    """Lightweight, non-blocking toast overlay."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "background: rgba(15,23,42,0.92); color: white; padding: 8px 12px;"
            "border-radius: 8px;"
        )
        self.setVisible(False)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self._kind_colors = {
            "info": "#2563eb",
            "success": "#16a34a",
            "error": "#dc2626",
        }
        self._durations = {"info": 2000, "success": 2400, "error": 3500}

    def show_toast(
        self, message: str, kind: str = "info", duration_ms: int | None = None
    ) -> None:
        color = self._kind_colors.get(kind, "#2563eb")
        self.setStyleSheet(
            f"background: rgba(15,23,42,0.92); color: white; padding: 8px 12px;"
            f"border: 1px solid {color}; border-radius: 8px;"
        )
        self.setText(message)
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        duration = (
            duration_ms if duration_ms is not None else self._durations.get(kind, 2200)
        )
        self._timer.start(duration)

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        x = max(8, (parent.width() - self.width()) // 2)
        y = max(8, parent.height() - self.height() - 24)
        self.move(x, y)


PROPERTIES = [
    "attenuation_length_cm",
    "dispersion_delta",
    "absorption_beta",
    "critical_angle_degrees",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
]

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("XRayLabTool GUI")
        self.resize(1100, 720)
        self.setMinimumSize(900, 620)

        self.threadpool = None  # Assigned on first use to avoid import cycles

        self.status_bar = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(18)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.status_bar.addPermanentWidget(self.progress)

        self.log_path_toggle = QPushButton("Log path")
        self.log_path_toggle.setProperty("class", "secondary")
        self.log_path_toggle.setToolTip("Show or hide the current log file path")
        self.log_path_toggle.setCheckable(True)
        self.log_path_toggle.setChecked(False)
        self.log_path_toggle.clicked.connect(self._toggle_log_path)
        self.status_bar.addPermanentWidget(self.log_path_toggle)
        self.status_bar.setSizeGripEnabled(True)
        self.setStatusBar(self.status_bar)

        self.toast = Toast(self)

        self.single_result = None
        self.multi_results = None
        self.multi_comparison = None
        self._workers: list = []

        self.material_presets = {
            "Si": 2.33,
            "SiO2": 2.2,
            "Al2O3": 3.95,
            "C": 3.52,
            "Au": 19.3,
            "Pt": 21.45,
            "Rh": 12.4,
            "Pd": 12.0,
            "CaCO3": 2.71,
        }
        self.energy_presets = {
            "10 keV": (10.0, 10.0, 1, False),
            "Cu Kalpha (8.048 keV)": (8.048, 8.048, 1, False),
            "1-30 keV log (100)": (1.0, 30.0, 100, True),
            "5-25 keV log (50)": (5.0, 25.0, 50, True),
        }

        tabs = QTabWidget()
        tabs.addTab(self._build_single_tab(), "Single Material")
        tabs.addTab(self._build_multi_tab(), "Multiple Materials")
        self.setCentralWidget(tabs)
        self._set_tab_order()
        self._tune_table_headers()

    # ------------------------------------------------------------------
    # Single tab
    def _build_single_tab(self) -> QWidget:
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
        self.single_sweep = SweepPlots()
        self.single_f1f2 = F1F2Plot()

        # Converter
        converter = QGroupBox("Energy ↔ Wavelength")
        conv_layout = QHBoxLayout()
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
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.addWidget(presets_box)
        left_layout.addWidget(input_box)
        left_layout.addStretch(1)

        self.single_plot_tabs = QTabWidget()
        self.single_plot_tabs.setMinimumHeight(320)
        self.single_plot_tabs.addTab(self.single_plot, "Property plot")
        self.single_plot_tabs.addTab(self.single_sweep, "Sweep")
        self.single_plot_tabs.addTab(self.single_f1f2, "f1 / f2")

        right_layout = QGridLayout()
        right_layout.setHorizontalSpacing(10)
        right_layout.setVerticalSpacing(8)
        right_layout.addLayout(plot_header, 0, 0)
        right_layout.addWidget(self.single_summary, 1, 0)
        right_layout.addWidget(self.single_plot_tabs, 2, 0)
        right_layout.setRowStretch(2, 2)

        layout = QGridLayout()
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)
        layout.addWidget(left_panel, 0, 0, 1, 1)
        layout.addLayout(right_layout, 0, 1, 1, 1)
        # Full-width rows below
        layout.addWidget(converter, 1, 0, 1, 2)
        layout.addWidget(self.single_table, 2, 0, 1, 2)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(2, 1)
        outer.addLayout(layout)
        return container

    def _tune_table_headers(self) -> None:
        def tune(table, default_size=110, min_size=80, stretch_last=True):
            hdr = table.horizontalHeader()
            hdr.setSectionResizeMode(QHeaderView.Interactive)
            hdr.setDefaultSectionSize(default_size)
            hdr.setMinimumSectionSize(min_size)
            hdr.setStretchLastSection(stretch_last)
            hdr.setTextElideMode(Qt.ElideMiddle)

        tune(self.single_table, default_size=110, min_size=80, stretch_last=False)
        self.single_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.single_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        tune(self.single_summary, default_size=120, min_size=90)
        tune(self.multi_summary, default_size=120, min_size=90)
        tune(self.multi_comp_table, default_size=110, min_size=80)

    def _run_single(self) -> None:
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
        self.single_save_png.setEnabled(False)
        self.single_export_csv.setEnabled(False)
        if self.threadpool is None:
            from PySide6.QtCore import QThreadPool

            self.threadpool = QThreadPool.globalInstance()
        worker = CalculationWorker(compute_single, formula, density, energy_cfg)
        worker.signals.finished.connect(self._on_single_finished)
        worker.signals.error.connect(self._on_single_error)
        self._track_worker(worker)
        self.threadpool.start(worker)

    def _on_single_finished(self, result) -> None:
        self.single_form.compute_button.setEnabled(True)
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

    def _on_single_error(self, message: str) -> None:
        self.single_form.compute_button.setEnabled(True)
        self.single_save_png.setEnabled(True)
        self.single_export_csv.setEnabled(True)
        logger.error("single_compute_failed", extra={"message": message})
        self._error(message)

    def _refresh_single_views(self) -> None:
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
        wl = self.single_result.wavelength_angstrom
        delta = self.single_result.dispersion_delta
        beta = self.single_result.absorption_beta
        crit = self.single_result.critical_angle_degrees
        atten = self.single_result.attenuation_length_cm
        resld = self.single_result.real_sld_per_ang2
        imsld = self.single_result.imaginary_sld_per_ang2
        self.single_table.setRowCount(len(energies))
        for i, e in enumerate(energies):
            mu = 1.0 / atten[i] if atten[i] != 0 else 0.0
            mrad = crit[i] * 3.141592653589793 / 180.0 * 1000.0
            cells = [
                f"{e:.4f}",
                f"{wl[i]:.5f}",
                f"{delta[i]:.3e}",
                f"{beta[i]:.3e}",
                f"{crit[i]:.4f}",
                f"{mrad:.3f}",
                f"{atten[i]:.4e}",
                f"{mu:.4e}",
                f"{self.single_result.scattering_factor_f1[i]:.3f}",
                f"{self.single_result.scattering_factor_f2[i]:.3f}",
                f"{resld[i]:.3e}",
                f"{imsld[i]:.3e}",
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col != 0 and col != 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.single_table.setItem(i, col, item)
        self.single_table.resizeColumnsToContents()

        # Summary row
        self.single_summary.setItem(0, 0, QTableWidgetItem(self.single_result.formula))
        self.single_summary.setItem(
            0, 1, QTableWidgetItem(f"{self.single_result.molecular_weight_g_mol:.4f}")
        )
        self.single_summary.setItem(
            0, 2, QTableWidgetItem(f"{self.single_result.density_g_cm3:.4f}")
        )
        self.single_summary.setItem(
            0,
            3,
            QTableWidgetItem(f"{self.single_result.electron_density_per_ang3:.4f}"),
        )
        self.single_summary.setItem(
            0, 4, QTableWidgetItem(f"{self.single_result.total_electrons:.2f}")
        )
        self.single_summary.resizeColumnsToContents()

        # Energy sweep plots only if >1 point
        if len(energies) > 1:
            self.single_sweep.render(self.single_result)
            self.single_f1f2.render(self.single_result)
        else:
            self.single_sweep.clear()
            self.single_f1f2.clear()

    # ------------------------------------------------------------------
    # Multi tab
    def _build_multi_tab(self) -> QWidget:
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
        entry_row.addWidget(add_btn, 0, 4)
        entry_row.addWidget(remove_btn, 0, 5)
        entry_row.addWidget(QLabel("Preset"), 1, 0)
        entry_row.addWidget(self.multi_preset, 1, 1, 1, 3)
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

        self.multi_plot = PlotCanvas()
        self.multi_bar_theta = ComparisonBars()
        self.multi_bar_atten = ComparisonBars()
        self.multi_summary = QTableWidget(0, 6)
        self.multi_summary.setAlternatingRowColors(True)
        self.multi_summary.setHorizontalHeaderLabels(
            [
                "Formula",
                "Energy (keV)",
                "θc (deg)",
                "Atten. length (cm)",
                "δ",
                "β",
            ]
        )
        self.multi_summary.verticalHeader().setVisible(False)
        self.multi_comp_table = QTableWidget()
        self.multi_comp_table.setColumnCount(0)
        self.multi_comp_table.setRowCount(0)
        self.multi_comp_table.setAlternatingRowColors(True)
        self.multi_comp_table.verticalHeader().setVisible(False)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        header_row.addWidget(QLabel("Property"))
        header_row.addWidget(self.multi_property)
        header_row.addWidget(self.multi_logx)
        header_row.addWidget(self.multi_logy)
        header_row.addStretch(1)
        header_row.addWidget(self.multi_save_png)
        header_row.addWidget(self.multi_export_csv)
        header_row.addWidget(self.multi_compute_btn)

        self.multi_plot_tabs = QTabWidget()
        self.multi_plot_tabs.addTab(self.multi_plot, "Curves")
        self.multi_plot_tabs.addTab(self.multi_bar_theta, "Critical angle bars")
        self.multi_plot_tabs.addTab(self.multi_bar_atten, "Attenuation bars")
        self.multi_plot_tabs.setMinimumHeight(320)

        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.addWidget(material_box)
        left_layout.addWidget(energy_box)
        left_layout.addWidget(self.multi_table)
        left_layout.addStretch(1)

        right_layout = QGridLayout()
        right_layout.setHorizontalSpacing(10)
        right_layout.setVerticalSpacing(8)
        right_layout.addLayout(header_row, 0, 0)
        right_layout.addWidget(self.multi_plot_tabs, 1, 0)
        right_layout.setRowStretch(1, 2)

        bottom_tables = QWidget()
        bottom_layout = QVBoxLayout(bottom_tables)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        bottom_layout.addWidget(self.multi_summary)
        bottom_layout.addWidget(self.multi_comp_table)

        layout = QGridLayout()
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(10)
        layout.addWidget(left_panel, 0, 0, 1, 1)
        layout.addLayout(right_layout, 0, 1, 1, 1)
        layout.addWidget(bottom_tables, 1, 0, 1, 2)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        outer.addLayout(layout)
        return container

    def _set_tab_order(self) -> None:
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
        self.setTabOrder(self.multi_logspace, self.multi_property)
        self.setTabOrder(self.multi_property, self.multi_logx)
        self.setTabOrder(self.multi_logx, self.multi_logy)
        self.setTabOrder(self.multi_logy, self.multi_save_png)
        self.setTabOrder(self.multi_save_png, self.multi_export_csv)
        self.setTabOrder(self.multi_export_csv, self.multi_compute_btn)

    def _add_material(self) -> None:
        formula = self.multi_formula.text().strip()
        density = float(self.multi_density.value())
        self.multi_table.add_material(formula, density)
        self.multi_formula.clear()
        self.multi_formula.setFocus()
        logger.info(
            "multi_add_material", extra={"formula": formula, "density": density}
        )

    def _add_multi_preset(self, name: str) -> None:
        if name in self.material_presets:
            self.multi_formula.setText(name)
            self.multi_density.setValue(self.material_presets[name])
            self._add_material()
        self.multi_preset.setCurrentIndex(0)

    def _remove_material(self) -> None:
        self.multi_table.remove_selected()
        logger.info(
            "multi_remove_material",
            extra={"remaining": len(self.multi_table.materials()[0])},
        )

    def _multi_energy_cfg(self) -> EnergyConfig:
        return EnergyConfig(
            start_kev=self.multi_energy_start.value(),
            end_kev=self.multi_energy_end.value(),
            points=self.multi_energy_points.value(),
            logspace=self.multi_logspace.isChecked(),
        )

    def _run_multi(self) -> None:
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
        self.multi_save_png.setEnabled(False)
        self.multi_export_csv.setEnabled(False)
        if self.threadpool is None:
            from PySide6.QtCore import QThreadPool

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
        self.threadpool.start(worker)

    def _on_multi_finished(self, results) -> None:
        self.multi_compute_btn.setEnabled(True)
        self.multi_save_png.setEnabled(True)
        self.multi_export_csv.setEnabled(True)
        self._show_progress(False, 0)
        self.multi_results = results
        logger.info(
            "multi_compute_complete",
            extra={"count": len(results), "first": next(iter(results.keys()), "")},
        )
        # Optional comparator summary for richer tables
        try:
            cmp = MaterialComparator()
            energies = list(next(iter(results.values())).energy_kev)
            densities = [res.density_g_cm3 for res in results.values()]
            self.multi_comparison = cmp.compare_materials(
                list(results.keys()), densities, energies, properties=PROPERTIES[:4]
            )
        except Exception:
            self.multi_comparison = None
        self._info("Multi-material comparison complete")
        self.toast.show_toast("Comparison done", "success")
        self._refresh_multi_views()

    def _on_multi_error(self, message: str) -> None:
        self.multi_compute_btn.setEnabled(True)
        self.multi_save_png.setEnabled(True)
        self.multi_export_csv.setEnabled(True)
        self._show_progress(False, 0)
        logger.error("multi_compute_failed", extra={"message": message})
        self._error(message)

    def _progress_multi(self, value: int) -> None:
        self._show_progress(True, value)

    def _refresh_multi_views(self) -> None:
        if not self.multi_results:
            return
        prop = self.multi_property.currentText()
        self.multi_plot.set_scales(
            self.multi_logx.isChecked(), self.multi_logy.isChecked()
        )
        ylabel = self._label_for_property(prop)
        self.multi_plot.plot_multi(self.multi_results, prop, ylabel)

        # Build summary at the first energy point using comparator data if present
        rows = []
        if self.multi_comparison:
            e0 = self.multi_comparison.energies[0]
            for material in self.multi_comparison.materials:
                rows.append(
                    (
                        material,
                        e0,
                        self.multi_comparison.data.get(
                            "critical_angle_degrees", {}
                        ).get(material, [None])[0],
                        self.multi_comparison.data.get("attenuation_length_cm", {}).get(
                            material, [None]
                        )[0],
                        self.multi_comparison.data.get("dispersion_delta", {}).get(
                            material, [None]
                        )[0],
                        self.multi_comparison.data.get("absorption_beta", {}).get(
                            material, [None]
                        )[0],
                    )
                )
        else:
            for formula, res in self.multi_results.items():
                rows.append(
                    (
                        formula,
                        res.energy_kev[0],
                        res.critical_angle_degrees[0],
                        res.attenuation_length_cm[0],
                        res.dispersion_delta[0],
                        res.absorption_beta[0],
                    )
                )

        self.multi_summary.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for col, val in enumerate(row):
                text = f"{val:.6g}" if isinstance(val, (float, int)) else str(val)
                item = QTableWidgetItem(text)
                if col > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.multi_summary.setItem(i, col, item)
        self.multi_summary.resizeColumnsToContents()

        # Bar charts at first energy point
        labels = []
        thetas = []
        attens = []
        for formula, res in self.multi_results.items():
            labels.append(formula)
            thetas.append(res.critical_angle_degrees[0])
            attens.append(res.attenuation_length_cm[0])
        if labels:
            self.multi_bar_theta.plot(
                labels, thetas, "Critical angle (deg)", logy=False
            )
            self.multi_bar_atten.plot(
                labels, attens, "Attenuation length (cm)", logy=True
            )

        # Comparator table
        if self.multi_comparison:
            df = MaterialComparator().create_comparison_table(self.multi_comparison)
            headers = list(df.columns)
            self.multi_comp_table.setColumnCount(len(headers))
            self.multi_comp_table.setHorizontalHeaderLabels(headers)
            self.multi_comp_table.setRowCount(len(df))
            for r, row in df.iterrows():
                for c, h in enumerate(headers):
                    self.multi_comp_table.setItem(r, c, QTableWidgetItem(str(row[h])))
            self.multi_comp_table.resizeColumnsToContents()
        else:
            self.multi_comp_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Status helpers
    def _info(self, message: str) -> None:
        self.status_bar.showMessage(message, 5000)
        self.toast.show_toast(message, "info")

    def _error(self, message: str) -> None:
        self.status_bar.showMessage(message, 10000)
        self.toast.show_toast(message, "error", duration_ms=3500)
        QMessageBox.critical(self, "Error", message)

    def _show_progress(self, active: bool, value: int = 0) -> None:
        if active:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, value)))
            self.progress.setVisible(True)
        else:
            self.progress.setVisible(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def _toggle_log_path(self) -> None:
        path = get_log_file_path()
        if path:
            if self.log_path_toggle.isChecked():
                self.status_bar.showMessage(f"Log file: {path}", 8000)
                logger.info("log_path_shown", extra={"path": path})
            else:
                self.status_bar.clearMessage()
        else:
            self.status_bar.showMessage("File logging is disabled", 5000)
            logger.info("log_path_missing")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "toast"):
            self.toast._reposition()

    # ------------------------------------------------------------------
    # Export helpers
    def _save_single_png(self) -> None:
        if self.single_result is None:
            self._error("No data to export yet")
            return
        prop = self.single_property.currentText()
        logger.info("single_save_png_clicked", extra={"property": prop})
        suggested = f"single_{self.single_result.formula}_{prop}.png"
        current_plot = self.single_plot_tabs.currentWidget()
        self._save_plot(current_plot, suggested)

    def _save_multi_png(self) -> None:
        if not self.multi_results:
            self._error("No data to export yet")
            return
        prop = self.multi_property.currentText()
        logger.info("multi_save_png_clicked", extra={"property": prop})
        current_plot = self.multi_plot_tabs.currentWidget()
        self._save_plot(current_plot, f"multi_{prop}.png")

    def _save_plot(self, plot_widget: QWidget, suggested: str) -> None:
        default_dir = QStandardPaths.writableLocation(QStandardPaths.PicturesLocation)
        if not default_dir:
            default_dir = os.path.expanduser("~")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            os.path.join(default_dir, suggested),
            "PNG Files (*.png)",
        )
        if not path:
            logger.info("save_plot_cancelled", extra={"suggested": suggested})
            return
        fig = getattr(plot_widget, "figure", None)
        if fig is None:
            self._error("Plot figure not available to save")
            logger.error(
                "plot_save_failed", extra={"path": path, "reason": "no figure"}
            )
            return
        fig.savefig(path, dpi=300)
        logger.info("plot_saved", extra={"path": path, "suggested": suggested})
        self._info(f"Saved plot to {path}")

    def _export_single_csv(self) -> str | None:
        if self.single_result is None:
            self._error("No data to export yet")
            return None
        prop = self.single_property.currentText()
        logger.info("single_export_csv_clicked", extra={"property": prop})
        fname = f"single_{self.single_result.formula}_{prop}.csv"
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        if not default_dir:
            default_dir = os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder to save CSV", default_dir
        )
        if not folder:
            logger.info("export_single_cancelled", extra={"suggested": fname})
            return None
        path = os.path.join(folder, fname)
        energies = self.single_result.energy_kev
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

    def _export_multi_csv(self) -> str | None:
        if not self.multi_results:
            self._error("No data to export yet")
            return None
        prop = self.multi_property.currentText()
        logger.info("multi_export_csv_clicked", extra={"property": prop})
        fname = f"multi_{prop}.csv"
        default_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        if not default_dir:
            default_dir = os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder to save CSV", default_dir
        )
        if not folder:
            logger.info("export_multi_cancelled", extra={"suggested": fname})
            return None
        path = os.path.join(folder, fname)

        # Build combined table: energy + one column per material for selected property
        first = next(iter(self.multi_results.values()))
        energies = first.energy_kev
        materials = list(self.multi_results.keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["energy_kev", *materials])
            for idx, energy in enumerate(energies):
                row = [energy]
                for formula in materials:
                    arr = getattr(self.multi_results[formula], prop)
                    value = arr[idx] if idx < len(arr) else ""
                    row.append(value)
                writer.writerow(row)

        # Also export comparator table if available
        if self.multi_comparison:
            comp_path = path.replace(".csv", "_comparator.csv")
            df = MaterialComparator().create_comparison_table(self.multi_comparison)
            df.to_csv(comp_path, index=False)
            logger.info(
                "export_multi_comparator_csv",
                extra={"path": comp_path, "materials": len(materials)},
            )
        self._info(f"Saved CSV to {path}")
        logger.info(
            "export_multi_csv",
            extra={"path": path, "materials": len(materials), "property": prop},
        )
        return path

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

    def _track_worker(self, worker):
        self._workers.append(worker)
        worker.signals.finished.connect(lambda _res, w=worker: self._cleanup_worker(w))
        worker.signals.error.connect(lambda _msg, w=worker: self._cleanup_worker(w))

    def _cleanup_worker(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)

    # ------------------------------------------------------------------
    # Presets helpers
    def _apply_single_preset(self, name: str) -> None:
        if name in self.material_presets:
            self.single_form.formula.setText(name)
            self.single_form.density.setValue(self.material_presets[name])
            logger.info("single_preset_applied", extra={"preset": name})
        else:
            return

    def _apply_energy_preset(self, name: str) -> None:
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

    def _convert_e2w(self) -> None:
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

    def _convert_w2e(self) -> None:
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
