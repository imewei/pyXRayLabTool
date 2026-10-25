"""Widgets and behavior for the 'Kramers-Kronig' tab (view layer only)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PySide6.QtCore import QStandardPaths, Qt, QThreadPool
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xraylabtool.logging_utils import get_logger

from ..services import compute_kk
from ..widgets.kk_plot import KKPlot
from ..workers import CalculationWorker

if TYPE_CHECKING:
    from ._protocol import MainWindowProtocol

logger = get_logger("xraylabtool.gui.main_window")

KK_INPUT_TYPES = ["nexafs", "beta", "asf"]  # as the `kk` CLI offers


def _kk_energy_grid(
    start: float,
    end: float,
    points: int,
    merge: tuple[float, float] | None,
    energy_unit: str,
) -> np.ndarray:
    """Return the requested keV grid, plus ``points`` samples of the merge window.

    A uniform grid over a wide range can step over a narrow near-edge window
    (or miss it when the range excludes it), so the window is always sampled.
    """
    grid = np.linspace(start, end, points)
    if merge is None:
        return grid
    scale = 1e-3 if energy_unit == "eV" else 1.0
    window = np.linspace(merge[0] * scale, merge[1] * scale, points)
    return np.unique(np.concatenate([grid, window]))


class KKTabMixin:
    """Widgets and behavior for the 'Kramers-Kronig' tab."""

    # Same declaration as SingleTabMixin: keeps mypy from narrowing it to non-Optional.
    threadpool: QThreadPool | None

    def _build_kk_tab(self: MainWindowProtocol) -> QWidget:
        """Build the tab: input form on the left, plot and table on the right."""
        self.kk_result = None

        self.kk_formula = QLineEdit("SiO2")
        self.kk_formula.setToolTip("Chemical formula, e.g. SiO2, Ca5(PO4)3F")

        self.kk_nearedge = QLineEdit()
        self.kk_nearedge.setPlaceholderText("optional near-edge data file")
        self.kk_browse_btn = QPushButton("Browse…")
        self.kk_browse_btn.setProperty("class", "secondary")
        self.kk_browse_btn.clicked.connect(self._browse_kk_nearedge)
        file_row = QHBoxLayout()
        file_row.addWidget(self.kk_nearedge, 1)
        file_row.addWidget(self.kk_browse_btn)

        self.kk_input_type = QComboBox()
        self.kk_input_type.addItems(KK_INPUT_TYPES)
        self.kk_input_type.setToolTip("Kind of values in the near-edge file")
        self.kk_energy_unit = QComboBox()
        self.kk_energy_unit.addItems(["eV", "keV"])
        self.kk_energy_unit.setToolTip("Energy unit of the near-edge file and merge")

        self.kk_merge = QLineEdit()
        self.kk_merge.setPlaceholderText("low, high (in the file's unit)")
        self.kk_merge.setToolTip(
            "Energies between which the data replaces the table; "
            "this window is always sampled with N points, even outside the range"
        )

        self.kk_refine = QDoubleSpinBox()
        self.kk_refine.setRange(0.0, 1.0)
        self.kk_refine.setDecimals(4)
        self.kk_refine.setSingleStep(0.01)
        self.kk_refine.setSpecialValueText("Off")
        self.kk_refine.setToolTip("Adaptive refinement tolerance; Off disables it")

        # The transform covers the data range 10 eV-500 keV (D3); CXRO f1 is
        # only drawn where it exists (0.03-30 keV).
        self.kk_energy_start = QDoubleSpinBox()
        self.kk_energy_start.setDecimals(4)
        self.kk_energy_start.setRange(0.01, 500.0)
        self.kk_energy_start.setValue(1.0)
        self.kk_energy_start.setSuffix(" keV")
        self.kk_energy_end = QDoubleSpinBox()
        self.kk_energy_end.setDecimals(4)
        self.kk_energy_end.setRange(0.01, 500.0)
        self.kk_energy_end.setValue(30.0)
        self.kk_energy_end.setSuffix(" keV")
        self.kk_points = QSpinBox()
        self.kk_points.setRange(1, 100_000)
        self.kk_points.setValue(500)

        self.kk_run_btn = QPushButton("Compute")
        self.kk_run_btn.setProperty("class", "primary")
        self.kk_run_btn.clicked.connect(self._run_kk)
        self.kk_export_csv = QPushButton("Export CSV")
        self.kk_export_csv.setProperty("class", "secondary")
        self.kk_export_csv.setEnabled(False)
        self.kk_export_csv.clicked.connect(self._export_kk_csv)

        form = QFormLayout()
        form.addRow("Formula", self.kk_formula)
        form.addRow("Near-edge file", file_row)
        form.addRow("Input type", self.kk_input_type)
        form.addRow("File energy unit", self.kk_energy_unit)
        form.addRow("Merge points", self.kk_merge)
        form.addRow("Refine tolerance", self.kk_refine)
        form.addRow("Start energy", self.kk_energy_start)
        form.addRow("End energy", self.kk_energy_end)
        form.addRow("Points", self.kk_points)
        input_box = QGroupBox("Kramers-Kronig input")
        input_layout = QVBoxLayout(input_box)
        input_layout.addLayout(form)
        buttons = QHBoxLayout()
        buttons.addWidget(self.kk_run_btn)
        buttons.addWidget(self.kk_export_csv)
        input_layout.addLayout(buttons)

        left_panel = QWidget()
        left_panel.setMinimumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(input_box)
        left_layout.addStretch(1)

        self.kk_plot = KKPlot()
        self.kk_table = QTableWidget(0, 3)
        self.kk_table.setAlternatingRowColors(True)
        self.kk_table.setHorizontalHeaderLabels(["Energy (keV)", "f1 (e)", "f2 (e)"])
        self.kk_table.verticalHeader().setVisible(False)
        self.kk_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.kk_plot, 2)
        right_layout.addWidget(self.kk_table, 1)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(left_panel, 0)
        layout.addLayout(right_layout, 1)
        return container

    def _browse_kk_nearedge(self: MainWindowProtocol) -> None:
        path, _ = QFileDialog.getOpenFileName(
            cast(QWidget, self), "Select near-edge data file"
        )
        if path:
            self.kk_nearedge.setText(path)

    def _run_kk(self: MainWindowProtocol) -> None:
        formula = self.kk_formula.text().strip()
        nearedge = self.kk_nearedge.text().strip() or None
        merge_text = self.kk_merge.text().strip()
        start = self.kk_energy_start.value()
        end = self.kk_energy_end.value()
        points = self.kk_points.value()
        if not formula:
            self._error("Please enter a chemical formula")
            return
        if points > 1 and end <= start:
            self._error("End energy must be greater than start energy")
            return
        merge: tuple[float, float] | None = None
        if merge_text:
            if nearedge is None:
                self._error("Merge points require a near-edge file")
                return
            try:
                low, high = (float(p) for p in merge_text.split(","))
            except ValueError:
                self._error("Merge points must be two numbers: low, high")
                return
            merge = (low, high)
        refine = self.kk_refine.value() or None
        input_type = self.kk_input_type.currentText() if nearedge else None
        logger.info(
            "kk_compute_clicked",
            extra={
                "formula": formula,
                "nearedge": nearedge,
                "points": points,
                "refine_tolerance": refine,
            },
        )
        self._info("Computing…")
        self.kk_run_btn.setEnabled(False)
        self.kk_run_btn.setText("Computing...")
        self.kk_export_csv.setEnabled(False)
        if self.threadpool is None:
            self.threadpool = QThreadPool.globalInstance()
        worker = CalculationWorker(
            compute_kk,
            formula,
            nearedge,
            input_type,
            self.kk_energy_unit.currentText(),
            merge,
            refine,
            energies_kev=_kk_energy_grid(
                start, end, points, merge, self.kk_energy_unit.currentText()
            ),
        )
        worker.signals.finished.connect(self._on_kk_finished)
        worker.signals.error.connect(self._on_kk_error)
        self._track_worker(worker)
        if self.threadpool is not None:
            self.threadpool.start(worker)

    def _on_kk_finished(self: MainWindowProtocol, computation: Any) -> None:
        self.kk_run_btn.setEnabled(True)
        self.kk_run_btn.setText("Compute")
        self.kk_export_csv.setEnabled(True)
        self.kk_result = computation
        res = computation.result
        self.kk_plot.set_data(computation)
        self.kk_table.setRowCount(len(res.energy_kev))
        for i, row in enumerate(zip(res.energy_kev, res.f1, res.f2, strict=True)):
            for col, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.6g}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.kk_table.setItem(i, col, item)
        self.kk_table.resizeColumnsToContents()
        logger.info("kk_compute_complete", extra={"points": len(res.energy_kev)})
        self._info("Kramers-Kronig calculation complete")
        self.toast.show_toast("Kramers-Kronig calculation done", "success")

    def _on_kk_error(self: MainWindowProtocol, message: str) -> None:
        self.kk_run_btn.setEnabled(True)
        self.kk_run_btn.setText("Compute")
        self.kk_export_csv.setEnabled(self.kk_result is not None)
        logger.error("kk_compute_failed", extra={"error_detail": message})
        self._error(message)

    def _export_kk_csv(self: MainWindowProtocol) -> str | None:
        """Write ``energy_keV, f1, f2`` of the last result to a chosen folder."""
        if self.kk_result is None:
            self._error("No data to export yet")
            return None
        res = self.kk_result.result
        fname = f"kk_{self._sanitize_filename(res.formula)}.csv"
        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        ) or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            cast(QWidget, self), "Select folder to save CSV", default_dir
        )
        if not folder:
            logger.info("export_kk_cancelled", extra={"suggested": fname})
            return None
        path = str(Path(folder) / fname)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["energy_keV", "f1", "f2"])
                writer.writerows(zip(res.energy_kev, res.f1, res.f2, strict=True))
        except OSError as exc:
            self._error(f"Could not save CSV: {exc}")
            logger.exception("export_kk_csv_failed", extra={"path": path})
            return None
        self._info(f"Saved CSV to {path}")
        logger.info("export_kk_csv", extra={"path": path, "rows": len(res.energy_kev)})
        return path
