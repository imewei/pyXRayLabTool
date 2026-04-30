"""Table widget for managing multiple materials."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from xraylabtool.validation import validate_chemical_formula, validate_density


class MaterialTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Formula", "Density (g/cm³)"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # type: ignore[attr-defined]
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(self.SelectionBehavior.SelectRows)
        self.setSelectionMode(self.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)  # type: ignore[attr-defined]
        hdr.setDefaultSectionSize(140)
        hdr.setMinimumSectionSize(90)
        hdr.setStretchLastSection(True)
        hdr.setTextElideMode(Qt.ElideMiddle)  # type: ignore[attr-defined]

    def add_material(self, formula: str, density: float) -> None:
        if not formula:
            raise ValueError("Formula cannot be empty")
        # Let the underlying validators raise exceptions if invalid
        validate_chemical_formula(formula)
        validate_density(density)

        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(formula))
        self.setItem(row, 1, QTableWidgetItem(f"{density:.4f}"))

    def remove_selected(self) -> None:
        rows = {idx.row() for idx in self.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.removeRow(row)

    def materials(self) -> tuple[list[str], list[float]]:
        formulas: list[str] = []
        densities: list[float] = []
        for row in range(self.rowCount()):
            formula_item = self.item(row, 0)
            density_item = self.item(row, 1)
            formula = formula_item.text().strip() if formula_item else ""
            density = float(density_item.text()) if density_item else 0.0
            if formula:
                formulas.append(formula)
                densities.append(density)
        return formulas, densities
