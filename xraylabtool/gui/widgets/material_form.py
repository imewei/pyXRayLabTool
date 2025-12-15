"""Input widgets for material definitions and energy ranges."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from xraylabtool.utils import parse_formula
from xraylabtool.validation import validate_chemical_formula, validate_density

from ..services import EnergyConfig


class MaterialInputForm(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.formula = QLineEdit()
        self.formula.setPlaceholderText("e.g. SiO2")

        self.density = QDoubleSpinBox()
        self.density.setRange(0.001, 100.0)
        self.density.setDecimals(4)
        self.density.setValue(2.2)
        self.density.setSuffix(" g/cm³")

        self.energy_start = QDoubleSpinBox()
        self.energy_start.setRange(0.03, 30.0)
        self.energy_start.setDecimals(3)
        self.energy_start.setValue(8.0)
        self.energy_start.setSuffix(" keV")

        self.energy_end = QDoubleSpinBox()
        self.energy_end.setRange(0.03, 30.0)
        self.energy_end.setDecimals(3)
        self.energy_end.setValue(12.0)
        self.energy_end.setSuffix(" keV")

        self.energy_points = QSpinBox()
        self.energy_points.setRange(1, 5000)
        self.energy_points.setValue(50)

        self.logspace = QCheckBox("Log-spaced energies")

        self.compute_button = QPushButton("Compute")
        self.compute_button.setProperty("class", "primary")

        self.formula_hint = QLabel("Format: H2O, SiO2, Ca5(PO4)3F")
        self.formula_hint.setProperty("role", "hint")
        self.composition_hint = QLabel("")
        self.composition_hint.setProperty("role", "hint")
        self.density_hint = QLabel("Density 0.001-30 g/cm³")
        self.density_hint.setProperty("role", "hint")
        self.energy_hint = QLabel("")
        self.energy_hint.setProperty("role", "hint")

        self.formula.textChanged.connect(self._validate_inputs)
        self.density.valueChanged.connect(self._validate_inputs)
        self.energy_start.valueChanged.connect(self._validate_inputs)
        self.energy_end.valueChanged.connect(self._validate_inputs)
        self.energy_points.valueChanged.connect(self._validate_inputs)
        self.logspace.stateChanged.connect(self._validate_inputs)

        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(6)
        layout.addRow("Formula", self.formula)
        layout.addRow("", self.formula_hint)
        layout.addRow("", self.composition_hint)
        layout.addRow("Density", self.density)
        layout.addRow("", self.density_hint)

        energy_row = QHBoxLayout()
        energy_row.addWidget(QLabel("Start"))
        energy_row.addWidget(self.energy_start)
        energy_row.addWidget(QLabel("End"))
        energy_row.addWidget(self.energy_end)
        energy_row.addWidget(QLabel("Points"))
        energy_row.addWidget(self.energy_points)
        layout.addRow("Energy grid", energy_row)
        layout.addRow("", self.logspace)
        layout.addRow("", self.energy_hint)
        layout.addRow(self.compute_button)

        self.setLayout(layout)
        self._validate_inputs()

    def energy_config(self) -> EnergyConfig:
        return EnergyConfig(
            start_kev=self.energy_start.value(),
            end_kev=self.energy_end.value(),
            points=self.energy_points.value(),
            logspace=self.logspace.isChecked(),
        )

    def values(self) -> tuple[str, float, EnergyConfig]:
        return (
            self.formula.text().strip(),
            float(self.density.value()),
            self.energy_config(),
        )

    def _validate_inputs(self) -> None:
        ok = True
        # Formula
        formula_text = self.formula.text().strip()
        try:
            if formula_text:
                validate_chemical_formula(formula_text)
                self.formula.setProperty("validation", "valid")
                self.formula_hint.setText("✓ Formula valid")
                self.formula_hint.setProperty("role", "success")
                try:
                    symbols, counts = parse_formula(formula_text)
                    comp = ", ".join(
                        f"{s}:{c:g}" for s, c in zip(symbols, counts, strict=False)
                    )
                    self.composition_hint.setText(f"Composition: {comp}")
                    self.composition_hint.setProperty("role", "success")
                except Exception:
                    self.composition_hint.setText("")
            else:
                ok = False
                self.formula.setProperty("validation", "invalid")
                self.formula_hint.setText("Enter a chemical formula")
                self.formula_hint.setProperty("role", "error")
        except Exception as exc:
            ok = False
            self.formula.setProperty("validation", "invalid")
            self.formula_hint.setText(str(exc))
            self.formula_hint.setProperty("role", "error")
            self.composition_hint.setText("")

        for w in (self.formula, self.formula_hint, self.composition_hint):
            w.style().unpolish(w)
            w.style().polish(w)

        # Density
        try:
            validate_density(self.density.value())
            self.density.setProperty("validation", "valid")
            self.density_hint.setText("✓ Density valid")
            self.density_hint.setProperty("role", "success")
        except Exception as exc:
            ok = False
            self.density.setProperty("validation", "invalid")
            self.density_hint.setText(str(exc))
            self.density_hint.setProperty("role", "error")

        for w in (self.density, self.density_hint):
            w.style().unpolish(w)
            w.style().polish(w)

        # Energy grid
        start = self.energy_start.value()
        end = self.energy_end.value()
        points = self.energy_points.value()
        logspace = self.logspace.isChecked()
        single_point = points == 1
        if (not single_point and end <= start) or (single_point and end < start):
            ok = False
            self.energy_hint.setText(
                "End energy must be greater than start energy (>= when 1 point)"
            )
            self.energy_hint.setProperty("role", "error")
        elif logspace and points < 3 and not single_point:
            ok = False
            self.energy_hint.setText("Log spacing needs at least 3 points")
            self.energy_hint.setProperty("role", "error")
        else:
            step = (end - start) / max(points - 1, 1)
            spacing = "log" if logspace else "linear"
            self.energy_hint.setText(
                f"{spacing} grid: {points} points from {start:.3f} to {end:.3f} keV (Δ≈{step:.3f})"
            )
            self.energy_hint.setProperty("role", "success")

        self.energy_hint.style().unpolish(self.energy_hint)
        self.energy_hint.style().polish(self.energy_hint)

        self.compute_button.setEnabled(ok)
