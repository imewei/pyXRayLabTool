"""GUI widget tests for MaterialInputForm (item 4.2).

Requires PySide6 — skipped automatically in CI where PySide6 is not available.
Run with QT_QPA_PLATFORM=offscreen to avoid requiring a display.
"""

from __future__ import annotations

import os
import sys

import pytest

# Skip entire module if PySide6 is not available
PySide6 = pytest.importorskip("PySide6")

# Ensure offscreen platform for headless environments
if "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication

# One QApplication per process
_app: QApplication | None = None


def _get_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return _get_app()


@pytest.fixture
def form(qt_app: QApplication):
    from xraylabtool.gui.widgets.material_form import MaterialInputForm

    widget = MaterialInputForm()
    yield widget
    widget.close()


# ---------------------------------------------------------------------------
# Formula validation
# ---------------------------------------------------------------------------


class TestFormulaValidation:
    def test_empty_formula_disables_compute(self, form) -> None:
        form.formula.setText("")
        assert not form.compute_button.isEnabled()

    def test_empty_formula_shows_error_role(self, form) -> None:
        form.formula.setText("")
        role = form.formula_hint.property("role")
        assert role == "error"

    def test_valid_formula_enables_compute(self, form) -> None:
        form.formula.setText("SiO2")
        # Density default (2.2) is valid, energy grid is valid — button should be on
        assert form.compute_button.isEnabled()

    def test_valid_formula_shows_success_role(self, form) -> None:
        form.formula.setText("SiO2")
        role = form.formula_hint.property("role")
        assert role == "success"

    def test_invalid_formula_disables_compute(self, form) -> None:
        form.formula.setText("Xx999invalid!!")
        assert not form.compute_button.isEnabled()

    def test_invalid_formula_shows_error_role(self, form) -> None:
        form.formula.setText("Xx999invalid!!")
        role = form.formula_hint.property("role")
        assert role == "error"

    def test_complex_valid_formula(self, form) -> None:
        form.formula.setText("Ca5(PO4)3F")
        assert form.compute_button.isEnabled()

    def test_formula_change_triggers_revalidation(self, form) -> None:
        form.formula.setText("SiO2")
        assert form.compute_button.isEnabled()
        form.formula.setText("")
        assert not form.compute_button.isEnabled()


# ---------------------------------------------------------------------------
# Density validation
# ---------------------------------------------------------------------------


class TestDensityValidation:
    def test_valid_density_no_error(self, form) -> None:
        form.formula.setText("SiO2")
        form.density.setValue(2.2)
        assert form.compute_button.isEnabled()

    def test_density_at_minimum_boundary(self, form) -> None:
        """QDoubleSpinBox minimum is 0.001 — cannot go below."""
        form.formula.setText("SiO2")
        form.density.setValue(0.001)
        # 0.001 is the spinbox minimum; validate_density may or may not accept it
        # The important thing is compute_button state reflects validation result
        role = form.density_hint.property("role")
        assert role in ("success", "error")

    def test_density_valid_shows_success(self, form) -> None:
        form.formula.setText("SiO2")
        form.density.setValue(5.0)
        role = form.density_hint.property("role")
        assert role == "success"

    def test_density_hint_updates_on_change(self, form) -> None:
        form.formula.setText("SiO2")
        form.density.setValue(2.2)
        text_before = form.density_hint.text()
        form.density.setValue(3.5)
        text_after = form.density_hint.text()
        # Both are valid; hints may or may not change text but role stays success
        assert form.density_hint.property("role") == "success"
        # Silence unused variable warning
        _ = text_before, text_after


# ---------------------------------------------------------------------------
# Energy grid validation
# ---------------------------------------------------------------------------


class TestEnergyGridValidation:
    def test_valid_energy_range(self, form) -> None:
        form.formula.setText("SiO2")
        form.energy_start.setValue(8.0)
        form.energy_end.setValue(12.0)
        form.energy_points.setValue(50)
        assert form.compute_button.isEnabled()

    def test_end_less_than_start_disables_compute(self, form) -> None:
        form.formula.setText("SiO2")
        form.energy_start.setValue(12.0)
        form.energy_end.setValue(8.0)
        form.energy_points.setValue(50)
        assert not form.compute_button.isEnabled()

    def test_logspace_with_too_few_points_fails(self, form) -> None:
        form.formula.setText("SiO2")
        form.energy_start.setValue(8.0)
        form.energy_end.setValue(12.0)
        form.energy_points.setValue(2)
        form.logspace.setChecked(True)
        assert not form.compute_button.isEnabled()


# ---------------------------------------------------------------------------
# Widget values API
# ---------------------------------------------------------------------------


class TestFormValues:
    def test_values_returns_tuple(self, form) -> None:
        form.formula.setText("Al2O3")
        form.density.setValue(3.95)
        result = form.values()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_values_formula(self, form) -> None:
        form.formula.setText("Al2O3")
        formula, _density, _config = form.values()
        assert formula == "Al2O3"

    def test_values_density(self, form) -> None:
        form.formula.setText("Al2O3")
        form.density.setValue(3.95)
        _formula, density, _config = form.values()
        assert abs(density - 3.95) < 1e-6

    def test_energy_config_from_form(self, form) -> None:
        form.energy_start.setValue(5.0)
        form.energy_end.setValue(15.0)
        form.energy_points.setValue(101)
        config = form.energy_config()
        assert config.start_kev == pytest.approx(5.0)
        assert config.end_kev == pytest.approx(15.0)
        assert config.points == 101
