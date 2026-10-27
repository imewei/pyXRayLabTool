"""Regression tests for defects found in the cross-model deep-debugging review.

Each test class pins one root cause (see the docstring for the symptom). They
were written before the fixes and fail on the code they were written against.
"""

from __future__ import annotations

import csv
import functools
import json
import os
import pathlib
import subprocess
import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    _validate_and_process_energies,
    calculate_multiple_xray_properties,
    calculate_single_material_properties,
    calculate_xray_properties,
)
from xraylabtool.exceptions import EnergyError, FormulaError

DATA_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "xraylabtool"
    / "data"
    / "AtomicScatteringFactor"
)


# ---------------------------------------------------------------------------
# Package-level lazy attribute access
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLazySubmoduleAccess:
    """`from xraylabtool import X` inside `__getattr__` recursed forever."""

    @pytest.mark.parametrize(
        "name",
        [
            "utils",
            "constants",
            "validation",
            "analysis",
            "io",
            "calculators",
            "data_handling",
        ],
    )
    def test_attribute_access_in_fresh_interpreter(self, name: str) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", f"import xraylabtool as x; assert x.{name}"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-400:]

    def test_missing_attribute_is_attribute_error_not_recursion(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import xraylabtool as x\n"
                "assert not hasattr(x, 'cleanup')\n"
                "assert getattr(x, 'export', None) is None\n",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-400:]


# ---------------------------------------------------------------------------
# Energy input contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEnergyInputContract:
    """Multi-dimensional, 0-d, complex and NaN energies were mishandled."""

    def test_2d_energy_is_rejected(self) -> None:
        with pytest.raises(EnergyError):
            calculate_single_material_properties("SiO2", np.array([[8.0, 10.0]]), 2.2)

    def test_2d_energy_rejected_by_multi_validator(self) -> None:
        with pytest.raises(EnergyError):
            _validate_and_process_energies(np.array([[8.0, 10.0]]))

    def test_zero_dimensional_array_is_a_scalar(self) -> None:
        result = calculate_single_material_properties("Si", np.array(8.0), 2.33)
        assert result.energy_kev.shape == (1,)
        reference = calculate_single_material_properties("Si", 8.0, 2.33)
        np.testing.assert_allclose(result.dispersion_delta, reference.dispersion_delta)

    def test_complex_array_is_rejected(self) -> None:
        with pytest.raises(EnergyError):
            calculate_single_material_properties("Si", np.array([8.0 + 4.0j]), 2.33)

    def test_nan_energy_is_an_energy_error(self) -> None:
        with pytest.raises(EnergyError):
            calculate_single_material_properties("Si", float("nan"), 2.33)

    def test_result_does_not_alias_caller_array(self) -> None:
        energies = np.array([8.0, 10.0])
        result = calculate_single_material_properties("Si", energies, 2.33)
        energies[:] = 1.0
        np.testing.assert_array_equal(result.energy_kev, [8.0, 10.0])


# ---------------------------------------------------------------------------
# Anomalous dispersion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnomalousDispersion:
    """Negative delta near absorption edges is physical and must not abort."""

    def test_carbon_k_edge_region_computes(self) -> None:
        result = calculate_single_material_properties("C", 0.2841, 2.0)
        assert result.dispersion_delta[0] < 0
        assert np.isfinite(result.attenuation_length_cm[0])
        assert result.critical_angle_degrees[0] == 0.0

    def test_every_element_computes_at_both_range_limits(self) -> None:
        failures = []
        for path in sorted(DATA_DIR.glob("*.nff")):
            symbol = path.stem.capitalize()
            for energy in (0.03, 30.0):
                try:
                    calculate_single_material_properties(symbol, energy, 5.0)
                except Exception as exc:
                    failures.append((symbol, energy, type(exc).__name__))
        assert not failures, failures[:10]


# ---------------------------------------------------------------------------
# Atomic data cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtomicDataCacheIsImmutable:
    """The memoized bulk loader returned a shared, mutable dict."""

    def test_returned_mapping_cannot_be_mutated(self) -> None:
        from xraylabtool.calculators.cache import get_bulk_atomic_data

        data = get_bulk_atomic_data(("Si",))
        with pytest.raises(TypeError):
            data["Si"] = {"atomic_number": 14, "atomic_weight": 1.0}  # type: ignore[index]
        with pytest.raises(TypeError):
            del data["Si"]  # type: ignore[attr-defined]

    def test_clear_cache_clears_inner_loader(self) -> None:
        from xraylabtool.calculators.cache import (
            clear_scattering_factor_cache,
            get_bulk_atomic_data,
        )
        from xraylabtool.data_handling.atomic_cache import get_bulk_atomic_data_fast

        get_bulk_atomic_data(("Si",))
        assert get_bulk_atomic_data_fast.cache_info().currsize > 0
        clear_scattering_factor_cache()
        assert get_bulk_atomic_data_fast.cache_info().currsize == 0


# ---------------------------------------------------------------------------
# Tabulated data integrity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScatteringDataIntegrity:
    """f.nff row 1 had energy 129.3 instead of 29.3 and was sorted into place."""

    def test_every_data_file_is_strictly_increasing_and_finite(self) -> None:
        problems = []
        for path in sorted(DATA_DIR.glob("*.nff")):
            data = np.loadtxt(path, delimiter=",", skiprows=1)
            if not np.all(np.isfinite(data)):
                problems.append((path.stem, "non-finite"))
            if np.any(np.diff(data[:, 0]) <= 0):
                problems.append((path.stem, "energies not strictly increasing"))
        assert not problems, problems

    def test_fluorine_has_no_spurious_dip_near_129_ev(self) -> None:
        result = calculate_single_material_properties("F", [0.128, 0.1293, 0.130], 1.5)
        f1 = result.scattering_factor_f1
        assert np.all(f1 > 6.5), f1
        assert np.max(np.abs(np.diff(f1))) < 0.5, f1


# ---------------------------------------------------------------------------
# Export correctness
# ---------------------------------------------------------------------------


def _two_results():
    return (
        calculate_single_material_properties("Si", [8.0, 10.0], 2.33),
        calculate_single_material_properties("C", [8.0], 2.0),
    )


@pytest.mark.unit
class TestExports:
    def test_unequal_energy_grids_are_rejected(self, tmp_path) -> None:
        from xraylabtool.io.file_operations import export_to_csv

        with pytest.raises(ValueError, match="energies"):
            export_to_csv(list(_two_results()), str(tmp_path / "o.csv"))

    def test_scalar_fields_get_their_own_column(self, tmp_path) -> None:
        from xraylabtool.io.file_operations import export_to_csv

        result = calculate_single_material_properties("Si", [8.0, 10.0], 2.33)
        path = tmp_path / "o.csv"
        export_to_csv(
            [result],
            str(path),
            fields=["formula", "density_g_cm3", "critical_angle_degrees"],
        )
        rows = list(csv.reader(path.open()))
        assert rows[0] == [
            "formula",
            "density_g_cm3",
            "critical_angle_degrees_energy_0",
            "critical_angle_degrees_energy_1",
        ]
        assert all(len(row) == len(rows[0]) for row in rows)
        assert float(rows[1][1]) == pytest.approx(2.33)
        assert float(rows[1][2]) == pytest.approx(result.critical_angle_degrees[0])

    def test_dict_of_results_csv_contains_numbers_not_repr(self, tmp_path) -> None:
        from xraylabtool.io.file_operations import save_calculation_results

        results = calculate_xray_properties(["SiO2"], [8.0, 10.0], [2.2])
        path = tmp_path / "o.csv"
        save_calculation_results(results, str(path), "csv")
        text = path.read_text()
        assert "XRayResult(" not in text
        rows = list(csv.DictReader(path.open()))
        assert float(rows[0]["dispersion_delta_energy_0"]) == pytest.approx(
            results["SiO2"].dispersion_delta[0]
        )

    def test_dict_of_results_json_contains_numbers_not_repr(self, tmp_path) -> None:
        from xraylabtool.io.file_operations import save_calculation_results

        results = calculate_xray_properties(["SiO2"], [8.0, 10.0], [2.2])
        path = tmp_path / "o.json"
        save_calculation_results(results, str(path), "json")
        data = json.loads(path.read_text())
        np.testing.assert_allclose(
            data["SiO2"]["dispersion_delta"], results["SiO2"].dispersion_delta
        )

    @pytest.mark.parametrize("order", [("energy", "formula"), ("formula", "energy")])
    def test_dict_csv_treats_strings_as_scalars(self, tmp_path, order) -> None:
        from xraylabtool.io.file_operations import save_calculation_results

        values = {"energy": [8.0, 10.0], "formula": "SiO2"}
        path = tmp_path / "o.csv"
        save_calculation_results({k: values[k] for k in order}, str(path), "csv")
        rows = list(csv.DictReader(path.open()))
        assert [r["formula"] for r in rows] == ["SiO2", "SiO2"]
        assert [float(r["energy"]) for r in rows] == [8.0, 10.0]

    @pytest.mark.parametrize("fmt", ["csv", "json"])
    def test_machine_readable_formatters_keep_small_values(self, fmt) -> None:
        from xraylabtool.interfaces.cli.formatting import (
            format_xray_result as cli_format,
        )
        from xraylabtool.io.data_export import format_xray_result as io_format

        result = calculate_single_material_properties("Si", [8.0, 10.0], 2.33)
        beta = result.absorption_beta
        assert beta.max() < 1e-6
        for fmt_fn in (cli_format, io_format):
            out = fmt_fn(result, fmt, fields=["energy_kev", "absorption_beta"])
            if fmt == "json":
                values = np.array(json.loads(out)["absorption_beta"], dtype=float)
            else:
                rows = list(csv.DictReader(out.splitlines()))
                values = np.array([float(r["absorption_beta"]) for r in rows])
            np.testing.assert_allclose(values, beta, rtol=1e-9)


# ---------------------------------------------------------------------------
# JIT behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBackendState:
    def test_device_program_compiles_once_across_materials(self) -> None:
        """Density and formula are traced arguments: one compile per bucket."""
        from xraylabtool.calculators import optics

        sizes = []
        for formula, density in [
            ("Si", 2.33),
            ("Fe", 7.87),
            ("Si", 2.34),
            ("Cu", 8.96),
        ]:
            calculate_single_material_properties(formula, [8.0, 10.0], density)
            sizes.append(optics._optical_program._cache_size())
        assert sizes[-1] == sizes[0], sizes


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBatchProcessing:
    FORMULAS = ["SiO2", "H2O", "Fe2O3", "Al2O3", "CaCO3", "MgO"]
    DENSITIES = [2.2, 1.0, 5.24, 3.95, 2.71, 3.58]

    def _config(self):
        from xraylabtool.data_handling.batch_processing import BatchConfig

        return BatchConfig(enable_progress=False, max_workers=4)

    def test_results_follow_input_order(self) -> None:
        from xraylabtool.data_handling.batch_processing import (
            calculate_batch_properties,
        )

        expected = [
            f"{f}@{d:.3f}" for f, d in zip(self.FORMULAS, self.DENSITIES, strict=True)
        ]
        for _ in range(6):
            results = calculate_batch_properties(
                self.FORMULAS, [8.0, 10.0], self.DENSITIES, self._config()
            )
            assert list(results) == expected

    def test_complex_scalar_energy_is_rejected(self) -> None:
        from xraylabtool.data_handling.batch_processing import (
            calculate_batch_properties,
        )

        with pytest.raises(EnergyError):
            calculate_batch_properties(["SiO2"], 8.0 + 4.0j, [2.2], self._config())

    def test_2d_energy_is_rejected(self) -> None:
        from xraylabtool.data_handling.batch_processing import (
            calculate_batch_properties,
        )

        with pytest.raises(EnergyError):
            calculate_batch_properties(
                ["SiO2"], np.array([[8.0, 10.0]]), [2.2], self._config()
            )

    def test_stalled_calculation_times_out(self, monkeypatch) -> None:
        import xraylabtool.data_handling.batch_processing as bp

        real = bp.process_single_calculation

        def slow(formula, energies, density):
            if formula == "H2O":
                time.sleep(4)
            return real(formula, energies, density)

        # Compile every bucket first: a cold compile would outlast the 0.5 s stall timeout.
        for formula, density in zip(self.FORMULAS, self.DENSITIES, strict=True):
            calculate_single_material_properties(formula, [8.0, 10.0], density)
        monkeypatch.setattr(bp, "process_single_calculation", slow)
        monkeypatch.setattr(bp, "_STALL_TIMEOUT_S", 0.5)
        start = time.monotonic()
        with pytest.warns(UserWarning, match="Timeout"):
            results = bp.calculate_batch_properties(
                self.FORMULAS, [8.0, 10.0], self.DENSITIES, self._config()
            )
        assert time.monotonic() - start < 3.5
        assert results["H2O"] is None
        assert results["SiO2@2.200"] is not None

    def test_process_pool_failure_falls_back_to_threads(self, monkeypatch) -> None:
        import concurrent.futures

        import xraylabtool.data_handling.batch_processing as bp

        class Broken:
            def __init__(self, *_args, **_kwargs):
                raise PermissionError("no semaphores in this environment")

        monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", Broken)
        formulas = [
            f"{s}"
            for s in ["SiO2", "H2O", "Fe2O3", "Al2O3", "CaCO3", "MgO", "TiO2", "ZnO"]
        ]
        densities = [2.2, 1.0, 5.24, 3.95, 2.71, 3.58, 4.23, 5.61]
        with pytest.warns(UserWarning, match="process pool"):
            results = bp.calculate_batch_properties(
                formulas, [8.0, 10.0], densities, self._config()
            )
        assert len(results) == 8
        assert all(v is not None for v in results.values())


# ---------------------------------------------------------------------------
# Formula parsing and validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormulaParsing:
    @pytest.mark.parametrize(
        "formula", ["Si(O-2)", "Si(O2+)", "Si()", "Si(O2)(", "Si(2)"]
    )
    def test_invalid_group_content_is_rejected(self, formula: str) -> None:
        from xraylabtool.utils import parse_formula

        with pytest.raises(FormulaError):
            parse_formula(formula)

    @pytest.mark.parametrize("formula", ["H0.00001O", "(H0.00001)O", "(H0.00001)1O"])
    def test_small_fractional_counts_survive_group_expansion(self, formula) -> None:
        from xraylabtool.utils import parse_formula

        symbols, counts = parse_formula(formula)
        assert symbols == ["H", "O"]
        assert counts == pytest.approx([1e-5, 1.0])

    def test_group_multiplier_applies(self) -> None:
        from xraylabtool.utils import parse_formula

        assert parse_formula("Ca(OH)2") == (["Ca", "O", "H"], [1.0, 2.0, 2.0])

    def test_validator_accepts_every_element_with_scattering_data(self) -> None:
        from xraylabtool.validation import validate_chemical_formula

        rejected = []
        for path in sorted(DATA_DIR.glob("*.nff")):
            symbol = path.stem.capitalize()
            try:
                validate_chemical_formula(symbol)
            except Exception:
                rejected.append(symbol)
        assert not rejected, rejected


# ---------------------------------------------------------------------------
# Multi-material keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRepeatedMaterials:
    def test_calculate_xray_properties_keeps_both_densities(self) -> None:
        results = calculate_xray_properties(["Si", "Si"], [8.0], [2.2, 2.33])
        assert len(results) == 2
        assert sorted(r.density_g_cm3 for r in results.values()) == [2.2, 2.33]

    def test_calculate_multiple_xray_properties_keeps_both_densities(self) -> None:
        results = calculate_multiple_xray_properties(["Si", "Si"], [8.0], [2.2, 2.33])
        assert len(results) == 2

    def test_gui_service_never_drops_a_material(self) -> None:
        services = pytest.importorskip("xraylabtool.gui.services")
        EnergyConfig, compute_multiple = (
            services.EnergyConfig,
            services.compute_multiple,
        )

        results = compute_multiple(["Si"] * 3, [2.0] * 3, EnergyConfig(points=1))
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Scattering-data text loader
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whitespace_delimited_loader_works_with_current_pandas(tmp_path) -> None:
    from xraylabtool.calculators.scattering_data import load_data_file

    path = tmp_path / "t.txt"
    path.write_text("E f1 f2\n10 1.0 0.1\n20 2.0 0.2\n")
    frame = load_data_file(str(path))
    assert list(frame.columns) == ["E", "f1", "f2"]
    assert len(frame) == 2


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGuiErrorHandling:
    @pytest.fixture(autouse=True)
    def _qt(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        pytest.importorskip("PySide6")

    @pytest.mark.parametrize(
        ("module", "mixin", "method"),
        [
            ("single_tab", "SingleTabMixin", "_on_single_error"),
            ("multi_tab", "MultiTabMixin", "_on_multi_error"),
        ],
    )
    def test_error_slot_reaches_the_user_notification(self, module, mixin, method):
        mod = pytest.importorskip(f"xraylabtool.gui.main_window.{module}")
        handler = getattr(getattr(mod, mixin), method)
        fake = MagicMock()
        handler(fake, "boom")
        fake._error.assert_called_once_with("boom")

    def test_worker_accepts_callables_without_code_attribute(self) -> None:
        CalculationWorker = pytest.importorskip(
            "xraylabtool.gui.workers"
        ).CalculationWorker

        done, errors = [], []
        worker = CalculationWorker(functools.partial(len, [1, 2, 3]))
        worker.signals.finished.connect(done.append)
        worker.signals.error.connect(errors.append)
        worker.run()
        assert done == [3] and not errors, errors

    def test_worker_ignores_local_variable_named_progress_cb(self) -> None:
        CalculationWorker = pytest.importorskip(
            "xraylabtool.gui.workers"
        ).CalculationWorker

        def task():
            progress_cb = 7
            return progress_cb

        done, errors = [], []
        worker = CalculationWorker(task)
        worker.signals.finished.connect(done.append)
        worker.signals.error.connect(errors.append)
        worker.run()
        assert done == [7] and not errors, errors

    def test_worker_injects_progress_callback_when_declared(self) -> None:
        CalculationWorker = pytest.importorskip(
            "xraylabtool.gui.workers"
        ).CalculationWorker

        def task(progress_cb=None):
            return progress_cb is not None

        done = []
        worker = CalculationWorker(task)
        worker.signals.finished.connect(done.append)
        worker.run()
        assert done == [True]


# ---------------------------------------------------------------------------
# Utility input validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUtilityValidation:
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_energy_to_wavelength_rejects_non_positive(self, value) -> None:
        from xraylabtool.utils import energy_to_wavelength

        with pytest.raises(ValueError):
            energy_to_wavelength(value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_wavelength_to_energy_rejects_non_positive(self, value) -> None:
        from xraylabtool.utils import wavelength_to_energy

        with pytest.raises(ValueError):
            wavelength_to_energy(value)

    def test_d_spacing_rejects_degenerate_inputs(self) -> None:
        from xraylabtool.utils import d_spacing_cubic

        with pytest.raises(ValueError):
            d_spacing_cubic(0, 0, 0, 5.431)
        with pytest.raises(ValueError):
            d_spacing_cubic(1, 1, 1, -5.431)

    @pytest.mark.parametrize("text", ["1-10:5:logg", "1-10:5:2", "1-10:5:log:extra"])
    def test_parse_energy_string_rejects_unknown_spacing(self, text) -> None:
        from xraylabtool.interfaces.cli.utils import parse_energy_string

        with pytest.raises(ValueError):
            parse_energy_string(text)
