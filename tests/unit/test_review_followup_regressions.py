"""Follow-up regressions from the second review of the codebase-review fixes."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from xraylabtool.calculators import host
from xraylabtool.calculators.derived_quantities import calculate_derived_quantities
from xraylabtool.calculators.xray_result import XRayResult
from xraylabtool.data_handling.atomic_cache import FastAtomicDataProvider
from xraylabtool.data_handling.batch_processing import load_batch_input
from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError
from xraylabtool.kk.spectrum import _atomic_numbers


@pytest.mark.parametrize(
    "count", [jnp.array(1.0), np.array(2.0), Fraction(1, 2), np.float32(1.5), 3]
)
def test_kk_element_counts_accept_real_scalar_types(count):
    assert _atomic_numbers(["Si"], [count]) == [14]


@pytest.mark.parametrize("count", [1 + 0j, np.complex64(1), "2", None, 0, -1, np.nan])
def test_kk_element_counts_reject_invalid_values(count):
    with pytest.raises(FormulaError):
        _atomic_numbers(["Si"], [count])


@pytest.mark.parametrize("energies", [np.nan, [8.0, np.inf], "abc", None])
def test_atomic_provider_rejects_non_finite_or_non_numeric_energies(energies):
    with pytest.raises(EnergyError):
        FastAtomicDataProvider().get_scattering_factors("Si", energies)


def test_batch_input_rejects_a_blank_energy_cell(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text('formula,density,energy\nSiO2,2.2,\nSi,2.33,"8,9"\n')
    with pytest.raises(ValueError, match="Non-finite energy"):
        load_batch_input(path, energy_column="energy")


def test_linear_absorption_survives_a_non_float_field():
    r = XRayResult.__new__(XRayResult)
    r.attenuation_length_cm = [0, 2]
    np.testing.assert_array_equal(r.linear_absorption_coefficient, [0.0, 0.5])


def test_mixed_shape_derived_calls_share_one_compiled_program():
    from xraylabtool.calculators import optics

    def call(n_wl, n_delta, n_beta):
        calculate_derived_quantities(
            np.full(n_wl, 1e-10),
            np.full(n_delta, 1e-6),
            np.full(n_beta, 1e-8),
            2.2,
            60.0,
            30.0,
        )

    optics._derived_fields_program.clear_cache()
    call(70, 70, 70)
    before = optics._derived_fields_program._cache_size()
    call(1, 70, 1)
    call(70, 1, 70)
    assert optics._derived_fields_program._cache_size() == before
    assert host.run_derived_fields  # exported helper still present


def _result():
    from xraylabtool import calculate_single_material_properties

    return calculate_single_material_properties("SiO2", np.array([8.0, 10.0]), 2.2)


def test_default_csv_keeps_the_derived_output_columns():
    from xraylabtool.io.data_export import format_xray_result

    header = format_xray_result(_result(), "csv").splitlines()[0].split(",")
    for name in ("energy_ev", "critical_angle_mrad", "linear_absorption_coefficient"):
        assert name in header
    assert "Formula" not in header  # deprecated aliases stay out


def test_unknown_export_fields_raise_instead_of_writing_a_ragged_csv(tmp_path):
    from xraylabtool.io.data_export import format_xray_result
    from xraylabtool.io.file_operations import export_to_csv

    with pytest.raises(ValidationError, match="nope"):
        export_to_csv([_result()], str(tmp_path / "o.csv"), fields=["formula", "nope"])
    with pytest.raises(ValidationError, match="nope"):
        format_xray_result(_result(), "csv", fields=["nope"])


def test_batch_rows_with_unquoted_extra_energies_are_not_truncated(capsys):
    from xraylabtool.interfaces.cli.commands.batch import _parse_batch_data

    rows = [
        {"formula": "Si", "density": "2.33", "energy": "8", None: ["10", "12"]},
        {"formula": "SiO2", "density": "2.2", "energy": "8,10"},
    ]
    formulas, _, energy_sets = _parse_batch_data(rows)
    assert formulas == ["SiO2"]
    assert energy_sets == [[8.0, 10.0]]
    assert "quote multiple energies" in capsys.readouterr().err


def test_batch_input_accepts_a_utf8_bom(tmp_path):
    from types import SimpleNamespace

    from xraylabtool.interfaces.cli.commands.batch import _validate_batch_input

    path = tmp_path / "in.csv"
    path.write_bytes(b"\xef\xbb\xbfformula,density,energy\nSi,2.33,8\n")
    rows = _validate_batch_input(SimpleNamespace(input_file=str(path)))
    assert rows is not None and rows[0]["formula"] == "Si"


def test_completion_descriptions_are_escaped_for_zsh_and_fish():
    from xraylabtool.interfaces.completion_v2.shells import _fish_text, _zsh_text

    nasty = 'it\'s $(echo PWNED) `x` "q" \\'
    assert _fish_text(nasty) == 'it\\\'s $(echo PWNED) `x` "q" \\\\'
    assert _zsh_text(nasty) == 'it\'s \\$(echo PWNED) \\`x\\` \\"q\\" \\\\'


def test_batch_summary_counts_rows_skipped_at_parse_time(tmp_path, capsys):
    from types import SimpleNamespace

    from xraylabtool.interfaces.cli.commands.batch import cmd_batch

    path = tmp_path / "in.csv"
    path.write_text("formula,density,energy\nSiO2,2.2,10\nBAD,x,10\nSi,2.33,8,10,12\n")
    args = SimpleNamespace(
        input_file=str(path),
        output=str(tmp_path / "out.csv"),
        format=None,
        fields=None,
        verbose=False,
        progress=False,
        no_progress=True,
        workers=None,
        debug=False,
    )
    assert cmd_batch(args) == 0  # some rows succeeded: warnings, not failure
    assert "2 of 3 input rows were skipped" in capsys.readouterr().err


def _batch_args(tmp_path, **overrides):
    from types import SimpleNamespace

    base = {
        "input_file": str(tmp_path / "in.csv"),
        "output": str(tmp_path / "out.csv"),
        "format": None,
        "fields": None,
        "verbose": False,
        "progress": False,
        "no_progress": True,
        "workers": None,
        "debug": False,
    }
    return SimpleNamespace(**{**base, **overrides})


def test_batch_strict_fails_and_writes_nothing_when_a_row_is_dropped(tmp_path):
    from xraylabtool.interfaces.cli.commands.batch import cmd_batch

    (tmp_path / "in.csv").write_text("formula,density,energy\nSiO2,2.2,10\nBAD,x,10\n")
    assert cmd_batch(_batch_args(tmp_path, strict=True)) == 2
    assert not (tmp_path / "out.csv").exists()


def test_batch_strict_passes_when_every_row_is_processed(tmp_path):
    from xraylabtool.interfaces.cli.commands.batch import cmd_batch

    (tmp_path / "in.csv").write_text('formula,density,energy\nSiO2,2.2,"8,10"\n')
    assert cmd_batch(_batch_args(tmp_path, strict=True)) == 0
    assert (tmp_path / "out.csv").exists()


def test_batch_rejects_duplicate_column_names(tmp_path, capsys):
    from xraylabtool.interfaces.cli.commands.batch import _validate_batch_input

    path = tmp_path / "in.csv"
    path.write_text("formula,density,energy,energy\nSi,2.33,8,12\n")
    assert _validate_batch_input(_batch_args(tmp_path, input_file=str(path))) is None
    assert "Duplicate column names: ['energy']" in capsys.readouterr().err


def test_calc_reports_a_zero_molecular_weight_formula(capsys):
    from types import SimpleNamespace

    from xraylabtool.interfaces.cli.commands.calc import cmd_calc

    args = SimpleNamespace(
        formula="Si0",
        energy="8",
        density=2.3,
        output=None,
        format="table",
        fields=None,
        precision=6,
        verbose=False,
        debug=False,
    )
    assert cmd_calc(args) == 1
    assert "zero molecular weight" in capsys.readouterr().err
