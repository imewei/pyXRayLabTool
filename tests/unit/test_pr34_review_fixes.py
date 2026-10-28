"""Regressions found by the PR #34 review: input validation and visible failures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from xraylabtool.analysis import comparator as comparator_mod
from xraylabtool.analysis.comparator import MaterialComparator
from xraylabtool.calculators import core, host
from xraylabtool.data_handling.batch_processing import BatchConfig
from xraylabtool.exceptions import EnergyError
from xraylabtool.interfaces.cli.commands.batch import _process_batch_materials


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
def test_batch_config_rejects_a_non_positive_chunk_size(bad):
    with pytest.raises(ValueError, match="chunk_size"):
        BatchConfig(chunk_size=bad)


def test_run_optical_blocks_rejects_a_non_positive_max_block():
    material = core._encode_material("SiO2")
    with pytest.raises(ValueError, match="max_block"):
        host.run_optical_blocks([material], [2.2], [10.0], max_block=0)


def test_cli_batch_reports_failed_materials_on_stderr_with_progress_on(capsys):
    formulas = ["SiO2"] * 10 + ["Xx9"]
    args = SimpleNamespace(verbose=False, progress=True, no_progress=False)
    results = _process_batch_materials(formulas, [2.2] * 11, [[10.0]] * 11, args)
    err = capsys.readouterr().err
    assert len(results) == 10
    assert "Failed to process Xx9" in err
    assert "1 of 11 materials failed" in err


def test_comparator_keeps_the_cause_of_a_per_formula_failure(monkeypatch):
    boom = RuntimeError("device exploded")

    def fail(**_kw):
        raise boom

    monkeypatch.setattr(comparator_mod, "calculate_xray_properties", fail)
    with pytest.raises(ValueError, match="SiO2") as info:
        MaterialComparator()._calculate_material_data(["SiO2"], [2.2], [10.0])
    assert info.value.__cause__ is boom


@pytest.mark.parametrize("energies", [[[10.0]], [[10.0, 11.0]]])
def test_calculate_multiple_raises_energy_error_for_a_2d_energy_list(energies):
    with pytest.raises(EnergyError):
        core.calculate_multiple_xray_properties(["SiO2"], energies, [2.2])


def test_critical_angle_gradient_is_finite_where_delta_is_negative():
    """Si has delta < 0 at some tabulated energies; the clamp must not give NaN grads."""
    import jax
    import jax.numpy as jnp

    from xraylabtool.backend.runtime import ensure_x64
    from xraylabtool.calculators import optics
    from xraylabtool.calculators.tables import load_tables

    ensure_x64()
    z, counts, mw, n_el = optics.encode_formula("Si")
    energies = jnp.geomspace(0.03, 30.0, 400)

    def total(rho, e):
        props = optics.optical_properties(load_tables(), z, counts, rho, mw, n_el, e)
        return props[0].critical_angle.sum()

    assert bool((jnp.asarray(jax.grad(total, 1)(5.0, energies)) == 0).any())
    assert bool(jnp.isfinite(jax.grad(total, 1)(5.0, energies)).all())
    assert bool(jnp.isfinite(jax.grad(total, 0)(5.0, energies)))
