"""Tests for ``kk_calculate_real``, ``KKResult`` and the lazy ``xraylabtool.kk`` API."""

from __future__ import annotations

import dataclasses
import doctest
import logging
from pathlib import Path
import subprocess
import sys
import typing

import numpy as np
import pytest

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool import kk
from xraylabtool.backend.array_ops import JaxBackend, get_backend, set_backend
from xraylabtool.exceptions import EnergyError, FormulaError, ValidationError
from xraylabtool.kk import api, kk_calculate_real, result
from xraylabtool.kk.result import KKResult
from xraylabtool.kk.spectrum import compound_spectrum
from xraylabtool.kk.transform import kk_real_spectrum, relativistic_correction
from xraylabtool.utils import parse_formula

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "characterization" / "kk" / "golden"


@pytest.mark.unit
@pytest.mark.parametrize("name", ["si", "sio2"])
def test_default_grid_clipped_to_0p03_30_kev_and_logged(
    name: str, kk_log: pytest.LogCaptureFixture
) -> None:
    with np.load(GOLDEN / f"{name}.npz") as g:
        formula = str(g["formula"])
        keep = (g["energy_ev"] >= 30.0) & (g["energy_ev"] <= 30000.0)
        golden_ev, golden_f1, golden_f2 = (
            g["energy_ev"][keep],
            g["f1"][keep],
            g["f2"][keep],
        )
    res = kk_calculate_real(formula)
    assert res.energy_kev.min() >= 0.03
    assert res.energy_kev.max() <= 30.0
    np.testing.assert_allclose(res.energy_kev * 1000.0, golden_ev, rtol=1e-15)
    seg, coeffs = compound_spectrum(*parse_formula(formula))
    assert_kk_close(res.f1, golden_f1, res.energy_kev * 1000.0, seg, coeffs, rtol=1e-10)
    np.testing.assert_allclose(res.f2, golden_f2, rtol=1e-12)

    records = [r for r in kk_log.records if r.name == "xraylabtool.kk"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert "clipped to 0.03-30 keV" in message
    assert f"kept {golden_ev.size} of " in message


@pytest.mark.unit
@pytest.mark.parametrize("energies_kev", [[0.01], [0.01, 1.0, 500.0], 12.5])
def test_explicit_energy_between_10ev_and_500kev_ok(energies_kev) -> None:
    res = kk_calculate_real("Au", energies_kev=energies_kev)
    expected = np.atleast_1d(np.asarray(energies_kev, dtype=float))
    assert np.array_equal(res.energy_kev, expected)
    assert np.isfinite(res.f1).all()
    assert np.isfinite(res.f2).all()


@pytest.mark.unit
def test_boundary_energies_are_accepted_and_f2_matches_the_table() -> None:
    from xraylabtool.kk.spectrum import evaluate_f2

    symbols, counts = parse_formula("Si")
    seg_ev, coeffs = compound_spectrum(symbols, counts)
    res = kk_calculate_real("Si", energies_kev=[0.01, 500.0])
    expected = evaluate_f2(np.array([10.0, 500_000.0]), seg_ev, coeffs)
    np.testing.assert_allclose(res.f2, expected, rtol=1e-12)
    assert np.isfinite(res.f1).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    "energy_kev", [np.nextafter(0.01, 0.0), np.nextafter(500.0, 1e3)]
)
def test_energy_one_ulp_outside_the_data_range_raises(energy_kev: float) -> None:
    with pytest.raises(EnergyError, match="tabulated range"):
        kk_calculate_real("Si", energies_kev=[energy_kev])


@pytest.mark.unit
@pytest.mark.parametrize("energy_kev", [0.009, 501.0, 0.0, -1.0])
def test_energy_out_of_data_range_raises_energy_error(energy_kev: float) -> None:
    with pytest.raises(EnergyError, match="tabulated range") as excinfo:
        kk_calculate_real("Si", energies_kev=[1.0, energy_kev])
    assert excinfo.value.valid_range == "0.01-500 keV"


@pytest.mark.unit
@pytest.mark.parametrize(
    "energies",
    [
        [float("nan")],
        [1.0, float("inf")],
        [-float("inf")],
        [],
        [[1.0, 2.0]],
        ["a"],
        ["1.0"],  # numeric strings are not energies
        [True, True],  # nor are booleans (1 keV would be a valid energy)
        [1 + 1j],  # complex input must not lose its imaginary part
        np.array([1 + 0j]),
    ],
)
def test_invalid_energies_raise_energy_error(energies) -> None:
    with pytest.raises(EnergyError):
        kk_calculate_real("Si", energies_kev=energies)


@pytest.mark.unit
@pytest.mark.parametrize("formula", ["Xx0", "", "H0", "Xx2", "Si-O", "SiO2junk", "()"])
def test_invalid_formulas_raise_formula_error(formula: str) -> None:
    with pytest.raises(FormulaError):
        kk_calculate_real(formula)


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [0, -3, 1.5, True, "8"])
def test_invalid_chunk_size_raises_validation_error(chunk_size) -> None:
    with pytest.raises(ValidationError, match="chunk_size"):
        kk_calculate_real("Si", energies_kev=[1.0], chunk_size=chunk_size)


@pytest.mark.unit
def test_keV_input_equals_internal_eV() -> None:
    """The keV API equals the eV kernel; the conversion happens once."""
    energies_kev = np.geomspace(0.05, 20.0, 37)
    res = kk_calculate_real("SiO2", energies_kev=energies_kev)
    seg, coeffs = compound_spectrum(*parse_formula("SiO2"))
    rel = relativistic_correction([14, 8], [1.0, 2.0])
    direct = kk_real_spectrum(energies_kev * 1000.0, seg, coeffs, rel)
    assert np.array_equal(res.f1, direct)
    assert res.relativistic_correction == rel

    # The default grid is converted the other way (eV / 1000) and back.
    default = kk_calculate_real("SiO2")
    assert_kk_close(
        kk_calculate_real("SiO2", energies_kev=default.energy_kev[::7]).f1,
        default.f1[::7],
        default.energy_kev[::7] * 1000.0,
        seg,
        coeffs,
        rtol=1e-10,
    )


@pytest.mark.unit
def test_result_fields_and_input_order_are_preserved() -> None:
    energies = np.array([5.0, 0.5, 5.0, 2.0])  # unsorted, duplicated
    res = kk_calculate_real("Fe", energies_kev=energies)
    assert isinstance(res, KKResult)
    assert res.formula == "Fe"
    assert np.array_equal(res.energy_kev, energies)
    assert res.f1[0] == res.f1[2]
    assert res.merge_points_kev is None
    assert res.n_refined_points == 0
    assert res.nearedge_scaled_kev_f2 is None
    assert res.relativistic_correction == relativistic_correction([26], [1.0])
    assert res.f1.dtype == np.float64
    assert res.f2.dtype == np.float64
    assert res.f1.shape == res.f2.shape == res.energy_kev.shape == (4,)


@pytest.mark.unit
def test_energies_are_copied_not_aliased() -> None:
    energies = np.array([1.0, 2.0])
    res = kk_calculate_real("Si", energies_kev=energies)
    energies[0] = 99.0
    assert res.energy_kev[0] == 1.0


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_is_recorded_and_chunk_size_does_not_matter(backend: str) -> None:
    if backend == "jax":
        pytest.importorskip("jax")
    original = "jax" if isinstance(get_backend(), JaxBackend) else "numpy"
    set_backend(backend)
    try:
        energies = np.geomspace(0.03, 30.0, 40)
        full = kk_calculate_real("SiO2", energies_kev=energies)
        chunked = kk_calculate_real("SiO2", energies_kev=energies, chunk_size=7)
    finally:
        set_backend(original)
    assert full.backend == backend
    seg, coeffs = compound_spectrum(*parse_formula("SiO2"))
    assert_kk_close(chunked.f1, full.f1, energies * 1000.0, seg, coeffs, rtol=1e-12)


@pytest.mark.unit
def test_kk_result_is_frozen() -> None:
    res = kk_calculate_real("Si", energies_kev=[1.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.f1 = np.zeros(1)  # type: ignore[misc]
    names = [f.name for f in dataclasses.fields(KKResult)]
    # eq=False: identity semantics, so array fields never reach == (which raises).
    assert res == res  # noqa: PLR0124
    assert res != kk_calculate_real("Si", energies_kev=[1.0])
    assert isinstance(hash(res), int)
    assert names == [
        "formula",
        "energy_kev",
        "f1",
        "f2",
        "relativistic_correction",
        "merge_points_kev",
        "n_refined_points",
        "backend",
        "nearedge_scaled_kev_f2",
    ]


@pytest.mark.unit
def test_import_xraylabtool_does_not_import_kk() -> None:
    code = (
        "import sys, xraylabtool\n"
        "assert 'xraylabtool.kk' not in sys.modules\n"
        "import xraylabtool.kk\n"  # the package itself is cheap ...
        "assert 'xraylabtool.kk.api' not in sys.modules\n"
        "assert 'xraylabtool.kk.transform' not in sys.modules\n"
        "xraylabtool.kk.kk_calculate_real\n"  # ... submodules load on first use
        "assert 'xraylabtool.kk.api' in sys.modules\n"
        "assert xraylabtool.kk is sys.modules['xraylabtool.kk']\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr


@pytest.mark.unit
def test_kk_is_a_lazy_submodule_of_xraylabtool() -> None:
    import xraylabtool

    assert "kk" in xraylabtool._LAZY_SUBMODULES
    assert xraylabtool.kk is kk


@pytest.mark.unit
def test_public_names_resolve_lazily() -> None:
    assert set(kk.__all__) == set(kk._EXPORTS)
    names = dir(kk)
    assert len(names) == len(set(names))
    for name in kk.__all__:
        assert getattr(kk, name) is not None
        assert name in dir(kk)
    names = dir(kk)  # after the lazy accesses above: still no duplicates
    assert len(names) == len(set(names))
    assert kk.kk_calculate_real is api.kk_calculate_real
    assert kk.KKResult is KKResult
    with pytest.raises(AttributeError, match="no_such_name"):
        kk.no_such_name  # noqa: B018


@pytest.mark.unit
@pytest.mark.parametrize("module", [api, result])
def test_docstring_examples_run(module) -> None:
    results = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0
    assert results.failed == 0


@pytest.mark.unit
def test_type_hints_resolve_at_runtime() -> None:
    """Sphinx autodoc calls ``get_type_hints``; it must not hit a NameError."""
    hints = typing.get_type_hints(KKResult)
    assert set(hints) == {f.name for f in dataclasses.fields(KKResult)}
    typing.get_type_hints(api.kk_calculate_real)


@pytest.mark.unit
def test_empty_default_grid_after_merge_raises() -> None:
    from xraylabtool.kk import NearEdgeSpectrum

    spec = NearEdgeSpectrum([10.0, 500000.0], [0.0, 1.0])
    with pytest.raises(EnergyError, match=r"0\.03-30 keV.*energies"):
        kk_calculate_real("Si", nearedge=spec, input_type="asf")
    # Explicit energies still work for the same merge.
    res = kk_calculate_real("Si", nearedge=spec, input_type="asf", energies_kev=[1.0])
    assert np.isfinite(res.f1).all()
