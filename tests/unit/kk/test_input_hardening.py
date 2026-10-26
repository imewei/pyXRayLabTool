"""Regressions for silent input corruption and misleading reports in ``xraylabtool.kk``."""

from __future__ import annotations

import numpy as np
import pytest

from xraylabtool.exceptions import FormulaError, NearEdgeDataError, ValidationError
from xraylabtool.kk import (
    KKResult,
    NearEdgeSpectrum,
    RefineOutcome,
    kk_calculate_real,
    load_asf,
    merge_spectra,
)
from xraylabtool.kk.nearedge import to_asf_scale
from xraylabtool.kk.refine import _log_slivers, refine_spectrum
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.kk.transform import kk_real_spectrum

SEG = [1.0, 2.0, 4.0]
COEFFS = [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0]]


@pytest.mark.unit
def test_kernel_rejects_complex_instead_of_dropping_the_imaginary_part() -> None:
    for args in (
        (np.array([1.5 + 9j]), SEG, COEFFS, 0.0),
        (np.array([1.5]), np.array(SEG, dtype=complex), COEFFS, 0.0),
        (np.array([1.5]), SEG, np.array(COEFFS, dtype=complex), 0.0),
    ):
        with pytest.raises(ValueError, match="real"):
            kk_real_spectrum(*args)


@pytest.mark.unit
def test_evaluate_f2_rejects_complex() -> None:
    with pytest.raises(ValueError, match="real"):
        evaluate_f2(np.array([1.5 + 9j]), SEG, COEFFS)
    with pytest.raises(ValueError, match="real"):
        evaluate_f2([1.5], np.array(SEG, dtype=complex), COEFFS)


@pytest.mark.unit
def test_refine_rejects_complex_inputs() -> None:
    e, f = np.array([1.5, 3.0]), np.array([1.0, 2.0])
    for args in (
        (e + 1j, f, f, SEG, COEFFS, 0.0),
        (e, f + 1j, f, SEG, COEFFS, 0.0),
        (e, f, f + 1j, SEG, COEFFS, 0.0),
    ):
        with pytest.raises(ValidationError, match="real"):
            refine_spectrum(*args, tolerance=0.1)
    # seg_ev errors are the kernel's ValueError, as refine_spectrum documents.
    with pytest.raises(ValueError, match="real"):
        refine_spectrum(
            e, f, f, np.array(SEG, dtype=complex), COEFFS, 0.0, tolerance=0.1
        )


@pytest.mark.unit
def test_result_containers_reject_complex_fields() -> None:
    fields = {
        "formula": "Si",
        "energy_kev": np.array([1.0]),
        "f1": np.array([1.0 + 2j]),
        "f2": np.array([1.0]),
        "relativistic_correction": 0.0,
        "merge_points_kev": None,
        "n_refined_points": 0,
        "backend": "numpy",
    }
    with pytest.raises(ValidationError, match="real"):
        KKResult(**fields)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="real"):
        RefineOutcome(np.array([1.0 + 1j]), np.array([1.0]), np.array([1.0]), 0, False)


@pytest.mark.unit
def test_merge_spectra_rejects_complex_inputs() -> None:
    seg, coeffs = compound_spectrum(["C"], [1.0])
    edge = NearEdgeSpectrum([280.0, 290.0, 300.0, 310.0], [0.1, 0.2, 0.4, 0.5])
    with pytest.raises(ValidationError, match="real"):
        merge_spectra(edge, seg.astype(complex), coeffs, merge_points_ev=(280.0, 310.0))
    with pytest.raises(ValidationError, match="real"):
        merge_spectra(edge, seg, coeffs.astype(complex), merge_points_ev=(280.0, 310.0))
    with pytest.raises(NearEdgeDataError, match="real"):
        merge_spectra(edge, seg, coeffs, merge_points_ev=(280.0 + 1j, 310.0))  # type: ignore[arg-type]


@pytest.mark.unit
def test_refinement_keeps_default_grid_knots_bit_exact() -> None:
    # Au has an edge at 126.175 eV where (k / 1000) * 1000 != k; recomputing eV from
    # keV moved f1 there by ~4e-8 although refinement added no point.
    plain = kk_calculate_real("Au")
    refined = kk_calculate_real("Au", refine_tolerance=1e9)
    assert refined.n_refined_points == 0
    np.testing.assert_array_equal(refined.energy_kev, plain.energy_kev)
    np.testing.assert_array_equal(refined.f1, plain.f1)


@pytest.mark.unit
def test_evaluate_f2_rejects_nan_segment_edge() -> None:
    # A NaN last edge used to defeat the upper-range check: [9.] came back for 100 eV.
    with pytest.raises(ValueError, match="seg_ev"):
        evaluate_f2([100.0], [1.0, 2.0, np.nan], COEFFS)


@pytest.mark.unit
def test_refinement_accepts_keV_values_that_collide_in_eV() -> None:
    low, high = 0.030000000000000027, 0.03000000000000003
    assert low != high
    assert low * 1000 == high * 1000
    res = kk_calculate_real("Si", energies_kev=[low, high], refine_tolerance=0.1)
    ev = res.energy_kev * 1000
    assert (np.diff(ev) > 0).all()
    assert res.energy_kev[0] == low


@pytest.mark.unit
def test_skipped_sliver_log_pairs_widest_width_with_its_own_energy(
    kk_log: pytest.LogCaptureFixture,
) -> None:
    _log_slivers(2, 1, 5.0e5, 8000.0, 0.05)
    assert "widest 8e+03 eV at 500000 eV" in kk_log.text


@pytest.mark.unit
@pytest.mark.parametrize("name", ["density", "formula_mass"])
def test_to_asf_scale_rejects_non_numeric_density_with_validation_error(
    name: str,
) -> None:
    spec = NearEdgeSpectrum([280.0, 290.0], [0.1, 0.2])
    with pytest.raises(ValidationError, match=name):
        to_asf_scale(spec, "beta", **{name: "heavy"})


@pytest.mark.unit
@pytest.mark.parametrize("z", [14.5, True, np.float64(3.0)])
def test_database_rejects_non_integer_atomic_number(z: object) -> None:
    db = load_asf()
    with pytest.raises(ValueError, match="atomic number"):
        db.energy_ev(z)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="atomic number"):
        db.im_coeffs(z)  # type: ignore[arg-type]


@pytest.mark.unit
def test_kk_result_stores_merge_points_as_an_immutable_tuple() -> None:
    res = KKResult(
        formula="Si",
        energy_kev=np.array([1.0]),
        f1=np.array([1.0]),
        f2=np.array([1.0]),
        relativistic_correction=0.0,
        merge_points_kev=[0.28, 0.32],  # type: ignore[arg-type]
        n_refined_points=0,
        backend="numpy",
    )
    assert res.merge_points_kev == (0.28, 0.32)
    assert isinstance(res.merge_points_kev, tuple)


@pytest.mark.unit
def test_reported_merge_points_are_the_ones_used_after_snapping() -> None:
    # 0.2800000000001 keV is 1e-10 eV above the data start and snaps to 280.0 eV.
    edge = NearEdgeSpectrum(
        [280.0, 290.0, 300.0, 310.0, 320.0], [0.05, 0.1, 0.3, 0.45, 0.5]
    )
    res = kk_calculate_real(
        "C",
        nearedge=edge,
        input_type="nexafs",
        merge_points_kev=(0.2800000000001, 0.32),
    )
    assert res.merge_points_kev == (0.28, 0.32)
    assert res.nearedge_scaled_kev_f2 is not None
    assert res.nearedge_scaled_kev_f2[0][0] == 0.28


@pytest.mark.unit
@pytest.mark.parametrize("formula", [5, b"Si", ["Si"], None])
def test_non_string_formula_raises_formula_error(formula: object) -> None:
    with pytest.raises(FormulaError, match="string"):
        kk_calculate_real(formula, energies_kev=[1.0])  # type: ignore[arg-type]
