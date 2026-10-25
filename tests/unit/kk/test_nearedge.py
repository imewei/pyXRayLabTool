"""Tests for near-edge loading, conversion and merging (``xraylabtool.kk.nearedge``)."""

from __future__ import annotations

import dataclasses
import doctest
import logging
from pathlib import Path
import typing

import numpy as np
import pytest

from xraylabtool import constants
from xraylabtool.exceptions import (
    DataFileError,
    NearEdgeDataError,
    ValidationError,
    XRayLabToolError,
)
from xraylabtool.kk import kk_calculate_real, nearedge
from xraylabtool.kk.nearedge import (
    NearEdgeSpectrum,
    load_nearedge,
    merge_spectra,
    to_asf_scale,
)
from xraylabtool.kk.spectrum import compound_spectrum, evaluate_f2
from xraylabtool.utils import parse_formula

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "tests" / "characterization" / "kk" / "nearedge_example.txt"

SEG_EV, COEFFS = compound_spectrum(*parse_formula("C8H8"))
MERGE = (280.0, 320.0)


def _edge_spectrum(
    lo: float = 270.0, hi: float = 330.0, n: int = 25
) -> NearEdgeSpectrum:
    """A rising edge-like NEXAFS curve on a regular eV grid."""
    energy = np.linspace(lo, hi, n)
    values = 0.05 + 0.5 / (1.0 + np.exp(-(energy - 300.0) / 4.0))
    return NearEdgeSpectrum(energy, values)


def _write(path: Path, rows: list[list[float]], header: str = "") -> Path:
    body = "\n".join("  ".join(repr(float(x)) for x in row) for row in rows)
    path.write_text(f"{header}{body}\n")
    return path


# Loading --------------------------------------------------------------------
@pytest.mark.unit
def test_load_reads_first_and_last_column_and_skips_header_lines(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "s.txt",
        [[1.0, 9.0, 0.5], [2.0, 9.0, 0.6], [3.0, 9.0, 0.7]],
        header="% comment\nEnergy (eV)  x  mu\n",
    )
    spec = load_nearedge(path)
    assert spec.energy_ev.tolist() == [1.0, 2.0, 3.0]
    assert spec.values.tolist() == [0.5, 0.6, 0.7]
    assert spec.energy_ev.dtype == spec.values.dtype == np.float64


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ([[1.0, 1.0], [1.0, 2.0], [2.0, 3.0]], "strictly increasing"),  # duplicate
        ([[2.0, 1.0], [1.0, 2.0]], "strictly increasing"),  # decreasing
        ([[1.0, 1.0], [3.0, 2.0], [2.0, 3.0]], "strictly increasing"),  # unsorted
        ([[1.0, 1.0], [2.0, float("nan")], [3.0, 3.0]], "finite"),
        ([[1.0, 1.0], [float("inf"), 2.0]], "finite"),
        ([[1.0, 1.0]], "at least 2"),
        ([[0.0, 1.0], [1.0, 2.0]], "positive"),
        ([[-1.0, 1.0], [1.0, 2.0]], "positive"),
    ],
)
def test_load_rejects_non_increasing_nan_and_short(
    tmp_path: Path, rows: list[list[float]], match: str
) -> None:
    with pytest.raises(NearEdgeDataError, match=match):
        load_nearedge(_write(tmp_path / "bad.txt", rows))


@pytest.mark.unit
def test_load_rejects_unreadable_or_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(DataFileError):
        load_nearedge(tmp_path / "missing.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a header\n")
    with pytest.raises(NearEdgeDataError, match="No numeric rows"):
        load_nearedge(empty)
    ragged = tmp_path / "ragged.txt"
    ragged.write_text("1 2\n2 3 4\n")
    with pytest.raises(NearEdgeDataError, match="columns"):
        load_nearedge(ragged)
    single = tmp_path / "single.txt"
    single.write_text("1\n2\n")
    with pytest.raises(NearEdgeDataError, match="columns"):
        load_nearedge(single)
    with pytest.raises(ValidationError, match="energy_unit"):
        load_nearedge(
            _write(tmp_path / "ok.txt", [[1.0, 1], [2.0, 2]]), energy_unit="MeV"
        )  # type: ignore[arg-type]


@pytest.mark.unit
def test_load_rejects_stray_text_inside_the_data_but_allows_comments(
    tmp_path: Path,
) -> None:
    ok = tmp_path / "ok.txt"
    ok.write_text("header words\n1 2\n# note\n\n2 3\n% another\n3 4\n")
    assert load_nearedge(ok).values.tolist() == [2.0, 3.0, 4.0]
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2\n2 3x\n3 4\n")
    with pytest.raises(NearEdgeDataError, match="line 2"):
        load_nearedge(bad)


@pytest.mark.unit
def test_load_example_file() -> None:
    spec = load_nearedge(EXAMPLE)
    assert spec.energy_ev.size == spec.values.size > 100
    assert spec.energy_ev[0] == 275.0
    assert np.all(np.diff(spec.energy_ev) > 0)


@pytest.mark.unit
def test_keV_unit_equals_eV_unit(tmp_path: Path) -> None:
    ev = load_nearedge(EXAMPLE)
    rows = [[e / 1000.0, v] for e, v in zip(ev.energy_ev, ev.values, strict=True)]
    kev = load_nearedge(_write(tmp_path / "kev.txt", rows), energy_unit="keV")
    np.testing.assert_allclose(kev.energy_ev, ev.energy_ev, rtol=1e-15, atol=0)
    assert np.array_equal(kev.values, ev.values)

    mp_ev, mp_kev = (280.0, 320.0), (0.28, 0.32)
    a = merge_spectra(to_asf_scale(ev, "nexafs"), SEG_EV, COEFFS, merge_points_ev=mp_ev)
    b = merge_spectra(
        to_asf_scale(kev, "nexafs"),
        SEG_EV,
        COEFFS,
        merge_points_ev=(mp_kev[0] * 1000.0, mp_kev[1] * 1000.0),
    )
    np.testing.assert_allclose(a[0], b[0], rtol=1e-14)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-9, atol=1e-12)


# Spectrum container ---------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ("energy", "values", "match"),
    [
        ([2.0, 1.0], [1.0, 1.0], "strictly increasing"),
        ([1.0, 1.0], [1.0, 1.0], "strictly increasing"),
        ([1.0, 2.0], [1.0, np.nan], "finite"),
        ([1.0, np.inf], [1.0, 1.0], "finite"),
        ([1.0, 2.0, 3.0], [1.0, 1.0], "same shape"),
        ([1.0], [1.0], "at least 2"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "1-D"),
        ([1.0 + 0j, 2.0], [1.0, 1.0], "real"),
        (["1", "2"], [1.0, 1.0], "real"),
        ([True, True], [1.0, 1.0], "real"),
    ],
)
def test_directly_constructed_spectrum_is_validated(energy, values, match) -> None:
    with pytest.raises(NearEdgeDataError, match=match):
        NearEdgeSpectrum(np.asarray(energy), np.asarray(values))


@pytest.mark.unit
def test_spectrum_is_immutable_and_copies_its_input() -> None:
    energy, values = np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0])
    spec = NearEdgeSpectrum(energy, values)
    energy[0], values[0] = 99, 99.0
    assert spec.energy_ev[0] == 1.0
    assert spec.values[0] == 1.0
    assert spec.energy_ev.dtype == np.float64
    assert not spec.energy_ev.flags.writeable
    assert not spec.values.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.values = values  # type: ignore[misc]
    assert spec == spec  # noqa: PLR0124  (eq=False: identity, never array ==)


@pytest.mark.unit
def test_nearedge_error_is_a_validation_error() -> None:
    assert issubclass(NearEdgeDataError, ValidationError)
    assert issubclass(NearEdgeDataError, XRayLabToolError)


@pytest.mark.unit
def test_type_hints_resolve_at_runtime() -> None:
    assert set(typing.get_type_hints(NearEdgeSpectrum)) == {"energy_ev", "values"}
    for func in (load_nearedge, to_asf_scale, merge_spectra):
        typing.get_type_hints(func)


# Conversion -----------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("name", ["nexafs", "photoabsorption", "xanes"])
def test_nexafs_conversion_matches_definition(name: str) -> None:
    spec = _edge_spectrum()
    hc = constants.SPEED_OF_LIGHT * constants.PLANCK / constants.ELEMENT_CHARGE
    expected = spec.energy_ev * spec.values / (2.0 * constants.THOMPSON * hc)
    out = to_asf_scale(spec, name)  # type: ignore[arg-type]
    np.testing.assert_allclose(out.values, expected, rtol=1e-14)
    assert np.array_equal(out.energy_ev, spec.energy_ev)


@pytest.mark.unit
def test_asf_input_is_unchanged_and_unknown_type_raises() -> None:
    spec = _edge_spectrum()
    out = to_asf_scale(spec, "asf")
    assert np.array_equal(out.values, spec.values)
    with pytest.raises(ValidationError, match="input_type"):
        to_asf_scale(spec, "mystery")  # type: ignore[arg-type]
    with pytest.raises(NearEdgeDataError, match="finite"):
        to_asf_scale(
            NearEdgeSpectrum(np.array([1e200, 2e200]), np.array([1e200, 1e200])),
            "nexafs",
        )


@pytest.mark.unit
def test_beta_round_trip_and_density_cancels_at_merge() -> None:
    spec = _edge_spectrum()
    hc = constants.SPEED_OF_LIGHT * constants.PLANCK / constants.ELEMENT_CHARGE
    density, mass = 1.05, 104.15
    out = to_asf_scale(spec, "beta", density=density, formula_mass=mass)
    number_density = density * constants.AVOGADRO / mass
    # Reverse conversion (f2 -> beta) recovers the input.
    beta = (
        1.0e6
        * out.values
        * number_density
        * constants.THOMPSON
        * hc**2
        / (2.0 * np.pi * out.energy_ev**2)
    )
    np.testing.assert_allclose(beta, spec.values, rtol=1e-13)

    merged = [
        merge_spectra(
            to_asf_scale(spec, "beta", density=d, formula_mass=m),
            SEG_EV,
            COEFFS,
            merge_points_ev=MERGE,
        )
        for d, m in [(1.05, 104.15), (0.3, 104.15), (2.0, 7.0), (None, None)]
    ]
    for e, c in merged[1:]:
        assert np.array_equal(e, merged[0][0])
        np.testing.assert_allclose(c, merged[0][1], rtol=1e-9, atol=1e-12)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValidationError, match="density"):
            to_asf_scale(spec, "beta", density=bad)
        with pytest.raises(ValidationError, match="formula_mass"):
            to_asf_scale(spec, "beta", formula_mass=bad)


# Merging --------------------------------------------------------------------
@pytest.mark.unit
def test_merge_builds_a_continuous_strictly_increasing_spectrum() -> None:
    spec = to_asf_scale(_edge_spectrum(), "nexafs")
    seg, coeffs = merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=MERGE)
    assert coeffs.shape == (seg.size - 1, 5)
    assert np.isfinite(seg).all()
    assert np.isfinite(coeffs).all()
    assert np.all(np.diff(seg) > 0)
    assert seg[0] == SEG_EV[0]
    assert seg[-1] == SEG_EV[-1]
    # Outside the merge window the tabulated spectrum is untouched.
    below, above = MERGE[0] > SEG_EV, MERGE[1] < SEG_EV
    assert np.array_equal(seg[seg < MERGE[0]], SEG_EV[below])
    assert np.array_equal(seg[seg > MERGE[1]], SEG_EV[above])
    # Continuity at both merge points: the tabulated value is reproduced.
    probe = np.array(MERGE)
    np.testing.assert_allclose(
        evaluate_f2(probe, seg, coeffs),
        evaluate_f2(probe, SEG_EV, COEFFS),
        rtol=1e-12,
    )
    # Inside the window the polyline passes through the scaled data points.
    inside = (spec.energy_ev > MERGE[0]) & (spec.energy_ev < MERGE[1])
    assert set(spec.energy_ev[inside]) <= set(seg)


@pytest.mark.unit
def test_default_merge_points_are_the_overlap_of_both_ranges() -> None:
    lo = float(SEG_EV[0])
    data = to_asf_scale(_edge_spectrum(lo / 2.0, lo * 30.0, 40), "nexafs")
    seg, coeffs = merge_spectra(data, SEG_EV, COEFFS, merge_points_ev=None)
    assert seg[0] == lo
    assert coeffs.shape == (seg.size - 1, 5)
    assert np.all(np.diff(seg) > 0)


@pytest.mark.unit
def test_merge_defaults_to_the_data_ends_and_accepts_knot_and_edge_points() -> None:
    spec = to_asf_scale(_edge_spectrum(), "nexafs")
    seg, coeffs = merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=None)
    assert seg.size - 1 == coeffs.shape[0]
    assert np.all(np.diff(seg) > 0)
    assert 270.0 in seg
    assert 330.0 in seg

    # Merge points exactly on tabulated knots: no zero-length segments.
    k = int(np.searchsorted(SEG_EV, 100.0))
    on_knots = (float(SEG_EV[k]), float(SEG_EV[k + 5]))
    wide = to_asf_scale(_edge_spectrum(on_knots[0] - 5, on_knots[1] + 5, 30), "nexafs")
    seg, coeffs = merge_spectra(wide, SEG_EV, COEFFS, merge_points_ev=on_knots)
    assert np.all(np.diff(seg) > 0)
    assert coeffs.shape == (seg.size - 1, 5)
    assert np.isfinite(coeffs).all()

    # Upper merge point on the last tabulated edge.
    top = float(SEG_EV[-1])
    data = to_asf_scale(_edge_spectrum(top - 100.0, top, 8), "nexafs")
    seg, coeffs = merge_spectra(data, SEG_EV, COEFFS, merge_points_ev=(top - 50.0, top))
    assert seg[-1] == top
    assert coeffs.shape == (seg.size - 1, 5)
    assert np.all(np.diff(seg) > 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("points", "match"),
    [
        ((320.0, 280.0), "increasing"),  # reversed
        ((300.0, 300.0), "increasing"),  # equal
        ((float("nan"), 320.0), "finite"),
        ((280.0, float("inf")), "finite"),
        ((260.0, 320.0), "inside the near-edge data"),  # below the data (270-330)
        ((280.0, 340.0), "inside the near-edge data"),  # above the data
        ((280.0,), "two"),
    ],
)
def test_merge_requires_ordered_points_inside_both_datasets(points, match) -> None:
    spec = to_asf_scale(_edge_spectrum(), "nexafs")
    with pytest.raises(NearEdgeDataError, match=match):
        merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=points)


@pytest.mark.unit
def test_merge_points_outside_data_raise() -> None:
    """Data reaching beyond the tabulated range: points there are rejected."""
    lo, hi = float(SEG_EV[0]), float(SEG_EV[-1])
    wide = to_asf_scale(_edge_spectrum(lo / 2.0, lo * 40.0, 40), "nexafs")
    with pytest.raises(NearEdgeDataError, match="tabulated") as below:
        merge_spectra(wide, SEG_EV, COEFFS, merge_points_ev=(lo / 1.5, lo * 20.0))
    assert f"{lo:g}" in str(below.value)
    far = to_asf_scale(_edge_spectrum(hi / 2.0, hi * 2.0, 40), "nexafs")
    with pytest.raises(NearEdgeDataError, match="tabulated"):
        merge_spectra(far, SEG_EV, COEFFS, merge_points_ev=(hi / 1.5, hi * 1.5))


@pytest.mark.unit
@pytest.mark.parametrize("flat", ["exact", "to_rounding", "tiny"])
def test_flat_endpoints_raise_not_inf(flat: str) -> None:
    energy = np.linspace(270.0, 330.0, 25)
    values = 0.05 + 0.5 * np.exp(-(((energy - 300.0) / 5.0) ** 2))  # peak, flat ends
    if flat == "exact":
        values[0] = values[-1] = 0.05
    elif flat == "to_rounding":
        values[-1] = values[0] * (1.0 + 1e-15)
    else:
        values[-1] = values[0] * (1.0 + 1e-9)
    spec = NearEdgeSpectrum(energy, values)  # already on the f2 scale
    with pytest.raises(NearEdgeDataError, match="flat endpoints"):
        merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=(270.0, 330.0))
    # Flat only because the merge points sit on a plateau of a real edge curve.
    plateau = to_asf_scale(_edge_spectrum(), "nexafs")
    with pytest.raises(NearEdgeDataError, match="flat endpoints"):
        merge_spectra(
            NearEdgeSpectrum(plateau.energy_ev, np.full_like(plateau.values, 3.0)),
            SEG_EV,
            COEFFS,
            merge_points_ev=MERGE,
        )


@pytest.mark.unit
def test_file_in_kev_declared_ev_raises_no_overlap(tmp_path: Path) -> None:
    ev = load_nearedge(EXAMPLE)
    rows = [[e / 1000.0, v] for e, v in zip(ev.energy_ev, ev.values, strict=True)]
    path = _write(tmp_path / "kev_file.txt", rows)
    wrong = load_nearedge(path)  # energy_unit defaults to eV
    spec = to_asf_scale(wrong, "nexafs")
    for points in (None, MERGE):
        with pytest.raises(NearEdgeDataError, match="overlap") as excinfo:
            merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=points)
        message = str(excinfo.value)
        assert "0.275" in message  # the file's parsed range ...
        assert "0.39" in message
        assert f"{SEG_EV[0]:g}" in message  # ... and the required range
        assert f"{SEG_EV[-1]:g}" in message
        assert "keV" in message  # hint at the likely cause


@pytest.mark.unit
def test_negative_normalised_f2_kept_and_warned(
    kk_log: pytest.LogCaptureFixture,
) -> None:
    """D6: a negative value after affine matching is kept, logged, never clipped."""
    a0, a1 = (float(x) for x in evaluate_f2(np.array(MERGE), SEG_EV, COEFFS))
    energy = np.linspace(270.0, 330.0, 25)  # 280, 300 and 320 are grid points
    i0, ic, i1 = (int(np.flatnonzero(energy == e)[0]) for e in (280.0, 300.0, 320.0))
    values = np.linspace(0.1, 0.6, energy.size)
    lo, hi = values[i0], values[i1]
    t = -1.5 * a0 / (a1 - a0)  # scaled value = a0 + t (a1 - a0) = -0.5 a0 < 0
    values[ic] = lo + t * (hi - lo)
    spec = NearEdgeSpectrum(energy, values)

    expected = (values - lo) * ((a1 - a0) / (hi - lo)) + a0
    inside = (energy > MERGE[0]) & (energy < MERGE[1])
    assert expected[ic] < 0
    n_neg, minimum = int((expected[inside] < 0).sum()), float(expected[inside].min())
    assert n_neg == 1

    seg, coeffs = merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=MERGE)
    assert np.isfinite(coeffs).all()
    got = evaluate_f2(energy[inside], seg, coeffs)
    np.testing.assert_allclose(got, expected[inside], rtol=1e-9, atol=1e-12)
    assert got.min() < 0  # not clipped

    warnings = [r for r in kk_log.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert f"{n_neg} of" in message
    assert "minimum" in message
    assert f"{minimum:.6g}" in message


@pytest.mark.unit
def test_non_negative_normalisation_logs_no_warning(
    kk_log: pytest.LogCaptureFixture,
) -> None:
    spec = to_asf_scale(_edge_spectrum(), "nexafs")
    merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=MERGE)
    assert not [r for r in kk_log.records if r.levelno >= logging.WARNING]


@pytest.mark.unit
def test_non_finite_after_scaling_raises() -> None:
    energy = np.linspace(270.0, 330.0, 25)
    values = np.linspace(0.0, 1e-3, energy.size)
    values[12] = 1e308  # (v - v0) * scale overflows
    spec = NearEdgeSpectrum(energy, values)
    with pytest.raises(NearEdgeDataError, match="finite"):
        merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=(270.0, 330.0))


# kk_calculate_real with near-edge data --------------------------------------
@pytest.mark.unit
def test_input_type_required_with_nearedge() -> None:
    spec = _edge_spectrum()
    with pytest.raises(TypeError, match="input_type"):
        kk_calculate_real("C8H8", nearedge=spec)
    with pytest.raises(TypeError, match="nearedge"):
        kk_calculate_real("C8H8", input_type="nexafs")
    with pytest.raises(TypeError, match="nearedge"):
        kk_calculate_real("C8H8", merge_points_kev=(0.28, 0.32))
    with pytest.raises(TypeError, match="NearEdgeSpectrum"):
        kk_calculate_real("C8H8", nearedge=np.ones((3, 2)), input_type="asf")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="input_type"):
        kk_calculate_real("C8H8", nearedge=spec, input_type="mystery")  # type: ignore[arg-type]


@pytest.mark.unit
def test_kk_calculate_real_with_nearedge_fills_the_result_fields() -> None:
    spec = _edge_spectrum()
    res = kk_calculate_real(
        "C8H8", nearedge=spec, input_type="nexafs", merge_points_kev=(0.28, 0.32)
    )
    assert res.merge_points_kev == (0.28, 0.32)
    assert np.isfinite(res.f1).all()
    assert np.isfinite(res.f2).all()
    assert res.nearedge_scaled_kev_f2 is not None
    e_kev, f2 = res.nearedge_scaled_kev_f2
    np.testing.assert_allclose(e_kev * 1000.0, spec.energy_ev, rtol=1e-15)
    assert f2.shape == e_kev.shape
    # Scaled data meets the tabulated f2 at the merge points.
    asf = evaluate_f2(np.array(MERGE), SEG_EV, COEFFS)
    np.testing.assert_allclose(np.interp(MERGE, spec.energy_ev, f2), asf, rtol=1e-12)
    # The merged grid contains the near-edge points and f2 is evaluated on it.
    assert 0.3 in res.energy_kev

    default_ends = kk_calculate_real("C8H8", nearedge=spec, input_type="nexafs")
    assert default_ends.merge_points_kev == (0.27, 0.33)


@pytest.mark.unit
def test_kk_calculate_real_nearedge_error_paths() -> None:
    with pytest.raises(NearEdgeDataError, match="increasing"):
        kk_calculate_real(
            "C8H8",
            nearedge=_edge_spectrum(),
            input_type="nexafs",
            merge_points_kev=(0.32, 0.28),
        )


@pytest.mark.unit
def test_docstring_examples_run() -> None:
    results = doctest.testmod(nearedge, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0
    assert results.failed == 0


@pytest.mark.unit
@pytest.mark.parametrize("start_ev", [1001, 1015, 1967])
def test_merge_points_at_data_ends_survive_the_kev_round_trip(start_ev: int) -> None:
    """(k / 1000) * 1000 != k for some integers; the data ends must still merge."""
    lo, hi = float(start_ev), float(start_ev + 60)
    assert (lo / 1000.0) * 1000.0 != lo or (hi / 1000.0) * 1000.0 != hi
    spec = _edge_spectrum(lo, hi, 13)
    res = kk_calculate_real(
        "C8H8",
        nearedge=spec,
        input_type="nexafs",
        merge_points_kev=(lo / 1000.0, hi / 1000.0),
    )
    assert np.isfinite(res.f1).all()
    # The merge points are snapped onto the data ends, not left a ulp inside.
    assert lo / 1000.0 in res.energy_kev
    assert hi / 1000.0 in res.energy_kev


@pytest.mark.unit
def test_genuinely_outside_merge_point_still_raises_with_distinct_digits() -> None:
    spec = to_asf_scale(_edge_spectrum(), "nexafs")  # 270-330 eV
    with pytest.raises(NearEdgeDataError, match="inside the near-edge") as excinfo:
        merge_spectra(spec, SEG_EV, COEFFS, merge_points_ev=(270.0 * (1 - 1e-9), 320.0))
    message = str(excinfo.value)
    assert "269.99999973" in message  # the offending value, to enough digits
    assert "270-330 eV" in message
    top = float(SEG_EV[-1])
    wide = to_asf_scale(_edge_spectrum(top / 2, top * 2, 20), "nexafs")
    with pytest.raises(NearEdgeDataError, match="tabulated"):
        merge_spectra(wide, SEG_EV, COEFFS, merge_points_ev=(top / 2, top * (1 + 1e-9)))
