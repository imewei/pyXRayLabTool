"""Tests for the piecewise-polynomial Kramers-Kronig transform kernel."""

from __future__ import annotations

import doctest
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from tests.fixtures.kk_rounding import assert_kk_close
from xraylabtool.kk import transform
from xraylabtool.kk.transform import (
    kk_piecewise_poly,
    kk_real_spectrum,
    relativistic_correction,
)

GOLDEN = Path(__file__).resolve().parents[2] / "characterization" / "kk" / "golden"

# Single segment used for the per-term checks: [a₁, a₀, a₋₁, a₋₂, a₋₃] on [2, 5].
BASIS_COEFFS = (0.3, -0.7, 1.1, 2.3, -1.9)
BASIS_SEG = np.array([2.0, 5.0])
BASIS_EVAL = (1.0, 3.3, 7.0)


def _golden(name: str) -> dict[str, np.ndarray]:
    with np.load(GOLDEN / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def _args(g: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    return g["seg_ev"], g["coeffs"], float(g["rel_corr"])


# --- independent references -------------------------------------------------


def _f2_poly(c: np.ndarray, x: float) -> float:
    return c[0] * x + c[1] + c[2] / x + c[3] / x**2 + c[4] / x**3


def _quadrature_f1(energy: float, seg: np.ndarray, coeffs: np.ndarray) -> float:
    """f1 − Z* = −(2/π) PV∫ x f2(x) / (x² − E²) dx by adaptive quadrature.

    Uses x/(x² − E²) = ½[1/(x − E) + 1/(x + E)] with QAWC (Cauchy weight) on
    the segment containing E, plain quadrature elsewhere.
    """
    total = 0.0
    opts = {"epsabs": 0.0, "epsrel": 1e-13, "limit": 200}
    for a, b, c in zip(seg[:-1], seg[1:], coeffs, strict=True):
        if a < energy < b:
            pv = quad(
                lambda x, c=c: _f2_poly(c, x),
                a,
                b,
                weight="cauchy",
                wvar=energy,
                **opts,
            )[0]
            regular = quad(lambda x, c=c: _f2_poly(c, x) / (x + energy), a, b, **opts)[
                0
            ]
            total += 0.5 * (pv + regular)
        else:
            total += quad(
                lambda x, c=c: x * _f2_poly(c, x) / (x**2 - energy**2), a, b, **opts
            )[0]
    return -2.0 / np.pi * total


def _eq10_corrected(
    energy: np.ndarray,
    seg: np.ndarray,
    coeffs: np.ndarray,
    rel_corr: float,
    a3_sign: float = 1.0,
) -> np.ndarray:
    """Plain transcription of Watts (2014) Eq. (10), a₋₃ term corrected.

    ``a3_sign=1`` gives the correct ``+(2a₋₃/E)(x_j⁻¹ − x_{j+1}⁻¹)``;
    ``a3_sign=-1`` gives the term as typeset in the published Eq. (10),
    ``+(2a₋₃/E)(x_{j+1}⁻¹ − x_j⁻¹)``, whose sign contradicts Table 1.
    """
    e = energy[:, None]
    x1, x2 = seg[:-1], seg[1:]
    a1, a0, am1, am2, am3 = coeffs.T
    plus_e = (e**2 * a1 - e * a0 + am1 - am2 / e + am3 / e**2) * np.log(
        np.abs((x2 + e) / (x1 + e))
    )
    minus_e = (e**2 * a1 + e * a0 + am1 + am2 / e + am3 / e**2) * np.log(
        np.abs((x2 - e) / (x1 - e))
    )
    rest = (
        -2 * e * a1 * (x2 - x1)
        + 2 * am2 / e * np.log(np.abs(x2 / x1))
        + a3_sign * 2 * am3 / e * (1 / x1 - 1 / x2)
    )
    return np.sum(plus_e - minus_e + rest, axis=1) / (np.pi * energy) + rel_corr


def _linear_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope = np.diff(y) / np.diff(x)
    coeffs = np.zeros((x.size - 1, 5))
    coeffs[:, 0] = slope
    coeffs[:, 1] = y[:-1] - slope * x[:-1]
    return coeffs


# --- relativistic correction ------------------------------------------------


@pytest.mark.unit
def test_relativistic_correction_matches_golden() -> None:
    assert relativistic_correction([14], [1.0]) == float(_golden("si")["rel_corr"])
    sio2 = relativistic_correction([14, 8], [1.0, 2.0])
    np.testing.assert_allclose(sio2, float(_golden("sio2")["rel_corr"]), rtol=1e-15)


@pytest.mark.unit
def test_relativistic_correction_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        relativistic_correction([14, 8], [1.0])


# --- analytic and quadrature references ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("energy", [0.5, 1.0, 7.0, 20.0])
def test_constant_band_closed_form(energy: float) -> None:
    a, b, c = 2.0, 5.0, 1.7
    out = kk_real_spectrum(
        np.array([energy]), np.array([a, b]), np.array([[0.0, c, 0.0, 0.0, 0.0]]), 0.0
    )
    expected = c / np.pi * np.log(abs((energy**2 - a**2) / (energy**2 - b**2)))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [expected], rtol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("column", range(5))
def test_each_basis_term_matches_quadrature(column: int) -> None:
    coeffs = np.zeros((1, 5))
    coeffs[0, column] = BASIS_COEFFS[column]
    energies = np.array(BASIS_EVAL)
    out = kk_real_spectrum(energies, BASIS_SEG, coeffs, 0.0)
    expected = [_quadrature_f1(e, BASIS_SEG, coeffs) for e in energies]
    assert np.isfinite(out).all()
    # Measured agreement: ≤ 8.4e-16 relative.
    np.testing.assert_allclose(out, expected, rtol=1e-12)


@pytest.mark.unit
def test_matches_independent_principal_value_quadrature() -> None:
    rng = np.random.default_rng(20261024)
    seg = np.array([2.0, 3.5, 5.0, 8.0, 13.0])
    coeffs = rng.uniform(-2.0, 2.0, size=(4, 5))
    assert np.all(coeffs != 0.0)
    energies = np.array([1.0, 2.7, 4.2, 6.1, 10.0, 20.0, 50.0])
    out = kk_real_spectrum(energies, seg, coeffs, 0.0)
    expected = [_quadrature_f1(e, seg, coeffs) for e in energies]
    assert np.isfinite(out).all()
    # Measured agreement: ≤ 7.8e-15 relative.
    np.testing.assert_allclose(out, expected, rtol=1e-8)


# Absolute tolerances: about 3x the measured |Eq. 10 − kernel| at generic
# energies (Si 1.8e-8, SiO2 4.9e-8, C8H8 9.6e-9, Au 3.6e-5). Both sides are
# float64 sums of large, mostly cancelling log terms; Au has the largest terms.
EQ10_ATOL = {"si": 6e-8, "sio2": 1.5e-7, "c8h8_nearedge": 3e-8, "au": 1.1e-4}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(EQ10_ATOL))
def test_matches_paper_eq10_as_corrected(name: str) -> None:
    g = _golden(name)
    energies = g["eval_ev"][g["eval_kind"] == 0]
    out = kk_real_spectrum(energies, *_args(g))
    expected = _eq10_corrected(energies, *_args(g))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, expected, rtol=0, atol=EQ10_ATOL[name])


@pytest.mark.unit
def test_printed_eq10_a3_sign_is_wrong() -> None:
    coeffs = np.array([[0.0, 0.0, 0.0, 0.0, BASIS_COEFFS[4]]])
    energies = np.array(BASIS_EVAL)
    reference = np.array([_quadrature_f1(e, BASIS_SEG, coeffs) for e in energies])
    corrected = _eq10_corrected(energies, BASIS_SEG, coeffs, 0.0)
    opposite = _eq10_corrected(energies, BASIS_SEG, coeffs, 0.0, a3_sign=-1.0)
    np.testing.assert_allclose(corrected, reference, rtol=1e-12)
    assert np.all(np.abs(opposite - reference) > 1e-3)
    np.testing.assert_allclose(
        kk_real_spectrum(energies, BASIS_SEG, coeffs, 0.0), reference, rtol=1e-12
    )


@pytest.mark.unit
def test_single_lorentz_oscillator_closed_form() -> None:
    amp, gamma, w0 = 1.0, 0.3, 1.0
    lo, hi = 1e-3 * w0, 1e3 * w0

    def denom(w: np.ndarray) -> np.ndarray:
        return (w0**2 - w**2) ** 2 + gamma**2 * w**2

    energies = np.array([0.5, 0.9, 0.99, 1.0003, 1.2, 3.0])
    exact = amp * (energies**2 - w0**2) / denom(energies)
    # X-ray sign convention: f1 − Z* < 0 below resonance, > 0 above.
    np.testing.assert_allclose(exact[[0, 4]], [-1.28205128, 1.36138614], rtol=1e-8)

    # Dropping f2 outside [lo, hi] changes f1 by at most (|x² − E²| ≥ ¾ of
    # the larger square, since lo ≤ E/2 and hi ≥ 2E):
    e_min = energies.min()
    tail = (2 / np.pi) * (4 / 3) * amp * gamma
    tail *= lo**3 / (3 * e_min**2 * (w0**2 - lo**2) ** 2) + 1 / (
        3 * hi**3 * (1 - (w0 / hi) ** 2) ** 2
    )
    assert tail < 1e-7

    errors = []
    for n_points in (2000, 4000):
        x = np.geomspace(lo, hi, n_points)
        coeffs = _linear_segments(x, amp * gamma * x / denom(x))
        out = kk_real_spectrum(energies, x, coeffs, 0.0)
        assert np.isfinite(out).all()
        errors.append(np.abs(out - exact).max())
    # Linear interpolation error is O(h²): measured max error 6.0e-4 (2000
    # points) and 1.55e-4 (4000 points), a 3.9x drop per halving of h.
    assert errors[0] < 1e-3 + tail
    assert errors[1] < 2.5e-4 + tail
    assert errors[1] < errors[0] / 3


# --- singular points (D5) ------------------------------------------------------

# Absolute bounds on |f1(nextafter(x_j, ±inf)) − f1(x_j)| at knots where f2 is
# continuous, about 3x measured (Si 3.0e-8, SiO2 3.0e-8, C8H8 3.7e-9, Au
# 6.1e-5): one ulp from the knot the two ln|x_j − E| ≈ −30 terms only cancel
# to the rounding of the adjacent polynomials.
KNOT_ATOL = {"si": 1e-7, "sio2": 1e-7, "c8h8_nearedge": 1.2e-8, "au": 2e-4}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(KNOT_ATOL))
def test_interior_knot_with_continuous_f2_is_finite_and_continuous(name: str) -> None:
    g = _golden(name)
    knots = g["eval_ev"][g["eval_kind"] == 1]
    assert knots.size > 0
    at = kk_real_spectrum(knots, *_args(g))
    above = kk_real_spectrum(np.nextafter(knots, np.inf), *_args(g))
    below = kk_real_spectrum(np.nextafter(knots, -np.inf), *_args(g))
    for out in (at, above, below):
        assert np.isfinite(out).all()
    np.testing.assert_allclose(above, at, rtol=0, atol=KNOT_ATOL[name])
    np.testing.assert_allclose(below, at, rtol=0, atol=KNOT_ATOL[name])


@pytest.mark.unit
def test_gradient_at_continuous_knot_is_finite() -> None:
    jnp = jax.numpy
    g = _golden("si")
    seg, coeffs, rel = (jnp.asarray(a) for a in _args(g))
    knot = float(g["eval_ev"][g["eval_kind"] == 1][0])
    points = jnp.asarray(
        [knot, np.nextafter(knot, np.inf), np.nextafter(knot, -np.inf)]
    )

    def f1_at(e: object) -> object:
        return kk_piecewise_poly(jnp.reshape(e, (1,)), seg, coeffs, rel)[0]

    grads = np.asarray(jax.vmap(jax.grad(f1_at))(points))
    assert np.isfinite(grads).all()


# Bound = rtol 1e-10 + ATOL. The NumPy-era bound was rtol=1e-10, atol=0 for all
# three. On JAX-CPU Ne at its 10 eV end point drifts by 7.6e-9 from the NumPy
# golden (XLA reassociates the cancelling log terms; the NumPy default backend
# used to mask it), so Ne gets atol=5e-8. Only the 500 keV end point
# discriminates D5: keeping the dropped boundary log would add a finite part of
# about 5.0e-5 to f1 there, against a bound of about 5e-8 + 1e-10*|f1| (about
# 1000x margin). At the 10 eV end point that term is only -2.8e-16 and at the
# 21.5 eV knot 2.7e-15, so the widened atol cannot hide a violation there either.
JUMP_ATOL = {"ar": 0.0, "ne": 5e-8, "synthetic": 0.0}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(JUMP_ATOL))
def test_jump_and_endpoint_follow_upstream_convention(name: str) -> None:
    g = _golden(name)
    mask = (g["eval_kind"] == 2) | (g["eval_kind"] == 3)
    assert np.any(g["eval_kind"] == 2)
    out = kk_real_spectrum(g["eval_ev"][mask], *_args(g))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(
        out, g["f1_kk_pp"][mask], rtol=1e-10, atol=JUMP_ATOL[name]
    )


# --- evaluation order, chunking ------------------------------------------


@pytest.mark.unit
def test_unsorted_duplicate_energies() -> None:
    g = _golden("synthetic")
    energies = g["eval_ev"]
    rng = np.random.default_rng(7)
    shuffled = rng.permutation(np.concatenate([energies, energies[::3]]))
    sorted_out = kk_real_spectrum(energies, *_args(g))
    by_energy = dict(zip(energies.tolist(), sorted_out.tolist(), strict=True))
    out = kk_real_spectrum(shuffled, *_args(g))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(
        out, [by_energy[e] for e in shuffled.tolist()], rtol=1e-13
    )


@pytest.mark.unit
def test_chunk_invariance() -> None:
    g = _golden("si")
    energies = g["eval_ev"]
    full = kk_real_spectrum(energies, *_args(g))
    for chunk in (1, 7, energies.size + 5):
        out = kk_real_spectrum(energies, *_args(g), chunk_size=chunk)
        assert out.shape == energies.shape
        assert_kk_close(out, full, energies, g["seg_ev"], g["coeffs"], rtol=1e-12)


@pytest.mark.unit
def test_rejects_bad_chunk_size() -> None:
    g = _golden("synthetic")
    with pytest.raises(ValueError, match="chunk_size"):
        kk_real_spectrum(g["eval_ev"], *_args(g), chunk_size=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("eval_ev", "seg_ev", "coeffs", "match"),
    [
        (np.array([1.0]), np.array([2.0, 5.0]), np.zeros((2, 5)), "coeffs"),
        (
            np.array([1.0]),
            np.array([5.0, 2.0]),
            np.zeros((1, 5)),
            "strictly increasing",
        ),
        (np.array([0.0]), np.array([2.0, 5.0]), np.zeros((1, 5)), "positive"),
        (np.array([np.nan]), np.array([2.0, 5.0]), np.zeros((1, 5)), "finite"),
        (np.array([1.0]), np.array([2.0, 5.0]), np.full((1, 5), np.inf), "finite"),
    ],
)
def test_rejects_malformed_inputs(
    eval_ev: np.ndarray, seg_ev: np.ndarray, coeffs: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        kk_real_spectrum(eval_ev, seg_ev, coeffs, 0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("eval_ev", "seg_ev", "coeffs", "rel", "message"),
    [
        (
            [0.0],
            [2.0, 5.0],
            [[0, 1, 0, 0, 0]],
            0.0,
            "eval_ev and seg_ev must be positive",
        ),
        (
            [1.0],
            [0.0, 5.0],
            [[0, 1, 0, 0, 0]],
            0.0,
            "eval_ev and seg_ev must be positive",
        ),
        ([1.0], [2.0, 5.0], [[0, 1, 0, 0, 0]], np.nan, "rel_corr must be finite"),
        ([np.inf], [2.0, 5.0], [[0, 1, 0, 0, 0]], 0.0, "eval_ev must be finite"),
        (
            [[1.0]],
            [2.0, 5.0],
            [[0, 1, 0, 0, 0]],
            0.0,
            r"eval_ev must be 1-D, got shape \(1, 1\)",
        ),
    ],
)
def test_legacy_error_messages(eval_ev, seg_ev, coeffs, rel, message) -> None:
    with pytest.raises(ValueError, match=message):
        kk_real_spectrum(eval_ev, seg_ev, coeffs, rel)


@pytest.mark.unit
def test_legacy_validation_order_with_several_invalid_inputs() -> None:
    ok_seg, ok_c = [2.0, 5.0], [[0, 1, 0, 0, 0]]
    # eval_ev shape precedes everything below it
    with pytest.raises(ValueError, match="eval_ev must be 1-D"):
        kk_real_spectrum([[1.0]], [5.0], ok_c, 0.0)
    # seg shape precedes coeffs shape, which precedes finiteness
    with pytest.raises(ValueError, match="seg_ev must be 1-D"):
        kk_real_spectrum([np.nan], [5.0], ok_c, 0.0)
    with pytest.raises(ValueError, match="coeffs must have shape"):
        kk_real_spectrum([np.nan], ok_seg, [[0, 0, 0, 0, 0]] * 2, 0.0)
    # finiteness: eval_ev, then seg_ev, then coeffs, then rel_corr
    with pytest.raises(ValueError, match="eval_ev must be finite"):
        kk_real_spectrum([np.nan], [np.nan, 5.0], [[np.nan, 1, 0, 0, 0]], np.nan)
    with pytest.raises(ValueError, match="seg_ev must be finite"):
        kk_real_spectrum([1.0], [np.nan, 5.0], [[np.nan, 1, 0, 0, 0]], np.nan)
    with pytest.raises(ValueError, match="coeffs must be finite"):
        kk_real_spectrum([1.0], ok_seg, [[np.nan, 1, 0, 0, 0]], np.nan)
    # finiteness (all arrays) precedes positivity precedes ordering precedes chunk_size
    with pytest.raises(ValueError, match="rel_corr must be finite"):
        kk_real_spectrum([0.0], [5.0, 2.0], ok_c, np.inf, chunk_size=0)
    with pytest.raises(ValueError, match="must be positive"):
        kk_real_spectrum([0.0], [5.0, 2.0], ok_c, 0.0, chunk_size=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        kk_real_spectrum([1.0], [5.0, 2.0], ok_c, 0.0, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        kk_real_spectrum([1.0], ok_seg, ok_c, 0.0, chunk_size=0)


@pytest.mark.unit
def test_empty_eval_returns_empty() -> None:
    g = _golden("synthetic")
    out = kk_real_spectrum(np.array([]), *_args(g))
    assert out.shape == (0,)
    assert out.dtype == np.float64


@pytest.mark.unit
def test_kernel_jits_without_tracer_error() -> None:
    g = _golden("synthetic")
    jitted = jax.jit(kk_piecewise_poly)
    out = np.asarray(jitted(g["eval_ev"], *_args(g)))
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, g["f1_kk_pp"], rtol=1e-8)


@pytest.mark.unit
def test_float32_inputs_give_float64_output(x64_restored) -> None:
    g = _golden("synthetic")
    e32 = g["eval_ev"].astype(np.float32)
    s32 = g["seg_ev"].astype(np.float32)
    c32 = g["coeffs"].astype(np.float32)
    jax.config.update("jax_enable_x64", False)
    out = kk_real_spectrum(e32, s32, c32, np.float32(g["rel_corr"]))
    assert jax.config.jax_enable_x64
    assert out.dtype == np.float64
    expected = kk_real_spectrum(
        e32.astype(np.float64),
        s32.astype(np.float64),
        c32.astype(np.float64),
        float(np.float32(g["rel_corr"])),
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)


@pytest.mark.unit
def test_docstring_examples_run() -> None:
    results = doctest.testmod(transform, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.attempted > 0
    assert results.failed == 0
