"""Legacy contracts of the device path found by the external review (F1 to F4).

Every expectation was read off the pre-switchover NumPy implementation (commit
``9b18d70``): broadcast errors and shapes, return types, the huge-formula values
and the ``ZeroDivisionError`` of an all-zero formula.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
import math

import numpy as np
import pytest

from xraylabtool.calculators import core
from xraylabtool.calculators.cache import create_scattering_factor_interpolators
from xraylabtool.calculators.core import (
    calculate_properties_from_factors,
    calculate_single_material_properties,
    calculate_xray_properties,
)
from xraylabtool.calculators.derived_quantities import (
    calculate_derived_quantities,
    calculate_scattering_factors,
)
from xraylabtool.constants import ENERGY_TO_WAVELENGTH_FACTOR, SCATTERING_FACTOR

getcontext().prec = 60
NON_FINITE = "Non-finite values"


def _si():
    f1, f2 = create_scattering_factor_interpolators("Si")
    return [(1.0, f1, f2)]


def _shapes(values):
    return [np.shape(v) for v in values]


# ---- F1: energy and wavelength broadcast together before padding -----------------


@pytest.mark.parametrize(
    ("n_energy", "n_wave", "message"),
    [(3, 2, r"shapes \(2,\) \(3,\)"), (2, 3, r"shapes \(3,\) \(2,\)")],
)
def test_incompatible_energy_and_wavelength_lengths_raise_numpys_error(
    n_energy, n_wave, message
):
    energy = np.linspace(8000.0, 10000.0, n_energy)
    wavelength = np.linspace(1e-10, 1.5e-10, n_wave)
    with pytest.raises(ValueError, match=f"operands could not be broadcast.*{message}"):
        calculate_scattering_factors(energy, wavelength, 2.33, 28.085, _si())


def test_length_one_and_scalar_wavelengths_still_broadcast():
    energy = np.linspace(8000.0, 10000.0, 3)
    full = calculate_scattering_factors(
        energy, np.full(3, 1.2e-10), 2.33, 28.085, _si()
    )
    for wavelength in (np.array([1.2e-10]), np.float64(1.2e-10), 1.2e-10):
        got = calculate_scattering_factors(energy, wavelength, 2.33, 28.085, _si())
        assert _shapes(got) == [(3,)] * 4
        for a, b in zip(got, full, strict=True):
            np.testing.assert_array_equal(a, b)


def test_one_energy_with_many_wavelengths_keeps_the_legacy_shapes():
    got = calculate_scattering_factors(
        np.array([8000.0]), np.linspace(1e-10, 1.5e-10, 3), 2.33, 28.085, _si()
    )
    assert _shapes(got) == [(3,), (3,), (1,), (1,)]


# ---- F2: no subnormal intermediate for a huge molecular weight --------------------


def _exact_reference(energy_kev, f1, f2, rho, mw):
    """δ and β from the formulas in 60-digit decimal arithmetic."""
    out = []
    for e, a, b in zip(energy_kev, f1, f2, strict=True):
        w2 = Decimal(ENERGY_TO_WAVELENGTH_FACTOR) ** 2 / Decimal(e) ** 2
        scale = w2 * Decimal(SCATTERING_FACTOR) * Decimal(rho) / Decimal(mw)
        out.append((float(scale * Decimal(a)), float(scale * Decimal(b))))
    return np.array(out).T


@pytest.mark.parametrize("zeros", [150, 305, 307])
def test_huge_atom_counts_keep_delta_and_beta(zeros):
    formula = "H1" + "0" * zeros
    energy = np.array([8.0, 11.919])
    res = calculate_single_material_properties(formula, energy, 1.0)
    delta, beta = _exact_reference(
        energy,
        res.scattering_factor_f1,
        res.scattering_factor_f2,
        1.0,
        res.molecular_weight_g_mol,
    )
    np.testing.assert_allclose(res.dispersion_delta, delta, rtol=1e-12)
    np.testing.assert_allclose(res.absorption_beta, beta, rtol=1e-12)
    assert np.all(res.dispersion_delta > 0)
    assert np.all(res.absorption_beta > 0)

    # The other fields follow the legacy formulas from the same δ and β.
    wl_m = ENERGY_TO_WAVELENGTH_FACTOR / energy
    ed = 1e6 * 1.0 / res.molecular_weight_g_mol * 6.02214076e23
    assert res.electron_density_per_ang3 == pytest.approx(
        ed * res.total_electrons / 1e30, rel=1e-6
    )
    crit = np.sqrt(np.maximum(2 * delta, 0.0)) * (180.0 / math.pi)
    np.testing.assert_allclose(res.critical_angle_degrees, crit, rtol=1e-12)
    att = wl_m / np.maximum(beta, 1e-30) / (4 * math.pi) * 1e2
    np.testing.assert_allclose(res.attenuation_length_cm, att, rtol=1e-12)
    sld = 2 * math.pi / 1e20
    np.testing.assert_allclose(res.real_sld_per_ang2, delta * sld / wl_m**2, rtol=1e-12)
    np.testing.assert_allclose(
        res.imaginary_sld_per_ang2, beta * sld / wl_m**2, rtol=1e-12
    )


def test_huge_molecular_weight_in_the_supplied_factor_tail():
    mw_formula = "H1" + "0" * 305
    energy = np.array([8.0, 11.919])
    f1 = np.array([1e305, 2e305])
    f2 = np.array([1e299, 3e299])
    res = calculate_properties_from_factors(mw_formula, energy, 1.0, f1, f2)
    delta, beta = _exact_reference(energy, f1, f2, 1.0, res.molecular_weight_g_mol)
    np.testing.assert_allclose(res.dispersion_delta, delta, rtol=1e-12)
    np.testing.assert_allclose(res.absorption_beta, beta, rtol=1e-12)


def test_huge_molecular_weight_in_the_standalone_scattering_factors():
    energy = np.array([8000.0, 11919.0])
    wavelength = ENERGY_TO_WAVELENGTH_FACTOR / (energy / 1000.0)
    mw = 1e305
    dispersion, absorption, f1, f2 = calculate_scattering_factors(
        energy,
        wavelength,
        1.0,
        mw,
        [(1e305, *create_scattering_factor_interpolators("H"))],
    )
    delta, beta = _exact_reference(energy / 1000.0, f1, f2, 1.0, mw)
    np.testing.assert_allclose(dispersion, delta, rtol=1e-12)
    np.testing.assert_allclose(absorption, beta, rtol=1e-12)


def test_tiny_molecular_weight_keeps_the_legacy_non_finite_failure():
    # S*rho/M overflowed to inf on the legacy path, so the call raised.
    with pytest.raises(ValueError, match=NON_FINITE):
        calculate_single_material_properties("H0." + "0" * 299 + "1", 8.0, 1.0)


def test_small_molecular_weights_compute_the_same_delta_as_ordinary_ones():
    ordinary = calculate_single_material_properties("H", 8.0, 1.0)
    for zeros in (30, 100):
        res = calculate_single_material_properties("H0." + "0" * zeros + "1", 8.0, 1.0)
        np.testing.assert_allclose(
            res.dispersion_delta, ordinary.dispersion_delta, rtol=1e-12
        )


# ---- F3: each derived field keeps the shape and type of its own expression --------

WL = 1.5e-10


def _derived(wl, delta, beta):
    return calculate_derived_quantities(wl, delta, beta, 2.2, 60.0, 30.0)


def test_all_scalar_derived_quantities_are_float_and_numpy_scalars():
    out = _derived(WL, 1e-6, 1e-8)
    assert type(out[0]) is float
    for value in out[1:]:
        assert type(value) is np.float64


def test_scalar_and_array_derived_quantities_agree():
    scalar = _derived(WL, 1e-6, 1e-8)
    array = _derived(np.array([WL]), np.array([1e-6]), np.array([1e-8]))
    assert scalar[0] == array[0]
    for s, a in zip(scalar[1:], array[1:], strict=True):
        assert np.shape(a) == (1,)
        assert s == a[0]


@pytest.mark.parametrize(
    ("wl", "delta", "beta", "shapes"),
    [
        (
            np.array([1e-10, 2e-10]),
            np.array(1e-6),
            np.array(1e-8),
            [(), (2,), (2,), (2,)],
        ),
        (WL, np.array([1e-6, 2e-6]), np.array([1e-8, 2e-8]), [(2,)] * 4),
        (np.array([1e-10, 2e-10]), np.array([1e-6, 2e-6]), 1e-8, [(2,)] * 4),
        (
            np.array([WL]),
            np.linspace(1e-6, 2e-6, 3),
            np.linspace(1e-8, 2e-8, 3),
            [(3,)] * 4,
        ),
        (
            np.full((2, 2), WL),
            np.full((2, 2), 1e-6),
            np.full((2, 2), 1e-8),
            [(2, 2)] * 4,
        ),
        (np.full(2, WL), np.full((2, 2), 1e-6), np.full((2, 2), 1e-8), [(2, 2)] * 4),
    ],
    ids=["wl_vector", "wl_scalar", "beta_scalar", "wl_one", "grid", "wl_row"],
)
def test_derived_field_shapes_follow_their_own_operands(wl, delta, beta, shapes):
    out = _derived(wl, delta, beta)
    assert _shapes(out[1:]) == shapes


def test_incompatible_derived_operands_raise_numpys_error():
    with pytest.raises(
        ValueError, match=r"operands could not be broadcast.*\(3,\) \(2,\)"
    ):
        _derived(
            np.linspace(1e-10, 2e-10, 3), np.linspace(1e-6, 2e-6, 3), np.ones(2) * 1e-8
        )


# ---- F4: all-zero formulas raise the legacy ZeroDivisionError ---------------------


@pytest.mark.parametrize("formula", ["H0", "H0O0"])
@pytest.mark.parametrize("density", [1.0, 1])
def test_all_zero_formula_raises_zero_division(formula, density):
    with pytest.raises(ZeroDivisionError, match=r"^division by zero$"):
        calculate_single_material_properties(formula, np.array([8.0, 9.0]), density)


def test_all_zero_formula_with_a_numpy_density_keeps_the_non_finite_failure():
    with pytest.raises(ValueError, match=NON_FINITE):
        calculate_single_material_properties("H0", 8.0, np.float64(1.0))


def test_partially_zero_formula_still_computes():
    res = calculate_single_material_properties("H0O", 8.0, 1.0)
    assert res.molecular_weight_g_mol == pytest.approx(15.999)
    assert np.isfinite(res.dispersion_delta).all()


def test_all_zero_formula_is_skipped_by_the_collection_calls():
    outcomes = core._calculate_materials(["H0", "SiO2", "H0O0"], [1.0, 2.2, 1.0], [8.0])
    assert isinstance(outcomes[0], ZeroDivisionError)
    assert isinstance(outcomes[2], ZeroDivisionError)
    assert isinstance(outcomes[1], dict)
    assert list(
        calculate_xray_properties(["H0", "SiO2"], np.array([8.0]), [1.0, 2.2])
    ) == ["SiO2"]


def test_standalone_scattering_factors_with_zero_molecular_weight():
    with pytest.raises(ZeroDivisionError, match=r"^division by zero$"):
        calculate_scattering_factors(
            np.array([8000.0]), np.array([1.5e-10]), 1.0, 0.0, _si()
        )
