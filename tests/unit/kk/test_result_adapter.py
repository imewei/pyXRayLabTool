"""Tests for ``KKResult.to_xray_result``."""

from __future__ import annotations

import doctest

import numpy as np
import pytest

from xraylabtool.calculators.core import calculate_properties_from_factors
from xraylabtool.exceptions import EnergyError
from xraylabtool.kk import kk_calculate_real, result
from xraylabtool.kk.result import KKResult


def _manual(energy_kev: list[float]) -> KKResult:
    e = np.asarray(energy_kev, dtype=np.float64)
    return KKResult(
        "Si", e, np.full_like(e, 14.0), np.full_like(e, 0.5), 0.0, None, 0, "numpy"
    )


@pytest.mark.unit
@pytest.mark.parametrize("energy", [[0.029, 1.0], [1.0, 30.001], [0.01, 50.0]])
def test_to_xray_result_rejects_out_of_range_energy(energy: list[float]) -> None:
    with pytest.raises(EnergyError, match=r"0\.03-30 keV"):
        _manual(energy).to_xray_result(2.33)


@pytest.mark.unit
def test_to_xray_result_accepts_range_ends() -> None:
    res = _manual([0.03, 30.0]).to_xray_result(2.33)
    assert res.energy_kev.shape == (2,)


@pytest.mark.unit
def test_to_xray_result_uses_kk_f1() -> None:
    energies = [2.0, 0.5, 8.0]  # deliberately unsorted: order is preserved
    kk = kk_calculate_real("SiO2", energies_kev=energies)
    res = kk.to_xray_result(2.2)
    np.testing.assert_array_equal(res.scattering_factor_f1, kk.f1)
    np.testing.assert_array_equal(res.scattering_factor_f2, kk.f2)
    np.testing.assert_array_equal(res.energy_kev, kk.energy_kev)
    ref = calculate_properties_from_factors("SiO2", kk.energy_kev, 2.2, kk.f1, kk.f2)
    np.testing.assert_allclose(
        res.dispersion_delta, ref.dispersion_delta, rtol=1e-12, atol=0.0
    )
    assert res.formula == "SiO2"
    assert res.density_g_cm3 == 2.2
    assert np.isfinite(res.critical_angle_degrees).all()


@pytest.mark.unit
def test_negative_f1_near_edge_reaches_xray_result() -> None:
    e = np.array([1.0, 2.0])
    kk = KKResult(
        "Si", e, np.array([-3.0, 14.0]), np.array([1.0, 1.0]), 0.0, None, 0, "numpy"
    )
    res = kk.to_xray_result(2.33)
    assert res.scattering_factor_f1[0] == -3.0
    assert res.critical_angle_degrees[0] == 0.0
    assert res.critical_angle_degrees[1] > 0.0


@pytest.mark.unit
def test_result_module_doctests() -> None:
    failed, tried = doctest.testmod(result, optionflags=doctest.ELLIPSIS)
    assert tried > 0
    assert failed == 0
