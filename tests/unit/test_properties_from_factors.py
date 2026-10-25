"""Tests for ``calculate_properties_from_factors`` (XRayResult from supplied f1/f2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    calculate_properties_from_factors,
    calculate_single_material_properties,
)
from xraylabtool.exceptions import ValidationError

ENERGIES = np.array([0.1, 1.0, 5.0, 8.048, 20.0])


@pytest.mark.unit
def test_helper_reproduces_lookup_path() -> None:
    ref = calculate_single_material_properties("SiO2", ENERGIES, 2.2)
    got = calculate_properties_from_factors(
        "SiO2",
        ENERGIES,
        2.2,
        ref.scattering_factor_f1,
        ref.scattering_factor_f2,
    )
    for field in dataclasses.fields(ref):
        a, b = getattr(ref, field.name), getattr(got, field.name)
        if isinstance(a, str):
            assert a == b
        else:
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=0.0, err_msg=field.name)
            assert np.isfinite(b).all()


@pytest.mark.unit
def test_negative_f1_preserved_critical_angle_zero() -> None:
    """Review Focus 4: negative f1 near an edge is physical, not an error."""
    energy = np.array([1.0, 2.0, 3.0])
    f1 = np.array([-5.0, 10.0, -0.5])
    f2 = np.array([1.0, 2.0, 3.0])
    res = calculate_properties_from_factors("Si", energy, 2.33, f1, f2)
    np.testing.assert_array_equal(res.scattering_factor_f1, f1)
    assert res.dispersion_delta[0] < 0
    assert res.dispersion_delta[1] > 0
    assert res.critical_angle_degrees[0] == 0.0
    assert res.critical_angle_degrees[1] > 0.0
    assert res.critical_angle_degrees[2] == 0.0
    assert np.isfinite(res.attenuation_length_cm).all()


@pytest.mark.unit
def test_negative_f2_kept_and_attenuation_clamped() -> None:
    """D6: negative f2 is kept; attenuation uses the existing 1e-30 clamp."""
    res = calculate_properties_from_factors(
        "Si", np.array([1.0, 2.0]), 2.33, np.array([14.0, 14.0]), np.array([-0.1, 0.1])
    )
    assert res.scattering_factor_f2[0] == -0.1
    assert res.absorption_beta[0] < 0
    assert np.isfinite(res.attenuation_length_cm).all()
    assert res.attenuation_length_cm[0] > res.attenuation_length_cm[1]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("energy", "f1", "f2", "density"),
    [
        ([1.0, 2.0], [1.0], [1.0, 1.0], 2.0),  # length mismatch f1
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0, 1.0], 2.0),  # length mismatch f2
        ([1.0, 2.0], [[1.0], [1.0]], [1.0, 1.0], 2.0),  # broadcastable wrong shape
        ([1.0, 2.0], [1.0, 1.0], [[1.0, 1.0]], 2.0),  # 2-D
        ([1.0, 2.0], 1.0, [1.0, 1.0], 2.0),  # scalar f1
        ([], [], [], 2.0),  # empty
        ([1.0, np.nan], [1.0, 1.0], [1.0, 1.0], 2.0),
        ([1.0, 2.0], [1.0, np.inf], [1.0, 1.0], 2.0),
        ([1.0, 2.0], [1.0, 1.0], [np.nan, 1.0], 2.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], 0.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], -1.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], np.nan),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], np.inf),
    ],
)
def test_helper_rejects_mismatched_nonfinite_or_bad_density(
    energy: object, f1: object, f2: object, density: float
) -> None:
    with pytest.raises(ValidationError):
        calculate_properties_from_factors(
            "Si",
            energy,
            density,
            f1,
            f2,  # type: ignore[arg-type]
        )
