"""Regressions for the energy-validation fast path and the XRayResult dtype rule."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from xraylabtool.calculators import core
from xraylabtool.calculators.xray_result import XRayResult
from xraylabtool.exceptions import EnergyError


@pytest.mark.parametrize(
    ("energies", "message"),
    [
        ([], "empty"),
        ([0.0, 5.0], "positive"),
        ([-1.0], "positive"),
        ([0.02, 5.0], "out of range"),
        ([5.0, 30.5], "out of range"),
        ([5.0, np.inf], "out of range"),
        ([5.0, np.nan], "finite"),
        ([np.nan], "finite"),
    ],
)
def test_invalid_energies_keep_their_exact_errors(energies, message):
    with pytest.raises(EnergyError, match=message):
        core._validate_energies(np.array(energies, dtype=float))


@pytest.mark.parametrize("edge", [0.03, 30.0])
def test_the_range_limits_are_inclusive(edge):
    got = core._validate_energies(np.array([edge, 10.0]))
    assert got.tolist() == [edge, 10.0]


def test_a_valid_input_is_returned_as_a_new_float64_array():
    original = np.linspace(5.0, 15.0, 7)
    got = core._validate_energies(original)
    assert got.dtype == np.float64
    assert not np.shares_memory(got, original)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_xray_result_normalises_array_fields_to_float64(dtype):
    result = core.calculate_single_material_properties("SiO2", np.array([8.0]), 2.2)
    fields = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    fields["dispersion_delta"] = np.asarray(fields["dispersion_delta"]).astype(dtype)
    rebuilt = XRayResult(**fields)
    assert rebuilt.dispersion_delta.dtype == np.float64
    assert rebuilt.dispersion_delta.tolist() == pytest.approx(
        np.asarray(fields["dispersion_delta"], dtype=np.float64).tolist()
    )
