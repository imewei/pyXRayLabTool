"""Invalid physical inputs must fail through the public validation errors."""

import warnings

import numpy as np
import pytest

from xraylabtool.data_handling.atomic_cache import FastAtomicDataProvider
from xraylabtool.exceptions import EnergyError, ValidationError
from xraylabtool.validation.validators import validate_density, validate_energy_range


@pytest.mark.parametrize("density", [1 + 2j, np.complex64(2 + 1j), np.complex128(2)])
def test_complex_density_is_rejected_without_lossy_conversion(density):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValidationError, match="real"):
            validate_density(density)


@pytest.mark.parametrize(
    "energies", [2 + 1j, np.array([2 + 0j]), "invalid", [], [None]]
)
def test_invalid_energy_input_raises_domain_error(energies):
    with pytest.raises(EnergyError):
        validate_energy_range(energies)


def test_real_energy_boundaries_are_preserved():
    np.testing.assert_array_equal(validate_energy_range([0.03, 30]), [0.03, 30])


def test_atomic_provider_rejects_complex_energies_before_real_cast():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(EnergyError, match="real"):
            FastAtomicDataProvider().get_scattering_factors("Si", np.array([10 + 1j]))
