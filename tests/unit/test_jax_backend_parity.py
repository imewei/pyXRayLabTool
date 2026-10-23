"""JAX backend parity and scalar-energy fallback tests."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from xraylabtool.backend.array_ops import get_backend, set_backend
from xraylabtool.backend.interpolation import InterpolationFactory
from xraylabtool.calculators.core import (
    _convert_energy_input,
    _validate_and_process_energies,
    calculate_single_material_properties,
)

ENERGIES_KEV = np.array([0.5, 1.0, 5.0, 10.0, 20.0, 29.0])
RESULT_FIELDS = (
    "dispersion_delta",
    "absorption_beta",
    "scattering_factor_f1",
    "scattering_factor_f2",
    "critical_angle_degrees",
    "attenuation_length_cm",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
)


@pytest.fixture
def restore_numpy_backend():
    yield
    set_backend("numpy")


@pytest.mark.unit
@pytest.mark.usefixtures("restore_numpy_backend")
class TestJaxBackendParity:
    """The JAX backend (jit kernels + interpax PCHIP) must match NumPy."""

    @pytest.mark.parametrize(
        ("formula", "density"), [("SiO2", 2.2), ("H2O", 1.0), ("Fe2O3", 5.24)]
    )
    def test_jax_matches_numpy(self, formula: str, density: float) -> None:
        pytest.importorskip("jax")
        pytest.importorskip("interpax")

        set_backend("numpy")
        expected = calculate_single_material_properties(formula, ENERGIES_KEV, density)
        set_backend("jax")
        assert type(get_backend()).__name__ == "JaxBackend"
        actual = calculate_single_material_properties(formula, ENERGIES_KEV, density)

        for field in RESULT_FIELDS:
            np.testing.assert_allclose(
                np.asarray(getattr(actual, field)),
                np.asarray(getattr(expected, field)),
                rtol=1e-12,
                err_msg=f"{formula}.{field}",
            )

    def test_pchip_factory_uses_interpax_and_matches_scipy(self) -> None:
        pytest.importorskip("jax")
        interpax = pytest.importorskip("interpax")

        x = np.linspace(1.0, 10.0, 12)
        y = np.log(x) + 0.1 * x
        query = np.linspace(1.5, 9.5, 25)

        set_backend("numpy")
        expected = InterpolationFactory.create_pchip(x, y)(query)
        set_backend("jax")
        interpolator = InterpolationFactory.create_pchip(x, y)

        assert isinstance(interpolator, interpax.PchipInterpolator)
        np.testing.assert_allclose(
            np.asarray(interpolator(query)), expected, rtol=1e-12
        )


@pytest.mark.unit
class TestScalarEnergyFallback:
    """Non-numeric scalars go through the float() fallback before validation."""

    @pytest.mark.parametrize("value", ["10.0", Fraction(10, 1)])
    def test_convert_energy_input_accepts_float_convertible(self, value) -> None:
        result = _convert_energy_input(value)
        np.testing.assert_allclose(np.asarray(result), [10.0])

    @pytest.mark.parametrize("value", ["10.0", Fraction(10, 1)])
    def test_validate_energies_accepts_float_convertible(self, value) -> None:
        np.testing.assert_allclose(_validate_and_process_energies(value), [10.0])

    def test_convert_energy_input_rejects_non_numeric_string(self) -> None:
        with pytest.raises(ValueError, match="Cannot convert energy to float"):
            _convert_energy_input("not-a-number")

    def test_validate_energies_rejects_non_numeric_string(self) -> None:
        with pytest.raises(ValueError, match="Cannot convert energy to float"):
            _validate_and_process_energies("not-a-number")
