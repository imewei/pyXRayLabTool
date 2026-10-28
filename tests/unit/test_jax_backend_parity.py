"""Golden parity of the device program and scalar-energy fallback tests."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from xraylabtool.backend.interpolation import JaxPchipInterpolator
from xraylabtool.calculators.core import (
    _convert_energy_input,
    _validate_and_process_energies,
    calculate_single_material_properties,
)

GOLDEN_NPZ = Path(__file__).parents[1] / "characterization" / "core" / "golden.npz"
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


@pytest.mark.unit
class TestGoldenParity:
    """The device program reproduces the frozen NumPy-path goldens (rtol 1e-12)."""

    @pytest.mark.parametrize("formula", ["SiO2", "H2O", "Fe2O3"])
    def test_single_material_matches_the_goldens(self, formula: str) -> None:
        with np.load(GOLDEN_NPZ, allow_pickle=False) as golden:
            prefix = f"single_three|{formula}|"
            energies = golden[prefix + "energy_kev"]
            density = float(golden[prefix + "density_g_cm3"])
            expected = {f: golden[prefix + f] for f in RESULT_FIELDS}

        actual = calculate_single_material_properties(formula, energies, density)

        for field in RESULT_FIELDS:
            np.testing.assert_allclose(
                np.asarray(getattr(actual, field)),
                expected[field],
                rtol=1e-12,
                err_msg=f"{formula}.{field}",
            )

    def test_jax_pchip_matches_scipy(self) -> None:
        from scipy.interpolate import PchipInterpolator

        x = np.linspace(1.0, 10.0, 12)
        y = np.log(x) + 0.1 * x
        query = np.linspace(1.5, 9.5, 25)

        np.testing.assert_allclose(
            np.asarray(JaxPchipInterpolator(x, y)(query)),
            PchipInterpolator(x, y)(query),
            rtol=1e-12,
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
