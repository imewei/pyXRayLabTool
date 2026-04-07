"""Characterization tests: PchipInterpolator golden values (P1-C).

This file is the highest-risk characterization test for the JAX migration.
SciPy's PchipInterpolator will be replaced by interpax (or a custom cubic
spline).  The values here are the numerical contract that the replacement must
match.

All values were captured from v0.3.0 by calling the live interpolators and
recording exact float64 output.  Tolerances are set to atol=1e-10 — tighter
than float32 machine epsilon — to catch any dtype or algorithm regression.
"""

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
)

# Sample energies used for all elements (eV)
SAMPLE_ENERGIES_EV = np.array([100.0, 1000.0, 5000.0, 10000.0, 20000.0])


@pytest.fixture(autouse=True)
def cold_cache():
    """Cold-path coverage: no cached state bleeds between tests."""
    clear_scattering_factor_cache()
    yield
    clear_scattering_factor_cache()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_element(element: str, expected_f1: np.ndarray, expected_f2: np.ndarray):
    """Assert interpolated f1 and f2 match expected arrays to atol=1e-10."""
    f1_interp, f2_interp = create_scattering_factor_interpolators(element)
    f1_vals = np.asarray(f1_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
    f2_vals = np.asarray(f2_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
    np.testing.assert_allclose(
        f1_vals,
        expected_f1,
        atol=1e-10,
        err_msg=f"{element} f1 mismatch — PCHIP interpolation may have changed",
    )
    np.testing.assert_allclose(
        f2_vals,
        expected_f2,
        atol=1e-10,
        err_msg=f"{element} f2 mismatch — PCHIP interpolation may have changed",
    )


# ---------------------------------------------------------------------------
# P1-C  Si — light element, well inside interpolation range
# ---------------------------------------------------------------------------


class TestPchipSi:
    """Golden PCHIP samples for Silicon at [100, 1000, 5000, 10000, 20000] eV."""

    EXPECTED_F1 = np.array(
        [-5.87426308, 12.99738071, 14.41953876, 14.27468999, 14.04805305]
    )
    EXPECTED_F2 = np.array([4.60160614, 1.06956209, 0.794992, 0.21351436, 0.05333107])

    def test_f1_golden(self):
        f1_interp, _ = create_scattering_factor_interpolators("Si")
        f1_vals = np.asarray(f1_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f1_vals, self.EXPECTED_F1, atol=1e-10)

    def test_f2_golden(self):
        _, f2_interp = create_scattering_factor_interpolators("Si")
        f2_vals = np.asarray(f2_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f2_vals, self.EXPECTED_F2, atol=1e-10)

    def test_output_dtype(self):
        f1_interp, f2_interp = create_scattering_factor_interpolators("Si")
        assert np.asarray(f1_interp(SAMPLE_ENERGIES_EV)).dtype in (
            np.float64,
            np.float32,
        ), "dtype must not silently become object or int"
        assert np.asarray(f2_interp(SAMPLE_ENERGIES_EV)).dtype in (
            np.float64,
            np.float32,
        )

    def test_output_shape(self):
        f1_interp, f2_interp = create_scattering_factor_interpolators("Si")
        assert np.asarray(f1_interp(SAMPLE_ENERGIES_EV)).shape == (5,)
        assert np.asarray(f2_interp(SAMPLE_ENERGIES_EV)).shape == (5,)


# ---------------------------------------------------------------------------
# P1-C  O — oxygen, low-Z, common compound constituent
# ---------------------------------------------------------------------------


class TestPchipO:
    """Golden PCHIP samples for Oxygen at [100, 1000, 5000, 10000, 20000] eV."""

    EXPECTED_F1 = np.array(
        [
            6.158406991848629,
            8.225556034039512,
            8.11077577684706,
            8.035817014448238,
            8.008253803189419,
        ]
    )
    EXPECTED_F2 = np.array(
        [
            2.58296024588449,
            1.754867817295257,
            0.09126998399717727,
            0.021094965936056333,
            0.004490099364666943,
        ]
    )

    def test_f1_golden(self):
        f1_interp, _ = create_scattering_factor_interpolators("O")
        f1_vals = np.asarray(f1_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f1_vals, self.EXPECTED_F1, atol=1e-10)

    def test_f2_golden(self):
        _, f2_interp = create_scattering_factor_interpolators("O")
        f2_vals = np.asarray(f2_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f2_vals, self.EXPECTED_F2, atol=1e-10)

    def test_output_shape(self):
        f1_interp, f2_interp = create_scattering_factor_interpolators("O")
        assert np.asarray(f1_interp(SAMPLE_ENERGIES_EV)).shape == (5,)
        assert np.asarray(f2_interp(SAMPLE_ENERGIES_EV)).shape == (5,)


# ---------------------------------------------------------------------------
# P1-C  Au — heavy element, tests edge-region PCHIP behaviour
# ---------------------------------------------------------------------------


class TestPchipAu:
    """Golden PCHIP samples for Gold at [100, 1000, 5000, 10000, 20000] eV.

    Heavy element data has absorption-edge discontinuities that stress
    PCHIP's monotonicity preservation — the hardest case for interpax.
    """

    EXPECTED_F1 = np.array(
        [21.0036754, 51.71952958, 74.54535322, 73.45286046, 78.10358641]
    )
    EXPECTED_F2 = np.array(
        [8.64431144, 25.30737386, 16.10210727, 5.42011705, 7.13588539]
    )

    def test_f1_golden(self):
        f1_interp, _ = create_scattering_factor_interpolators("Au")
        f1_vals = np.asarray(f1_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f1_vals, self.EXPECTED_F1, atol=1e-10)

    def test_f2_golden(self):
        _, f2_interp = create_scattering_factor_interpolators("Au")
        f2_vals = np.asarray(f2_interp(SAMPLE_ENERGIES_EV), dtype=np.float64)
        np.testing.assert_allclose(f2_vals, self.EXPECTED_F2, atol=1e-10)

    def test_output_shape(self):
        f1_interp, f2_interp = create_scattering_factor_interpolators("Au")
        assert np.asarray(f1_interp(SAMPLE_ENERGIES_EV)).shape == (5,)
        assert np.asarray(f2_interp(SAMPLE_ENERGIES_EV)).shape == (5,)


# ---------------------------------------------------------------------------
# Cross-element consistency: interpolators are independent objects
# ---------------------------------------------------------------------------


def test_si_o_au_interpolators_are_independent():
    """Loading three elements must return three distinct interpolator pairs."""
    f1_si, _f2_si = create_scattering_factor_interpolators("Si")
    f1_o, _f2_o = create_scattering_factor_interpolators("O")
    f1_au, _f2_au = create_scattering_factor_interpolators("Au")

    e = np.array([10000.0])
    # Si f1 at 10 keV must differ from O and Au
    assert not np.allclose(f1_si(e), f1_o(e)), "Si and O f1 must not be identical"
    assert not np.allclose(f1_si(e), f1_au(e)), "Si and Au f1 must not be identical"
