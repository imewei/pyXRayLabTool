"""Characterization tests: derived_quantities.py module (P2-A through P2-D).

The `calculators/derived_quantities.py` module had zero test coverage before
v0.3.0.  These tests lock down its formula implementations so that the JAX
rewrite of each function can be validated against exact expected values.

All golden values were computed from the live numpy/scipy stack on 2026-04-06.
"""

import math

import numpy as np
import pytest

from xraylabtool.calculators.derived_quantities import (
    calculate_attenuation_length,
    calculate_critical_angle,
    calculate_scattering_length_density,
    calculate_transmission,
)
from xraylabtool.constants import PI

# ---------------------------------------------------------------------------
# P2-A  calculate_critical_angle
# Formula: theta_c_rad = sqrt(2*delta); return theta_c_rad * (180/pi)
# ---------------------------------------------------------------------------


class TestGoldenCriticalAngle:
    """Two canonical delta values from the test plan."""

    def test_delta_1e6_golden(self):
        """delta=1e-6 -> ~0.08103 degrees."""
        result = calculate_critical_angle(np.array([1e-6]))
        np.testing.assert_allclose(result[0], 0.08102846845413955, atol=1e-10)

    def test_delta_5e6_golden(self):
        """delta=5e-6 -> ~0.18119 degrees."""
        result = calculate_critical_angle(np.array([5e-6]))
        np.testing.assert_allclose(result[0], 0.18118516357615333, atol=1e-10)

    def test_formula_correctness(self):
        """Verify formula: theta = sqrt(2*delta) * (180/pi)."""
        delta = np.array([3e-6])
        expected = math.sqrt(2.0 * 3e-6) * (180.0 / math.pi)
        result = calculate_critical_angle(delta)
        np.testing.assert_allclose(result[0], expected, atol=1e-14)

    def test_array_input(self):
        """Array of delta values returns array of the same length."""
        delta = np.array([1e-6, 5e-6, 1e-5])
        result = calculate_critical_angle(delta)
        assert result.shape == (3,)

    def test_monotone_with_delta(self):
        """Critical angle must increase monotonically with delta."""
        deltas = np.array([1e-7, 1e-6, 1e-5, 1e-4])
        angles = calculate_critical_angle(deltas)
        assert np.all(np.diff(angles) > 0), "Critical angle must increase with delta"

    def test_output_dtype(self):
        result = calculate_critical_angle(np.array([1e-6]))
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# P2-B  calculate_attenuation_length
# Formula: length_m = (wl_ang * 1e-10) / (4*pi*beta); return length_m * 100
# ---------------------------------------------------------------------------


class TestGoldenAttenuationLength:
    """Known wavelength/beta inputs -> known output, locked at rtol=1e-10."""

    # Inputs derived from locked SiO2@10keV run
    WL_ANG = np.array([1.2398417166827826])  # Angstroms
    BETA = np.array([3.887268047879803e-08])

    def test_golden_value(self):
        result = calculate_attenuation_length(self.WL_ANG, self.BETA)
        np.testing.assert_allclose(
            result[0], 0.025381184861793808, rtol=1e-10
        )

    def test_formula_correctness(self):
        """Verify Λ = lambda_m / (4*pi*beta) * 100."""
        wl_ang = 1.2398417166827826
        beta = 3.887268047879803e-08
        wl_m = wl_ang * 1e-10
        expected_cm = wl_m / (4.0 * PI * beta) * 100.0
        result = calculate_attenuation_length(
            np.array([wl_ang]), np.array([beta])
        )
        np.testing.assert_allclose(result[0], expected_cm, atol=1e-14)

    def test_larger_beta_gives_shorter_length(self):
        """More absorbing material -> shorter attenuation length."""
        wl = np.array([1.24])
        al_low = calculate_attenuation_length(wl, np.array([1e-8]))
        al_high = calculate_attenuation_length(wl, np.array([1e-6]))
        assert al_low[0] > al_high[0]

    def test_output_dtype(self):
        result = calculate_attenuation_length(self.WL_ANG, self.BETA)
        assert result.dtype == np.float64

    def test_output_positive(self):
        result = calculate_attenuation_length(self.WL_ANG, self.BETA)
        assert result[0] > 0.0


# ---------------------------------------------------------------------------
# P2-C  calculate_scattering_length_density
# Note: this function takes wavelength in ANGSTROMS (not meters) and uses a
# different formula than core.calculate_derived_quantities.
# Formula:
#   real_sld  = -delta / (pi * wl_ang^2) * 1e20
#   im_sld    =  beta  / (pi * wl_ang^2) * 1e20
# ---------------------------------------------------------------------------


class TestGoldenScatteringLengthDensity:
    """Scalar inputs -> known SLD, locked at atol=1e-12."""

    DELTA = np.array([4.613309228943556e-06])
    BETA = np.array([3.887268047879803e-08])
    WL_ANG = np.array([1.2398417166827826])  # Angstroms

    @pytest.fixture(scope="class")
    def sld_outputs(self):
        return calculate_scattering_length_density(
            self.DELTA, self.BETA, self.WL_ANG
        )

    def test_real_sld_golden(self, sld_outputs):
        re_sld, _ = sld_outputs
        np.testing.assert_allclose(re_sld[0], -95527894079547.34, atol=1e-12)

    def test_im_sld_golden(self, sld_outputs):
        _, im_sld = sld_outputs
        np.testing.assert_allclose(im_sld[0], 804937436248.356, atol=1e-12)

    def test_real_sld_is_negative(self, sld_outputs):
        """real_sld = -delta/(...) is always negative for positive delta."""
        re_sld, _ = sld_outputs
        assert re_sld[0] < 0.0

    def test_im_sld_is_positive(self, sld_outputs):
        """im_sld = beta/(...) is always positive for positive beta."""
        _, im_sld = sld_outputs
        assert im_sld[0] > 0.0

    def test_formula_correctness(self):
        """Verify exact formula: -delta / (pi*wl^2) * 1e20."""
        delta = 4.613309228943556e-06
        beta = 3.887268047879803e-08
        wl = 1.2398417166827826
        expected_re = -delta / (PI * wl**2) * 1e20
        expected_im = beta / (PI * wl**2) * 1e20
        re_sld, im_sld = calculate_scattering_length_density(
            np.array([delta]), np.array([beta]), np.array([wl])
        )
        np.testing.assert_allclose(re_sld[0], expected_re, atol=1e-14)
        np.testing.assert_allclose(im_sld[0], expected_im, atol=1e-14)

    def test_output_dtypes(self, sld_outputs):
        re_sld, im_sld = sld_outputs
        assert re_sld.dtype == np.float64
        assert im_sld.dtype == np.float64

    def test_output_shapes(self, sld_outputs):
        re_sld, im_sld = sld_outputs
        assert re_sld.shape == (1,)
        assert im_sld.shape == (1,)


# ---------------------------------------------------------------------------
# P2-D  calculate_transmission
# Formula: exp(-t / Lambda)
# ---------------------------------------------------------------------------


class TestGoldenTransmission:
    """exp(-t/Lambda) must be bit-for-bit identical; atol=1e-14."""

    # SiO2 attenuation length at 10 keV (cm)
    LAMBDA_CM = np.array([0.025381184861850772])

    def test_transmission_1mm_golden(self):
        """t=0.1 cm (1 mm) through SiO2 at 10 keV."""
        t = calculate_transmission(0.1, self.LAMBDA_CM)
        np.testing.assert_allclose(
            t[0], math.exp(-0.1 / 0.025381184861850772), atol=1e-14
        )

    def test_transmission_10um_golden(self):
        """t=0.001 cm (10 um) — thin-film regime."""
        result = calculate_transmission(0.001, self.LAMBDA_CM)
        expected = math.exp(-0.001 / 0.025381184861850772)
        np.testing.assert_allclose(result[0], expected, atol=1e-14)

    def test_transmission_equals_exp_formula(self):
        """Direct formula check: T = exp(-t/Lambda)."""
        t_cm = 0.01
        lambda_cm = np.array([0.025381184861850772])
        result = calculate_transmission(t_cm, lambda_cm)
        np.testing.assert_allclose(
            result[0], np.exp(-t_cm / lambda_cm[0]), atol=1e-14
        )

    def test_zero_thickness_is_unity(self):
        """Zero thickness must give T=1 exactly."""
        result = calculate_transmission(0.0, self.LAMBDA_CM)
        np.testing.assert_allclose(result[0], 1.0, atol=1e-15)

    def test_thick_sample_approaches_zero(self):
        """Very thick sample (100x Lambda) must give T~0."""
        result = calculate_transmission(100.0 * self.LAMBDA_CM[0], self.LAMBDA_CM)
        assert result[0] < 1e-30

    def test_monotone_with_thickness(self):
        """Transmission must decrease strictly with thickness."""
        thicknesses = np.array([0.0, 0.001, 0.01, 0.1, 1.0])
        transmissions = np.array(
            [calculate_transmission(t, self.LAMBDA_CM)[0] for t in thicknesses]
        )
        assert np.all(np.diff(transmissions) < 0)

    def test_output_dtype(self):
        result = calculate_transmission(0.01, self.LAMBDA_CM)
        assert result.dtype == np.float64
