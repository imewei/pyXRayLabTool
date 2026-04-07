"""Characterization tests: core pipeline golden values (P1-A, P1-B, P1-D).

Golden values were captured from xraylabtool v0.3.0 (numpy/scipy stack) by
running the live functions and recording the exact float64 results.  They must
not be changed without explicit sign-off.

Tolerances are set tighter than float32 machine epsilon (~1.2e-7) so that a
silent float32 regression from JAX's default dtype is immediately caught.
"""

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    calculate_derived_quantities,
    calculate_scattering_factors,
    calculate_single_material_properties,
    clear_scattering_factor_cache,
    create_scattering_factor_interpolators,
)
from xraylabtool.constants import ENERGY_TO_WAVELENGTH_FACTOR

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def cold_cache():
    """Ensure every test runs against the cold code-path (no cached state)."""
    clear_scattering_factor_cache()
    yield
    clear_scattering_factor_cache()


# ---------------------------------------------------------------------------
# P1-A  calculate_single_material_properties — end-to-end golden values
# ---------------------------------------------------------------------------


class TestGoldenSiO2:
    """SiO2 at 10.0 keV, density 2.2 g/cm³."""

    @pytest.fixture(scope="class")
    def result(self):
        clear_scattering_factor_cache()
        return calculate_single_material_properties("SiO2", 10.0, 2.2)

    def test_molecular_weight(self, result):
        assert abs(result.molecular_weight_g_mol - 60.083) < 1e-2

    def test_total_electrons(self, result):
        np.testing.assert_allclose(result.total_electrons, 30.0, rtol=1e-10)

    def test_electron_density(self, result):
        np.testing.assert_allclose(
            result.electron_density_per_ang3, 0.6615205155201972, rtol=1e-10
        )

    def test_wavelength(self, result):
        np.testing.assert_allclose(
            np.asarray(result.wavelength_angstrom)[0],
            1.2398417166827826,
            rtol=1e-10,
        )
        assert np.asarray(result.wavelength_angstrom).dtype == np.float64

    def test_dispersion_delta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.dispersion_delta)[0],
            4.613309228943556e-06,
            rtol=1e-10,
        )
        assert np.asarray(result.dispersion_delta).dtype == np.float64

    def test_absorption_beta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.absorption_beta)[0],
            3.887268047879803e-08,
            rtol=1e-10,
        )
        assert np.asarray(result.absorption_beta).dtype == np.float64

    def test_critical_angle(self, result):
        np.testing.assert_allclose(
            np.asarray(result.critical_angle_degrees)[0],
            0.1740379316778023,
            rtol=1e-8,
        )

    def test_attenuation_length(self, result):
        np.testing.assert_allclose(
            np.asarray(result.attenuation_length_cm)[0],
            0.025381184861850772,
            rtol=1e-6,
        )

    def test_real_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.real_sld_per_ang2)[0],
            1.885645047668597e-05,
            rtol=1e-10,
        )

    def test_imaginary_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.imaginary_sld_per_ang2)[0],
            1.5888828126796716e-07,
            rtol=1e-10,
        )

    def test_scattering_factor_f1(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f1)[0],
            30.346324020501555,
            rtol=1e-10,
        )

    def test_scattering_factor_f2(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f2)[0],
            0.2557042892234579,
            rtol=1e-10,
        )


class TestGoldenSi:
    """Si at 8.047 keV (Cu Ka), density 2.33 g/cm³."""

    @pytest.fixture(scope="class")
    def result(self):
        clear_scattering_factor_cache()
        return calculate_single_material_properties("Si", 8.047, 2.33)

    def test_molecular_weight(self, result):
        assert abs(result.molecular_weight_g_mol - 28.085) < 1e-2

    def test_total_electrons(self, result):
        np.testing.assert_allclose(result.total_electrons, 14.0, rtol=1e-10)

    def test_electron_density(self, result):
        np.testing.assert_allclose(
            result.electron_density_per_ang3, 0.699456192678654, rtol=1e-10
        )

    def test_wavelength(self, result):
        np.testing.assert_allclose(
            np.asarray(result.wavelength_angstrom)[0],
            1.5407502382040295,
            rtol=1e-10,
        )

    def test_dispersion_delta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.dispersion_delta)[0],
            7.604877922767226e-06,
            rtol=1e-10,
        )
        assert np.asarray(result.dispersion_delta).dtype == np.float64

    def test_absorption_beta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.absorption_beta)[0],
            1.728678959151362e-07,
            rtol=1e-10,
        )
        assert np.asarray(result.absorption_beta).dtype == np.float64

    def test_critical_angle(self, result):
        np.testing.assert_allclose(
            np.asarray(result.critical_angle_degrees)[0],
            0.22345174662966666,
            rtol=1e-8,
        )

    def test_attenuation_length(self, result):
        np.testing.assert_allclose(
            np.asarray(result.attenuation_length_cm)[0],
            0.007092641903866529,
            rtol=1e-6,
        )

    def test_real_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.real_sld_per_ang2)[0],
            2.012832463334068e-05,
            rtol=1e-10,
        )

    def test_imaginary_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.imaginary_sld_per_ang2)[0],
            4.5754069466986127e-07,
            rtol=1e-10,
        )

    def test_scattering_factor_f1(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f1)[0],
            14.296948499381564,
            rtol=1e-10,
        )

    def test_scattering_factor_f2(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f2)[0],
            0.32498659810121494,
            rtol=1e-10,
        )


class TestGoldenAu:
    """Au at 10.0 keV, density 19.3 g/cm³."""

    @pytest.fixture(scope="class")
    def result(self):
        clear_scattering_factor_cache()
        return calculate_single_material_properties("Au", 10.0, 19.3)

    def test_molecular_weight(self, result):
        assert abs(result.molecular_weight_g_mol - 196.97) < 1e-2

    def test_total_electrons(self, result):
        np.testing.assert_allclose(result.total_electrons, 79.0, rtol=1e-10)

    def test_electron_density(self, result):
        np.testing.assert_allclose(
            result.electron_density_per_ang3, 4.6616032350880845, rtol=1e-10
        )

    def test_wavelength(self, result):
        np.testing.assert_allclose(
            np.asarray(result.wavelength_angstrom)[0],
            1.2398417166827826,
            rtol=1e-10,
        )

    def test_dispersion_delta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.dispersion_delta)[0],
            2.9881427527391494e-05,
            rtol=1e-10,
        )
        assert np.asarray(result.dispersion_delta).dtype == np.float64

    def test_absorption_beta(self, result):
        np.testing.assert_allclose(
            np.asarray(result.absorption_beta)[0],
            2.204962934040285e-06,
            rtol=1e-10,
        )
        assert np.asarray(result.absorption_beta).dtype == np.float64

    def test_critical_angle(self, result):
        np.testing.assert_allclose(
            np.asarray(result.critical_angle_degrees)[0],
            0.44293326818909534,
            rtol=1e-8,
        )

    def test_attenuation_length(self, result):
        np.testing.assert_allclose(
            np.asarray(result.attenuation_length_cm)[0],
            0.0004474608956351756,
            rtol=1e-6,
        )

    def test_real_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.real_sld_per_ang2)[0],
            0.00012213741381302763,
            rtol=1e-10,
        )

    def test_imaginary_sld(self, result):
        np.testing.assert_allclose(
            np.asarray(result.imaginary_sld_per_ang2)[0],
            9.012570435947148e-06,
            rtol=1e-10,
        )

    def test_scattering_factor_f1(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f1)[0],
            73.45286046055072,
            rtol=1e-10,
        )

    def test_scattering_factor_f2(self, result):
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f2)[0],
            5.4201170464925905,
            rtol=1e-10,
        )


class TestGoldenFe2O3:
    """Fe2O3 at [8, 10, 12] keV, density 5.24 g/cm³ — full array assertions."""

    @pytest.fixture(scope="class")
    def result(self):
        clear_scattering_factor_cache()
        return calculate_single_material_properties("Fe2O3", [8.0, 10.0, 12.0], 5.24)

    def test_molecular_weight(self, result):
        assert abs(result.molecular_weight_g_mol - 159.687) < 1e-2

    def test_total_electrons(self, result):
        np.testing.assert_allclose(result.total_electrons, 76.0, rtol=1e-10)

    def test_electron_density(self, result):
        np.testing.assert_allclose(
            result.electron_density_per_ang3, 1.501849133678759, rtol=1e-10
        )

    def test_wavelength_array(self, result):
        expected = np.array([1.54980215, 1.23984172, 1.03320143])
        np.testing.assert_allclose(
            np.asarray(result.wavelength_angstrom), expected, rtol=1e-7
        )
        assert np.asarray(result.wavelength_angstrom).dtype == np.float64

    def test_dispersion_delta_array(self, result):
        expected = np.array([1.5699325422722898e-05, 1.0362034024426724e-05, 7.248916179697252e-06])
        np.testing.assert_allclose(
            np.asarray(result.dispersion_delta), expected, rtol=1e-9
        )
        assert np.asarray(result.dispersion_delta).dtype == np.float64

    def test_absorption_beta_array(self, result):
        expected = np.array([1.4005422853421068e-06, 6.262873111508109e-07, 3.1858563221542804e-07])
        np.testing.assert_allclose(
            np.asarray(result.absorption_beta), expected, rtol=1e-9
        )
        assert np.asarray(result.absorption_beta).dtype == np.float64

    def test_critical_angle_array(self, result):
        expected = np.array([0.321054030308344, 0.26083155903542726, 0.21815951991793991])
        np.testing.assert_allclose(
            np.asarray(result.critical_angle_degrees), expected, rtol=1e-9
        )

    def test_attenuation_length_array(self, result):
        expected = np.array([0.0008805827389451401, 0.0015753707152314435, 0.002580767904396232])
        np.testing.assert_allclose(
            np.asarray(result.attenuation_length_cm), expected, rtol=1e-9
        )

    def test_real_sld_array(self, result):
        expected = np.array([4.1068453056634865e-05, 4.235380108349727e-05, 4.2666100172979545e-05])
        np.testing.assert_allclose(
            np.asarray(result.real_sld_per_ang2), expected, rtol=1e-9
        )

    def test_imaginary_sld_array(self, result):
        expected = np.array([3.6637309916611986e-06, 2.5598881585478413e-06, 1.8751501825674536e-06])
        np.testing.assert_allclose(
            np.asarray(result.imaginary_sld_per_ang2), expected, rtol=1e-9
        )

    def test_f1_array(self, result):
        expected = np.array([73.75030772760337, 76.05852256071249, 76.61934606971113])
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f1), expected, rtol=1e-9
        )

    def test_f2_array(self, result):
        expected = np.array([6.579290622258248, 4.597020958661374, 3.3673755086199737])
        np.testing.assert_allclose(
            np.asarray(result.scattering_factor_f2), expected, rtol=1e-9
        )


# ---------------------------------------------------------------------------
# P1-B  calculate_scattering_factors — isolated golden values
# ---------------------------------------------------------------------------


class TestGoldenScatteringFactors:
    """Lock intermediate values of calculate_scattering_factors for Si.

    These values are the target the JAX einsum rewrite must reproduce.
    """

    @pytest.fixture(scope="class")
    def outputs(self):
        clear_scattering_factor_cache()
        energy_kev = np.array([8.0, 10.0, 12.0])
        energy_ev = energy_kev * 1000.0
        wavelength = ENERGY_TO_WAVELENGTH_FACTOR / energy_kev  # meters
        f1_interp, f2_interp = create_scattering_factor_interpolators("Si")
        element_data = [(1.0, f1_interp, f2_interp)]
        return calculate_scattering_factors(
            energy_ev, wavelength, 2.33, 28.085, element_data
        )

    def test_dispersion(self, outputs):
        disp, _, _, _ = outputs
        expected = np.array([7.69519539e-06, 4.91681177e-06, 3.38362871e-06])
        np.testing.assert_allclose(disp, expected, atol=1e-12)
        assert disp.dtype == np.float64

    def test_absorption(self, outputs):
        _, absorb, _, _ = outputs
        expected = np.array([1.76880597e-07, 7.35434468e-08, 3.57212114e-08])
        np.testing.assert_allclose(absorb, expected, atol=1e-12)
        assert absorb.dtype == np.float64

    def test_f1_total(self, outputs):
        _, _, f1t, _ = outputs
        expected = np.array([14.2982448, 14.27468999, 14.1458255])
        np.testing.assert_allclose(f1t, expected, atol=1e-12)
        assert f1t.dtype == np.float64

    def test_f2_total(self, outputs):
        _, _, _, f2t = outputs
        expected = np.array([0.32865729, 0.21351436, 0.1493385])
        np.testing.assert_allclose(f2t, expected, atol=1e-12)
        assert f2t.dtype == np.float64


# ---------------------------------------------------------------------------
# P1-D  calculate_derived_quantities — hardcoded synthetic inputs
# ---------------------------------------------------------------------------


class TestGoldenDerivedQuantities:
    """Isolate the derived-quantities formula from the interpolation layer.

    Inputs are hardcoded SiO2@10keV golden values so the test fails if the
    formula changes, regardless of interpolator behaviour.
    """

    # Synthetic inputs taken from the locked SiO2@10keV run
    WAVELENGTH_M = np.array([1.2398417166827826e-10])  # meters
    DISPERSION = np.array([4.613309228943556e-06])
    ABSORPTION = np.array([3.887268047879803e-08])
    DENSITY = 2.2
    MW = 60.083
    ELECTRONS = 30.0

    @pytest.fixture(scope="class")
    def outputs(self):
        clear_scattering_factor_cache()
        return calculate_derived_quantities(
            self.WAVELENGTH_M,
            self.DISPERSION,
            self.ABSORPTION,
            self.DENSITY,
            self.MW,
            self.ELECTRONS,
        )

    def test_electron_density(self, outputs):
        ed, _, _, _, _ = outputs
        np.testing.assert_allclose(ed, 0.6615205155201972, rtol=1e-12)

    def test_critical_angle(self, outputs):
        _, ca, _, _, _ = outputs
        np.testing.assert_allclose(ca, np.array([0.1740379316778023]), rtol=1e-12)
        assert ca.dtype == np.float64

    def test_attenuation_length(self, outputs):
        _, _, al, _, _ = outputs
        np.testing.assert_allclose(al, np.array([0.025381184861850772]), rtol=1e-12)
        assert al.dtype == np.float64

    def test_re_sld(self, outputs):
        _, _, _, rs, _ = outputs
        np.testing.assert_allclose(rs, np.array([1.885645047668597e-05]), rtol=1e-12)
        assert rs.dtype == np.float64

    def test_im_sld(self, outputs):
        _, _, _, _, is_ = outputs
        np.testing.assert_allclose(is_, np.array([1.5888828126796716e-07]), rtol=1e-12)
        assert is_.dtype == np.float64
