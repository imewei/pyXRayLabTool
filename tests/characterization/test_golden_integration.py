"""
Characterization tests for end-to-end integration paths (Plan section 3).

Golden values captured from xraylabtool v0.3.0 numpy/scipy stack on 2026-04-06.
These tests lock the complete pipeline output — from formula parsing through PCHIP
interpolation to every XRayResult field — so any regression introduced during JAX
migration is immediately visible.

Tolerances (see plan section 4):
    delta, beta          : atol=1e-12 (catches float32 regression at ~1e-7 relative)
    critical_angle       : atol=1e-8 degrees
    attenuation_length   : rtol=1e-6 (large dynamic range)
    real/imag SLD        : atol=1e-12
    multi-material equiv : atol=1e-15 (should be bitwise identical)
    batch equiv          : atol=1e-12
"""

from __future__ import annotations

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    XRayResult,
    calculate_multiple_xray_properties,
    calculate_single_material_properties,
    clear_scattering_factor_cache,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Cold-path for every test — prevents cache artifacts from polluting values."""
    clear_scattering_factor_cache()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _single(formula: str, energy_kev, density: float) -> XRayResult:
    clear_scattering_factor_cache()
    return calculate_single_material_properties(formula, energy_kev, density)


# ---------------------------------------------------------------------------
# Full pipeline: SiO2 @ 10 keV — lock every XRayResult field
# ---------------------------------------------------------------------------


class TestFullPipelineSiO2:
    """
    End-to-end characterization for SiO2 at 10 keV with density 2.2 g/cm³.
    Golden values captured 2026-04-06.
    """

    @pytest.fixture(scope="class", autouse=True)
    def result(self, request) -> None:
        clear_scattering_factor_cache()
        request.cls.r = calculate_single_material_properties("SiO2", 10.0, 2.2)

    # --- Scalar material properties ---

    def test_formula(self) -> None:
        assert self.r.formula == "SiO2"

    def test_molecular_weight(self) -> None:
        np.testing.assert_allclose(
            self.r.molecular_weight_g_mol, 60.083, atol=1e-2,
            err_msg="SiO2 MW regression"
        )

    def test_total_electrons(self) -> None:
        np.testing.assert_allclose(
            self.r.total_electrons, 30.0, atol=1e-10,
            err_msg="SiO2 Z_total regression"
        )

    def test_density(self) -> None:
        assert self.r.density_g_cm3 == 2.2

    def test_electron_density(self) -> None:
        # Captured: 0.6615205155201972 e/Å³
        np.testing.assert_allclose(
            self.r.electron_density_per_ang3, 0.6615205155201972, atol=1e-12,
            err_msg="SiO2 electron_density_per_ang3 regression"
        )

    # --- Array shape and dtype ---

    def test_array_length_is_one(self) -> None:
        assert len(self.r.energy_kev) == 1

    def test_dispersion_delta_dtype_float64(self) -> None:
        assert self.r.dispersion_delta.dtype == np.float64

    def test_absorption_beta_dtype_float64(self) -> None:
        assert self.r.absorption_beta.dtype == np.float64

    # --- Wavelength ---

    def test_wavelength_angstrom(self) -> None:
        # Captured: 1.2398417166827826 Å
        np.testing.assert_allclose(
            self.r.wavelength_angstrom[0], 1.2398417166827826, atol=1e-8,
            err_msg="SiO2 wavelength_angstrom regression"
        )

    # --- Optical constants ---

    def test_dispersion_delta(self) -> None:
        # Captured: 4.613309228943556e-06
        np.testing.assert_allclose(
            self.r.dispersion_delta[0], 4.613309228943556e-06, atol=1e-12,
            err_msg="SiO2 dispersion_delta regression"
        )

    def test_absorption_beta(self) -> None:
        # Captured: 3.887268047879803e-08
        np.testing.assert_allclose(
            self.r.absorption_beta[0], 3.887268047879803e-08, atol=1e-12,
            err_msg="SiO2 absorption_beta regression"
        )

    def test_scattering_factor_f1(self) -> None:
        # Captured: 30.346324020501555
        np.testing.assert_allclose(
            self.r.scattering_factor_f1[0], 30.346324020501555, atol=1e-10,
            err_msg="SiO2 f1 regression"
        )

    def test_scattering_factor_f2(self) -> None:
        # Captured: 0.2557042892234579
        np.testing.assert_allclose(
            self.r.scattering_factor_f2[0], 0.2557042892234579, atol=1e-10,
            err_msg="SiO2 f2 regression"
        )

    # --- Derived quantities ---

    def test_critical_angle_degrees(self) -> None:
        # Captured: 0.1740379316778023 degrees
        np.testing.assert_allclose(
            self.r.critical_angle_degrees[0], 0.1740379316778023, atol=1e-8,
            err_msg="SiO2 critical_angle_degrees regression"
        )

    def test_attenuation_length_cm(self) -> None:
        # Captured: 0.025381184861850772 cm
        np.testing.assert_allclose(
            self.r.attenuation_length_cm[0], 0.025381184861850772, rtol=1e-6,
            err_msg="SiO2 attenuation_length_cm regression"
        )

    def test_real_sld(self) -> None:
        # Captured: 1.885645047668597e-05 Å⁻²
        np.testing.assert_allclose(
            self.r.real_sld_per_ang2[0], 1.885645047668597e-05, atol=1e-12,
            err_msg="SiO2 real_sld_per_ang2 regression"
        )

    def test_imaginary_sld(self) -> None:
        # Captured: 1.5888828126796716e-07 Å⁻²
        np.testing.assert_allclose(
            self.r.imaginary_sld_per_ang2[0], 1.5888828126796716e-07, atol=1e-12,
            err_msg="SiO2 imaginary_sld_per_ang2 regression"
        )

    # --- Physical sanity ---

    def test_delta_positive(self) -> None:
        assert float(self.r.dispersion_delta[0]) > 0

    def test_beta_positive(self) -> None:
        assert float(self.r.absorption_beta[0]) > 0

    def test_critical_angle_positive(self) -> None:
        assert float(self.r.critical_angle_degrees[0]) > 0

    def test_attenuation_length_positive(self) -> None:
        assert float(self.r.attenuation_length_cm[0]) > 0


# ---------------------------------------------------------------------------
# Multi-material: calculate_multiple_xray_properties vs independent calls
# ---------------------------------------------------------------------------


class TestMultiMaterialEquivalence:
    """
    calculate_multiple_xray_properties with 3 materials must return values
    identical to 3 independent single-material calls (atol=1e-15 — should be
    numerically exact since both paths call the same underlying function).

    Note: calculate_multiple_xray_properties returns dicts, not XRayResult objects.
    """

    MATERIALS = [
        ("SiO2",   10.0, 2.2),
        ("Al2O3",  10.0, 3.95),
        ("Fe2O3",  10.0, 5.24),
    ]

    @pytest.fixture(scope="class", autouse=True)
    def results(self, request) -> None:
        formulas = [m[0] for m in self.MATERIALS]
        densities = [m[2] for m in self.MATERIALS]

        clear_scattering_factor_cache()
        request.cls.multi = calculate_multiple_xray_properties(
            formulas, 10.0, densities
        )

        singles = {}
        for formula, energy, density in self.MATERIALS:
            clear_scattering_factor_cache()
            singles[formula] = calculate_single_material_properties(
                formula, energy, density
            )
        request.cls.singles = singles

    @pytest.mark.parametrize("formula", ["SiO2", "Al2O3", "Fe2O3"])
    def test_dispersion_matches_single(self, formula: str) -> None:
        multi_val = self.multi[formula]["dispersion"][0]
        single_val = float(self.singles[formula].dispersion_delta[0])
        np.testing.assert_allclose(
            multi_val, single_val, atol=1e-15,
            err_msg=f"{formula} dispersion mismatch between multi and single"
        )

    @pytest.mark.parametrize("formula", ["SiO2", "Al2O3", "Fe2O3"])
    def test_absorption_matches_single(self, formula: str) -> None:
        multi_val = self.multi[formula]["absorption"][0]
        single_val = float(self.singles[formula].absorption_beta[0])
        np.testing.assert_allclose(
            multi_val, single_val, atol=1e-15,
            err_msg=f"{formula} absorption mismatch between multi and single"
        )

    @pytest.mark.parametrize("formula", ["SiO2", "Al2O3", "Fe2O3"])
    def test_molecular_weight_matches_single(self, formula: str) -> None:
        multi_val = self.multi[formula]["molecular_weight"]
        single_val = self.singles[formula].molecular_weight_g_mol
        np.testing.assert_allclose(
            multi_val, single_val, atol=1e-15,
            err_msg=f"{formula} MW mismatch"
        )

    def test_multi_returns_all_three_keys(self) -> None:
        assert set(self.multi.keys()) == {"SiO2", "Al2O3", "Fe2O3"}

    # Golden values for Al2O3 from multi call
    def test_al2o3_dispersion_golden(self) -> None:
        # Captured: 8.110134538736214e-06
        np.testing.assert_allclose(
            self.multi["Al2O3"]["dispersion"][0], 8.110134538736214e-06, atol=1e-12
        )

    def test_fe2o3_absorption_golden(self) -> None:
        # Captured: 6.262873111508109e-07
        np.testing.assert_allclose(
            self.multi["Fe2O3"]["absorption"][0], 6.262873111508109e-07, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Batch processor equivalence (calculate_batch_properties)
# ---------------------------------------------------------------------------


class TestBatchProcessorEquivalence:
    """
    calculate_batch_properties parallel result must match single-material calls
    to atol=1e-12.
    """

    def test_batch_matches_single_sio2(self) -> None:
        from xraylabtool.data_handling.batch_processing import (
            BatchConfig,
            calculate_batch_properties,
        )

        energy = np.array([10.0])
        clear_scattering_factor_cache()
        batch_results = calculate_batch_properties(
            ["SiO2"], energy, [2.2],
            config=BatchConfig(max_workers=1, enable_progress=False),
        )
        key = "SiO2@2.200"
        assert key in batch_results
        batch_r = batch_results[key]

        clear_scattering_factor_cache()
        single_r = calculate_single_material_properties("SiO2", energy, 2.2)

        np.testing.assert_allclose(
            batch_r.dispersion_delta[0], single_r.dispersion_delta[0], atol=1e-12,
            err_msg="Batch vs single dispersion_delta mismatch for SiO2"
        )
        np.testing.assert_allclose(
            batch_r.absorption_beta[0], single_r.absorption_beta[0], atol=1e-12,
            err_msg="Batch vs single absorption_beta mismatch for SiO2"
        )

    def test_batch_three_materials_all_match_singles(self) -> None:
        from xraylabtool.data_handling.batch_processing import (
            BatchConfig,
            calculate_batch_properties,
        )

        materials = [("SiO2", 2.2), ("Al2O3", 3.95), ("Fe2O3", 5.24)]
        formulas = [m[0] for m in materials]
        densities = [m[1] for m in materials]
        energy = np.array([10.0])

        clear_scattering_factor_cache()
        batch_results = calculate_batch_properties(
            formulas, energy, densities,
            config=BatchConfig(max_workers=2, enable_progress=False),
        )

        for formula, density in materials:
            clear_scattering_factor_cache()
            single_r = calculate_single_material_properties(formula, energy, density)
            key = f"{formula}@{density:.3f}"
            assert key in batch_results, f"Key {key!r} missing from batch results"
            batch_r = batch_results[key]
            np.testing.assert_allclose(
                batch_r.dispersion_delta[0], single_r.dispersion_delta[0],
                atol=1e-12,
                err_msg=f"Batch vs single dispersion mismatch for {formula}"
            )


# ---------------------------------------------------------------------------
# Energy sweep: 500-point linspace 0.03–30 keV for Si
# ---------------------------------------------------------------------------


class TestEnergySweepSi:
    """
    500-point energy sweep for Si. Lock snapshot values at indices 0, 249, 499
    and assert no NaN/Inf across the full range.

    Golden values captured 2026-04-06.
    """

    ENERGIES = np.linspace(0.03, 30.0, 500)
    DENSITY = 2.33  # g/cm³

    @pytest.fixture(scope="class", autouse=True)
    def sweep_result(self, request) -> None:
        clear_scattering_factor_cache()
        request.cls.r = calculate_single_material_properties(
            "Si", self.ENERGIES, self.DENSITY
        )

    # --- Finiteness ---

    def test_no_nan_dispersion(self) -> None:
        assert not np.any(np.isnan(self.r.dispersion_delta))

    def test_no_inf_dispersion(self) -> None:
        assert not np.any(np.isinf(self.r.dispersion_delta))

    def test_no_nan_absorption(self) -> None:
        assert not np.any(np.isnan(self.r.absorption_beta))

    def test_no_inf_absorption(self) -> None:
        assert not np.any(np.isinf(self.r.absorption_beta))

    def test_no_nan_critical_angle(self) -> None:
        assert not np.any(np.isnan(self.r.critical_angle_degrees))

    def test_no_nan_attenuation_length(self) -> None:
        assert not np.any(np.isnan(self.r.attenuation_length_cm))

    def test_output_length(self) -> None:
        assert len(self.r.dispersion_delta) == 500

    # --- Snapshot at index 0 (0.03 keV — soft X-ray) ---

    def test_snapshot_index0_dispersion(self) -> None:
        # Captured: 0.1453834710364258
        np.testing.assert_allclose(
            self.r.dispersion_delta[0], 0.1453834710364258, atol=1e-10,
            err_msg="Si sweep dispersion_delta[0] regression"
        )

    def test_snapshot_index0_absorption(self) -> None:
        # Captured: 0.014289187705435344
        np.testing.assert_allclose(
            self.r.absorption_beta[0], 0.014289187705435344, atol=1e-10,
            err_msg="Si sweep absorption_beta[0] regression"
        )

    def test_snapshot_index0_wavelength(self) -> None:
        # Captured: 413.2805722275943 Å
        np.testing.assert_allclose(
            self.r.wavelength_angstrom[0], 413.2805722275943, atol=1e-6,
            err_msg="Si sweep wavelength_angstrom[0] regression"
        )

    def test_snapshot_index0_critical_angle(self) -> None:
        # Captured: 30.895494231525966 degrees
        np.testing.assert_allclose(
            self.r.critical_angle_degrees[0], 30.895494231525966, atol=1e-8,
            err_msg="Si sweep critical_angle_degrees[0] regression"
        )

    def test_snapshot_index0_attenuation_length(self) -> None:
        # Captured: 2.301588001704564e-05 cm
        np.testing.assert_allclose(
            self.r.attenuation_length_cm[0], 2.301588001704564e-05, rtol=1e-6,
            err_msg="Si sweep attenuation_length_cm[0] regression"
        )

    # --- Snapshot at index 249 (mid-range ~15 keV) ---

    def test_snapshot_index249_dispersion(self) -> None:
        # Captured: 2.1610269670902516e-06
        np.testing.assert_allclose(
            self.r.dispersion_delta[249], 2.1610269670902516e-06, atol=1e-12,
            err_msg="Si sweep dispersion_delta[249] regression"
        )

    def test_snapshot_index249_absorption(self) -> None:
        # Captured: 1.4722650524080965e-08
        np.testing.assert_allclose(
            self.r.absorption_beta[249], 1.4722650524080965e-08, atol=1e-12,
            err_msg="Si sweep absorption_beta[249] regression"
        )

    def test_snapshot_index249_wavelength(self) -> None:
        # Captured: 0.8273901927445118 Å
        np.testing.assert_allclose(
            self.r.wavelength_angstrom[249], 0.8273901927445118, atol=1e-8,
            err_msg="Si sweep wavelength_angstrom[249] regression"
        )

    def test_snapshot_index249_critical_angle(self) -> None:
        # Captured: 0.11911534787852067 degrees
        np.testing.assert_allclose(
            self.r.critical_angle_degrees[249], 0.11911534787852067, atol=1e-8,
            err_msg="Si sweep critical_angle_degrees[249] regression"
        )

    def test_snapshot_index249_attenuation_length(self) -> None:
        # Captured: 0.04472130844431125 cm
        np.testing.assert_allclose(
            self.r.attenuation_length_cm[249], 0.04472130844431125, rtol=1e-6,
            err_msg="Si sweep attenuation_length_cm[249] regression"
        )

    # --- Snapshot at index 499 (30 keV — hard X-ray upper bound) ---

    def test_snapshot_index499_dispersion(self) -> None:
        # Captured: 5.365535935522566e-07
        np.testing.assert_allclose(
            self.r.dispersion_delta[499], 5.365535935522566e-07, atol=1e-12,
            err_msg="Si sweep dispersion_delta[499] regression"
        )

    def test_snapshot_index499_absorption(self) -> None:
        # Captured: 8.743484809443038e-10
        np.testing.assert_allclose(
            self.r.absorption_beta[499], 8.743484809443038e-10, atol=1e-14,
            err_msg="Si sweep absorption_beta[499] regression"
        )

    def test_snapshot_index499_wavelength(self) -> None:
        # Captured: 0.41328057222759423 Å
        np.testing.assert_allclose(
            self.r.wavelength_angstrom[499], 0.41328057222759423, atol=1e-8,
            err_msg="Si sweep wavelength_angstrom[499] regression"
        )

    def test_snapshot_index499_critical_angle(self) -> None:
        # Captured: 0.059353206215586506 degrees
        np.testing.assert_allclose(
            self.r.critical_angle_degrees[499], 0.059353206215586506, atol=1e-8,
            err_msg="Si sweep critical_angle_degrees[499] regression"
        )

    def test_snapshot_index499_attenuation_length(self) -> None:
        # Captured: 0.3761409059853941 cm
        np.testing.assert_allclose(
            self.r.attenuation_length_cm[499], 0.3761409059853941, rtol=1e-6,
            err_msg="Si sweep attenuation_length_cm[499] regression"
        )

    # --- Physical monotonicity ---

    def test_dispersion_all_positive(self) -> None:
        assert np.all(self.r.dispersion_delta > 0)

    def test_absorption_all_positive(self) -> None:
        assert np.all(self.r.absorption_beta > 0)

    def test_wavelength_decreasing(self) -> None:
        """Higher energy → shorter wavelength."""
        assert np.all(np.diff(self.r.wavelength_angstrom) < 0)
