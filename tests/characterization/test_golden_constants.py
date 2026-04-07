"""
Characterization tests for physical constants and unit conversions (Plan P2-F).

Golden values captured from xraylabtool v0.3.0 numpy/scipy stack on 2026-04-06.
These tests lock every constant defined in constants.py to its exact float64
representation.  A regression here means a constant was re-typed, a dependency
changed its value, or a floating-point operation was reordered during migration.

Tolerances:
    Fundamental constants  : exact equality (they are hardcoded literals)
    Derived constants      : atol=0 / equal (computed at import time from literals)
    energy_to_wavelength   : atol=1e-8 (8th decimal place in Angstroms)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import xraylabtool.constants as xconst
from xraylabtool.constants import energy_to_wavelength_angstrom

# ---------------------------------------------------------------------------
# Fundamental physical constants — must be exact literals
# ---------------------------------------------------------------------------


class TestFundamentalConstants:
    """Lock the hardcoded float literals in constants.py."""

    def test_thompson(self) -> None:
        assert xconst.THOMPSON == 2.8179403227e-15

    def test_avogadro(self) -> None:
        assert xconst.AVOGADRO == 6.02214199e23

    def test_planck(self) -> None:
        assert xconst.PLANCK == 6.626068e-34

    def test_speed_of_light(self) -> None:
        assert xconst.SPEED_OF_LIGHT == 299792458.0

    def test_element_charge(self) -> None:
        assert xconst.ELEMENT_CHARGE == 1.60217646e-19


# ---------------------------------------------------------------------------
# Derived constants — exact float64 values captured at v0.3.0
# ---------------------------------------------------------------------------


class TestDerivedConstants:
    """Lock derived constants to their exact captured float64 values."""

    def test_energy_to_wavelength_factor(self) -> None:
        # Captured: 1.2398417166827828e-09 m·keV
        assert xconst.ENERGY_TO_WAVELENGTH_FACTOR == 1.2398417166827828e-09

    def test_scattering_factor(self) -> None:
        # Captured: 270086523204316.84
        assert xconst.SCATTERING_FACTOR == 270086523204316.84

    def test_scattering_factor_derivation(self) -> None:
        """SCATTERING_FACTOR must equal THOMPSON * AVOGADRO * 1e6 / (2π)."""
        expected = xconst.THOMPSON * xconst.AVOGADRO * 1e6 / (2 * math.pi)
        np.testing.assert_allclose(xconst.SCATTERING_FACTOR, expected, rtol=1e-14)

    def test_energy_to_wavelength_factor_derivation(self) -> None:
        """ENERGY_TO_WAVELENGTH_FACTOR must equal (h·c/e) / 1000."""
        expected = (
            xconst.SPEED_OF_LIGHT * xconst.PLANCK / xconst.ELEMENT_CHARGE
        ) / 1000.0
        np.testing.assert_allclose(
            xconst.ENERGY_TO_WAVELENGTH_FACTOR, expected, rtol=1e-14
        )


# ---------------------------------------------------------------------------
# Mathematical constants — exact Python float representations
# ---------------------------------------------------------------------------


class TestMathematicalConstants:
    def test_pi(self) -> None:
        assert xconst.PI == 3.141592653589793

    def test_two_pi(self) -> None:
        assert xconst.TWO_PI == 6.283185307179586

    def test_sqrt_2(self) -> None:
        assert xconst.SQRT_2 == 1.4142135623730951


# ---------------------------------------------------------------------------
# Unit conversion constants — exact values
# ---------------------------------------------------------------------------


class TestUnitConversionConstants:
    def test_kev_to_ev(self) -> None:
        assert xconst.KEV_TO_EV == 1000.0

    def test_ev_to_kev(self) -> None:
        assert xconst.EV_TO_KEV == 0.001

    def test_kev_ev_round_trip(self) -> None:
        np.testing.assert_allclose(xconst.KEV_TO_EV * xconst.EV_TO_KEV, 1.0, rtol=1e-15)

    def test_angstrom_to_meter(self) -> None:
        assert xconst.ANGSTROM_TO_METER == 1e-10

    def test_meter_to_angstrom(self) -> None:
        assert xconst.METER_TO_ANGSTROM == 1e10

    def test_angstrom_meter_round_trip(self) -> None:
        np.testing.assert_allclose(
            xconst.ANGSTROM_TO_METER * xconst.METER_TO_ANGSTROM, 1.0, rtol=1e-15
        )

    def test_cm_to_meter(self) -> None:
        assert xconst.CM_TO_METER == 0.01

    def test_meter_to_cm(self) -> None:
        assert xconst.METER_TO_CM == 100.0

    def test_degrees_to_radians(self) -> None:
        assert xconst.DEGREES_TO_RADIANS == 0.017453292519943295

    def test_radians_to_degrees(self) -> None:
        assert xconst.RADIANS_TO_DEGREES == 57.29577951308232

    def test_degrees_radians_round_trip(self) -> None:
        np.testing.assert_allclose(
            xconst.DEGREES_TO_RADIANS * xconst.RADIANS_TO_DEGREES, 1.0, rtol=1e-15
        )


# ---------------------------------------------------------------------------
# energy_to_wavelength_angstrom — functional golden values
# ---------------------------------------------------------------------------


class TestEnergyToWavelength:
    """
    Lock energy_to_wavelength_angstrom() at specific energies to atol=1e-8 Å.
    This is tight enough to catch float32 regression (float32 gives ~1e-6
    relative error, meaning ~1e-6 Å error at 1 Å wavelength).
    """

    @pytest.mark.parametrize(
        "energy_kev, expected_angstrom",
        [
            # (energy keV, captured wavelength Å)
            (10.0, 1.2398417166827826),  # reference energy
            (8.047, 1.5407502382040295),  # Cu Kα
            (17.479, 0.7093321795770826),  # Mo Kα
            (0.03, 413.2805722275943),  # soft X-ray lower bound
            (30.0, 0.41328057222759423),  # hard X-ray upper bound
        ],
    )
    def test_golden_values(self, energy_kev: float, expected_angstrom: float) -> None:
        result = energy_to_wavelength_angstrom(energy_kev)
        np.testing.assert_allclose(
            result,
            expected_angstrom,
            atol=1e-8,
            err_msg=f"energy_to_wavelength_angstrom({energy_kev} keV) regression",
        )

    def test_return_type_is_float(self) -> None:
        result = energy_to_wavelength_angstrom(10.0)
        assert isinstance(result, float), (
            f"Expected Python float, got {type(result).__name__}"
        )

    def test_negative_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_wavelength_angstrom(-1.0)

    def test_zero_energy_raises(self) -> None:
        with pytest.raises(ValueError):
            energy_to_wavelength_angstrom(0.0)

    def test_inverse_consistency(self) -> None:
        """Round-trip: wavelength_angstrom_to_energy(energy_to_wavelength_angstrom(E)) ≈ E."""
        from xraylabtool.constants import wavelength_angstrom_to_energy

        for energy in [0.5, 1.0, 8.047, 10.0, 20.0]:
            wl = energy_to_wavelength_angstrom(energy)
            recovered = wavelength_angstrom_to_energy(wl)
            np.testing.assert_allclose(
                recovered,
                energy,
                rtol=1e-12,
                err_msg=f"round-trip failed at {energy} keV",
            )


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


class TestDtypePreservation:
    """Assert float64 is preserved through constant arithmetic."""

    def test_energy_to_wavelength_factor_dtype(self) -> None:
        val = np.array(xconst.ENERGY_TO_WAVELENGTH_FACTOR)
        assert val.dtype == np.float64

    def test_scattering_factor_dtype(self) -> None:
        val = np.array(xconst.SCATTERING_FACTOR)
        assert val.dtype == np.float64

    def test_thompson_dtype(self) -> None:
        val = np.array(xconst.THOMPSON)
        assert val.dtype == np.float64
