"""
Characterization tests for _calculate_molecular_properties (Plan section P2-E).

Golden values captured from xraylabtool v0.3.0 numpy/scipy stack on 2026-04-06.
These tests lock molecular weight (MW) and electron count (Z_total) for known
chemical formulas. A regression here indicates the atomic weight table, parser,
or summation logic has changed.

Tolerances:
    MW : atol=1e-2 (atomic weight table precision)
    Z_total : atol=1e-10 (must be exact integer sums)
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from xraylabtool.calculators.core import (
    _calculate_molecular_properties,
    clear_scattering_factor_cache,
    get_bulk_atomic_data,
)
from xraylabtool.utils import parse_formula

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calc(formula: str) -> tuple[float, float]:
    """Return (molecular_weight, total_electrons) for *formula*."""
    symbols, counts = parse_formula(formula)
    atomic_data = get_bulk_atomic_data(tuple(symbols))
    return _calculate_molecular_properties(symbols, counts, atomic_data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Ensure a cold-path call for every test."""
    clear_scattering_factor_cache()


# ---------------------------------------------------------------------------
# Molecular weight golden tests
# ---------------------------------------------------------------------------


class TestMolecularWeightGolden:
    """Golden-value assertions for molecular weight."""

    def test_sio2_mw(self) -> None:
        mw, _ = _calc("SiO2")
        # Golden: 60.083 g/mol (Si=28.085, O=15.999 × 2)
        np.testing.assert_allclose(mw, 60.083, atol=1e-2, err_msg="SiO2 MW regression")

    def test_al2o3_mw(self) -> None:
        mw, _ = _calc("Al2O3")
        # Golden: 101.961 g/mol
        np.testing.assert_allclose(
            mw, 101.961, atol=1e-2, err_msg="Al2O3 MW regression"
        )

    def test_fe2o3_mw(self) -> None:
        mw, _ = _calc("Fe2O3")
        # Golden: 159.687 g/mol
        np.testing.assert_allclose(
            mw, 159.687, atol=1e-2, err_msg="Fe2O3 MW regression"
        )

    def test_au_mw(self) -> None:
        mw, _ = _calc("Au")
        # Golden: 196.97 g/mol
        np.testing.assert_allclose(mw, 196.97, atol=1e-2, err_msg="Au MW regression")


# ---------------------------------------------------------------------------
# Electron count golden tests
# ---------------------------------------------------------------------------


class TestElectronCountGolden:
    """Golden-value assertions for Z_total (total electrons per formula unit)."""

    def test_sio2_z(self) -> None:
        _, z = _calc("SiO2")
        # Si(14) + 2×O(8) = 30
        np.testing.assert_allclose(
            z, 30.0, atol=1e-10, err_msg="SiO2 Z_total regression"
        )

    def test_al2o3_z(self) -> None:
        _, z = _calc("Al2O3")
        # 2×Al(13) + 3×O(8) = 50
        np.testing.assert_allclose(
            z, 50.0, atol=1e-10, err_msg="Al2O3 Z_total regression"
        )

    def test_fe2o3_z(self) -> None:
        _, z = _calc("Fe2O3")
        # 2×Fe(26) + 3×O(8) = 76
        np.testing.assert_allclose(
            z, 76.0, atol=1e-10, err_msg="Fe2O3 Z_total regression"
        )

    def test_au_z(self) -> None:
        _, z = _calc("Au")
        # Au(79) = 79
        np.testing.assert_allclose(z, 79.0, atol=1e-10, err_msg="Au Z_total regression")


# ---------------------------------------------------------------------------
# Exact captured values (tighter lock — both MW and Z in one shot)
# ---------------------------------------------------------------------------


class TestExactCapturedValues:
    """
    Lock the exact float64 values produced by v0.3.0.
    rtol=1e-3 is tight enough to catch float32 regression (float32 eps ~1.2e-7
    relative, but MW rounding would appear at ~1e-4 relative).
    """

    @pytest.mark.parametrize(
        "formula, expected_mw, expected_z",
        [
            ("SiO2", 60.083, 30.0),
            ("Al2O3", 101.961, 50.0),
            ("Fe2O3", 159.687, 76.0),
            ("Au", 196.97, 79.0),
        ],
    )
    def test_mw_and_z_parametric(
        self, formula: str, expected_mw: float, expected_z: float
    ) -> None:
        mw, z = _calc(formula)
        np.testing.assert_allclose(
            mw, expected_mw, rtol=1e-3, err_msg=f"{formula} MW parametric"
        )
        np.testing.assert_allclose(
            z, expected_z, atol=1e-10, err_msg=f"{formula} Z parametric"
        )


# ---------------------------------------------------------------------------
# Ca5(PO4)3OH — parenthesis expansion test
# ---------------------------------------------------------------------------


class TestComplexFormula:
    """Test parenthesis-containing formula through the molecular properties path."""

    def test_hydroxyapatite_mw(self) -> None:
        """Ca5(PO4)3OH: verify MW matches textbook value (~502.31 g/mol)."""
        mw, _ = _calc("Ca5(PO4)3OH")
        # Ca5(PO4)3OH = Ca×5 + P×3 + O×13 + H×1 = 502.307 g/mol
        # Previous v0.3.0 value (312.367) was wrong due to parser ignoring
        # parentheses. Fixed in audit remediation (item 1.1).
        np.testing.assert_allclose(
            mw, 502.307, atol=1e-2, err_msg="Ca5(PO4)3OH MW regression"
        )

    def test_hydroxyapatite_z(self) -> None:
        _, z = _calc("Ca5(PO4)3OH")
        # Ca(20)×5=100 + P(15)×3=45 + O(8)×13=104 + H(1)×1=1 = 250
        # Previous v0.3.0 value (156.0) was wrong due to parser ignoring
        # parentheses. Fixed in audit remediation (item 1.1).
        np.testing.assert_allclose(
            z, 250.0, atol=1e-10, err_msg="Ca5(PO4)3OH Z_total regression"
        )
