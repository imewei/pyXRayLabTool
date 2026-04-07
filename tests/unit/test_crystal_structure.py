"""
Tests for the CrystalStructure class in xraylabtool.calculators.scattering_data.

Physics background
------------------
The structure factor for a set of atoms at fractional positions r_j = (x_j, y_j, z_j)
and Miller indices (h, k, l) is:

    F(hkl) = Σ_j  f_j · occ_j · exp(2πi (h·x_j + k·y_j + l·z_j))

where f_j is the atomic form factor evaluated at the relevant q.

In these unit tests we use f_j = 1 for simplicity (the placeholder
`get_scattering_factor` already returns ones), which lets us isolate the
geometric phase factor.

Notes on the current implementation
-------------------------------------
`calculate_structure_factor` is a stub that always returns complex(1, 0).
Tests that verify physically meaningful structure factors are marked
``xfail(strict=False)`` so they document the required behaviour and will
automatically start passing once the real implementation is merged.
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from xraylabtool.calculators.scattering_data import CrystalStructure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cubic(a: float = 4.0) -> CrystalStructure:
    """Return a simple cubic CrystalStructure with lattice constant *a* Å."""
    return CrystalStructure((a, a, a, 90.0, 90.0, 90.0))


def _phase(h: int, k: int, miller_l: int, x: float, y: float, z: float) -> complex:
    """Return the geometric phase factor exp(2*pi*i*(hx + ky + lz))."""
    return cmath.exp(2j * math.pi * (h * x + k * y + miller_l * z))


# ---------------------------------------------------------------------------
# Structural / API tests (these pass against the current stub)
# ---------------------------------------------------------------------------


class TestCrystalStructureAPI:
    """Verify the public interface of CrystalStructure."""

    def test_instantiation_stores_lattice_parameters(self):
        cs = CrystalStructure((3.5, 3.5, 3.5, 90.0, 90.0, 90.0))
        assert cs.a == pytest.approx(3.5)
        assert cs.b == pytest.approx(3.5)
        assert cs.c == pytest.approx(3.5)
        assert cs.alpha == pytest.approx(90.0)
        assert cs.beta == pytest.approx(90.0)
        assert cs.gamma == pytest.approx(90.0)

    def test_atoms_list_initially_empty(self):
        cs = _make_cubic()
        assert cs.atoms == []

    def test_add_atom_appends_to_atoms_list(self):
        cs = _make_cubic()
        cs.add_atom("Si", (0.0, 0.0, 0.0))
        assert len(cs.atoms) == 1

    def test_add_atom_stores_element_and_position(self):
        cs = _make_cubic()
        cs.add_atom("Fe", (0.5, 0.5, 0.5), occupancy=0.8)
        atom = cs.atoms[0]
        assert atom["element"] == "Fe"
        assert atom["position"] == (0.5, 0.5, 0.5)
        assert atom["occupancy"] == pytest.approx(0.8)

    def test_add_atom_default_occupancy_is_one(self):
        cs = _make_cubic()
        cs.add_atom("O", (0.25, 0.25, 0.25))
        assert cs.atoms[0]["occupancy"] == pytest.approx(1.0)

    def test_add_multiple_atoms(self):
        cs = _make_cubic()
        for pos in [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
            cs.add_atom("Cu", pos)
        assert len(cs.atoms) == 4

    def test_calculate_structure_factor_returns_complex(self):
        cs = _make_cubic()
        cs.add_atom("Si", (0.0, 0.0, 0.0))
        result = cs.calculate_structure_factor((1, 0, 0))
        assert isinstance(result, complex)

    def test_calculate_structure_factor_no_atoms_returns_zero_or_complex(self):
        """An empty structure should return a complex number (zero or stub)."""
        cs = _make_cubic()
        result = cs.calculate_structure_factor((0, 0, 0))
        assert isinstance(result, complex)


# ---------------------------------------------------------------------------
# Physics tests — marked xfail until the real implementation lands
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="calculate_structure_factor is a placeholder; physics tests will "
    "pass once the real summation over phase factors is implemented",
)
class TestStructureFactorPhysics:
    """Physics-correct structure factor tests.

    All tests use f_j = 1 (the current placeholder behaviour of
    AtomicScatteringFactor.get_scattering_factor) to isolate the geometric
    phase contribution.  When f_j = 1, the structure factor reduces to the
    pure geometric factor:

        F(hkl) = Σ_j occ_j · exp(2πi(h·x_j + k·y_j + l·z_j))
    """

    def test_single_atom_at_origin_equals_form_factor(self):
        """F(hkl) = f · exp(0) = f for a single atom at the origin.

        With f = 1 the structure factor magnitude must equal 1 for any (hkl).
        """
        cs = _make_cubic()
        cs.add_atom("Si", (0.0, 0.0, 0.0))

        for hkl in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (0, 2, 2)]:
            F = cs.calculate_structure_factor(hkl)
            assert abs(F) == pytest.approx(1.0, abs=1e-10), (
                f"Single atom at origin: |F{hkl}| should be 1, got {abs(F)}"
            )

    def test_bcc_constructive_interference(self):
        """BCC: constructive interference when h+k+l is even.

        BCC basis: atom 1 at (0,0,0), atom 2 at (1/2,1/2,1/2).
        F(hkl) = f · (1 + exp(iπ(h+k+l)))
        When h+k+l is even: F = 2f  →  |F| = 2
        """
        cs = _make_cubic()
        cs.add_atom("Fe", (0.0, 0.0, 0.0))
        cs.add_atom("Fe", (0.5, 0.5, 0.5))

        constructive_cases = [(1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 2, 0)]
        for hkl in constructive_cases:
            F = cs.calculate_structure_factor(hkl)
            assert abs(F) == pytest.approx(2.0, abs=1e-10), (
                f"BCC constructive: |F{hkl}| should be 2, got {abs(F)}"
            )

    def test_bcc_destructive_interference(self):
        """BCC: destructive interference (systematic absence) when h+k+l is odd.

        When h+k+l is odd: F = f * (1 - 1) = 0 -> |F| = 0
        """
        cs = _make_cubic()
        cs.add_atom("Fe", (0.0, 0.0, 0.0))
        cs.add_atom("Fe", (0.5, 0.5, 0.5))

        forbidden = [(1, 0, 0), (0, 1, 0), (1, 1, 1), (3, 0, 0)]
        for hkl in forbidden:
            F = cs.calculate_structure_factor(hkl)
            assert abs(F) == pytest.approx(0.0, abs=1e-10), (
                f"BCC systematic absence: |F{hkl}| should be 0, got {abs(F)}"
            )

    def test_fcc_systematic_absences(self):
        """FCC: systematic absences when h,k,l are mixed (not all even or all odd).

        FCC basis atoms: (0,0,0), (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2)
        Allowed reflections: h,k,l all even or all odd  →  |F| = 4
        Forbidden: mixed parity  →  |F| = 0
        """
        cs = _make_cubic()
        fcc_positions = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ]
        for pos in fcc_positions:
            cs.add_atom("Cu", pos)

        # Allowed reflections
        allowed = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]
        for hkl in allowed:
            F = cs.calculate_structure_factor(hkl)
            assert abs(F) == pytest.approx(4.0, abs=1e-10), (
                f"FCC allowed: |F{hkl}| should be 4, got {abs(F)}"
            )

        # Forbidden (mixed-parity) reflections
        forbidden = [(1, 0, 0), (1, 1, 0), (2, 1, 0), (3, 0, 0)]
        for hkl in forbidden:
            F = cs.calculate_structure_factor(hkl)
            assert abs(F) == pytest.approx(0.0, abs=1e-10), (
                f"FCC systematic absence: |F{hkl}| should be 0, got {abs(F)}"
            )

    def test_structure_factor_phase_nontrivial(self):
        """Verify the phase of F(hkl) for a single off-origin atom.

        For one atom at position (x, y, z) with f = 1:
            F(hkl) = exp(2πi(hx + ky + lz))

        The phase must match the analytic formula exactly.
        """
        x, y, z = 0.25, 0.1, 0.4
        cs = _make_cubic()
        cs.add_atom("C", (x, y, z))

        test_cases = [
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (1, 1, 1),
            (2, 3, 1),
        ]
        for hkl in test_cases:
            h, k, ml = hkl
            expected = _phase(h, k, ml, x, y, z)
            F = cs.calculate_structure_factor(hkl)
            assert abs(F - expected) == pytest.approx(0.0, abs=1e-10), (
                f"Phase mismatch for {hkl}: expected {expected}, got {F}"
            )

    def test_structure_factor_with_partial_occupancy(self):
        """Partial occupancy scales the atomic contribution linearly.

        Two atoms at the same position with occupancy 0.5 each should give
        the same structure factor as one atom with full occupancy.
        """
        cs_full = _make_cubic()
        cs_full.add_atom("Si", (0.0, 0.0, 0.0), occupancy=1.0)

        cs_half = _make_cubic()
        cs_half.add_atom("Si", (0.0, 0.0, 0.0), occupancy=0.5)
        cs_half.add_atom("Si", (0.0, 0.0, 0.0), occupancy=0.5)

        for hkl in [(1, 0, 0), (1, 1, 1), (2, 2, 0)]:
            F_full = cs_full.calculate_structure_factor(hkl)
            F_half = cs_half.calculate_structure_factor(hkl)
            assert abs(F_full - F_half) == pytest.approx(0.0, abs=1e-10), (
                f"Occupancy scaling failed for {hkl}: "
                f"full={F_full:.4f}, half+half={F_half:.4f}"
            )
