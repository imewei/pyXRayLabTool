"""Tests for the Qt-free KK compute service (``xraylabtool.gui.services``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Importing xraylabtool.gui.services runs xraylabtool.gui.__init__, which imports
# Qt; skip on headless runners without libEGL (as test_gui_services.py does).
pytest.importorskip("PySide6.QtWidgets")

from xraylabtool.calculators.core import calculate_single_material_properties
from xraylabtool.gui.services import KKComputation, compute_kk
from xraylabtool.kk import kk_calculate_real, load_nearedge

# Carbon K-edge style NEXAFS rows (eV, absorption).
_EDGE_ROWS_EV = [(280.0, 0.05), (290.0, 0.1), (300.0, 0.3), (310.0, 0.45), (320.0, 0.5)]
_ENERGIES = [0.1, 0.29, 0.3, 1.0]


def _write_edge(path: Path, *, kev: bool = False) -> Path:
    scale = 1e-3 if kev else 1.0
    path.write_text("\n".join(f"{e * scale!r} {v!r}" for e, v in _EDGE_ROWS_EV) + "\n")
    return path


@pytest.mark.unit
def test_compute_kk_service_matches_api_without_nearedge() -> None:
    comp = compute_kk("Si", None, None, "eV", None, None, energies_kev=[1.0, 2.0, 5.0])
    ref = kk_calculate_real("Si", energies_kev=[1.0, 2.0, 5.0])
    assert isinstance(comp, KKComputation)
    np.testing.assert_allclose(comp.result.f1, ref.f1, rtol=1e-12)
    np.testing.assert_allclose(comp.result.f2, ref.f2, rtol=1e-12)
    assert comp.result.nearedge_scaled_kev_f2 is None
    # CXRO f1 totals on the same energies, independent of density.
    cxro = calculate_single_material_properties("Si", np.array([1.0, 2.0, 5.0]), 1.0)
    np.testing.assert_allclose(comp.cxro_energy_kev, [1.0, 2.0, 5.0])
    np.testing.assert_allclose(comp.cxro_f1, cxro.scattering_factor_f1, rtol=1e-12)
    assert np.isfinite(comp.cxro_f1).all()


@pytest.mark.unit
def test_compute_kk_service_matches_api_with_nearedge(tmp_path: Path) -> None:
    edge = _write_edge(tmp_path / "edge.txt")
    comp = compute_kk(
        "C", str(edge), "nexafs", "eV", None, None, energies_kev=_ENERGIES
    )
    ref = kk_calculate_real(
        "C",
        energies_kev=_ENERGIES,
        nearedge=load_nearedge(edge),
        input_type="nexafs",
    )
    np.testing.assert_allclose(comp.result.f1, ref.f1, rtol=1e-12)
    np.testing.assert_allclose(comp.result.f2, ref.f2, rtol=1e-12)
    assert comp.result.nearedge_scaled_kev_f2 is not None


@pytest.mark.unit
def test_compute_kk_merge_points_follow_energy_unit(tmp_path: Path) -> None:
    ev_file = _write_edge(tmp_path / "ev.txt")
    kev_file = _write_edge(tmp_path / "kev.txt", kev=True)
    a = compute_kk(
        "C", str(ev_file), "nexafs", "eV", (285.0, 315.0), None, energies_kev=_ENERGIES
    )
    b = compute_kk(
        "C",
        str(kev_file),
        "nexafs",
        "keV",
        (0.285, 0.315),
        None,
        energies_kev=_ENERGIES,
    )
    assert a.result.merge_points_kev == pytest.approx((0.285, 0.315))
    np.testing.assert_allclose(a.result.f1, b.result.f1, rtol=1e-12)


@pytest.mark.unit
def test_compute_kk_forwards_energies_kev() -> None:
    grid = np.array([0.5, 1.5, 3.0, 8.0])
    comp = compute_kk("SiO2", None, None, "eV", None, None, energies_kev=grid)
    np.testing.assert_allclose(comp.result.energy_kev, grid)
    np.testing.assert_allclose(comp.cxro_energy_kev, grid)


@pytest.mark.unit
def test_compute_kk_forwards_refine_tolerance() -> None:
    comp = compute_kk("Si", None, None, "eV", None, 1e-3, energies_kev=[1.0, 2.0])
    assert comp.result.n_refined_points > 0
    assert comp.cxro_energy_kev.size == comp.result.energy_kev.size


@pytest.mark.unit
def test_cxro_comparison_empty_outside_tabulated_range() -> None:
    comp = compute_kk("Si", None, None, "eV", None, None, energies_kev=[0.02, 100.0])
    assert comp.cxro_energy_kev.size == 0
    assert comp.cxro_f1.size == 0
    assert comp.result.energy_kev.size == 2


@pytest.mark.unit
def test_cxro_comparison_restricted_to_tabulated_range() -> None:
    comp = compute_kk(
        "Si", None, None, "eV", None, None, energies_kev=[0.02, 1.0, 100.0]
    )
    np.testing.assert_allclose(comp.cxro_energy_kev, [1.0])
    assert comp.cxro_f1.shape == (1,)


@pytest.mark.unit
def test_compute_kk_default_grid_cxro_matches_result_energies() -> None:
    comp = compute_kk("Si", None, None, "eV", None, None)
    assert comp.cxro_energy_kev.size > 0
    assert comp.cxro_f1.shape == comp.cxro_energy_kev.shape
    assert np.isfinite(comp.cxro_f1).all()


@pytest.mark.unit
def test_kk_f1_agrees_with_cxro_f1_for_sio2() -> None:
    """Independent CXRO check; also guards an energy/order mismatch."""
    comp = compute_kk("SiO2", None, None, "eV", None, None, energies_kev=[1.0, 5.0])
    np.testing.assert_allclose(comp.cxro_energy_kev, [1.0, 5.0])
    np.testing.assert_allclose(comp.result.f1, comp.cxro_f1, rtol=5e-3)


@pytest.mark.unit
def test_compute_kk_rejects_unknown_energy_unit() -> None:
    with pytest.raises(ValueError, match="energy_unit"):
        compute_kk("Si", None, None, "MeV", None, None)
