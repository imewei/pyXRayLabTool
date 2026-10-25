"""Tests for the vendored kkcalc ASF database."""

from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

import xraylabtool
from xraylabtool.kk.database import AsfDatabase, load_asf

ROOT = Path(__file__).resolve().parents[3]
NFF_DIR = Path(xraylabtool.__file__).parent / "data" / "AtomicScatteringFactor"

# Elements whose kkcalc (Henke) f2 differs from the repo's CXRO .nff f2.
EXPECTED_DIVERGENT = {"Sc", "Mo", "Ru", "Be", "Ti", "Si", "La", "Gd", "W"}


def test_asf_has_92_elements() -> None:
    db = load_asf()
    assert isinstance(db, AsfDatabase)
    assert len(db.symbols) == 92
    assert db.symbols[0] == "H"
    assert db.symbols[-1] == "U"
    assert db.mass.shape == (92,)
    assert np.isfinite(db.mass).all()


def test_load_asf_is_cached() -> None:
    assert load_asf() is load_asf()


def test_asf_grid_spans_10ev_to_500kev() -> None:
    db = load_asf()
    for z in range(1, 93):
        energy = db.energy_ev(z)
        assert energy[0] == 10.0
        assert energy[-1] == 500000.0
        assert np.all(np.diff(energy) > 0)


def test_im_coeffs_shape() -> None:
    db = load_asf()
    for z in range(1, 93):
        energy = db.energy_ev(z)
        coeffs = db.im_coeffs(z)
        assert coeffs.shape == (len(energy) - 1, 5)
        assert np.isfinite(coeffs).all()


def test_pyproject_package_data_lists_kk_files() -> None:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    patterns = cfg["tool"]["setuptools"]["package-data"]["xraylabtool"]
    assert "data/kkcalc/asf.npz" in patterns
    assert "data/kkcalc/LICENSE-kkcalc.txt" in patterns


def _asf_f2(db: AsfDatabase, z: int, energy: np.ndarray) -> np.ndarray:
    """Evaluate the ASF piecewise polynomial f2 (right-continuous segments)."""
    knots = db.energy_ev(z)
    seg = np.clip(np.searchsorted(knots, energy, side="right") - 1, 0, len(knots) - 2)
    c = db.im_coeffs(z)[seg]
    return (
        c[:, 0] * energy
        + c[:, 1]
        + c[:, 2] / energy
        + c[:, 3] / energy**2
        + c[:, 4] / energy**3
    )


def test_nff_divergent_elements() -> None:
    """Elements where the ASF f2 and the repo ``.nff`` f2 disagree.

    Procedure: for every element, read the ``.nff`` file (CXRO, 30 eV-30 keV)
    and evaluate the ASF piecewise polynomial f2 at the file's own tabulated
    energies, with no interpolation of the ``.nff`` data (PCHIP or linear
    interpolation alone differs by more than 1e-6 for every element). An
    element is divergent if any relative deviation
    ``|f2_asf - f2_nff| / |f2_nff|`` exceeds 1e-6. The set must equal
    ``{Sc, Mo, Ru, Be, Ti, Si, La, Gd, W}``.
    """
    db = load_asf()
    divergent: set[str] = set()
    for z, symbol in enumerate(db.symbols, start=1):
        table = np.genfromtxt(
            NFF_DIR / f"{symbol.lower()}.nff", delimiter=",", skip_header=1
        )
        energy, f2_nff = table[:, 0], table[:, 2]
        f2_asf = _asf_f2(db, z, energy)
        rel = np.abs(f2_asf - f2_nff) / np.abs(f2_nff)
        if np.nanmax(rel) > 1e-6:
            divergent.add(symbol)
    assert divergent == EXPECTED_DIVERGENT
