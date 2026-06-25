"""Unit tests for xraylabtool.gui.services (Qt-free helper layer)."""

from __future__ import annotations

import pytest

# services.py is logically Qt-free, but importing it goes through
# xraylabtool.gui.__init__, which eagerly imports the Qt launcher. Skip on
# headless runners without the Qt stack (libEGL), matching the other GUI tests.
pytest.importorskip("PySide6.QtWidgets")

from xraylabtool.gui.services import EnergyConfig, compute_multiple


def test_duplicate_formula_different_density_not_dropped() -> None:
    """Regression: same formula at two densities must both survive.

    compute_multiple used to key results solely by formula string, so a
    legitimate comparison (e.g. amorphous vs crystalline SiO2) silently
    overwrote the first material. Both must now be preserved.
    """
    cfg = EnergyConfig(8.0, 12.0, 3, False)
    results = compute_multiple(["SiO2", "SiO2"], [2.2, 2.65], cfg)

    assert len(results) == 2, "duplicate-formula material was silently dropped"
    densities = sorted(r.density_g_cm3 for r in results.values())
    assert densities == [2.2, 2.65]
