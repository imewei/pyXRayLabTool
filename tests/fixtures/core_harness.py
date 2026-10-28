"""Thin host harness over the pure optics programs, for the tests of Task 1.

It encodes formulas and calls the production wrappers of
:mod:`xraylabtool.calculators.host` (pad to buckets, one ``runtime.upload``, the
program, one ``runtime.fetch``, slice off padding), so the tests exercise the
programs the way production calls them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from xraylabtool.calculators import host, optics

# Result key of the golden files for each row of the program's ``out``.
GOLDEN_FIELDS = (
    "wavelength_angstrom",
    "dispersion_delta",
    "absorption_beta",
    "scattering_factor_f1",
    "scattering_factor_f2",
    "critical_angle_degrees",
    "attenuation_length_cm",
    "real_sld_per_ang2",
    "imaginary_sld_per_ang2",
)
assert len(GOLDEN_FIELDS) == len(optics.OUT_FIELDS)


def run_program(
    materials: Sequence[tuple[str, float]],
    energies: Any,
    *,
    k_bucket: int | None = None,
    f_bucket: int | None = None,
    m_bucket: int | None = None,
) -> dict[str, Any]:
    """Run ``_optical_program`` for ``materials`` and return the unpadded result.

    Returns a dict with ``out`` (``[F, 9, M]``), ``electron_density`` (``[F]``),
    ``ok`` (``[F]``) and the host scalars ``mw``, ``n_electrons`` (``[F]``).
    """
    encoded = [optics.encode_formula(f) for f, _ in materials]
    got = host.run_optical(
        encoded,
        [density for _, density in materials],
        energies,
        k_bucket=k_bucket,
        f_bucket=f_bucket,
        m_bucket=m_bucket,
    )
    return {
        **got,
        "mw": np.array([e[2] for e in encoded]),
        "n_electrons": np.array([e[3] for e in encoded]),
    }


def run_tail(
    formula: str,
    density: float,
    energies: Any,
    f1: Any,
    f2: Any,
    *,
    m_bucket: int | None = None,
    f_bucket: int = 1,
) -> dict[str, Any]:
    """Run ``_tail_program`` for one material with supplied f1/f2 (row 0 of ``f_bucket``)."""
    _, _, mw, ne = optics.encode_formula(formula)
    got = host.run_tail(
        [mw],
        [density],
        [ne],
        np.atleast_2d(f1),
        np.atleast_2d(f2),
        energies,
        f_bucket=f_bucket,
        m_bucket=m_bucket,
    )
    return got
