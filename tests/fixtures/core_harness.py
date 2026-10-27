"""Thin host harness over the pure optics programs, for the tests of Task 1.

It does what the public wrappers of Task 2 do (encode, pad to buckets, one
``runtime.upload``, the program under a ``disallow_explicit`` transfer guard,
one ``runtime.fetch``, slice off padding), so the tests exercise the programs
the way production calls them. Padded material rows are ``z=0, counts=0,
mw=1, rho=1`` and padded energies repeat the last real energy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from xraylabtool.backend import runtime
from xraylabtool.backend.shapes import bucket, bucket_pow2
from xraylabtool.calculators import optics
from xraylabtool.calculators.tables import load_tables

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


def _energies(energies: Any, m_bucket: int | None) -> tuple[np.ndarray, int]:
    e = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    m = e.size
    width = bucket(m, minimum=64) if m_bucket is None else m_bucket
    return np.concatenate([e, np.full(width - m, e[-1])]), m


def _fetch(program: Any, build: Any, upload: Any) -> Any:
    runtime.ensure_x64()
    with jax.enable_x64(True):
        device = runtime.upload(upload)
        with jax.transfer_guard("disallow_explicit"):
            result = program(*build(device))
        return runtime.fetch(result)


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
    f = len(materials)
    k = max(len(z) for z, *_ in encoded)
    k_pad = bucket_pow2(k, minimum=2) if k_bucket is None else k_bucket
    f_pad = bucket_pow2(f, minimum=1) if f_bucket is None else f_bucket
    z = np.zeros((f_pad, k_pad), dtype=np.int32)
    counts = np.zeros((f_pad, k_pad))
    mw = np.ones(f_pad)
    rho = np.ones(f_pad)
    ne = np.zeros(f_pad)
    for i, ((zi, ci, mwi, nei), (_, density)) in enumerate(
        zip(encoded, materials, strict=True)
    ):
        z[i, : len(zi)] = zi
        counts[i, : len(ci)] = ci
        mw[i], ne[i], rho[i] = mwi, nei, density
    e_pad, m = _energies(energies, m_bucket)
    tables = load_tables()
    got = _fetch(
        optics._optical_program,
        lambda dev: (tables, dev[:5], dev[5]),
        (z, counts, mw, rho, ne, e_pad),
    )
    return {
        "out": got["out"][:f, :, :m],
        "electron_density": got["electron_density"][:f],
        "ok": got["ok"][:f],
        "mw": mw[:f],
        "n_electrons": ne[:f],
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
    e_pad, m = _energies(energies, m_bucket)
    pad = e_pad.size - m

    def padded(v: Any) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        row = np.concatenate([v, np.full(pad, v[-1])])
        return np.vstack([row, np.zeros((f_bucket - 1, row.size))])

    def column(value: float, fill: float) -> np.ndarray:
        return np.array([value] + [fill] * (f_bucket - 1))

    got = _fetch(
        optics._tail_program,
        lambda dev: ((dev[0], dev[1], dev[2]), dev[3], dev[4], dev[5]),
        (
            column(mw, 1.0),
            column(density, 1.0),
            column(ne, 0.0),
            padded(f1),
            padded(f2),
            e_pad,
        ),
    )
    return {
        "out": got["out"][:1, :, :m],
        "electron_density": got["electron_density"][:1],
        "ok": got["ok"][:1],
    }
