"""Host wrappers of the device programs in :mod:`xraylabtool.calculators.optics`.

Each wrapper pads its inputs to the shape buckets (so a handful of compiled
programs serve every call), makes **one** :func:`xraylabtool.backend.runtime.upload`,
runs one program, makes **one** :func:`xraylabtool.backend.runtime.fetch` and
slices the padding off. Every call checks the scattering tables are device
arrays and that it made exactly one upload and one fetch.

Padded material rows are ``z=0, counts=0, mw=1, rho=1, n_electrons=0`` (finite,
contribute nothing); padded energies repeat the last real energy so no NaN enters.

This module imports ``jax`` at module level, so callers import it inside
functions; importing it has no side effect (no ``ensure_x64()``).

References
----------
Design: ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md`` §1 and §3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np

from xraylabtool.backend import runtime
from xraylabtool.backend.shapes import bucket, bucket_pow2
from xraylabtool.calculators import optics
from xraylabtool.calculators.tables import ScatteringTables, load_tables

__all__ = [
    "run_derived",
    "run_optical",
    "run_rows",
    "run_scattering",
    "run_tail",
]

Encoded = tuple[np.ndarray, np.ndarray, float, float]


def _device_tables() -> ScatteringTables:
    tables = load_tables()
    if not all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(tables)):
        raise TypeError("the scattering tables must be device arrays (jax.Array)")
    return tables


def _run(program: Callable[[Any], Any], payload: Any) -> Any:
    """Upload ``payload``, run ``program`` on the device tree, fetch the result."""
    runtime.ensure_x64()
    with jax.enable_x64(True):
        before = runtime.thread_transfer_counts()
        device = runtime.upload(payload)
        with runtime.worker_guard():
            result = program(device)
        host = runtime.fetch(result)
        after = runtime.thread_transfer_counts()
    spent = (after["upload"] - before["upload"], after["fetch"] - before["fetch"])
    if spent != (1, 1):
        raise RuntimeError(f"expected 1 upload and 1 fetch, made {spent}")
    return host


def _pad_last(values: Any, width: int) -> np.ndarray:
    """``values`` (1-D) extended to ``width`` with its last entry (1.0 if empty)."""
    v = np.asarray(values, dtype=np.float64)
    fill = v[-1] if v.size else 1.0
    return np.concatenate([v, np.full(width - v.size, fill)])


def _width(m: int, m_bucket: int | None) -> int:
    return bucket(m, minimum=64) if m_bucket is None else m_bucket


def run_optical(
    encoded: Sequence[Encoded],
    densities: Sequence[float],
    energies: Any,
    *,
    k_bucket: int | None = None,
    f_bucket: int | None = None,
    m_bucket: int | None = None,
) -> dict[str, np.ndarray]:
    """Run ``_optical_program`` for ``F`` materials at shared ``energies`` (keV).

    Parameters
    ----------
    encoded : sequence of tuple
        ``encode_formula`` results ``(z, counts, mw, n_electrons)``, one per material.
    densities : sequence of float
        Mass densities in g/cm³.
    energies : array_like, shape ``(M,)``
        keV.
    k_bucket, f_bucket, m_bucket : int, optional
        Override the element, material and energy buckets (tests of padding
        invariance); by default ``bucket_pow2(k, 2)``, ``bucket_pow2(f, 1)`` and
        ``bucket(m, 64)``.

    Returns
    -------
    dict
        Unpadded ``out`` (``[F, 9, M]``, rows in ``OUT_FIELDS`` order),
        ``electron_density`` (``[F]``) and ``ok`` (``[F]``).

    Examples
    --------
    >>> from xraylabtool.calculators.host import run_optical
    >>> from xraylabtool.calculators.optics import encode_formula
    >>> got = run_optical([encode_formula("SiO2")], [2.2], [8.0, 9.0])
    >>> got["out"].shape, bool(got["ok"][0])
    ((1, 9, 2), True)
    """
    f = len(encoded)
    k_pad = max(len(z) for z, *_ in encoded)
    k_pad = bucket_pow2(k_pad, minimum=2) if k_bucket is None else k_bucket
    f_pad = bucket_pow2(f, minimum=1) if f_bucket is None else f_bucket
    z = np.zeros((f_pad, k_pad), dtype=np.int32)
    counts = np.zeros((f_pad, k_pad))
    mw = np.ones(f_pad)
    rho = np.ones(f_pad)
    n_electrons = np.zeros(f_pad)
    for i, ((zi, ci, mwi, nei), density) in enumerate(
        zip(encoded, densities, strict=True)
    ):
        z[i, : len(zi)] = zi
        counts[i, : len(ci)] = ci
        mw[i], n_electrons[i], rho[i] = mwi, nei, density
    e = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    m = e.size
    tables = _device_tables()
    got = _run(
        lambda dev: optics._optical_program(tables, dev[:5], dev[5]),
        (z, counts, mw, rho, n_electrons, _pad_last(e, _width(m, m_bucket))),
    )
    return {
        "out": got["out"][:f, :, :m],
        "electron_density": got["electron_density"][:f],
        "ok": got["ok"][:f],
    }


def run_tail(
    mw: Any,
    rho: Any,
    n_electrons: Any,
    f1: Any,
    f2: Any,
    energies: Any,
    *,
    f_bucket: int | None = None,
    m_bucket: int | None = None,
) -> dict[str, np.ndarray]:
    """Run ``_tail_program``: optical properties from supplied f1/f2 totals.

    Parameters
    ----------
    mw, rho, n_electrons : array_like, shape ``(F,)``
    f1, f2 : array_like, shape ``(F, M)``
    energies : array_like, shape ``(M,)``
        keV.
    f_bucket, m_bucket : int, optional
        As for :func:`run_optical`. Padded material rows are ``mw=1, rho=1,
        n_electrons=0`` with zero f1/f2.

    Returns
    -------
    dict
        Unpadded ``out`` (``[F, 9, M]``), ``electron_density`` and ``ok`` (``[F]``).
    """
    mw_a = np.atleast_1d(np.asarray(mw, dtype=np.float64))
    f = mw_a.size
    f_pad = bucket_pow2(f, minimum=1) if f_bucket is None else f_bucket
    e = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    m = e.size
    width = _width(m, m_bucket)

    def column(values: Any, fill: float) -> np.ndarray:
        v = np.atleast_1d(np.asarray(values, dtype=np.float64))
        return np.concatenate([v, np.full(f_pad - f, fill)])

    def rows(values: Any) -> np.ndarray:
        v = np.atleast_2d(np.asarray(values, dtype=np.float64))
        v = np.stack([_pad_last(r, width) for r in v])
        return np.vstack([v, np.zeros((f_pad - f, width))])

    got = _run(
        lambda dev: optics._tail_program(
            (dev[0], dev[1], dev[2]), dev[3], dev[4], dev[5]
        ),
        (
            column(mw_a, 1.0),
            column(rho, 1.0),
            column(n_electrons, 0.0),
            rows(f1),
            rows(f2),
            _pad_last(e, width),
        ),
    )
    return {
        "out": got["out"][:f, :, :m],
        "electron_density": got["electron_density"][:f],
        "ok": got["ok"][:f],
    }


def run_scattering(
    z: Any, counts: Any, e_ev: Any, wavelength: Any, factor: float
) -> np.ndarray:
    """Run ``_scattering_program`` for one material; returns ``[4, M]`` (δ, β, f1, f2)."""
    zs = np.asarray(z, dtype=np.int32)
    k_pad = bucket_pow2(zs.size, minimum=2)
    z_pad = np.zeros(k_pad, dtype=np.int32)
    c_pad = np.zeros(k_pad)
    z_pad[: zs.size] = zs
    c_pad[: zs.size] = np.asarray(counts, dtype=np.float64)
    e = np.atleast_1d(np.asarray(e_ev, dtype=np.float64))
    width = _width(e.size, None)
    tables = _device_tables()
    got = _run(
        lambda dev: optics._scattering_program(
            tables, (dev[0], dev[1]), dev[2], dev[3], dev[4]
        ),
        (
            z_pad,
            c_pad,
            _pad_last(e, width),
            _pad_last(np.atleast_1d(wavelength), width),
            np.float64(factor),
        ),
    )
    return np.asarray(got[:, : e.size])


def run_derived(
    wavelength: Any,
    dispersion: Any,
    absorption: Any,
    rho: float,
    mw: float,
    n_electrons: float,
) -> tuple[float, np.ndarray]:
    """Run ``_derived_program`` on equal-length 1-D arrays.

    Returns
    -------
    electron_density : float
        Electrons per Å³.
    out : ndarray, shape ``(4, M)``
        Critical angle, attenuation length, real and imaginary SLD.
    """
    wl = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
    width = _width(wl.size, None)
    got = _run(
        optics._derived_program,
        (
            _pad_last(wl, width),
            _pad_last(dispersion, width),
            _pad_last(absorption, width),
            np.float64(rho),
            np.float64(mw),
            np.float64(n_electrons),
        ),
    )
    return float(got["electron_density"]), np.asarray(got["out"][:, : wl.size])


def run_rows(z: int, e_ev: Any) -> np.ndarray:
    """f1 and f2 of table row ``z`` at ``e_ev`` (eV) as a host ``[2, M]`` array."""
    e = np.atleast_1d(np.asarray(e_ev, dtype=np.float64))
    tables = _device_tables()
    got = _run(
        lambda dev: optics._row_program(tables, dev[0], dev[1]),
        (np.int32(z), _pad_last(e, _width(e.size, None))),
    )
    return np.asarray(got[:, : e.size])
