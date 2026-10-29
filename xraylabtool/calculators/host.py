"""Host wrappers of the device programs in :mod:`xraylabtool.calculators.optics`.

Each wrapper pads its inputs to the shape buckets (so a handful of compiled
programs serve every call), makes **one** :func:`xraylabtool.backend.runtime.upload`,
runs one program, makes **one** :func:`xraylabtool.backend.runtime.fetch` and
slices the padding off. Every call goes through :func:`execute`, which checks that
it made exactly one upload and one fetch; the wrappers that read the scattering
tables (``run_optical``, ``run_scattering``, ``run_rows``) also check they are
device arrays.

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
import functools
from typing import Any

import jax
import numpy as np
from numpy.typing import ArrayLike, NDArray

from xraylabtool.backend import runtime
from xraylabtool.backend.shapes import block_for, bucket_energy, bucket_pow2
from xraylabtool.calculators import optics
from xraylabtool.calculators.tables import ScatteringTables, load_tables

# A 0-d field comes back as a NumPy scalar, any other shape as an array.
_Field = NDArray[np.float64] | np.float64

__all__ = [
    "broadcast_shape",
    "check_molecular_weight",
    "device_tables",
    "execute",
    "restrict_to",
    "run_columns",
    "run_derived_fields",
    "run_map",
    "run_optical",
    "run_optical_blocks",
    "run_rows",
    "run_scattering",
    "run_series",
    "run_tail",
]

Encoded = tuple[np.ndarray, np.ndarray, float, float]


def device_tables() -> ScatteringTables:
    """The resident tables, checked to be device arrays."""
    tables = load_tables()
    if not all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(tables)):
        raise TypeError("the scattering tables must be device arrays (jax.Array)")
    return tables


def execute(program: Callable[[Any], Any], payload: Any) -> Any:
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
    return bucket_energy(m) if m_bucket is None else m_bucket


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
        ``bucket_energy(m)``.

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
    # One flat float64 buffer (material rows, then the energies): one leaf
    # uploads in ~26 us, six in ~108 us. The row layout lives in ``optics``.
    e = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    m = e.size
    rows = optics.pack_materials(encoded, densities, f_pad, k_pad)
    packed = np.concatenate([rows.ravel(), _pad_last(e, _width(m, m_bucket))])
    tables = device_tables()
    got = execute(
        lambda dev: optics._optical_program(tables, dev, f_pad, k_pad), packed
    )
    return {
        "out": got["out"][:f, :, :m],
        "electron_density": got["electron_density"][:f],
        "ok": got["ok"][:f],
    }


def run_optical_blocks(
    encoded: Sequence[Encoded],
    densities: Sequence[float],
    energies: Any,
    *,
    max_block: int | None = None,
    on_block: Callable[[int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run ``_optical_program`` over any number of materials as ``F``-blocks.

    Materials are grouped by element bucket ``K`` (so a one-element material is
    never padded to a six-element one) and each group is cut into blocks of
    :func:`~xraylabtool.backend.shapes.block_for` materials, so every block
    keeps the padded ``F·K·M`` within the memory budget; blocks of one group share a shape
    (up to the last, smaller one) and so one compiled program.
    Each block costs one upload and one fetch; its rows are written back to the
    positions of the input, so the output order equals the input order.

    Parameters
    ----------
    encoded : sequence of tuple
        ``encode_formula`` results, one per material.
    densities : sequence of float
        Mass densities in g/cm³.
    energies : array_like, shape ``(M,)``
        keV, shared by all materials.
    max_block : int, optional
        Upper bound of materials per block (``BatchConfig.chunk_size``).
    on_block : callable, optional
        Called with the number of materials after each block (progress).

    Returns
    -------
    dict
        ``out`` (``[N, 9, M]``), ``electron_density`` and ``ok`` (``[N]``), as
        :func:`run_optical` but for all ``N`` materials; empty for ``N = 0``.
    """
    n = len(encoded)
    e = np.atleast_1d(np.asarray(energies, dtype=np.float64))
    m_pad = _width(e.size, None)
    out = np.zeros((n, len(optics.OUT_FIELDS), e.size))
    electron_density = np.zeros(n)
    ok = np.zeros(n, dtype=bool)  # a row no block wrote reads as not ok
    groups: dict[int, list[int]] = {}
    for i, (z, *_rest) in enumerate(encoded):
        groups.setdefault(bucket_pow2(len(z), minimum=2), []).append(i)
    for k_pad, members in groups.items():
        # ``block_for`` rounds down, so ask for the padded count: a group that fits
        # the memory budget is one block, a larger one is cut at the budget.
        # Exception to the budget: ``block_for`` never returns less than 1, so one
        # material whose own ``K·M`` exceeds the budget (e.g. 9 elements at
        # ~300k energies, 65 at ~40k) still runs as an F=1 block, about 1.3x over the
        # planning bound. A material cannot be split without a second program call
        # (spec section 1); see "Known limitations" in ADR-006.
        block = block_for(None, k_pad * m_pad, bucket_pow2(len(members)))
        # A ladder rung (640, 800, ...) would be padded up to the next power of two
        # by run_optical and exceed the budget, so floor to the power of two; a cap
        # below it then pads to at most that power of two.
        block = 1 << (block.bit_length() - 1)
        if max_block is not None:
            if max_block < 1:
                raise ValueError(f"max_block must be at least 1, got {max_block}")
            block = min(block, max_block)
        for start in range(0, len(members), block):
            rows = members[start : start + block]
            got = run_optical(
                [encoded[i] for i in rows],
                [densities[i] for i in rows],
                e,
                k_bucket=k_pad,
            )
            out[rows], electron_density[rows], ok[rows] = (
                got["out"],
                got["electron_density"],
                got["ok"],
            )
            if on_block is not None:
                on_block(len(rows))
    return {"out": out, "electron_density": electron_density, "ok": ok}


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

    got = execute(
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


def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast shape of ``shapes`` with NumPy's own ``ValueError`` on a mismatch.

    The legacy kernels multiplied the operand arrays, so an incompatible pair raised
    ``operands could not be broadcast together with shapes (a,) (b,)``; multiplying
    zero-stride views raises the identical error without allocating the operands.
    """
    out = np.broadcast_to(np.float64(0.0), shapes[0])
    for shape in shapes[1:]:
        out = np.multiply(out, np.broadcast_to(np.float64(0.0), shape))
    return tuple(np.shape(out))


def restrict_to(full: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The part of ``full`` that broadcasts to ``shape`` (size-1 axes keep entry 0).

    ``full`` is a result on a broadcast shape whose value does not depend on the axes
    ``shape`` lacks or has as size 1; the legacy kernels returned it at ``shape``.
    """
    pad = (1,) * (full.ndim - len(shape)) + tuple(shape)
    index = tuple(slice(0, 1) if p == 1 else slice(None) for p in pad)
    return full[index].reshape(shape)


def check_molecular_weight(mass_density: Any, molecular_weight: Any) -> None:
    """Raise the legacy ``ZeroDivisionError`` for a zero molecular weight.

    The legacy code evaluated ``SCATTERING_FACTOR * mass_density / molecular_weight``
    on Python numbers, so a formula whose atom counts are all zero raised
    ``ZeroDivisionError("division by zero")``. A NumPy-scalar or array density gave
    ``inf`` and the non-finite status instead, which the device program still does.
    """
    if molecular_weight == 0 and not isinstance(mass_density, np.generic | np.ndarray):
        raise ZeroDivisionError("division by zero")


def run_scattering(
    z: Any,
    counts: Any,
    e_ev: Any,
    wavelength: Any,
    mass_density: float,
    molecular_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``_scattering_program`` for one material.

    Returns
    -------
    tuple of ndarray
        ``(dispersion, absorption, f1_total, f2_total)``. Energy and wavelength
        broadcast as the legacy kernel did (a mismatch raises NumPy's
        ``ValueError``): δ and β take the broadcast shape, f1 and f2 the shape of
        ``e_ev``.
    """
    zs = np.asarray(z, dtype=np.int32)
    k_pad = bucket_pow2(zs.size, minimum=2)
    z_pad = np.zeros(k_pad, dtype=np.int32)
    c_pad = np.zeros(k_pad)
    z_pad[: zs.size] = zs
    c_pad[: zs.size] = np.asarray(counts, dtype=np.float64)
    e = np.asarray(e_ev, dtype=np.float64)
    wl = np.asarray(wavelength, dtype=np.float64)
    shape = broadcast_shape(wl.shape, e.shape)
    e_flat = np.broadcast_to(e, shape).ravel()
    wl_flat = np.broadcast_to(wl, shape).ravel()
    width = _width(e_flat.size, None)
    tables = device_tables()
    got = execute(
        lambda dev: optics._scattering_program(
            tables, (dev[0], dev[1]), dev[2], dev[3], dev[4], dev[5]
        ),
        (
            z_pad,
            c_pad,
            _pad_last(e_flat, width),
            _pad_last(wl_flat, width),
            np.float64(mass_density),
            np.float64(molecular_weight),
        ),
    )
    rows = [np.array(r[: e_flat.size]).reshape(shape) for r in got]
    dispersion, absorption = rows[0], rows[1]
    return (
        dispersion,
        absorption,
        restrict_to(rows[2], e.shape),
        restrict_to(rows[3], e.shape),
    )


def run_derived_fields(
    wavelength: ArrayLike,
    dispersion: ArrayLike,
    absorption: ArrayLike,
    rho: float,
    mw: float,
    n_electrons: float,
) -> tuple[float, tuple[_Field, _Field, _Field, _Field]]:
    """Run derived quantities on each field's own broadcast grid in one call.

    Critical angle depends only on dispersion; attenuation and imaginary SLD
    share the wavelength/absorption grid; real SLD uses wavelength/dispersion.
    Each grid is flattened and bucketed independently, retaining scalar and
    empty outputs without requiring dispersion and absorption to broadcast.

    Parameters
    ----------
    wavelength : array_like
        X-ray wavelengths in metres.
    dispersion, absorption : array_like
        Dimensionless optical coefficients delta and beta. Each must broadcast
        with ``wavelength``; they need not broadcast with each other.
    rho : float
        Material mass density in g/cm³.
    mw : float
        Molecular weight in g/mol.
    n_electrons : float
        Number of electrons per formula unit.

    Returns
    -------
    electron_density : float
        Electron density in electrons/Å³.
    fields : tuple of numpy.ndarray
        Critical angle in degrees, attenuation length in cm, real SLD in Å⁻²,
        and imaginary SLD in Å⁻², in that order. Critical angle has the shape of
        ``dispersion``; attenuation length and imaginary SLD have the broadcast
        shape of ``wavelength`` and ``absorption``; real SLD has the broadcast
        shape of ``wavelength`` and ``dispersion``. Scalar fields are NumPy
        scalars; empty field shapes are preserved.
    """
    delta = np.asarray(dispersion, dtype=np.float64)
    wl_beta, beta = np.broadcast_arrays(wavelength, absorption)
    wl_delta, delta_sld = np.broadcast_arrays(wavelength, delta)
    grids = (delta, wl_beta, beta, wl_delta, delta_sld)
    # One common width, so the program is compiled per bucket, not per shape mix.
    width = max(_width(a.size, None) for a in grids)
    payload = tuple(_pad_last(a.ravel(), width) for a in grids)
    ed, fields = execute(
        optics._derived_fields_program,
        (*payload, np.float64(rho), np.float64(mw), np.float64(n_electrons)),
    )
    shapes = (delta.shape, wl_beta.shape, wl_delta.shape, wl_beta.shape)
    critical, attenuation, re_sld, im_sld = (
        np.array(field[: int(np.prod(shape))]).reshape(shape)[()]
        for field, shape in zip(fields, shapes, strict=True)
    )
    return float(ed), (critical, attenuation, re_sld, im_sld)


@functools.cache
def _jitted(fn: Callable[..., Any]) -> Callable[..., Any]:
    return jax.jit(fn)


def run_map(fn: Callable[..., Any], *arrays: Any) -> Any:
    """Apply the elementwise function ``fn`` to host arrays on the device.

    The arrays are broadcast against each other, flattened, padded to the energy
    bucket (so one compile serves every length in it), run as one jitted call and
    reshaped to the broadcast shape. ``fn`` may return an array or a tuple of
    arrays; it must act elementwise.

    Returns
    -------
    ndarray or tuple of ndarray
        Host results with the broadcast shape.
    """
    flat = [np.asarray(a, dtype=np.float64) for a in np.broadcast_arrays(*arrays)]
    shape = flat[0].shape
    m = flat[0].size
    width = _width(m, None)
    got = execute(
        lambda dev: _jitted(fn)(*dev), tuple(_pad_last(a.ravel(), width) for a in flat)
    )
    return jax.tree_util.tree_map(lambda x: np.array(x[:m]).reshape(shape), got)


def run_series(
    fn: Callable[..., Any], *series: np.ndarray, extra: Sequence[Any] = ()
) -> np.ndarray:
    """Apply ``fn(*padded, n, *extra)`` to equally long 1-D host arrays on the device.

    Each series is padded to the energy bucket with its last entry (so one compile
    serves every length in it) and the true length ``n`` is the final argument;
    ``fn`` masks the padding. Returns the first ``n`` entries of the result.
    """
    n = series[0].size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    width = _width(n, None)
    padded = tuple(_pad_last(s, width) for s in series)
    got = execute(lambda dev: _jitted(fn)(*dev), (*padded, np.int32(n), *extra))
    return np.array(got[:n])


def run_columns(fn: Callable[..., Any], columns: np.ndarray) -> np.ndarray:
    """Apply ``fn(padded, n)`` to the series in the columns of ``columns`` (``[N, k]``).

    Rows are padded to the energy bucket and columns to a power of two with edge
    values (finite, and every column is independent), so one compile serves every
    shape in the bucket pair; one upload, one fetch. Returns the ``[N, k]`` result.
    """
    n, k = columns.shape
    padded = np.pad(
        columns,
        ((0, _width(n, None) - n), (0, bucket_pow2(k, minimum=2) - k)),
        mode="edge",
    )
    got = execute(lambda dev: _jitted(fn)(*dev), (padded, np.int32(n)))
    return np.array(got[:n, :k])


def run_rows(z: int, e_ev: Any) -> np.ndarray:
    """f1 and f2 of table row ``z`` at ``e_ev`` (eV) as a host ``[2, M]`` array."""
    e = np.atleast_1d(np.asarray(e_ev, dtype=np.float64))
    tables = device_tables()
    got = execute(
        lambda dev: optics._row_program(tables, dev[0], dev[1]),
        (np.int32(z), _pad_last(e, _width(e.size, None))),
    )
    return np.array(got[:, : e.size])
